#include "toy/parser/Parser.h"
#include "ast/ASTWalker.h"
#include "toy/ast/ToyExpr.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

namespace toy {

Token *Parser::Advance() { return Tok = Lexer.GetNextToken(); }
void Parser::Skip() { Advance(); }
Token *Parser::Peek(unsigned lookAt) { return Lexer.PeekNextToken(lookAt); }
Token *Parser::PrevTok() { return Lexer.PeekPrevToken(); }

Module Parser::Parse() { return parseModule(); }

void Parser::recovery(bool allowFunction) {
  while (true) {
    auto peekTok = Peek();
    if (peekTok->is<Token::Tok_return, Token::Tok_var, Token::Tok_EOF>())
      return;

    if (allowFunction && peekTok->is<Token::Tok_def>())
      return;

    if (peekTok->is<Token::Tok_semicolon>()) {
      Skip();
      return;
    }

    Skip();
  }
}

Module Parser::parseModule() {
  FixLocationScope scope(*this);
  llvm::SmallVector<Stmt> stmts;
  bool fail = false;
  while (!Peek()->is<Token::Tok_EOF>()) {
    auto funDecl = parseFunctionOrStructDecl();
    if (!funDecl) {
      fail = true;
      recovery();
    }
    stmts.emplace_back(funDecl);
  }

  if (fail)
    return nullptr;
  return Module::create(scope.CreateRange(), context, stmts);
}

Stmt Parser::parseFunctionOrStructDecl() {
  if (Peek()->is<Token::Tok_def>())
    return parseFunctionDecl();

  if (Peek()->is<Token::Tok_struct>())
    return parseStructDecl();

  Report(Peek()->getRange(), Reporter::Diag::err_unexpected_token,
         Peek()->getSymbol());
  return nullptr;
}

FuncDecl Parser::parseFunctionDecl() {
  FixLocationScope scope(*this);
  if (Consume<Token::Tok_def>())
    return nullptr;

  auto id = Peek();
  if (Expect<Token::Tok_identifier>(id))
    return nullptr;
  Skip();

  if (Consume<Token::Tok_lparen>())
    return nullptr;

  auto params = parseParamList();
  if (!params)
    return nullptr;

  if (Consume<Token::Tok_rparen>())
    return nullptr;

  auto blockStmt = parseBlockStmt();
  if (!blockStmt)
    return nullptr;

  llvm::DenseMap<std::size_t, StructDecl> structTag;
  llvm::SmallVector<std::pair<std::optional<std::string>, std::string>>
      paramList;
  paramList.reserve(params->size());

  for (const auto &[idx, param] : llvm::enumerate(*params)) {
    const auto &[structOpt, name] = param;
    if (param.first) {
      if (auto iter = structTable.find(param.first->str());
          iter != structTable.end()) {
        structTag.try_emplace(idx, iter->second);
      } else {
        Report(id->getRange(), Reporter::Diag::err_undeclared_struct_type,
               param.first->str());
        return nullptr;
      }
    }

    paramList.emplace_back(
        structOpt ? std::optional(structOpt->str()) : std::nullopt, name.str());
  }

  auto funDecl = FuncDecl::create(scope.CreateRange(), context, id->getSymbol(),
                                  paramList, blockStmt);
  funDecl.setParamStructDeclsTag(structTag);
  return funDecl;
}

#define RECOVERY                                                               \
  fail = true;                                                                 \
  recovery();                                                                  \
  continue

StructDecl Parser::parseStructDecl() {
  FixLocationScope scope(*this);
  if (Consume<Token::Tok_struct>())
    return nullptr;

  auto id = Peek();
  if (Expect<Token::Tok_identifier>(id))
    return nullptr;
  Skip();
  auto structName = id->getSymbol();

  auto [iter, inserted] = structTable.try_emplace(structName, nullptr);
  if (!inserted) {
    Report(id->getRange(), Reporter::Diag::err_redeclared_struct_type,
           structName);
    return nullptr;
  }

  if (Consume<Token::Tok_lbrace>())
    return nullptr;

  llvm::SmallVector<std::string> fields;
  llvm::DenseMap<llvm::StringRef, llvm::SMRange> fieldTable;
  bool fail = false;

  while (!Peek()->is<Token::Tok_rbrace>() && !Peek()->is<Token::Tok_EOF>()) {
    if (Consume<Token::Tok_var>()) {
      RECOVERY;
    }
    auto fieldTok = Peek();
    if (Expect<Token::Tok_identifier>(fieldTok)) {
      RECOVERY;
    }
    auto range = fieldTok->getRange();
    Skip();

    if (Consume<Token::Tok_semicolon>()) {
      RECOVERY;
    }

    if (!fail) {
      auto [fieldIter, fieldInserted] =
          fieldTable.try_emplace(fieldTok->getSymbol(), range);
      if (!fieldInserted) {
        Report(range, Reporter::Diag::err_duplicated_field_name,
               fieldTok->getSymbol());
        Report(fieldIter->second,
               Reporter::Diag::note_previous_declared_field_name);
        RECOVERY;
      }
      fields.emplace_back(fieldTok->getSymbol().str());
    }
  }

  if (fail)
    return nullptr;

  if (Consume<Token::Tok_rbrace>())
    return nullptr;

  auto structDecl =
      StructDecl::create(scope.CreateRange(), context, structName, fields);
  return iter->second = structDecl;
}

#undef RECOVERY

BlockStmt Parser::parseBlockStmt() {
  FixLocationScope scope(*this);
  if (Consume<Token::Tok_lbrace>())
    return nullptr;

  llvm::SmallVector<Stmt> stmts;
  bool fail = false;
  while (!Peek()->is<Token::Tok_rbrace>() && !Peek()->is<Token::Tok_EOF>()) {
    auto stmt = parseStmt();
    if (!stmt) {
      fail = true;
      recovery(false);
    }
    stmts.emplace_back(stmt);
  }

  if (Consume<Token::Tok_rbrace>())
    return nullptr;

  if (fail)
    return nullptr;
  return BlockStmt::create(scope.CreateRange(), context, stmts);
}

Stmt Parser::parseStmt() {
  if (Peek()->is<Token::Tok_lbrace>())
    return parseBlockStmt();
  if (Peek()->is<Token::Tok_return>())
    return parseReturnStmt();
  if (Peek()->is<Token::Tok_var>())
    return parseVarDecl();
  if (Peek()->is<Token::Tok_identifier>() &&
      structTable.contains(Peek()->getSymbol()))
    return parseStructVarDecl();
  return parseExprStmt();
}

ReturnStmt Parser::parseReturnStmt() {
  FixLocationScope scope(*this);
  if (Consume<Token::Tok_return>())
    return nullptr;

  if (ConsumeIf<Token::Tok_semicolon>())
    return ReturnStmt::create(scope.CreateRange(), context, std::nullopt);

  auto expr = parseExpr();
  if (!expr)
    return nullptr;

  if (Consume<Token::Tok_semicolon>())
    return nullptr;

  return ReturnStmt::create(scope.CreateRange(), context, expr);
}

ExprStmt Parser::parseExprStmt() {
  FixLocationScope scope(*this);
  auto expr = parseExpr();
  if (!expr)
    return nullptr;

  if (Consume<Token::Tok_semicolon>())
    return nullptr;

  return ExprStmt::create(scope.CreateRange(), context, expr);
}

VarDecl Parser::parseVarDecl() {
  FixLocationScope scope(*this);
  if (Consume<Token::Tok_var>())
    return nullptr;

  auto id = Peek();
  if (Expect<Token::Tok_identifier>(id))
    return nullptr;
  Skip();

  std::optional<ShapeInfo> shape;
  if (ConsumeIf<Token::Tok_lt>()) {
    auto parseShape = parseShapeList();
    if (!parseShape)
      return nullptr;
    if (Consume<Token::Tok_gt>())
      return nullptr;
    shape = parseShape;
  }

  if (Consume<Token::Tok_equal>())
    return nullptr;

  auto expr = parseExpr();
  if (!expr)
    return nullptr;

  if (Consume<Token::Tok_semicolon>())
    return nullptr;

  return VarDecl::create(scope.CreateRange(), context, id->getSymbol(), shape,
                         expr);
}

StructVarDecl Parser::parseStructVarDecl() {
  FixLocationScope scope(*this);
  auto structTok = Peek();
  if (Expect<Token::Tok_identifier>(structTok))
    return nullptr;
  Skip();
  auto structName = structTok->getSymbol();

  auto varNameTok = Peek();
  if (Expect<Token::Tok_identifier>(varNameTok))
    return nullptr;
  Skip();

  auto varName = varNameTok->getSymbol();

  if (Consume<Token::Tok_equal>())
    return nullptr;

  if (Consume<Token::Tok_lbrace>())
    return nullptr;

  llvm::SmallVector<Expr> exprs;
  auto expr = parseExpr();
  if (!expr)
    return nullptr;
  exprs.emplace_back(expr);

  while (Peek()->is<Token::Tok_comma>()) {
    Skip();
    if (Peek()->is<Token::Tok_rbrace>())
      break;
    expr = parseExpr();
    if (!expr)
      return nullptr;
    exprs.emplace_back(expr);
  }

  if (Consume<Token::Tok_rbrace>())
    return nullptr;

  if (Consume<Token::Tok_semicolon>())
    return nullptr;

  auto structDecl = structTable.at(structName);
  if (structDecl.getFields().size() != exprs.size()) {
    Report(
        llvm::SMRange{
            exprs.front().getLoc().Start,
            exprs.back().getLoc().End,
        },
        Reporter::Diag::err_unmatched_struct_field_count,
        structDecl.getFields().size(), exprs.size());
    Report(structDecl.getLoc(),
           Reporter::Diag::note_previous_struct_declaration);
    return nullptr;
  }

  if (llvm::any_of(exprs,
                   [](Expr expr) { return !expr.isa<Number, Tensor>(); })) {
    Report(exprs.front().getLoc(), Reporter::Diag::err_unexpected_struct_init);
    return nullptr;
  }

  auto structVarDecl = StructVarDecl::create(scope.CreateRange(), context,
                                             structName, varName, exprs);
  structVarDecl.setStructDeclTag(structDecl);
  return structVarDecl;
}

Expr Parser::parseExpr() { return parseAddOperation(); }

Expr Parser::parseAddOperation() {
  FixLocationScope scope(*this);
  auto lhs = parseMulOperation();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_plus, Token::Tok_minus>()) {
    auto op = Advance();

    auto rhs = parseMulOperation();
    if (!rhs)
      return nullptr;

    lhs = BinaryOp::create(scope.CreateRange(), context, lhs, rhs,
                           op->is<Token::Tok_plus>() ? BinaryOpKind::Add
                                                     : BinaryOpKind::Sub);
  }

  return lhs;
}

Expr Parser::parseMulOperation() {
  FixLocationScope scope(*this);
  auto lhs = parseStructAccess();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_star, Token::Tok_slash>()) {
    auto op = Advance();

    auto rhs = parseStructAccess();
    if (!rhs)
      return nullptr;

    lhs = BinaryOp::create(scope.CreateRange(), context, lhs, rhs,
                           op->is<Token::Tok_star>() ? BinaryOpKind::Mul
                                                     : BinaryOpKind::Div);
  }

  return lhs;
}

Expr Parser::parseStructAccess() {
  FixLocationScope scope(*this);
  auto lhs = parseFunctionCall();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_dot>()) {
    Skip();

    auto fieldName = Peek();
    if (Expect<Token::Tok_identifier>(fieldName))
      return nullptr;
    Skip();
    lhs = StructAccess::create(scope.CreateRange(), context, lhs,
                               fieldName->getSymbol());
  }

  return lhs;
}

Expr Parser::parseFunctionCall() {
  FixLocationScope scope(*this);
  auto lhs = parseConstant();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_lparen>()) {
    Skip();

    if (!lhs.isa<Identifier>()) {
      Report(lhs.getLoc(), Reporter::Diag::err_unexpected_function_call);
      return nullptr;
    }
    auto functionName = lhs.cast<Identifier>().getName();

    llvm::SmallVector<Expr> args;
    if (!ConsumeIf<Token::Tok_rparen>()) {
      auto expr = parseExpr();
      if (!expr)
        return nullptr;
      args.emplace_back(expr);
      while (Peek()->is<Token::Tok_comma>()) {
        Skip();
        expr = parseExpr();
        if (!expr)
          return nullptr;
        args.emplace_back(expr);
      }

      if (Consume<Token::Tok_rparen>())
        return nullptr;
    }
    lhs =
        FunctionCall::create(scope.CreateRange(), context, functionName, args);
  }

  return lhs;
}

Expr Parser::parseConstant() {
  FixLocationScope scope(*this);
  auto *peekTok = Peek();
  if (peekTok->is<Token::Tok_number>()) {
    Skip();
    return Number::create(scope.CreateRange(), context, peekTok->getSymbol());
  }

  if (peekTok->is<Token::Tok_identifier>()) {
    Skip();
    return Identifier::create(scope.CreateRange(), context,
                              peekTok->getSymbol());
  }

  if (peekTok->is<Token::Tok_lparen>()) {
    Skip();
    auto expr = parseExpr();
    if (!expr)
      return nullptr;
    if (Consume<Token::Tok_rparen>())
      return nullptr;
    return expr;
  }

  if (peekTok->is<Token::Tok_lbracket>()) {
    Skip();
    llvm::SmallVector<Expr> exprs;

    auto expr = parseExpr();
    if (!expr)
      return nullptr;
    exprs.emplace_back(expr);

    while (Peek()->is<Token::Tok_comma>()) {
      Skip();
      if (Peek()->is<Token::Tok_rbracket>())
        break;
      expr = parseExpr();
      if (!expr)
        return nullptr;
      exprs.emplace_back(expr);
    }

    if (Consume<Token::Tok_rbracket>())
      return nullptr;

    auto tensorConst = Tensor::create(scope.CreateRange(), context, exprs);
    if (!tensorConstantSema(tensorConst))
      return nullptr;
    return tensorConst;
  }

  if (peekTok->is<Token::Tok_transpose, Token::Tok_print>()) {
    return parseBuiltinFunction();
  }

  Report(peekTok->getRange(), Reporter::Diag::err_unparsable_token,
         peekTok->getSymbol());
  return nullptr;
}

Expr Parser::parseBuiltinFunction() {
  FixLocationScope scope(*this);
  auto peekTok = Peek();
  if (peekTok->is<Token::Tok_transpose>()) {
    Skip();
    if (Consume<Token::Tok_lparen>())
      return nullptr;
    auto expr = parseExpr();
    if (!expr)
      return nullptr;
    if (Consume<Token::Tok_rparen>())
      return nullptr;
    return Transpose::create(scope.CreateRange(), context, expr);
  }

  if (peekTok->is<Token::Tok_print>()) {
    Skip();
    if (Consume<Token::Tok_lparen>())
      return nullptr;
    auto expr = parseExpr();
    if (!expr)
      return nullptr;
    if (Consume<Token::Tok_rparen>())
      return nullptr;
    return Print::create(scope.CreateRange(), context, expr);
  }

  llvm_unreachable("All builtin functions are handled");
}

std::optional<llvm::SmallVector<
    std::pair<std::optional<llvm::StringRef>, llvm::StringRef>>>
Parser::parseParamList() {
  if (Peek()->is<Token::Tok_rparen>())
    return llvm::SmallVector<
        std::pair<std::optional<llvm::StringRef>, llvm::StringRef>>{};

  llvm::SmallVector<std::pair<std::optional<llvm::StringRef>, llvm::StringRef>>
      params;
  if (Consume<Token::Tok_identifier>())
    return std::nullopt;
  if (structTable.contains(Tok->getSymbol())) {
    auto structName = Tok->getSymbol();
    if (Consume<Token::Tok_identifier>())
      return std::nullopt;
    auto varName = Tok->getSymbol();
    params.emplace_back(structName, varName);
  } else
    params.emplace_back(std::nullopt, Tok->getSymbol());

  while (Peek()->is<Token::Tok_comma>()) {
    Skip();
    if (Consume<Token::Tok_identifier>())
      return std::nullopt;
    if (structTable.contains(Tok->getSymbol())) {
      auto structName = Tok->getSymbol();
      if (Consume<Token::Tok_identifier>())
        return std::nullopt;
      auto varName = Tok->getSymbol();
      params.emplace_back(structName, varName);
    } else
      params.emplace_back(std::nullopt, Tok->getSymbol());
  }

  return params;
}

std::optional<ShapeInfo> Parser::parseShapeList() {
  if (Peek()->is<Token::Tok_gt>())
    return ShapeInfo{};

  ShapeInfo shape;
  if (Consume<Token::Tok_number>())
    return std::nullopt;
  auto numberStr = Tok->getSymbol();
  if (numberStr.contains('.')) {
    Report(Tok->getRange(), Reporter::Diag::err_unappropriate_shape_number);
    return std::nullopt;
  }

  shape.emplace_back(std::stoul(numberStr.str()));

  while (Peek()->is<Token::Tok_comma>()) {
    Skip();
    if (Consume<Token::Tok_number>())
      return std::nullopt;
    numberStr = Tok->getSymbol();
    if (numberStr.contains('.')) {
      Report(Tok->getRange(), Reporter::Diag::err_unappropriate_shape_number);
      return std::nullopt;
    }
    shape.emplace_back(std::stoll(numberStr.str()));
  }

  return shape;
}

bool Parser::tensorConstantSema(Tensor tensor) {
  auto elements = tensor.getElements();
  if (elements.empty())
    return false;

  llvm::SmallVector<std::optional<ShapeInfo>> shape;
  if (!llvm::all_of(elements, [&](Expr expr) {
        if (auto childTensor = expr.dyn_cast<Tensor>()) {
          if (!tensorConstantSema(childTensor))
            return false;
          shape.emplace_back(childTensor.getShapeTag());
          return true;
        }

        if (auto childNumber = expr.dyn_cast<Number>()) {
          shape.emplace_back(std::nullopt);
          return true;
        }

        Report(expr.getLoc(), Reporter::Diag::err_unexpected_tensor_element);
        return false;
      }))
    return false;

  if (!llvm::all_equal(shape)) {
    Report(tensor.getLoc(), Reporter::Diag::err_invalid_tensor_shape);
    return false;
  }

  ShapeInfo tensorShape{static_cast<int64_t>(elements.size())};
  if (shape[0])
    tensorShape.append(shape[0]->begin(), shape[0]->end());

  tensor.setShapeTag(tensorShape);
  return true;
}

static llvm::SourceMgr::DiagKind parserDiagKinds[] = {
#define DIAG(ID, Kind, Msg) llvm::SourceMgr::DK_##Kind,
#include "toy/parser/ParserDiagnostic.def"
};

static llvm::StringLiteral parserDiagMsg[] = {
#define DIAG(ID, Kind, Msg) Msg,
#include "toy/parser/ParserDiagnostic.def"
};

llvm::SourceMgr::DiagKind Parser::Reporter::getDiagKind(Diag diag) {
  return parserDiagKinds[diag];
}

llvm::StringLiteral Parser::Reporter::getDiagMsg(Diag diag) {
  return parserDiagMsg[diag];
}

} // namespace toy
