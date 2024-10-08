#include "toy/parser/Parser.h"
#include "toy/ast/ToyExpr.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <llvm-18/llvm/ADT/StringRef.h>
#include <llvm-18/llvm/Support/ErrorHandling.h>
#include <llvm-18/llvm/Support/SourceMgr.h>
#include <optional>

namespace toy {

Token *Parser::Advance() { return Tok = Lexer.GetNextToken(); }
void Parser::Skip() { Advance(); }
Token *Parser::Peek(unsigned lookAt) { return Lexer.PeekNextToken(lookAt); }
Token *Parser::PrevTok() { return Lexer.PeekPrevToken(); }

Module Parser::Parse() { return parseModule(); }

void Parser::recovery() {
  while (true) {
    auto peekTok = Peek();
    if (peekTok->is<Token::Tok_def, Token::Tok_return, Token::Tok_var,
                    Token::Tok_EOF>())
      return;

    if (peekTok->is<Token::Tok_semicolon>()) {
      Skip();
      return;
    }

    Skip();
  }
}

Module Parser::parseModule() {
  llvm::SmallVector<Stmt> stmts;
  bool fail = false;
  while (!Peek()->is<Token::Tok_EOF>()) {
    auto funDecl = parseFunctionDecl();
    if (!funDecl) {
      fail = true;
      recovery();
    }
    stmts.emplace_back(funDecl);
  }

  if (fail)
    return nullptr;
  return Module::create(context, stmts);
}

FuncDecl Parser::parseFunctionDecl() {
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

  auto paramList = llvm::map_to_vector(
      *params, [](llvm::StringRef param) { return param.str(); });
  return FuncDecl::create(context, id->getSymbol(), paramList, blockStmt);
}

BlockStmt Parser::parseBlockStmt() {
  if (Consume<Token::Tok_lbrace>())
    return nullptr;

  llvm::SmallVector<Stmt> stmts;
  bool fail = false;
  while (!Peek()->is<Token::Tok_rbrace>() && !Peek()->is<Token::Tok_EOF>()) {
    auto stmt = parseStmt();
    if (!stmt) {
      fail = true;
      recovery();
    }
    stmts.emplace_back(stmt);
  }

  if (Consume<Token::Tok_rbrace>())
    return nullptr;

  if (fail)
    return nullptr;
  return BlockStmt::create(context, stmts);
}

Stmt Parser::parseStmt() {
  if (Peek()->is<Token::Tok_lbrace>())
    return parseBlockStmt();
  if (Peek()->is<Token::Tok_return>())
    return parseReturnStmt();
  if (Peek()->is<Token::Tok_var>())
    return parseVarDecl();
  return parseExprStmt();
}

ReturnStmt Parser::parseReturnStmt() {
  if (Consume<Token::Tok_return>())
    return nullptr;

  if (ConsumeIf<Token::Tok_semicolon>())
    return ReturnStmt::create(context, std::nullopt);

  auto expr = parseExpr();
  if (!expr)
    return nullptr;

  if (Consume<Token::Tok_semicolon>())
    return nullptr;

  return ReturnStmt::create(context, expr);
}

ExprStmt Parser::parseExprStmt() {
  auto expr = parseExpr();
  if (!expr)
    return nullptr;

  if (Consume<Token::Tok_semicolon>())
    return nullptr;

  return ExprStmt::create(context, expr);
}

VarDecl Parser::parseVarDecl() {
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

  return VarDecl::create(context, id->getSymbol(), shape, expr);
}

Expr Parser::parseExpr() { return parseAddOperation(); }

Expr Parser::parseAddOperation() {
  auto lhs = parseMulOperation();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_plus, Token::Tok_minus>()) {
    auto op = Advance();

    auto rhs = parseMulOperation();
    if (!rhs)
      return nullptr;

    lhs = BinaryOp::create(context, lhs, rhs,
                           op->is<Token::Tok_plus>() ? BinaryOpKind::Add
                                                     : BinaryOpKind::Sub);
  }

  return lhs;
}

Expr Parser::parseMulOperation() {
  auto lhs = parseFunctionCall();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_star, Token::Tok_slash>()) {
    auto op = Advance();

    auto rhs = parseFunctionCall();
    if (!rhs)
      return nullptr;

    lhs = BinaryOp::create(context, lhs, rhs,
                           op->is<Token::Tok_star>() ? BinaryOpKind::Mul
                                                     : BinaryOpKind::Div);
  }

  return lhs;
}

Expr Parser::parseFunctionCall() {
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

    lhs = FunctionCall::create(context, functionName, args);
  }

  return lhs;
}

Expr Parser::parseConstant() {
  auto *peekTok = Peek();
  if (peekTok->is<Token::Tok_number>()) {
    Skip();
    return Number::create(context, peekTok->getSymbol());
  }

  if (peekTok->is<Token::Tok_identifier>()) {
    Skip();
    return Identifier::create(context, peekTok->getSymbol());
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

    return Tensor::create(context, exprs);
  }

  if (peekTok->is<Token::Tok_transpose, Token::Tok_print>()) {
    return parseBuiltinFunction();
  }

  Report(peekTok->getRange(), Reporter::Diag::err_unparsable_token,
         peekTok->getSymbol());
  return nullptr;
}

Expr Parser::parseBuiltinFunction() {
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
    return Transpose::create(context, expr);
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
    return Print::create(context, expr);
  }

  llvm_unreachable("All builtin functions are handled");
}

std::optional<llvm::SmallVector<llvm::StringRef>> Parser::parseParamList() {
  if (Peek()->is<Token::Tok_rparen>())
    return llvm::SmallVector<llvm::StringRef>();

  llvm::SmallVector<llvm::StringRef> params;
  if (Consume<Token::Tok_identifier>())
    return std::nullopt;
  params.emplace_back(Tok->getSymbol());

  while (Peek()->is<Token::Tok_comma>()) {
    Skip();
    if (Consume<Token::Tok_identifier>())
      return std::nullopt;
    params.emplace_back(Tok->getSymbol());
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
    shape.emplace_back(std::stoul(numberStr.str()));
  }

  return shape;
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