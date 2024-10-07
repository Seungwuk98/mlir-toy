#include "toy/parser/Parser.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace toy {

Token *Parser::Advance() { return Tok = Lexer.GetNextToken(); }
void Parser::Skip() { Advance(); }
Token *Parser::Peek(unsigned lookAt) { return Lexer.PeekNextToken(lookAt); }
Token *Parser::PrevTok() { return Lexer.PeekPrevToken(); }

Module Parser::parseModule() {
  llvm::SmallVector<Stmt> stmts;
  bool fail = false;
  while (!Peek()->is<Token::Tok_EOF>()) {
    auto funDecl = parseFunctionDecl();
    if (!funDecl) {
      fail = true;
      recovery();
    }
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
  }

  if (!Consume<Token::Tok_rbrace>())
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
  auto lhs = parseConstant();
  if (!lhs)
    return nullptr;

  while (Peek()->is<Token::Tok_star, Token::Tok_slash>()) {
    auto op = Advance();

    auto rhs = parseConstant();
    if (!rhs)
      return nullptr;

    lhs = BinaryOp::create(context, lhs, rhs,
                           op->is<Token::Tok_star>() ? BinaryOpKind::Mul
                                                     : BinaryOpKind::Div);
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

  /// TODO report
  return nullptr;
}

} // namespace toy
