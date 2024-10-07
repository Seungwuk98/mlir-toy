#ifndef TOY_PARSER_H
#define TOY_PARSER_H

#include "toy/ast/ToyAST.h"
#include "toy/ast/ToyExpr.h"
#include "toy/ast/ToyStmt.h"
#include "toy/parser/Lexer.h"
#include "toy/parser/Token.h"

namespace toy {

using ast::AST;

class Parser {
public:
  Parser(Lexer &lexer, llvm::SourceMgr &srcMgr)
      : Lexer(lexer), srcMgr(srcMgr), context(lexer.getContext()) {}

  Module Parse();

private:
  Token *Advance();
  void Skip();
  Token *Peek(unsigned lookAt = 0);
  Token *PrevTok();
  template <Token::Kind Kind> bool PeekExpect();
  template <Token::Kind Kind> bool Expect(Token *token);
  template <Token::Kind Kind> bool Consume();
  template <Token::Kind Kind> bool ConsumeIf();

  /// <module> ::= <function decl>+
  Module parseModule();

  /// <function decl> ::= `def` <identifier> `(` <param list>* `)` <block stmt>
  FuncDecl parseFunctionDecl();

  /// <block stmt> ::= `{` <stmt>* `}`
  BlockStmt parseBlockStmt();

  /// <stmt> ::= (
  ///   <block stmt>
  ///   | <return stmt>
  ///   | <expr stmt>
  ///   | <var decl>
  /// )
  Stmt parseStmt();

  /// <return stmt> ::= `return` <expr> `;`
  ReturnStmt parseReturnStmt();

  /// <expr stmt> ::= <expr> `;`
  ExprStmt parseExprStmt();

  /// <var decl> ::= `var` <identifier> (`<` <shape list> `>`) `=` <expr> `;`
  VarDecl parseVarDecl();

  /// <expr> ::= <additive operation>
  Expr parseExpr();

  /// <add operation>> ::= <add operation> ((`+` | `-`) <add operation>)?
  Expr parseAddOperation();

  /// <mul operation> ::= <constant> ((`*` | `/`) <constant>)?
  Expr parseMulOperation();

  /// <constant> ::= <number> | <identifier> | `(` <expr> `)`
  Expr parseConstant();

  /// <param list> ::= <identifier> (`,` <identifier>)*
  std::optional<llvm::SmallVector<llvm::StringRef>> parseParamList();

  /// <shape list> ::= <number> (`,` <number>)*
  std::optional<llvm::SmallVector<std::uint64_t>> parseShapeList();

  void recovery();

private:
  Token *Tok;
  Lexer &Lexer;
  llvm::SourceMgr &srcMgr;
  ToyContext *context;
};

template <Token::Kind Kind> bool Parser::PeekExpect() {
  return !Peek()->is<Kind>();
}

template <Token::Kind Kind> bool Parser::Expect(Token *token) {
  if (!token->is<Kind>()) {
    /// TODO report
    return true;
  }
  return false;
}

template <Token::Kind Kind> bool Parser::Consume() {
  if (PeekExpect<Kind>()) {
    /// TODO report
    return true;
  }
  Advance();
  return false;
}

template <Token::Kind Kind> bool Parser::ConsumeIf() {
  if (Peek()->is<Kind>()) {
    Advance();
    return true;
  }
  return false;
}

} // namespace toy

#endif // TOY_PARSER_H
