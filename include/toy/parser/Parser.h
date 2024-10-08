#ifndef TOY_PARSER_H
#define TOY_PARSER_H

#include "toy/ast/ToyAST.h"
#include "toy/ast/ToyExpr.h"
#include "toy/ast/ToyStmt.h"
#include "toy/parser/Lexer.h"
#include "toy/parser/Token.h"
#include "toy/reporter/DiagnosticReporter.h"
#include "llvm/ADT/StringRef.h"
#include <iterator>

namespace toy {

using ast::AST;

class Parser {
public:
  Parser(Lexer &lexer, llvm::SourceMgr &srcMgr)
      : Lexer(lexer), srcMgr(srcMgr), context(lexer.getContext()),
        reporter(lexer.getReporter()) {}

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

  /// <mul operation> ::= <function call> ((`*` | `/`) <function call>)?
  Expr parseMulOperation();

  /// <function call> ::= <constant> ( `(` <expr list> `)`) ?
  Expr parseFunctionCall();

  /// <constant> ::= <number> | <identifier> | <tensor> | `(` <expr> `)`
  Expr parseConstant();

  /// <builtin function> ::=
  ///   `transpose` `(` <expr> `)`
  Expr parseBuiltinFunction();

  /// <param list> ::= <identifier> (`,` <identifier>)*
  std::optional<llvm::SmallVector<llvm::StringRef>> parseParamList();

  /// <shape list> ::= <number> (`,` <number>)*
  std::optional<llvm::SmallVector<std::uint64_t>> parseShapeList();

  void recovery();

  struct Reporter {
    enum Diag {
#define DIAG(ID, ...) ID,
#include "toy/parser/ParserDiagnostic.def"
    };

    static llvm::SourceMgr::DiagKind getDiagKind(Diag diag);
    static llvm::StringLiteral getDiagMsg(Diag diag);
  };

  template <typename... Args>
  void Report(llvm::SMRange loc, Reporter::Diag diag, Args &&...args) {
    auto kind = Reporter::getDiagKind(diag);
    auto msg = Reporter::getDiagMsg(diag);
    auto evaledMsg = llvm::formatv(msg.data(), std::forward<Args>(args)...);
    reporter.Report(loc, kind, evaledMsg.str());
  }

private:
  Token *Tok;
  Lexer &Lexer;
  llvm::SourceMgr &srcMgr;
  ToyContext *context;
  DiagnosticReporter &reporter;
};

template <Token::Kind Kind> bool Parser::PeekExpect() {
  return Expect<Kind>(Peek());
}

template <Token::Kind Kind> bool Parser::Expect(Token *token) {
  if (!token->is<Kind>()) {
    Report(token->getRange(), Reporter::Diag::err_unexpected_token,
           Token::getTokenKindName(token->getKind()),
           Token::getTokenKindName(Kind));
    return true;
  }
  return false;
}

template <Token::Kind Kind> bool Parser::Consume() {
  if (PeekExpect<Kind>()) {
    return true;
  }
  Skip();
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
