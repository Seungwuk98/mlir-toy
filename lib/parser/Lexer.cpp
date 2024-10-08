#include "toy/parser/Lexer.h"
#include <llvm-18/llvm/ADT/StringSwitch.h>
#include <llvm-18/llvm/Support/SourceMgr.h>

namespace toy {

char Lexer::advance() {
  if (Pos == Buffer.size())
    return eof;

  auto ch = Buffer[Pos++];

  if (ch == '\n') {
    ++Row;
    Col = 0;
  } else
    ++Col;

  return ch;
}

void Lexer::capture() {
  lastFixedPos = Pos;
  lastFixedCol = Col;
  lastFixedRow = Row;
  TokenKind = Token::Tok_unknown;
}

Token *Lexer::create() const {
  auto symbol = Buffer.slice(lastFixedPos, Pos);
  auto range =
      llvm::SMRange(llvm::SMLoc::getFromPointer(Buffer.data() + lastFixedPos),
                    llvm::SMLoc::getFromPointer(Buffer.data() + Pos));
  return context->Alloc<Token>(symbol, range, TokenKind, lastFixedCol,
                               lastFixedRow, TokenStream.size());
}

char Lexer::peek(int lookAt) const {
  auto lookPos = Pos + lookAt;
  if (lookPos >= Buffer.size())
    return eof;
  return Buffer[lookPos];
}

void Lexer::skipWhitespace() {
  while (std::isspace(peek()))
    advance();
}

void Lexer::Lex() {
  skipWhitespace();
  capture();

  auto ch = advance();
  switch (ch) {
  case '(':
    TokenKind = Token::Tok_lparen;
    parenStack.emplace_back(lastFixedPos);
    break;
  case ')':
    TokenKind = Token::Tok_rparen;
    if (parenStack.empty() || Buffer[parenStack.back()] != '(') {
      Report(llvm::SMRange(
                 llvm::SMLoc::getFromPointer(Buffer.data() + lastFixedPos),
                 llvm::SMLoc::getFromPointer(Buffer.data() + Pos)),
             Reporter::Diag::err_unmatched_paren);
    } else {
      parenPairs[parenStack.back()] = lastFixedPos;
      parenStack.pop_back();
    }
    break;
  case '[':
    TokenKind = Token::Tok_lbracket;
    parenStack.emplace_back(lastFixedPos);
    break;
  case ']':
    TokenKind = Token::Tok_rbracket;
    if (parenStack.empty() || Buffer[parenStack.back()] != '[') {
      Report(llvm::SMRange(
                 llvm::SMLoc::getFromPointer(Buffer.data() + lastFixedPos),
                 llvm::SMLoc::getFromPointer(Buffer.data() + Pos)),
             Reporter::Diag::err_unmatched_paren);
    } else {
      parenPairs[parenStack.back()] = lastFixedPos;
      parenStack.pop_back();
    }
    break;
  case '{':
    TokenKind = Token::Tok_lbrace;
    parenStack.emplace_back(lastFixedPos);
    break;
  case '}':
    TokenKind = Token::Tok_rbrace;
    if (parenStack.empty() || Buffer[parenStack.back()] != '{') {
      Report(llvm::SMRange(
                 llvm::SMLoc::getFromPointer(Buffer.data() + lastFixedPos),
                 llvm::SMLoc::getFromPointer(Buffer.data() + Pos)),
             Reporter::Diag::err_unmatched_paren);
    } else {
      parenPairs[parenStack.back()] = lastFixedPos;
      parenStack.pop_back();
    }
    break;
  case '<':
    TokenKind = Token::Tok_lt;
    break;
  case '>':
    TokenKind = Token::Tok_gt;
    break;
  case '+':
    TokenKind = Token::Tok_plus;
    break;
  case '-':
    TokenKind = Token::Tok_minus;
    break;
  case '*':
    TokenKind = Token::Tok_star;
    break;
  case '/':
    TokenKind = Token::Tok_slash;
    break;
  case '=':
    TokenKind = Token::Tok_equal;
    break;
  case ',':
    TokenKind = Token::Tok_comma;
    break;
  case ';':
    TokenKind = Token::Tok_semicolon;
    break;
  case eof:
    TokenKind = Token::Tok_EOF;
    return;
  case '#':
    while (peek() != '\n' && peek() != eof)
      advance();
    Lex();
    return;
  default:
    if (std::isdigit(ch)) {
      TokenKind = Token::Tok_number;
      lexNumber();
    } else if (std::isalpha(ch) || ch == '_') {
      lexIdentifierOrKeyword();
    } else {
      Report(llvm::SMRange(
                 llvm::SMLoc::getFromPointer(Buffer.data() + lastFixedPos),
                 llvm::SMLoc::getFromPointer(Buffer.data() + Pos)),
             Reporter::Diag::err_unlexable_char, ch);
    }
  }
}

void Lexer::lexNumber() {
  while (std::isdigit(peek()))
    advance();

  if (peek() == '.') {
    advance();
    while (std::isdigit(peek()))
      advance();
  }
}

void Lexer::lexIdentifierOrKeyword() {
  while (std::isalnum(peek()) || peek() == '_')
    advance();

  auto symbol = Buffer.slice(lastFixedPos, Pos);

  TokenKind = llvm::StringSwitch<Token::Kind>(symbol)
#define KEYWORD(ID) .Case(#ID, Token::Tok_##ID)
#include "toy/parser/TokenKind.def"

                  .Default(Token::Tok_identifier);
}

Token *Lexer::GetNextToken() {
  lexToPosToken(tokenPos);
  return TokenStream[tokenPos++];
}

Token *Lexer::PeekNextToken(unsigned lookAt) {
  lexToPosToken(tokenPos + lookAt);
  return TokenStream[tokenPos + lookAt];
}

Token *Lexer::PeekPrevToken() {
  assert(tokenPos > 0 && "No previous token");
  return TokenStream[tokenPos - 1];
}

void Lexer::lexToPosToken(std::size_t pos) {
  while (pos >= TokenStream.size()) {
    if (!TokenStream.empty() && TokenStream.back()->is<Token::Tok_EOF>())
      return;
    Lex();
    TokenStream.emplace_back(create());
  }
}

Token *Lexer::GetNextToken(Token *token, unsigned lookAt) {
  lexToPosToken(token->getIndex() + lookAt);
  if (token->getIndex() + lookAt >= TokenStream.size())
    return TokenStream.back();
  return TokenStream[token->getIndex() + lookAt];
}

static llvm::SourceMgr::DiagKind lexerDiagKinds[] = {
#define DIAG(ID, Kind, Msg) llvm::SourceMgr::DK_##Kind,
#include "toy/parser/LexerDiagnostic.def"
};

llvm::SourceMgr::DiagKind Lexer::Reporter::getDiagKind(Diag diag) {
  return lexerDiagKinds[diag];
}

static llvm::StringLiteral lexerDiagMsgs[] = {
#define DIAG(ID, Kind, Msg) Msg,
#include "toy/parser/LexerDiagnostic.def"
};

llvm::StringLiteral Lexer::Reporter::getDiagMsg(Diag diag) {
  return lexerDiagMsgs[diag];
}

} // namespace toy
