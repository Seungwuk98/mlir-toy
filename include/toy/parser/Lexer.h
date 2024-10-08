#ifndef TOY_LEXER_H
#define TOY_LEXER_H

#include "toy/context/ToyContext.h"
#include "toy/parser/Token.h"
#include "toy/reporter/DiagnosticReporter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"

namespace toy {

class Lexer {
public:
  Lexer(llvm::StringRef buffer, ToyContext *context, llvm::SourceMgr &srcMgr,
        DiagnosticReporter &diagReporter)
      : Buffer(buffer), context(context), srcMgr(srcMgr),
        diagReporter(diagReporter) {}

  Token *GetNextToken();
  Token *PeekNextToken(unsigned lookAt = 0);
  Token *PeekPrevToken();
  Token *GetNextToken(Token *token, unsigned lookAt = 0);

  llvm::ArrayRef<Token *> getTokenStream() const { return TokenStream; }
  llvm::SourceMgr &getSourceMgr() { return srcMgr; }
  ToyContext *getContext() { return context; }
  DiagnosticReporter &getReporter() { return diagReporter; }

  static constexpr char eof = '\0';

  struct Reporter {
    enum Diag {
#define DIAG(ID, ...) ID,
#include "toy/parser/LexerDiagnostic.def"
    };

    static llvm::SourceMgr::DiagKind getDiagKind(Diag diag);
    static llvm::StringLiteral getDiagMsg(Diag diag);
  };

  template <typename... Args>
  void Report(llvm::SMRange loc, Reporter::Diag diag, Args &&...args) {
    auto diagKind = Reporter::getDiagKind(diag);
    llvm::StringRef msg = Reporter::getDiagMsg(diag);
    auto evaledMsg = llvm::formatv(msg.data(), std::forward<Args>(args)...);
    diagReporter.Report(loc, diagKind, evaledMsg.str());
  }

private:
  void capture();
  Token *create() const;
  void Lex();
  char advance();
  char peek(int lookAt = 0) const;
  void skipWhitespace();

  void lexNumber();
  void lexIdentifierOrKeyword();
  void lexToPosToken(std::size_t pos);

private:
  llvm::StringRef Buffer;
  llvm::SmallVector<Token *> TokenStream;

  std::size_t Pos = 0;
  std::size_t Col = 0;
  std::size_t Row = 0;
  Token::Kind TokenKind = Token::Tok_unknown;
  std::size_t lastFixedPos = 0;
  std::size_t lastFixedCol = 0;
  std::size_t lastFixedRow = 0;
  std::size_t tokenPos = 0;

  std::vector<std::size_t> parenStack;
  llvm::DenseMap<std::size_t, std::size_t> parenPairs;

  ToyContext *context;
  llvm::SourceMgr &srcMgr;
  DiagnosticReporter &diagReporter;
};

} // namespace toy

#endif // TOY_LEXER_H
