#ifndef TOY_TOKEN_H
#define TOY_TOKEN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace toy {

class Token {
public:
  enum Kind {
#define TOKEN(ID) Tok_##ID,
#include "toy/parser/TokenKind.def"
  };
  static llvm::StringRef getTokenKindName(Kind kind);

  Token(llvm::StringRef symbol, llvm::SMRange range, Kind kind, std::size_t col,
        std::size_t row, std::size_t index)
      : symbol(symbol), range(range), kind(kind), col(col), row(row),
        index(index) {}

  llvm::StringRef getSymbol() const { return symbol; }
  llvm::SMRange getRange() const { return range; }
  Kind getKind() const { return kind; }
  std::size_t getCol() const { return col; }
  std::size_t getRow() const { return row; }
  std::size_t getIndex() const { return index; }

  template <Kind... kind> bool is() const {
    return ((getKind() == kind) || ...);
  }

  std::string toString() const;

private:
  llvm::StringRef symbol;
  llvm::SMRange range;
  Kind kind;
  std::size_t col;
  std::size_t row;
  std::size_t index;
};

} // namespace toy

#endif // TOY_TOKEN_H
