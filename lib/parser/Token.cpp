#include "toy/parser/Token.h"
#include "llvm/Support/raw_ostream.h"

namespace toy {

static llvm::StringRef tokenKindNames[] = {
#define TOKEN(ID) #ID,
#include "toy/parser/TokenKind.def"
};

llvm::StringRef Token::getTokenKindName(Kind kind) {
  return tokenKindNames[kind];
}

std::string Token::toString() const {
  std::string result;
  llvm::raw_string_ostream ss(result);
  ss << getTokenKindName(getKind()) << " \"" << getSymbol() << "\"" << " at "
     << getRow() << ":" << getCol();
  return result;
}

} // namespace toy
