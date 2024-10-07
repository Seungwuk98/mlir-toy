#ifndef TOY_AST_H
#define TOY_AST_H

#include "ast/AST.h"
#include "llvm/Support/SMLoc.h"

namespace toy {

using ast::AST;

class ToyAST : public AST {
public:
  using AST::AST;

  void setRange(llvm::SMRange range) { this->range = range; }
  llvm::SMRange getRange() const { return range; }

private:
  llvm::SMRange range;
};

} // namespace toy

#endif // TOY_AST_H
