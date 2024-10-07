#ifndef TOY_AST_SET_H
#define TOY_AST_SET_H

#include "ast/ASTSet.h"

namespace toy {

class ToyASTSet : public ast::ASTSet {
public:
  using ASTSet::ASTSet;

  void RegisterSet() override;

  llvm::StringRef getASTSetName() const override { return "Toy"; }
};

} // namespace toy

#endif // TOY_AST_SET_H
