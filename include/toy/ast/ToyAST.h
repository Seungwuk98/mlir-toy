#ifndef TOY_AST_H
#define TOY_AST_H

#include "ast/AST.h"
#include "llvm/Support/SMLoc.h"

namespace toy {

using ast::AST;

class ToyAST : public AST {
public:
  using AST::AST;
};

using ShapeInfo = llvm::SmallVector<std::int64_t>;
} // namespace toy

#endif // TOY_AST_H
