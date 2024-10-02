#ifndef TOY_CONTEXT_H
#define TOY_CONTEXT_H

#include "ast/ASTContext.h"
#include "mlir/IR/MLIRContext.h"

namespace toy {

class ToyContextImpl;
class ToyContext : public mlir::MLIRContext, public ast::ASTContext {
public:
  ToyContext();
  ~ToyContext();

private:
  ToyContextImpl *impl;
};

} // namespace toy

#endif // TOY_CONTEXT_H
