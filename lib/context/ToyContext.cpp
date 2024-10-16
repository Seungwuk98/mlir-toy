#include "toy/context/ToyContext.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/DialectRegistry.h"
#include "toy/ast/ToyASTSet.h"
#include "toy/mlir/Dialect/ToyDialect.h"

namespace toy {

class ToyContextImpl {
public:
private:
};

ToyContext::ToyContext() : impl(new ToyContextImpl) {
  // load AST sets
  GetOrRegisterASTSet<ToyASTSet>();

  // load MLIR Dialects
  getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::DialectRegistry registry;
  // load Extensions
  mlir::func::registerAllExtensions(registry);
  appendDialectRegistry(registry);
}
ToyContext::~ToyContext() { delete impl; }

} // namespace toy
