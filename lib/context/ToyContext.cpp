#include "toy/context/ToyContext.h"
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
}
ToyContext::~ToyContext() { delete impl; }

} // namespace toy
