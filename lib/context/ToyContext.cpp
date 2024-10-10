#include "toy/context/ToyContext.h"
#include "toy/ast/ToyASTSet.h"
namespace toy {

class ToyContextImpl {
public:
private:
};

ToyContext::ToyContext() : impl(new ToyContextImpl) {
  // load AST sets
  GetOrRegisterASTSet<ToyASTSet>();

  // load MLIR Dialects
  getOrLoadDialect<ToyDialect>();
}
ToyContext::~ToyContext() { delete impl; }

} // namespace toy
