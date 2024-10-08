#include "toy/context/ToyContext.h"
#include "toy/ast/ToyASTSet.h"
namespace toy {

class ToyContextImpl {
public:
private:
};

ToyContext::ToyContext() : impl(new ToyContextImpl) {
  GetOrRegisterASTSet<ToyASTSet>();
}
ToyContext::~ToyContext() { delete impl; }

} // namespace toy
