#include "toy/context/ToyContext.h"

namespace toy {

class ToyContextImpl {
public:
private:
};

ToyContext::ToyContext() : impl(new ToyContextImpl) {}
ToyContext::~ToyContext() { delete impl; }

} // namespace toy
