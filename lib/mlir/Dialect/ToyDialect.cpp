#include "toy/mlir/Dialect/ToyDialect.h"
#include "toy/mlir/Dialect/ToyOp.h"

/// tablegen generated .inc file
#include "toy/mlir/Dialect/ToyDialect.cpp.inc"

namespace mlir::toy {

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/mlir/Dialect/ToyOp.cpp.inc"
      >();
}

} // namespace mlir::toy
