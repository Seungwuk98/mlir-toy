#ifndef TOY_MLIR_INTERFACE_H
#define TOY_MLIR_INTERFACE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::toy {
#include "toy/mlir/Dialect/ToyInterface.h.inc"
}

#endif // TOY_MLIR_INTERFACE_H
