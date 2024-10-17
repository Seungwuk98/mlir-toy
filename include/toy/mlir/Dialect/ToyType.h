#ifndef TOY_MLIR_TYPE_H
#define TOY_MLIR_TYPE_H

#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"

namespace mlir::toy::detail {
struct StructTypeStorage;
}

#define GET_TYPEDEF_CLASSES
#include "toy/mlir/Dialect/ToyType.h.inc"

#endif // TOY_MLIR_TYPE_H
