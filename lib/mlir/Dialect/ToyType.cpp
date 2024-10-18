#include "toy/mlir/Dialect/ToyType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "toy/mlir/Dialect/ToyDialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::toy {

StructType StructType::get(TypeRange elementTypes) {
  assert(!elementTypes.empty() && "expected at least one element type");
  return Base::get(elementTypes[0].getContext(), elementTypes);
}

llvm::ArrayRef<Type> StructType::getElementTypes() const {
  return getImpl()->getTypes();
}

std::size_t StructType::size() const { return getImpl()->size(); }

StructType::iterator StructType::begin() const {
  return getElementTypes().begin();
}

StructType::iterator StructType::end() const { return getElementTypes().end(); }

Type StructType::getType(std::size_t index) const {
  assert(index < size() && "index out of range");
  return getElementTypes()[index];
}

} // namespace mlir::toy

#define GET_TYPEDEF_CLASSES
#include "toy/mlir/Dialect/ToyType.cpp.inc"
