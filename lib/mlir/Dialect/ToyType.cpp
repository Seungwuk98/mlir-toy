#include "toy/mlir/Dialect/ToyType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "toy/mlir/Dialect/ToyDialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::toy {

namespace detail {
struct StructTypeStorage final
    : public mlir::TypeStorage,
      public llvm::TrailingObjects<StructTypeStorage, Type> {
  using KeyTy = TypeRange;
  using TObjs = llvm::TrailingObjects<StructTypeStorage, Type>;

  StructTypeStorage(std::size_t numElements) : numElements(numElements) {}

  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    auto bytes = totalSizeToAlloc<Type>(key.size());
    auto rawMem = allocator.allocate(bytes, alignof(StructTypeStorage));
    auto *storage = new (rawMem) StructTypeStorage(key.size());

    std::uninitialized_copy(key.begin(), key.end(),
                            storage->getTrailingObjects<Type>());
    return storage;
  }

  bool operator==(const KeyTy &key) const { return key == getTypes(); }

  llvm::ArrayRef<Type> getTypes() const {
    return {getTrailingObjects<Type>(), size()};
  }

  std::size_t size() const { return numElements; }

  std::size_t numElements;
};
} // namespace detail

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
