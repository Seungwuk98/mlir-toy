#ifndef TOY_TYPE_TD
#define TOY_TYPE_TD

include "toy/mlir/Dialect/ToyDialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"


/// Base class for all LLVM dialect types.
class ToyTypeBase<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Toy_Dialect, name, traits> {
  let mnemonic = ?;
  let typeName = "toy." # typeMnemonic;
}

def Toy_StructType : ToyTypeBase<"Struct", "struct"> {
  let description = [{
    The struct type represents a structure of elements.
  }];

  let parameters = (ins "::llvm::ArrayRef<Type>":$elementTypes);
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::TypeRange":$elementTypes)>, 
  ];

  let skipDefaultBuilders = 1;
  let genStorageClass = 0;
  let extraClassDeclaration = [{
    
    std::size_t size() const;
    using iterator = ::llvm::ArrayRef<Type>::iterator;
    
    iterator begin() const;
    iterator end() const;

    Type getType(std::size_t index) const;

  }];
}

def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;


#endif // TOY_TYPE_TD
