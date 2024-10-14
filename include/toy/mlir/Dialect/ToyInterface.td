#ifndef TOY_INTERFACE_TD
#define TOY_INTERFACE_TD

include "mlir/IR/Interfaces.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.", "void", "inferShapes">   
  ];
}

#endif // TOY_INTERFACE_TD
