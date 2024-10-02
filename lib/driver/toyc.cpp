#include "mlir/IR/MLIRContext.h"
#include "toy/context/ToyContext.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

int main() {
  std::cout << 1 << '\n';
  llvm::errs() << 1 << '\n';
  mlir::MLIRContext context;
  toy::ToyContext toyContext;
}
