#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "toy/mlir/Dialect/ToyDialect.h"
#include "toy/mlir/Pass/Passes.h"

int main(int arc, char **argv) {
  mlir::registerAllPasses();
  mlir::toy::registerPrintOpLoweringPass();
  mlir::toy::registerAffineToLLVMLoweringPass();
  mlir::toy::registerToyToAffineLoweringPass();
  mlir::toy::registerShapeInferencePass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::toy::ToyDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(arc, argv, "Toy optimizer driver\n", registry));
}
