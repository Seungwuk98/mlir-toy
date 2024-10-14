#include "mlir/IR/OperationSupport.h"
#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "llvm/Support/raw_ostream.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "MLIRTestUtils.h"
#include "TestExtras.h"
#include "doctest/doctest.h"
#include "toy/parser/Parser.h"

namespace toy::test {

TEST_SUITE("MLIR Generation Tests") {}

TEST_CASE("Gen Test" * doctest::test_suite("MLIR Generation Tests")) {
  MLIR_GEN_TEST("Toy Simple Program 1", R"(

def main() {
  var a = [1, 2, 3, 4, 5, 6]; 
  return;
}

)",
                R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    toy.return
  }
}
)");

  MLIR_GEN_TEST("Toy Simple Program 2", R"(

def transpose_transpose(a) {
  return transpose(transpose(a));
}

def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  transpose_transpose(a);
  return;
}
)",
                R"(
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
    %2 = toy.generic_call @transpose_transpose(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
    toy.return
  }
}

)");
}

} // namespace toy::test
