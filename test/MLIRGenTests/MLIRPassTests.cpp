#include "MLIRTestUtils.h"
#include "doctest/doctest.h"

namespace toy::test {

TEST_CASE("MLIR Pass Tests" * doctest::test_suite("MLIR Generation Tests")) {

  MLIR_PASS_TEST("Nested Transpose", R"(
def transpose_transpose(a) {
  return transpose(transpose(a));
}
)",
                 R"(
module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
}
)");

  MLIR_PASS_TEST("Redundant Reshape", R"(
def main() {
  var a<2, 1> = [1, 2];
  var b<2, 1> = a;
  var c<2, 1> = b;
  print(c);
}
)",
                 R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
)");

  MLIR_PASS_TEST("Multiple Reshape", R"(
def main() {
  var a = [1, 2, 3, 4, 5, 6];
  var b<2, 3> = a;
  var c<3, 2> = b;
  print(c);
}
)",
                 R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]> : tensor<3x2xf64>
    toy.print %0 : tensor<3x2xf64>
    toy.return
  }
}
)");
}

} // namespace toy::test
