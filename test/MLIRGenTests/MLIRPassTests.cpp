#include "MLIRTestUtils.h"
#include "doctest/doctest.h"

namespace toy::test {

TEST_CASE("MLIR Canonicalizer Pass Tests" *
          doctest::test_suite("MLIR Generation Tests")) {

  MLIR_CANO_PASS_TEST("Nested Transpose", R"(
def transpose_transpose(a) {
  return transpose(transpose(a));
}
)",
                      R"(
module {
  toy.func private @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
}
)");

  MLIR_CANO_PASS_TEST("Redundant Reshape", R"(
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

  MLIR_CANO_PASS_TEST("Multiple Reshape", R"(
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

TEST_CASE("MLIR Inliner Pass Tests" *
          doctest::test_suite("MLIR Generation Tests")) {
  MLIR_INLINE_PASS_TEST("Inline Tests", R"(
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var x = multiply_transpose(a, b);
  var y = multiply_transpose(b, a);
  print(y);
}
)",
                        R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
    %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %6 = toy.mul %4, %5 : tensor<*xf64>
    toy.print %6 : tensor<*xf64>
    toy.return
  }
}
)");
}

TEST_CASE("MLIR Shape Inference Pass Tests" *
          doctest::test_suite("MLIR Generation Tests")) {

  MLIR_SI_PASS_TEST("Simple Shape Inference", R"(
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var x = multiply_transpose(a, b);
  var y = multiply_transpose(b, a);
  print(y);
})",
                    R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
    %3 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %4 = toy.mul %2, %3 : tensor<3x2xf64>
    toy.print %4 : tensor<3x2xf64>
    toy.return
  }
}

)");
}

TEST_CASE("MLIR Toy To Affine Lowering Pass Tests" *
          doctest::test_suite("MLIR Generation Tests")) {}
} // namespace toy::test
