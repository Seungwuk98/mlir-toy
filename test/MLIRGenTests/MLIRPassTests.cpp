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
    %1 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %2 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %3 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %3, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
)");

  MLIR_INLINE_PASS_TEST("Inline Tests 2", R"(
struct X {
  var x;
  var y;
}

def multiply_x(X a, X b) {
  return a.x * b.x;
}
def main() {
  X a = {[1, 2, 3], [4, 5, 6]};
  X b = {[7, 8, 9], [10, 11, 12]};
  var c = multiply_x(a, b);
  print(c);
}
)",
                        R"(
module {
  toy.func @main() {
    %0 = toy.struct_constant [dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>, dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf64>] : !toy.struct<tensor<3xf64>, tensor<3xf64>>
    %1 = toy.struct_constant [dense<[7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<3xf64>, dense<[1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<3xf64>] : !toy.struct<tensor<3xf64>, tensor<3xf64>>
    %2 = toy.cast %0 : !toy.struct<tensor<3xf64>, tensor<3xf64>> to !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %3 = toy.cast %1 : !toy.struct<tensor<3xf64>, tensor<3xf64>> to !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %4 = toy.struct_access %2[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %5 = toy.struct_access %3[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
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
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %3 = toy.mul %1, %2 : tensor<3x2xf64>
    toy.print %3 : tensor<3x2xf64>
    toy.return
  }
}
)");

  MLIR_SI_PASS_TEST("Simple Shape Inference 2", R"(
struct X {
  var x;
  var y;
}

def multiply_x(X a, X b) {
  return a.x * b.x;
}
def main() {
  X a = {[1, 2, 3], [4, 5, 6]};
  X b = {[7, 8, 9], [10, 11, 12]};
  var c = multiply_x(a, b);
  print(c);
})",
                    R"(
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
    %1 = toy.constant dense<[7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<3xf64>
    %2 = toy.mul %0, %1 : tensor<3xf64>
    toy.print %2 : tensor<3xf64>
    toy.return
  }
}
)");
}

} // namespace toy::test
