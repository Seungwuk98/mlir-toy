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
  toy.func private @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
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

  MLIR_GEN_TEST("Toy Simple Program 3", R"(
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
)",
                R"(
module {
  toy.func private @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.struct_access %arg0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.mul %1, %3 : tensor<*xf64>
    toy.return %4 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.struct_constant [dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>, dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>] : !toy.struct<tensor<2x3xf64>, tensor<2x3xf64>>
    %3 = toy.generic_call @multiply_transpose(%2) : (!toy.struct<tensor<2x3xf64>, tensor<2x3xf64>>) -> tensor<*xf64>
    toy.print %3 : tensor<*xf64>
    toy.return
  }
}
)");

  MLIR_GEN_TEST("Toy Simple Program 4", R"(
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
  toy.func private @multiply_x(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>, %arg1: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.struct_access %arg1[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
    %1 = toy.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf64>
    %2 = toy.struct_constant [dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>, dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf64>] : !toy.struct<tensor<3xf64>, tensor<3xf64>>
    %3 = toy.constant dense<[7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<3xf64>
    %4 = toy.constant dense<[1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<3xf64>
    %5 = toy.struct_constant [dense<[7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<3xf64>, dense<[1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<3xf64>] : !toy.struct<tensor<3xf64>, tensor<3xf64>>
    %6 = toy.generic_call @multiply_x(%2, %5) : (!toy.struct<tensor<3xf64>, tensor<3xf64>>, !toy.struct<tensor<3xf64>, tensor<3xf64>>) -> tensor<*xf64>
    toy.print %6 : tensor<*xf64>
    toy.return
  }
}

)");
}

} // namespace toy::test
