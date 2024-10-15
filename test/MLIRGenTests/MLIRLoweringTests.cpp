#include "MLIRTestUtils.h"
#include "TestExtras.h"
#include "doctest/doctest.h"

namespace toy::test {

TEST_CASE("MLIR Lowering Tests" *
          doctest::test_suite("MLIR Generation Tests")) {

  MLIR_AFFINE_LOWER_TEST("One Transpose", R"(
def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6]; 
  var b = transpose(a);
  print(b);                
}
)",
                         R"(
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_0 = memref.alloc() : memref<2x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 4.000000e+00 : f64
    %cst_4 = arith.constant 5.000000e+00 : f64
    %cst_5 = arith.constant 6.000000e+00 : f64
    affine.store %cst, %alloc_0[%c0, %c0] : memref<2x3xf64>
    affine.store %cst_1, %alloc_0[%c0, %c1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_0[%c0, %c2] : memref<2x3xf64>
    affine.store %cst_3, %alloc_0[%c1, %c0] : memref<2x3xf64>
    affine.store %cst_4, %alloc_0[%c1, %c1] : memref<2x3xf64>
    affine.store %cst_5, %alloc_0[%c1, %c2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_0[%arg1, %arg0] : memref<2x3xf64>
        memref.store %0, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_0 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
)");

  MLIR_AFFINE_LOWER_TEST("Transpose Multiplication", R"(
def transpose_multiply(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  var b<2, 3> = [6, 5, 4, 3, 2, 1];
  print(transpose_multiply(a, b));
}
)",
                         R"(
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_0 = memref.alloc() : memref<3x2xf64>
    %alloc_1 = memref.alloc() : memref<3x2xf64>
    %alloc_2 = memref.alloc() : memref<2x3xf64>
    %alloc_3 = memref.alloc() : memref<2x3xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 3.000000e+00 : f64
    %cst_6 = arith.constant 4.000000e+00 : f64
    %cst_7 = arith.constant 5.000000e+00 : f64
    %cst_8 = arith.constant 6.000000e+00 : f64
    affine.store %cst, %alloc_3[%c0, %c0] : memref<2x3xf64>
    affine.store %cst_4, %alloc_3[%c0, %c1] : memref<2x3xf64>
    affine.store %cst_5, %alloc_3[%c0, %c2] : memref<2x3xf64>
    affine.store %cst_6, %alloc_3[%c1, %c0] : memref<2x3xf64>
    affine.store %cst_7, %alloc_3[%c1, %c1] : memref<2x3xf64>
    affine.store %cst_8, %alloc_3[%c1, %c2] : memref<2x3xf64>
    %c0_9 = arith.constant 0 : index
    %c1_10 = arith.constant 1 : index
    %c2_11 = arith.constant 2 : index
    %cst_12 = arith.constant 6.000000e+00 : f64
    %cst_13 = arith.constant 5.000000e+00 : f64
    %cst_14 = arith.constant 4.000000e+00 : f64
    %cst_15 = arith.constant 3.000000e+00 : f64
    %cst_16 = arith.constant 2.000000e+00 : f64
    %cst_17 = arith.constant 1.000000e+00 : f64
    affine.store %cst_12, %alloc_2[%c0_9, %c0_9] : memref<2x3xf64>
    affine.store %cst_13, %alloc_2[%c0_9, %c1_10] : memref<2x3xf64>
    affine.store %cst_14, %alloc_2[%c0_9, %c2_11] : memref<2x3xf64>
    affine.store %cst_15, %alloc_2[%c1_10, %c0_9] : memref<2x3xf64>
    affine.store %cst_16, %alloc_2[%c1_10, %c1_10] : memref<2x3xf64>
    affine.store %cst_17, %alloc_2[%c1_10, %c2_11] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_3[%arg1, %arg0] : memref<2x3xf64>
        memref.store %0, %alloc_1[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_2[%arg1, %arg0] : memref<2x3xf64>
        memref.store %0, %alloc_0[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_1[%arg0, %arg1] : memref<3x2xf64>
        %1 = affine.load %alloc_0[%arg0, %arg1] : memref<3x2xf64>
        %2 = arith.mulf %0, %1 : f64
        memref.store %2, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_3 : memref<2x3xf64>
    memref.dealloc %alloc_2 : memref<2x3xf64>
    memref.dealloc %alloc_1 : memref<3x2xf64>
    memref.dealloc %alloc_0 : memref<3x2xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
)");

  MLIR_AFFINE_LOWER_TEST("Add", R"(
def main() {
  var a = [1, 2, 3, 4, 5, 6];
  var b = [6, 5, 4, 3, 2, 1];
  print(a + b);
}
)",
                         R"(
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<6xf64>
    %alloc_0 = memref.alloc() : memref<6xf64>
    %alloc_1 = memref.alloc() : memref<6xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 1.000000e+00 : f64
    %cst_2 = arith.constant 2.000000e+00 : f64
    %cst_3 = arith.constant 3.000000e+00 : f64
    %cst_4 = arith.constant 4.000000e+00 : f64
    %cst_5 = arith.constant 5.000000e+00 : f64
    %cst_6 = arith.constant 6.000000e+00 : f64
    affine.store %cst, %alloc_1[%c0] : memref<6xf64>
    affine.store %cst_2, %alloc_1[%c1] : memref<6xf64>
    affine.store %cst_3, %alloc_1[%c2] : memref<6xf64>
    affine.store %cst_4, %alloc_1[%c3] : memref<6xf64>
    affine.store %cst_5, %alloc_1[%c4] : memref<6xf64>
    affine.store %cst_6, %alloc_1[%c5] : memref<6xf64>
    %c0_7 = arith.constant 0 : index
    %c1_8 = arith.constant 1 : index
    %c2_9 = arith.constant 2 : index
    %c3_10 = arith.constant 3 : index
    %c4_11 = arith.constant 4 : index
    %c5_12 = arith.constant 5 : index
    %cst_13 = arith.constant 6.000000e+00 : f64
    %cst_14 = arith.constant 5.000000e+00 : f64
    %cst_15 = arith.constant 4.000000e+00 : f64
    %cst_16 = arith.constant 3.000000e+00 : f64
    %cst_17 = arith.constant 2.000000e+00 : f64
    %cst_18 = arith.constant 1.000000e+00 : f64
    affine.store %cst_13, %alloc_0[%c0_7] : memref<6xf64>
    affine.store %cst_14, %alloc_0[%c1_8] : memref<6xf64>
    affine.store %cst_15, %alloc_0[%c2_9] : memref<6xf64>
    affine.store %cst_16, %alloc_0[%c3_10] : memref<6xf64>
    affine.store %cst_17, %alloc_0[%c4_11] : memref<6xf64>
    affine.store %cst_18, %alloc_0[%c5_12] : memref<6xf64>
    affine.for %arg0 = 0 to 6 {
      %0 = affine.load %alloc_1[%arg0] : memref<6xf64>
      %1 = affine.load %alloc_0[%arg0] : memref<6xf64>
      %2 = arith.addf %0, %1 : f64
      memref.store %2, %alloc[%arg0] : memref<6xf64>
    }
    toy.print %alloc : memref<6xf64>
    memref.dealloc %alloc_1 : memref<6xf64>
    memref.dealloc %alloc_0 : memref<6xf64>
    memref.dealloc %alloc : memref<6xf64>
    return
  }
}
)");

  MLIR_AFFINE_LOWER_TEST("Add a value", R"(
def main() {
  var a = 1;
  var b = 2;
  print(a + b);
}
)",
                         R"(
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<f64>
    %alloc_0 = memref.alloc() : memref<f64>
    %alloc_1 = memref.alloc() : memref<f64>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    affine.store %cst, %alloc_1[] : memref<f64>
    %c0_2 = arith.constant 0 : index
    %cst_3 = arith.constant 2.000000e+00 : f64
    affine.store %cst_3, %alloc_0[] : memref<f64>
    %0 = affine.load %alloc_1[] : memref<f64>
    %1 = affine.load %alloc_0[] : memref<f64>
    %2 = arith.addf %0, %1 : f64
    memref.store %2, %alloc[] : memref<f64>
    toy.print %alloc : memref<f64>
    memref.dealloc %alloc_1 : memref<f64>
    memref.dealloc %alloc_0 : memref<f64>
    memref.dealloc %alloc : memref<f64>
    return
  }
})");
}

} // namespace toy::test
