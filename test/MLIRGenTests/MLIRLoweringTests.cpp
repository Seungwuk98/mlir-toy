#include "MLIRTestUtils.h"
#include "TestExtras.h"
#include "doctest/doctest.h"

namespace toy::test {

TEST_CASE("MLIR Affine Lowering Tests" *
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

TEST_CASE("MLIR LLVM Lowering Tests" *
          doctest::test_suite("MLIR Generation Tests")) {
  MLIR_LLVM_LOWER_TEST("Print tensor", R"(

def main() {
  print([1, 2, 3, 4, 5, 6]);
}
)",
                       R"(
module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @"@printf.newline"("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"@printf.float.space"("%f \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"@printf.float"("%f\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.mlir.constant(6 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %5, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.insertvalue %9, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %0, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %1, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(3 : index) : i64
    %17 = llvm.mlir.constant(4 : index) : i64
    %18 = llvm.mlir.constant(5 : index) : i64
    %19 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %20 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %21 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %22 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %23 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %24 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %25 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.getelementptr %25[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %19, %26 : f64, !llvm.ptr
    %27 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.getelementptr %27[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %20, %28 : f64, !llvm.ptr
    %29 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.getelementptr %29[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %21, %30 : f64, !llvm.ptr
    %31 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.getelementptr %31[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %22, %32 : f64, !llvm.ptr
    %33 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.getelementptr %33[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %23, %34 : f64, !llvm.ptr
    %35 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.getelementptr %35[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %24, %36 : f64, !llvm.ptr
    %37 = llvm.mlir.constant(6 : index) : i64
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.mlir.addressof @"@printf.float" : !llvm.ptr
    %41 = llvm.mlir.constant(0 : i64) : i64
    %42 = llvm.getelementptr %40[%41] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<3 x i8>
    %43 = llvm.mlir.addressof @"@printf.float.space" : !llvm.ptr
    %44 = llvm.mlir.constant(0 : i64) : i64
    %45 = llvm.getelementptr %43[%44] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %46 = llvm.mlir.addressof @"@printf.newline" : !llvm.ptr
    %47 = llvm.mlir.constant(0 : i64) : i64
    %48 = llvm.getelementptr %46[%47] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    llvm.br ^bb1(%38 : i64)
  ^bb1(%49: i64):  // 2 preds: ^bb0, ^bb5
    %50 = llvm.icmp "slt" %49, %37 : i64
    llvm.cond_br %50, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %51 = llvm.mlir.constant(5 : index) : i64
    %52 = llvm.icmp "ne" %49, %51 : i64
    %53 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.getelementptr %53[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %55 = llvm.load %54 : !llvm.ptr -> f64
    llvm.cond_br %52, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %56 = llvm.call @printf(%45, %55) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    %57 = llvm.call @printf(%42, %55) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %58 = llvm.add %49, %39  : i64
    llvm.br ^bb1(%58 : i64)
  ^bb6:  // pred: ^bb1
    %59 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%59) : (!llvm.ptr) -> ()
    llvm.return
  }
})");

  MLIR_LLVM_LOWER_TEST("Print tensor - 2", R"(
def main() {
  print([[1, 2, 3], [4, 5, 6]]);
}
)",
                       R"(
module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @"@printf.newline"("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"@printf.float.space"("%f \00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @"@printf.float"("%f\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(3 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(6 : index) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %6 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %0, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %1, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %1, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %2, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.constant(2 : index) : i64
    %20 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %21 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %22 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %23 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %24 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %25 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %26 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(3 : index) : i64
    %28 = llvm.mul %17, %27  : i64
    %29 = llvm.add %28, %17  : i64
    %30 = llvm.getelementptr %26[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %20, %30 : f64, !llvm.ptr
    %31 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.constant(3 : index) : i64
    %33 = llvm.mul %17, %32  : i64
    %34 = llvm.add %33, %18  : i64
    %35 = llvm.getelementptr %31[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %21, %35 : f64, !llvm.ptr
    %36 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(3 : index) : i64
    %38 = llvm.mul %17, %37  : i64
    %39 = llvm.add %38, %19  : i64
    %40 = llvm.getelementptr %36[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %22, %40 : f64, !llvm.ptr
    %41 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(3 : index) : i64
    %43 = llvm.mul %18, %42  : i64
    %44 = llvm.add %43, %17  : i64
    %45 = llvm.getelementptr %41[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %23, %45 : f64, !llvm.ptr
    %46 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.mlir.constant(3 : index) : i64
    %48 = llvm.mul %18, %47  : i64
    %49 = llvm.add %48, %18  : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %24, %50 : f64, !llvm.ptr
    %51 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(3 : index) : i64
    %53 = llvm.mul %18, %52  : i64
    %54 = llvm.add %53, %19  : i64
    %55 = llvm.getelementptr %51[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %25, %55 : f64, !llvm.ptr
    %56 = llvm.mlir.constant(2 : index) : i64
    %57 = llvm.mlir.constant(3 : index) : i64
    %58 = llvm.mlir.constant(0 : index) : i64
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.mlir.addressof @"@printf.float" : !llvm.ptr
    %61 = llvm.mlir.constant(0 : i64) : i64
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<3 x i8>
    %63 = llvm.mlir.addressof @"@printf.float.space" : !llvm.ptr
    %64 = llvm.mlir.constant(0 : i64) : i64
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %66 = llvm.mlir.addressof @"@printf.newline" : !llvm.ptr
    %67 = llvm.mlir.constant(0 : i64) : i64
    %68 = llvm.getelementptr %66[%67] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i8>
    llvm.br ^bb1(%58 : i64)
  ^bb1(%69: i64):  // 2 preds: ^bb0, ^bb10
    %70 = llvm.icmp "slt" %69, %56 : i64
    llvm.cond_br %70, ^bb2, ^bb11
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%58 : i64)
  ^bb3(%71: i64):  // 2 preds: ^bb2, ^bb7
    %72 = llvm.icmp "slt" %71, %57 : i64
    llvm.cond_br %72, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %73 = llvm.mlir.constant(2 : index) : i64
    %74 = llvm.icmp "ne" %71, %73 : i64
    %75 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mlir.constant(3 : index) : i64
    %77 = llvm.mul %69, %76  : i64
    %78 = llvm.add %77, %71  : i64
    %79 = llvm.getelementptr %75[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %80 = llvm.load %79 : !llvm.ptr -> f64
    llvm.cond_br %74, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %81 = llvm.call @printf(%65, %80) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.br ^bb7
  ^bb6:  // pred: ^bb4
    %82 = llvm.call @printf(%62, %80) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %83 = llvm.add %71, %59  : i64
    llvm.br ^bb3(%83 : i64)
  ^bb8:  // pred: ^bb3
    %84 = llvm.mlir.constant(1 : index) : i64
    %85 = llvm.icmp "ne" %69, %84 : i64
    llvm.cond_br %85, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %86 = llvm.call @printf(%68) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %87 = llvm.add %69, %59  : i64
    llvm.br ^bb1(%87 : i64)
  ^bb11:  // pred: ^bb1
    %88 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%88) : (!llvm.ptr) -> ()
    llvm.return
  }
})");
}

} // namespace toy::test
