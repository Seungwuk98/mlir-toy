// RUN: toy-opt %s --toy-shape-inference | FileCheck %s

module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
    %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
    %3 = toy.mul %1, %2 : tensor<*xf64>
    toy.print %3 : tensor<*xf64>
    toy.return
  }
}

// CHECK-LABEL:   toy.func @main() 
// CHECK-NEXT:     %0 = toy.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
// CHECK-NEXT:     %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
// CHECK-NEXT:     %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
// CHECK-NEXT:     %3 = toy.mul %1, %2 : tensor<3x2xf64>
// CHECK-NEXT:     toy.print %3 : tensor<3x2xf64>
// CHECK-NEXT:     toy.return
