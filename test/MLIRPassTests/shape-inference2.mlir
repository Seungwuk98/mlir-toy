// RUN: toy-opt %s --toy-shape-inference | FileCheck %s

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

// CHECK-LABEL:   toy.func @main() 
// CHECK-NEXT:     %0 = toy.struct_constant [dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>, dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf64>] : !toy.struct<tensor<3xf64>, tensor<3xf64>>
// CHECK-NEXT:     %1 = toy.struct_constant [dense<[7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<3xf64>, dense<[1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<3xf64>] : !toy.struct<tensor<3xf64>, tensor<3xf64>>
// CHECK-NEXT:     %2 = toy.cast %0 : !toy.struct<tensor<3xf64>, tensor<3xf64>> to !toy.struct<tensor<3xf64>, tensor<3xf64>>
// CHECK-NEXT:     %3 = toy.cast %1 : !toy.struct<tensor<3xf64>, tensor<3xf64>> to !toy.struct<tensor<3xf64>, tensor<3xf64>>
// CHECK-NEXT:     %4 = toy.struct_access %2[0] : !toy.struct<tensor<3xf64>, tensor<3xf64>> -> tensor<3xf64>
// CHECK-NEXT:     %5 = toy.struct_access %3[0] : !toy.struct<tensor<3xf64>, tensor<3xf64>> -> tensor<3xf64>
// CHECK-NEXT:     %6 = toy.mul %4, %5 : tensor<3xf64>
// CHECK-NEXT:     toy.print %6 : tensor<3xf64>
// CHECK-NEXT:     toy.return
