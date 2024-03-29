// module {
//   func.func @add_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
//     linalg.add ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
//     return
//   }
// }

module {
//   func.func @add_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
//     linalg.add ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
//     return
//   }


  func.func @conv1d_no_symbols(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    linalg.conv_1d ins(%arg0, %arg1 : memref<?xf32>, memref<?xf32>) outs(%arg2 : memref<?xf32>)
    return
  }

  //   func.func @conv1d_no_symbols(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  //   // arith.constant dense
  //   // %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  //   linalg.conv_1d ins(%arg0, %arg1 : memref<?xf32>, memref<?xf32>) outs(%arg2 : memref<?xf32>)
  //   return
  // }

    //   func.func @conv1d_no_symbols(%arg0: memref<2xf32>, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
    // // arith.constant dense
    // // %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
    // linalg.conv_1d ins(%arg0, %arg1 : memref<2xf32>, memref<2xf32>) outs(%arg2 : memref<4xf32>)
    // linalg.add ins(%lhs, %rhs : memref<7x14x21xf32>, memref<7x14x21xf32>)
    //          outs(%out : memref<7x14x21xf32>)
    // return
    // }

    func.func @generalize_add() -> memref<2x2xf32> {
      %lhs = arith.constant dense<[[1, 2], [3, 4]]> : memref<2x2xf32>
      %rhs = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : memref<2x2xf32>
      %0 = linalg.add ins(%lhs, %rhs : memref<2x2xf32>, memref<2x2xf32>)
             outs(%out : memref<2x2xf32>)

      return %0 : memref<2x2xf32>
}
  
}


