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

  
}


