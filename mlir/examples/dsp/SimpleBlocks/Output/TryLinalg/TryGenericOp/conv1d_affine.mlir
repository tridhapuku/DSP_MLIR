#map = affine_map<(d0, d1) -> (d0 + d1)>  //from linalg.generic
//#map = affine_map<(d0, d1) -> (d0 - d1)>  //from the actual definition
module {
  func.func @main() {
    //input from else
    %cst = arith.constant 6.000000e+01 : f64
    %cst_0 = arith.constant 5.000000e+01 : f64
    %cst_1 = arith.constant 4.000000e+01 : f64
    %cst_2 = arith.constant 3.000000e+01 : f64
    %cst_3 = arith.constant 2.000000e+01 : f64
    %cst_4 = arith.constant 1.000000e+01 : f64
    // %alloc = memref.alloc() : memref<3xf64>
    %alloc = memref.alloc() : memref<4xf64>
    %alloc_5 = memref.alloc() : memref<3xf64>
    %alloc_6 = memref.alloc() : memref<3xf64>
    affine.store %cst_4, %alloc_6[0] : memref<3xf64>
    affine.store %cst_3, %alloc_6[1] : memref<3xf64>
    affine.store %cst_2, %alloc_6[2] : memref<3xf64>
    affine.store %cst_1, %alloc_5[0] : memref<3xf64>
    affine.store %cst_0, %alloc_5[1] : memref<3xf64>
    affine.store %cst, %alloc_5[2] : memref<3xf64>
    
    %c0 = arith.constant 0 : index
    %dim = memref.dim %alloc_5, %c0 : memref<3xf64>
    %dim_0 = memref.dim %alloc_6, %c0 : memref<3xf64>
    affine.for %arg3 = 0 to %dim_0 {
      affine.for %arg4 = 0 to %dim {
        %0 = affine.apply #map(%arg3, %arg4)
        %1 = affine.load %alloc_5[%0] : memref<3xf64>
        %2 = affine.load %alloc_6[%arg4] : memref<3xf64>
        %3 = affine.load %alloc[%arg3] : memref<4xf64>
        %4 = arith.mulf %1, %2 : f64
        %5 = arith.addf %3, %4 : f64
        affine.store %5, %alloc[%arg3] : memref<4xf64>
      }
    }
    dsp.print %alloc : memref<4xf64>
    memref.dealloc %alloc_6 : memref<3xf64>
    memref.dealloc %alloc_5 : memref<3xf64>
    memref.dealloc %alloc : memref<4xf64>
    return
  }
}

  //#map = affine_map<(d0, d1) -> (d0 + d1)>
  // func.func @conv1d_no_symbols(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  //   %c0 = arith.constant 0 : index
  //   %dim = memref.dim %arg1, %c0 : memref<?xf32>
  //   %dim_0 = memref.dim %arg2, %c0 : memref<?xf32>
  //   affine.for %arg3 = 0 to %dim_0 {
  //     affine.for %arg4 = 0 to %dim {
  //       %0 = affine.apply #map(%arg3, %arg4)
  //       %1 = affine.load %arg0[%0] : memref<?xf32>
  //       %2 = affine.load %arg1[%arg4] : memref<?xf32>
  //       %3 = affine.load %arg2[%arg3] : memref<?xf32>
  //       %4 = arith.mulf %1, %2 : f32
  //       %5 = arith.addf %3, %4 : f32
  //       affine.store %5, %arg2[%arg3] : memref<?xf32>
  //     }
  //   }
  //   return
  // }

  // for i = to n
      //  for j = 0 to k
      //    o[i] = o[i] + h[i+j] * x[j]
