module {
  func.func @test_static_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant 6.28318548 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_memref %arg1 : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_2, %alloc[%arg2, %arg3, %arg4] : memref<1x4x8xf32>
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_2, %alloc_3[%arg2, %arg3, %arg4] : memref<1x4x8xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 4 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 8 {
              %4 = affine.load %1[%arg2, %arg5, %arg6] : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>
              %5 = affine.load %0[%arg2, %arg5, %arg6] : memref<1x4x8xf32, strided<[?, ?, ?], offset: ?>>
              %6 = affine.load %alloc[%arg2, %arg3, %arg4] : memref<1x4x8xf32>
              %7 = affine.load %alloc_3[%arg2, %arg3, %arg4] : memref<1x4x8xf32>
              %8 = arith.index_castui %arg3 : index to i32
              %9 = arith.uitofp %8 : i32 to f32
              %10 = arith.index_castui %arg4 : index to i32
              %11 = arith.uitofp %10 : i32 to f32
              %12 = arith.index_castui %arg5 : index to i32
              %13 = arith.uitofp %12 : i32 to f32
              %14 = arith.index_castui %arg6 : index to i32
              %15 = arith.uitofp %14 : i32 to f32
              %16 = arith.mulf %13, %9 : f32
              %17 = arith.mulf %15, %11 : f32
              %18 = arith.divf %16, %cst_0 : f32
              %19 = arith.divf %17, %cst : f32
              %20 = arith.addf %18, %19 : f32
              %21 = arith.mulf %20, %cst_1 : f32
              %22 = math.cos %21 : f32
              %23 = math.sin %21 : f32
              %24 = arith.mulf %4, %22 : f32
              %25 = arith.mulf %5, %23 : f32
              %26 = arith.addf %24, %25 : f32
              %27 = arith.mulf %5, %22 : f32
              %28 = arith.mulf %4, %23 : f32
              %29 = arith.subf %27, %28 : f32
              %30 = arith.addf %6, %26 : f32
              %31 = arith.addf %7, %29 : f32
              affine.store %30, %alloc[%arg2, %arg3, %arg4] : memref<1x4x8xf32>
              affine.store %31, %alloc_3[%arg2, %arg3, %arg4] : memref<1x4x8xf32>
            }
          }
        }
      }
    }
    %2 = bufferization.to_tensor %alloc_3 : memref<1x4x8xf32>
    %3 = bufferization.to_tensor %alloc : memref<1x4x8xf32>
    return %3, %2 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
  }
}

