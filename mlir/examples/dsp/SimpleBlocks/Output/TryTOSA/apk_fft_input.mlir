module {
  func.func @test_fft2d(%arg0: memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>) -> (memref<1x2x2xf32>, memref<1x2x2xf32>) {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 6.28318548 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 2 {
          affine.store %cst_1, %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 2 {
          affine.store %cst_1, %alloc_2[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 2 {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 2 {
              %0 = affine.load %arg0[%arg2, %arg5, %arg6] : memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>
              %1 = affine.load %arg1[%arg2, %arg5, %arg6] : memref<1x2x2xf32, strided<[?, ?, ?], offset: ?>>
              %2 = affine.load %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
              %3 = affine.load %alloc_2[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
              %4 = arith.index_castui %arg3 : index to i32
              %5 = arith.uitofp %4 : i32 to f32
              %6 = arith.index_castui %arg4 : index to i32
              %7 = arith.uitofp %6 : i32 to f32
              %8 = arith.index_castui %arg5 : index to i32
              %9 = arith.uitofp %8 : i32 to f32
              %10 = arith.index_castui %arg6 : index to i32
              %11 = arith.uitofp %10 : i32 to f32
              %12 = arith.mulf %9, %5 : f32
              %13 = arith.mulf %11, %7 : f32
              %14 = arith.divf %12, %cst : f32
              %15 = arith.divf %13, %cst : f32
              %16 = arith.addf %14, %15 : f32
              %17 = arith.mulf %16, %cst_0 : f32
              %18 = math.cos %17 : f32
              %19 = math.sin %17 : f32
              %20 = arith.mulf %0, %18 : f32
              %21 = arith.mulf %1, %19 : f32
              %22 = arith.addf %20, %21 : f32
              %23 = arith.mulf %1, %18 : f32
              %24 = arith.mulf %0, %19 : f32
              %25 = arith.subf %23, %24 : f32
              %26 = arith.addf %2, %22 : f32
              %27 = arith.addf %3, %25 : f32
              affine.store %26, %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
              affine.store %27, %alloc_2[%arg2, %arg3, %arg4] : memref<1x2x2xf32>
            }
          }
        }
      }
    }
    return %alloc, %alloc_2 : memref<1x2x2xf32>, memref<1x2x2xf32>
  }
}

