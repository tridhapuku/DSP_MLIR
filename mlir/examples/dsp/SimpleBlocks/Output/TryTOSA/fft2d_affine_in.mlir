module {
  func.func @main() {
    //Input from other file 
    %cst_11 = arith.constant 0.000000e+01 : f64
    %cst_12 = arith.constant 0.000000e+01 : f64
    %cst_13 = arith.constant 0.000000e+01 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %cst_3 = arith.constant 3.000000e+01 : f64
    %cst_4 = arith.constant 2.000000e+01 : f64
    %cst_5 = arith.constant 1.000000e+01 : f64
    // %alloc = memref.alloc() : memref<1x2x2xf64>
    %arg0 = memref.alloc() : memref<1x2x2xf64>
    %arg1 = memref.alloc() : memref<1x2x2xf64>
    affine.store %cst_5, %arg1[0, 0, 0] : memref<1x2x2xf64>
    affine.store %cst_4, %arg1[0, 0, 1] : memref<1x2x2xf64>
    affine.store %cst_3, %arg1[0, 1, 0] : memref<1x2x2xf64>
    affine.store %cst_2, %arg1[0, 1, 1] : memref<1x2x2xf64>
    affine.store %cst_2, %arg0[0, 0, 0] : memref<1x2x2xf64>
    affine.store %cst_13, %arg0[0, 0, 1] : memref<1x2x2xf64>
    affine.store %cst_12, %arg0[0, 1, 0] : memref<1x2x2xf64>
    affine.store %cst_11, %arg0[0, 1, 1] : memref<1x2x2xf64>


    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 6.28318548 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf64>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 2 {
          affine.store %cst_1, %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf64>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf64>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 2 {
          affine.store %cst_1, %alloc_2[%arg2, %arg3, %arg4] : memref<1x2x2xf64>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 2 {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 2 {
              %0 = affine.load %arg0[%arg2, %arg5, %arg6] : memref<1x2x2xf64>
              %1 = affine.load %arg1[%arg2, %arg5, %arg6] : memref<1x2x2xf64>
              %2 = affine.load %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf64>
              %3 = affine.load %alloc_2[%arg2, %arg3, %arg4] : memref<1x2x2xf64>
              %4 = arith.index_castui %arg3 : index to i32
              %5 = arith.uitofp %4 : i32 to f64
              %6 = arith.index_castui %arg4 : index to i32
              %7 = arith.uitofp %6 : i32 to f64
              %8 = arith.index_castui %arg5 : index to i32
              %9 = arith.uitofp %8 : i32 to f64
              %10 = arith.index_castui %arg6 : index to i32
              %11 = arith.uitofp %10 : i32 to f64
              %12 = arith.mulf %9, %5 : f64
              %13 = arith.mulf %11, %7 : f64
              %14 = arith.divf %12, %cst : f64
              %15 = arith.divf %13, %cst : f64
              %16 = arith.addf %14, %15 : f64
              %17 = arith.mulf %16, %cst_0 : f64
              %18 = math.cos %17 : f64
              %19 = math.sin %17 : f64
              %20 = arith.mulf %0, %18 : f64
              %21 = arith.mulf %1, %19 : f64
              %22 = arith.addf %20, %21 : f64
              %23 = arith.mulf %1, %18 : f64
              %24 = arith.mulf %0, %19 : f64
              %25 = arith.subf %23, %24 : f64
              %26 = arith.addf %2, %22 : f64
              %27 = arith.addf %3, %25 : f64
              affine.store %26, %alloc[%arg2, %arg3, %arg4] : memref<1x2x2xf64>
              affine.store %27, %alloc_2[%arg2, %arg3, %arg4] : memref<1x2x2xf64>
            }
          }
        }
      }
    }
    dsp.print %alloc : memref<1x2x2xf64>
    dsp.print %alloc_2 : memref<1x2x2xf64>
    memref.dealloc %arg0 : memref<1x2x2xf64>
    memref.dealloc %arg1 : memref<1x2x2xf64>
    memref.dealloc %alloc : memref<1x2x2xf64>
    //return %alloc, %alloc_2 : memref<1x2x2xf64>, memref<1x2x2xf64>
    return 
  }
}

