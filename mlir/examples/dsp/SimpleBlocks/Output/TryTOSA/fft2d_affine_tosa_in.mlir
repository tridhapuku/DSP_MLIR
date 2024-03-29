module {
  // memref.global "private" constant @__constant_1x2x2xf64_0 : memref<1x2x2xf64> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x2x2xf64_0 : memref<1x2x2xf64> = dense<[[[1.000000e+01, 2.000000e+01], [3.000000e+01, 4.000000e+01]]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x2x2xf64 : memref<1x2x2xf64> = dense<[[[1.000000e+01, 2.000000e+01], [3.000000e+01, 4.000000e+01]]]> {alignment = 64 : i64}
  func.func @main()  {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 6.28318548 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %0 = memref.get_global @__constant_1x2x2xf64 : memref<1x2x2xf64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf64>
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 2 {
        affine.for %arg2 = 0 to 2 {
          affine.store %cst_1, %alloc[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf64>
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 2 {
        affine.for %arg2 = 0 to 2 {
          affine.store %cst_1, %alloc_2[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
        }
      }
    }
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 2 {
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 2 {
              %1 = affine.load %0[%arg0, %arg3, %arg4] : memref<1x2x2xf64>
              %2 = affine.load %alloc[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
              %3 = affine.load %alloc_2[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
              %4 = arith.index_castui %arg1 : index to i32
              %5 = arith.uitofp %4 : i32 to f64
              %6 = arith.index_castui %arg2 : index to i32
              %7 = arith.uitofp %6 : i32 to f64
              %8 = arith.index_castui %arg3 : index to i32
              %9 = arith.uitofp %8 : i32 to f64
              %10 = arith.index_castui %arg4 : index to i32
              %11 = arith.uitofp %10 : i32 to f64
              %12 = arith.mulf %9, %5 : f64
              %13 = arith.mulf %11, %7 : f64
              %14 = arith.divf %12, %cst : f64
              %15 = arith.divf %13, %cst : f64
              %16 = arith.addf %14, %15 : f64
              %17 = arith.mulf %16, %cst_0 : f64
              %18 = math.cos %17 : f64
              %19 = math.sin %17 : f64
              %20 = arith.mulf %1, %18 : f64
              %21 = arith.mulf %19, %cst_1 : f64
              %22 = arith.addf %20, %21 : f64
              %23 = arith.mulf %18, %cst_1 : f64
              %24 = arith.mulf %1, %19 : f64
              %25 = arith.subf %23, %24 : f64
              %26 = arith.addf %2, %22 : f64
              %27 = arith.addf %3, %25 : f64
              affine.store %26, %alloc[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
              affine.store %27, %alloc_2[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
            }
          }
        }
      }
    }
    dsp.print %alloc : memref<1x2x2xf64>
    dsp.print %alloc_2 : memref<1x2x2xf64>
    // memref.dealloc %arg0 : memref<1x2x2xf64>
    // memref.dealloc %arg1 : memref<1x2x2xf64>
    memref.dealloc %alloc : memref<1x2x2xf64>
    memref.dealloc %alloc_2 : memref<1x2x2xf64>
    // return %alloc, %alloc_2 : memref<1x2x2xf64>, memref<1x2x2xf64>
    return 
  }
}

