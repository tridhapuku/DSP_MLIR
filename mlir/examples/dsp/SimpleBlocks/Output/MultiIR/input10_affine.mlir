module {
  // func.func @main() {
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 4.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+02 : f64
    %cst_3 = arith.constant 9.000000e+01 : f64
    %cst_4 = arith.constant 8.000000e+01 : f64
    %cst_5 = arith.constant 7.000000e+01 : f64
    %cst_6 = arith.constant 6.000000e+01 : f64
    %cst_7 = arith.constant 5.000000e+01 : f64
    %cst_8 = arith.constant 4.000000e+01 : f64
    %cst_9 = arith.constant 3.000000e+01 : f64
    %cst_10 = arith.constant 2.000000e+01 : f64
    %cst_11 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<10xf64>
    %alloc_12 = memref.alloc() : memref<10xf64>
    %alloc_13 = memref.alloc() : memref<f64>
    %alloc_14 = memref.alloc() : memref<f64>
    %alloc_15 = memref.alloc() : memref<10xf64>
    affine.store %cst_11, %alloc_15[0] : memref<10xf64>
    affine.store %cst_10, %alloc_15[1] : memref<10xf64>
    affine.store %cst_9, %alloc_15[2] : memref<10xf64>
    affine.store %cst_8, %alloc_15[3] : memref<10xf64>
    affine.store %cst_7, %alloc_15[4] : memref<10xf64>
    affine.store %cst_6, %alloc_15[5] : memref<10xf64>
    affine.store %cst_5, %alloc_15[6] : memref<10xf64>
    affine.store %cst_4, %alloc_15[7] : memref<10xf64>
    affine.store %cst_3, %alloc_15[8] : memref<10xf64>
    affine.store %cst_2, %alloc_15[9] : memref<10xf64>
    affine.store %cst_1, %alloc_14[] : memref<f64>
    affine.store %cst_0, %alloc_13[] : memref<f64>
    affine.for %arg0 = 0 to 2 {
      affine.store %cst, %alloc_12[%arg0] : memref<10xf64>
    }
    affine.for %arg0 = 0 to 4 {
      affine.store %cst, %alloc[%arg0] : memref<10xf64>
    }
    dsp.print %alloc : memref<10xf64>
    // toy.print %alloc : memref<10xf64>
    memref.dealloc %alloc_15 : memref<10xf64>
    memref.dealloc %alloc_14 : memref<f64>
    memref.dealloc %alloc_13 : memref<f64>
    memref.dealloc %alloc_12 : memref<10xf64>
    memref.dealloc %alloc : memref<10xf64>
    dsp.return
  }
}
