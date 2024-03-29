module {
  func.func @main() {
    %cst = arith.constant 7.000000e+01 : f64
    %cst_0 = arith.constant 6.000000e+01 : f64
    %cst_1 = arith.constant 5.000000e+01 : f64
    %cst_2 = arith.constant 4.000000e+01 : f64
    %cst_3 = arith.constant 3.000000e+01 : f64
    %cst_4 = arith.constant 2.000000e+01 : f64
    %cst_5 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<1x2x2xf64>
    %alloc_6 = memref.alloc() : memref<1x2x2xf64>
    %alloc_7 = memref.alloc() : memref<1x2x2xf64>
    affine.store %cst_5, %alloc_7[0, 0, 0] : memref<1x2x2xf64>
    affine.store %cst_4, %alloc_7[0, 0, 1] : memref<1x2x2xf64>
    affine.store %cst_3, %alloc_7[0, 1, 0] : memref<1x2x2xf64>
    affine.store %cst_2, %alloc_7[0, 1, 1] : memref<1x2x2xf64>
    affine.store %cst_2, %alloc_6[0, 0, 0] : memref<1x2x2xf64>
    affine.store %cst_1, %alloc_6[0, 0, 1] : memref<1x2x2xf64>
    affine.store %cst_0, %alloc_6[0, 1, 0] : memref<1x2x2xf64>
    affine.store %cst, %alloc_6[0, 1, 1] : memref<1x2x2xf64>

    
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 2 {
        affine.for %arg2 = 0 to 2 {
          %0 = affine.load %alloc_7[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
          %1 = affine.load %alloc_6[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
          %2 = arith.subf %0, %1 : f64
          affine.store %2, %alloc[%arg0, %arg1, %arg2] : memref<1x2x2xf64>
        }
      }
    }
    dsp.print %alloc : memref<1x2x2xf64>
    memref.dealloc %alloc_7 : memref<1x2x2xf64>
    memref.dealloc %alloc_6 : memref<1x2x2xf64>
    memref.dealloc %alloc : memref<1x2x2xf64>
    return
  }
}
