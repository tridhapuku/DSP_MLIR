module {
  func.func @main() {
    %cst = arith.constant 6.000000e+01 : f64
    %cst_0 = arith.constant 5.000000e+01 : f64
    %cst_1 = arith.constant 4.000000e+01 : f64
    %cst_2 = arith.constant 3.000000e+01 : f64
    %cst_3 = arith.constant 2.000000e+01 : f64
    %cst_4 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<3xf64>
    %alloc_5 = memref.alloc() : memref<3xf64>
    %alloc_6 = memref.alloc() : memref<3xf64>
    affine.store %cst_4, %alloc_6[0] : memref<3xf64>
    affine.store %cst_3, %alloc_6[1] : memref<3xf64>
    affine.store %cst_2, %alloc_6[2] : memref<3xf64>
    affine.store %cst_1, %alloc_5[0] : memref<3xf64>
    affine.store %cst_0, %alloc_5[1] : memref<3xf64>
    affine.store %cst, %alloc_5[2] : memref<3xf64>
    affine.for %arg0 = 0 to 3 {
      %0 = affine.load %alloc_6[%arg0] : memref<3xf64>
      %1 = affine.load %alloc_5[%arg0] : memref<3xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %alloc[%arg0] : memref<3xf64>
    }
    dsp.print %alloc : memref<3xf64>
    memref.dealloc %alloc_6 : memref<3xf64>
    memref.dealloc %alloc_5 : memref<3xf64>
    memref.dealloc %alloc : memref<3xf64>
    return
  }
}
