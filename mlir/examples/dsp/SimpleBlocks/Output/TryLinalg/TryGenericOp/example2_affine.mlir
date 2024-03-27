module {
    //   func.func @main(%arg0: memref<2x2xf64>, %arg1: memref<2x2xf64>) {
  func.func @main() {

    //Get input from outside
    %cst = arith.constant 7.000000e+01 : f64
    %cst_0 = arith.constant 6.000000e+01 : f64
    %cst_1 = arith.constant 5.000000e+01 : f64
    %cst_2 = arith.constant 4.000000e+01 : f64
    %cst_3 = arith.constant 3.000000e+01 : f64
    %cst_4 = arith.constant 2.000000e+01 : f64
    %cst_5 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<2x2xf64>
    %alloc_6 = memref.alloc() : memref<2x2xf64>
    %alloc_7 = memref.alloc() : memref<2x2xf64>
    affine.store %cst_5, %alloc_7[0, 0] : memref<2x2xf64>
    affine.store %cst_4, %alloc_7[0, 1] : memref<2x2xf64>
    affine.store %cst_3, %alloc_7[1, 0] : memref<2x2xf64>
    affine.store %cst_2, %alloc_7[1, 1] : memref<2x2xf64>
    affine.store %cst_2, %alloc_6[0, 0] : memref<2x2xf64>
    affine.store %cst_1, %alloc_6[0, 1] : memref<2x2xf64>
    affine.store %cst_0, %alloc_6[1, 0] : memref<2x2xf64>
    affine.store %cst, %alloc_6[1, 1] : memref<2x2xf64>

    //code from linalg.generic
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %alloc_6, %c0 : memref<2x2xf64>
    %dim_0 = memref.dim %alloc_6, %c1 : memref<2x2xf64>
    affine.for %arg2 = 0 to %dim_0 {
      affine.for %arg3 = 0 to %dim {
        %0 = affine.load %alloc_6[%arg2, %arg3] : memref<2x2xf64>
        %1 = affine.load %alloc_7[%arg2, %arg3] : memref<2x2xf64>
        %2 = arith.addf %0, %1 :  f64
        affine.store %2, %alloc[%arg2, %arg3] : memref<2x2xf64>
      }
    }
    dsp.print %alloc : memref<2x2xf64>
    memref.dealloc %alloc_7 : memref<2x2xf64>
    memref.dealloc %alloc_6 : memref<2x2xf64>
    memref.dealloc %alloc : memref<2x2xf64>
    return
  }
}
