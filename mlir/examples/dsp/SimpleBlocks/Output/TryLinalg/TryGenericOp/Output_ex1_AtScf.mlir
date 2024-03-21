module {
  func.func @main(){
   
   //From separate input
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
   
   
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %alloc_15, %c0 : memref<10xf64>
    scf.for %arg3 = %c0 to %dim step %c1 {
      %0 = memref.load %alloc_15[%arg3] : memref<10xf64>
      %1 = memref.load %alloc_15[%arg3] : memref<10xf64>
      %2 = arith.addf %0, %1 : f64
      memref.store %2, %alloc[%arg3] : memref<10xf64>
    }

    dsp.print %alloc : memref<10xf64>
    memref.dealloc %alloc_15 : memref<10xf64>
    return
  }
}
