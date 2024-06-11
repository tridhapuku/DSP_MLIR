// ./bin/mlir-opt  --convert-affine-to-standard --convert-scf-to-std --convert-memref-to-llvm --convert-std-to-llvm --convert-func-to-llvm ../mlir/examples/dsp/SimpleBlocks/Output/TryMoreBlocksDsp/rleMlirAffineFinal.mlir -o out.mlir
// ./bin/mlir-opt --lower-affine --convert-scf-to-cf --convert-func-to-llvm ../mlir/examples/dsp/SimpleBlocks/Output/TryMoreBlocksDsp/rleMlirAffineFinal.mlir -o out.mlir
// ./bin/mlir-opt --lower-affine --convert-scf-to-cf --convert-func-to-llvm
module {
  func.func @main() {
    %c6 = arith.constant 6 : index
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+01 : f64
    %cst_1 = arith.constant 8.000000e-01 : f64
    %cst_2 = arith.constant 3.200000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<12xf64>
    %alloc_3 = memref.alloc() : memref<6xf64>
    affine.store %cst_2, %alloc_3[0] : memref<6xf64>
    affine.store %cst_2, %alloc_3[1] : memref<6xf64>
    affine.store %cst_1, %alloc_3[2] : memref<6xf64>
    affine.store %cst_1, %alloc_3[3] : memref<6xf64>
    affine.store %cst_1, %alloc_3[4] : memref<6xf64>
    affine.store %cst_0, %alloc_3[5] : memref<6xf64>
    %0 = affine.load %alloc_3[0] : memref<6xf64>
    affine.store %0, %alloc[0] : memref<12xf64>
    %1:2 = affine.for %arg0 = 1 to 6 iter_args(%arg1 = %cst, %arg2 = %c0) -> (f64, index) {
      %3 = affine.load %alloc_3[%arg0] : memref<6xf64>
      %4 = affine.load %alloc_3[%arg0 - 1] : memref<6xf64>
      %5 = arith.cmpf oeq, %4, %3 : f64
      %6:2 = scf.if %5 -> (f64, index) {
        %7 = arith.addf %arg1, %cst : f64
        scf.yield %7, %arg2 : f64, index
      } else {
        %7 = arith.addi %arg2, %c6 : index
        memref.store %arg1, %alloc[%7] : memref<12xf64>
        memref.store %3, %alloc[%arg2] : memref<12xf64>
        %8 = arith.addi %arg2, %c1 : index
        scf.yield %cst, %8 : f64, index
      }
      affine.yield %6#0, %6#1 : f64, index
    }
    %2 = arith.cmpf ogt, %1#0, %cst : f64
    scf.if %2 {
      memref.store %1#0, %alloc[%1#1] : memref<12xf64>
    }
    // dsp.print %alloc : memref<12xf64>
    memref.dealloc %alloc_3 : memref<6xf64>
    memref.dealloc %alloc : memref<12xf64>
    return
  }
}
