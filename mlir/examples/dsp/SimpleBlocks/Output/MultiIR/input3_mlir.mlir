module {
  dsp.func @main() {
    %0 = dsp.constant dense<[1.00e+00, 2.00e+00, 3.00e+00]> : tensor<3xf64>
    %1 = dsp.constant dense<2.00e+00> : tensor<f64>
    %2 = "dsp.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<*xf64>
    dsp.print %2 : tensor<*xf64>
    dsp.return
  }
}


  
  // affine.for %arg0 = 0 to 1 {
  //     %0 = affine.load %alloc_4[%arg0] : memref<3xf64>
  //     affine.store %0, %alloc[%arg0 + 2] : memref<3xf64>
  //   }

  //   %4 = llvm.mlir.constant(3 : index) : i64
  //   %5 = llvm.mlir.constant(1 : index) : i64
  //   %6 = llvm.mlir.null : !llvm.ptr
  //   %7 = llvm.getelementptr %6[%4] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  //   %8 = llvm.ptrtoint %7 : !llvm.ptr to i64


  //   define void @main() !dbg !3 {
  //     %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (double, ptr null, i64 3) to i64)), !dbg !6
  //     %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %1, 0, !dbg !6


  //     0.000000 0.000000 1.000000