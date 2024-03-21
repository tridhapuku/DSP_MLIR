module {
  func.func @main() {
    %cst = arith.constant 6.000000e+01 : f64
    %alloc_5 = memref.alloc() : memref<3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.store %cst, %alloc_5[%arg0] : memref<3xf64>
    }

    dsp.print %alloc_5 : memref<3xf64>
    memref.dealloc %alloc_5 : memref<3xf64>
    return
  }
}
