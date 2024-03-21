#map = affine_map<() -> ()>
module {

  func.func @main() -> f64 {
    %cst = arith.constant 1.000000e+01 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %alloc = memref.alloc() : memref<f64>
    %alloc_1 = memref.alloc() : memref<f64>
    %alloc_2 = memref.alloc() : memref<f64>
    affine.store %cst_0, %alloc_2[] : memref<f64>
    affine.store %cst, %alloc_1[] : memref<f64>
    %0 = affine.load %alloc_2[] : memref<f64>
    %1 = affine.load %alloc_1[] : memref<f64>
    // %2 = arith.addf %0, %1 : f64
    // %2 = linalg.add %0, %1 : tensor<f64>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%0, %1 : f64, f64) outs(%alloc : memref<f64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %3 = arith.addf %in, %in_0 : f64
      linalg.yield %3 : f64
    } -> tensor<f64>  //memref not expected

    //   %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%0, %1 : memref<f64>, f64) outs(%alloc : memref<f64>) {
    // ^bb0(%in: memref<f64>, %in_0: f64, %out: f64):
    //   %3 = arith.addf %in, %in_0 : memref<f64>
    //   linalg.yield %3 : f64
    // } -> tensor<f64> 
    // affine.store %2, %alloc[] : memref<f64>
    // return %2 : tensor<f64>
    // dsp.print %alloc : memref<f64>
    memref.dealloc %alloc_2 : memref<f64>
    memref.dealloc %alloc_1 : memref<f64>
    memref.dealloc %alloc : memref<f64>
    return %2 : tensor<f64>
    // return
  }
}