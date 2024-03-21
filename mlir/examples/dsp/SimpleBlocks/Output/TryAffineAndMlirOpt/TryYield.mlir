#set = affine_set<(d0) : (d0 - 10 >= 0)>
module {
  func.func @affine_if_not_invariant(%arg0: memref<1024xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = affine.for %arg1 = 0 to 10 step 2 iter_args(%arg2 = %cst) -> (f32) {
      %1 = affine.load %arg0[%arg1] : memref<1024xf32>
      %2 = affine.if #set(%arg1) -> f32 {
        %4 = arith.addf %arg2, %1 : f32
        affine.yield %4 : f32
      } else {
        affine.yield %arg2 : f32
      }
      %3 = arith.addf %2, %cst_0 : f32
      affine.yield %3 : f32
    }
    return %0 : f32
  }
}