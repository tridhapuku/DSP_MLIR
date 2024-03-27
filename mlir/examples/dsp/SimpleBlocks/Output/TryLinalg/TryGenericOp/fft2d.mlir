#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @test_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant 6.28318548 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %2 = tensor.empty() : tensor<1x4x8xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%1, %3 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32, %out_4: f32):
      %5 = linalg.index 1 : index
      %6 = arith.index_castui %5 : index to i32
      %7 = arith.uitofp %6 : i32 to f32
      %8 = linalg.index 2 : index
      %9 = arith.index_castui %8 : index to i32
      %10 = arith.uitofp %9 : i32 to f32
      %11 = linalg.index 3 : index
      %12 = arith.index_castui %11 : index to i32
      %13 = arith.uitofp %12 : i32 to f32
      %14 = linalg.index 4 : index
      %15 = arith.index_castui %14 : index to i32
      %16 = arith.uitofp %15 : i32 to f32
      %17 = arith.mulf %13, %7 : f32
      %18 = arith.mulf %16, %10 : f32
      %19 = arith.divf %17, %cst_0 : f32
      %20 = arith.divf %18, %cst : f32
      %21 = arith.addf %19, %20 : f32
      %22 = arith.mulf %21, %cst_1 : f32
      %23 = math.cos %22 : f32
      %24 = math.sin %22 : f32
      %25 = arith.mulf %in, %23 : f32
      %26 = arith.mulf %in_3, %24 : f32
      %27 = arith.addf %25, %26 : f32
      %28 = arith.mulf %in_3, %23 : f32
      %29 = arith.mulf %in, %24 : f32
      %30 = arith.subf %28, %29 : f32
      %31 = arith.addf %out, %27 : f32
      %32 = arith.addf %out_4, %30 : f32
      linalg.yield %31, %32 : f32, f32
    } -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
    return %4#0, %4#1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
  }
}

