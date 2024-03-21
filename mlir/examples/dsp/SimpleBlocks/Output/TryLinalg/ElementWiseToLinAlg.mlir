#map = affine_map<() -> ()>
module {
  func.func @addf_rank0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%arg0 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func @addf_rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg0 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}


// -----
#map = affine_map<() -> ()>
module {
  func.func @exp(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%arg0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = math.exp %in : f32
      linalg.yield %1 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }
}


// -----
#map = affine_map<() -> ()>
module {
  func.func @select(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>, tensor<i32>) outs(%arg1 : tensor<i32>) {
    ^bb0(%in: i1, %in_0: i32, %in_1: i32, %out: i32):
      %1 = arith.select %in, %in_0, %in_1 : i32
      linalg.yield %1 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }
}


// -----
#map = affine_map<() -> ()>
module {
  func.func @cmpf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
    %0 = tensor.empty() : tensor<i1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<i1>) {
    ^bb0(%in: f32, %in_0: f32, %out: i1):
      %2 = arith.cmpf olt, %in, %in_0 : f32
      linalg.yield %2 : i1
    } -> tensor<i1>
    return %1 : tensor<i1>
  }
}


// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @cmpf(%arg0: tensor<4x?x?x8x2x?xf32>, %arg1: tensor<4x?x?x8x2x?xf32>) -> tensor<4x?x?x8x2x?xi1> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<4x?x?x8x2x?xf32>
    %c2 = arith.constant 2 : index
    %dim_0 = tensor.dim %arg0, %c2 : tensor<4x?x?x8x2x?xf32>
    %c5 = arith.constant 5 : index
    %dim_1 = tensor.dim %arg0, %c5 : tensor<4x?x?x8x2x?xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1) : tensor<4x?x?x8x2x?xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x?x?x8x2x?xf32>, tensor<4x?x?x8x2x?xf32>) outs(%0 : tensor<4x?x?x8x2x?xi1>) {
    ^bb0(%in: f32, %in_2: f32, %out: i1):
      %2 = arith.cmpf olt, %in, %in_2 : f32
      linalg.yield %2 : i1
    } -> tensor<4x?x?x8x2x?xi1>
    return %1 : tensor<4x?x?x8x2x?xi1>
  }
}

