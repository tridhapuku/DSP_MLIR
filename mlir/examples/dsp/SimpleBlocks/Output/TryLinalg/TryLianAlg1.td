#map = affine_map<() -> ()>
module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%arg0 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }
}

//By chatgpt 
// #map = affine_map<(i, j) -> (i, j)>
// module {
//   func.func @elementwise_add(%input1: tensor<4x4xf32>, %input2: tensor<4x4xf32>) -> tensor<4x4xf32> {
//     %result = linalg.generic {
//       indexing_maps = [#map, #map, #map],
//       iterator_types = ["parallel", "parallel"]
//     } ins(%input1, %input2 : tensor<4x4xf32>, tensor<4x4xf32>)
//       outs(%input1 : tensor<4x4xf32>) {
//         ^bb0(%arg0: f32, %arg1: f32, %out: f32):
//           %sum = arith.addf %arg0, %arg1 : f32
//           linalg.yield %sum : f32
//       } -> tensor<4x4xf32>
//     return %result : tensor<4x4xf32>
//   }
// }

// Given the following IR:
// #map = affine_map<(d0) -> (d0)>
// module {
//   func.func @addf_rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg0 : tensor<?xf32>) {
//     ^bb0(%in: f32, %in_0: f32, %out: f32):
//       %1 = arith.addf %in, %in_0 : f32
//       linalg.yield %1 : f32
//     } -> tensor<?xf32>
//     return %0 : tensor<?xf32>
//   }
// }

// explain the following : 1) what are different iterator types in linalg? 2) Explain following lines
// ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg0 : tensor<?xf32>) &  ^bb0(%in: f32, %in_0: f32, %out: f32):

// -----
// #map = affine_map<() -> ()>
// module {
//   func.func @exp(%arg0: tensor<f32>) -> tensor<f32> {
//     %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%arg0 : tensor<f32>) {
//     ^bb0(%in: f32, %out: f32):
//       %1 = math.exp %in : f32
//       linalg.yield %1 : f32
//     } -> tensor<f32>
//     return %0 : tensor<f32>
//   }
// }