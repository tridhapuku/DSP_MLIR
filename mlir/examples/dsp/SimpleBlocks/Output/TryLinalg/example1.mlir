// // File name: example1.mlir
// #accesses = [
//   affine_map<(m) -> (m)>,
//   affine_map<(m) -> (m)>
// ]

// #attrs = {
//   indexing_maps = #accesses,
//   iterator_types = ["parallel"]
// }

// func.func @example(%A: memref<?xf32, strided<[1]>>,
//               %B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
//   linalg.generic #attrs
//   ins(%A: memref<?xf32, strided<[1]>>)
//   outs(%B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
//   ^bb0(%a: f32, %b: vector<4xf32>):
//     %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
//     linalg.yield %c: vector<4xf32>
//   }
//   return
// }



// File name: example1.mlir -- Modifying and checking 
//1) Remove stride
//2) Try iterator type as reduction
//3) Try input and output tensors
//4) 
#accesses = [
  affine_map<(m) -> (m)>,
  affine_map<(m) -> (m)>
]

#attrs = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

func.func @example(%A: memref<?xf32, strided<[1]>>,
              %B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  linalg.generic #attrs
  ins(%A: memref<?xf32, strided<[1]>>)
  outs(%B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  ^bb0(%a: f32, %b: vector<4xf32>):
    %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
    linalg.yield %c: vector<4xf32>
  }
  return
}

// #map = affine_map<(d0) -> (d0)>

// module {
//   func.func @add_vectors(%input1: tensor<4xf32>, %input2: tensor<4xf32>) -> tensor<4xf32> {
//     %result = linalg.generic
//       {indexing_maps = [#map, #map,#map], iterator_types = ["parallel"]}
//       ins(%input1, %input2: tensor<4xf32>, tensor<4xf32>)
//       outs(%result: tensor<4xf32>) {
//         ^bb(%in1: f32, %in2: f32, %out: f32):
//           %res = arith.addf %in1, %in2 : f32
//           linalg.yield %res : f32
//       } -> tensor<4xf32>
//     return %result : tensor<4xf32>
//   }
// }