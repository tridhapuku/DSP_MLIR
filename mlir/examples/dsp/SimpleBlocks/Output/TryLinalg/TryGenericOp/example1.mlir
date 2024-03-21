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

//Not working
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



// File name: example1.mlir -- Modifying and checking 
//1) Remove stride
//2) Try iterator type as reduction
//3) Try input and output tensors
//4) 
  // #accesses = [
  //   affine_map<(m) -> (m)>,
  //   affine_map<(m) -> (m)>
  // ]

  // #attrs = {
  //   indexing_maps = #accesses,
  //   iterator_types = ["parallel"]
  // }

  // func.func @example(%A: memref<?xf32, strided<[3]>>,
  //               %B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  //   linalg.generic #attrs
  //   ins(%A: memref<?xf32, strided<[3]>>)
  //   outs(%B: memref<?xvector<4xf32>, strided<[2], offset: 1>>) {
  //   ^bb0(%a: f32, %b: vector<4xf32>):
  //     %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
  //     linalg.yield %c: vector<4xf32>
  //   }
  //   return
  // }

// File name: example1.mlir -- Modifying and checking 
//1) Remove stride
  //   #accesses = [
  //   affine_map<(m) -> (m)>,
  //   affine_map<(m) -> (m)>
  // ]

  // #attrs = {
  //   indexing_maps = #accesses,
  //   iterator_types = ["parallel"]
  // }

  // func.func @example(%A: memref<?xf32>,
  //               %B: memref<?xvector<4xf32>>) {
  //   linalg.generic #attrs
  //   ins(%A: memref<?xf32>)
  //   outs(%B: memref<?xvector<4xf32>>) {
  //   ^bb0(%a: f32, %b: vector<4xf32>):
  //     %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
  //     // %c = arith.addf(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
  //     // %c = arith.addf %a, %b: f32
  //     linalg.yield %c: vector<4xf32>
  //   }
  //   return
  // }

//1) Try use arith.addf instead of some-compute : input & output tensors
//2) Try iterator type as reduction
//3) Try input and output tensors -- new variable for output
//4) 

//3) Try input and output tensors -- new variable for output
  // #accesses = [
  //   affine_map<(m) -> (m)>,
  //   affine_map<(m) -> (m)>,
  //   affine_map<(m) -> (m)>
  // ]

  // #attrs = {
  //   indexing_maps = #accesses,
  //   iterator_types = ["parallel"]
  // }

  // func.func @example(%A: memref<?xf32>,
  //               %B: memref<?xf32>)
  //                //* %D: memref<?xf32>) 
  // {
  //   // %C = memref.alloc
  //   // %alloc = memref.alloc() : memref<?xf32>
  //   %D = memref.alloc() : memref<?xf32>
  //   linalg.generic #attrs
  //   ins(%A ,%B : memref<?xf32> , memref<?xf32>)
  //   outs(%D: memref<?xf32>) {
  //   ^bb0(%a: f32, %b: f32 , %e: f32):
  //     // %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
  //     // %c = arith.addf %a, %b: (f32, f32) -> (f32)
  //     %c = arith.addf %a, %b: f32
  //     linalg.yield %c: f32
  //   }
  //   return
  // }


//3) Try input and output tensors -- new variable for output
    // With fixed no of inputs
  // #accesses = [
  //   affine_map<(m) -> (m)>,
  //   affine_map<(m) -> (m)>,
  //   affine_map<(m) -> (m)>
  // ]

  // #attrs = {
  //   indexing_maps = #accesses,
  //   iterator_types = ["parallel"]
  // }

  // func.func @example(%A: memref<?xf32>,
  //               %B: memref<?xf32>)
  //                //* %D: memref<?xf32>) 
  // {
  //   // %C = memref.alloc
  //   // %alloc = memref.alloc() : memref<?xf32>
  //   %D = memref.alloc() : memref<6xf32>
  //   linalg.generic #attrs
  //   ins(%A ,%B : memref<?xf32> , memref<?xf32>)
  //   outs(%D: memref<6xf32>) {
  //   ^bb0(%a: f32, %b: f32 , %e: f32):
  //     // %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
  //     // %c = arith.addf %a, %b: (f32, f32) -> (f32)
  //     %c = arith.addf %a, %b: f32
  //     linalg.yield %c: f32
  //   }
  //   return
  // }

//2) Try iterator type as reduction
//3) Try input and output tensors -- new variable for output
//4)

//2) Try iterator type as reduction
   #accesses = [
    affine_map<(m) -> (m)>,
    affine_map<(m) -> (m)>,
    affine_map<(m) -> (m)>
  ]

  #attrs = {
    indexing_maps = #accesses,
    iterator_types = ["reduction"]
  }

  func.func @example(%A: memref<?xf32>,
                %B: memref<?xf32>,
                  %D: memref<?xf32>) 
  {
    // %C = memref.alloc
    // %alloc = memref.alloc() : memref<?xf32>
    // %D = memref.alloc() : memref<10xf32>
    linalg.generic #attrs
    ins(%A ,%B : memref<?xf32> , memref<?xf32>)
    outs(%D: memref<?xf32>) {
    ^bb0(%a: f32, %b: f32 , %e: f32):
      // %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
      // %c = arith.addf %a, %b: (f32, f32) -> (f32)
      %c = arith.addf %a, %b: f32
      linalg.yield %c: f32
    }
    return
  }


  //Output
  module {
  func.func @example(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf32>
    scf.for %arg3 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<?xf32>
      %1 = memref.load %arg1[%arg3] : memref<?xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}
