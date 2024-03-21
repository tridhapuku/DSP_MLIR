
//1) Remove strides
//2) Try affine-map indexing
//3) Try input and output tensors -- new variable for output
//4) Try iterator type as reduction

// File name: example2.mlir
    // #indexing_maps = [
    //   affine_map<(i, j) -> (j, i)>,
    //   affine_map<(i, j) -> (j)>
    // ]

    // #attrs = {
    //   indexing_maps = #indexing_maps,
    //   iterator_types = ["parallel", "parallel"]
    // }

    // func.func @example(%A: memref<8x?xf32, strided<[2, 2], offset: 0>>,
    //               %B: memref<?xvector<4xf32>>) {
    //   linalg.generic #attrs
    //   ins(%A: memref<8x?xf32, strided<[2, 2], offset: 0>>)
    //   outs(%B: memref<?xvector<4xf32>>) {
    //   ^bb0(%a: f32, %b: vector<4xf32>):
    //     %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
    //     linalg.yield %c: vector<4xf32>
    //   }
    //   return
    // }


//1) Remove strides
//2) Try affine-map indexing
//3) Try input and output tensors -- new variable for output
//4) Try iterator type as reduction

    // #indexing_maps = [
    // // affine_map<(i, j) -> (j, i)>,
    // affine_map<(i, j) -> (i, j)>,
    // // affine_map<(i, j) -> (i + 1, j)>,
    // // affine_map<(i, j) -> (j)>
    // affine_map<(i, j) -> (i)>
    // // affine_map<(i, j) -> (j,i)>  //not working -- operand rank (1) to match the result rank of indexing_map #1
    // ]

    // #attrs = {
    // indexing_maps = #indexing_maps,
    // iterator_types = ["parallel", "parallel"]
    // }

    // func.func @example(%A: memref<8x?xf32>,
    //             %B: memref<?xvector<4xf32>>) {
    // linalg.generic #attrs
    // ins(%A: memref<8x?xf32>)
    // outs(%B: memref<?xvector<4xf32>>) {
    // ^bb0(%a: f32, %b: vector<4xf32>):
    //     %c = "some_compute"(%a, %b): (f32, vector<4xf32>) -> (vector<4xf32>)
    //     linalg.yield %c: vector<4xf32>
    // }
    // return
    // }


//---------------

//2) Try affine-map indexing
//  Try affine-map indexing with individual element operation
//3) Try input and output tensors -- new variable for output
//4) Try iterator type as reduction

    // #indexing_maps = [
    // // affine_map<(i, j) -> (j, i)>,
    // affine_map<(i, j) -> (i, j)>,
    // // affine_map<(i, j) -> (i + 1, j)>,
    // // affine_map<(i, j) -> (j)>
    // affine_map<(i, j) -> (i)>
    // // affine_map<(i, j) -> (j,i)>  //not working -- operand rank (1) to match the result rank of indexing_map #1
    // ]

    // #attrs = {
    // indexing_maps = #indexing_maps,
    // iterator_types = ["parallel", "parallel"]
    // }

    // func.func @example(%A: memref<8x?xf32>,
    //             %B: memref<6x4xf32>) {
    // linalg.generic #attrs
    // ins(%A: memref<8x?xf32>)
    // outs(%B: memref<6x4xf32>) {
    // ^bb0(%a: f32, %b: tensor<4xf32>):
    //     %c = "some_compute"(%a, %b): (f32, tensor<4xf32>) -> (tensor<4xf32>)
    //     linalg.yield %c: tensor<4xf32>
    // }
    // return
    // }


//---------------

//2) Try affine-map indexing
//  Try affine-map indexing with individual element operation
//3) Try input and output tensors -- new variable for output
//4) Try iterator type as reduction
//5) Try using std operations - arith.addf , memref.alloc 
//
    //    #indexing_maps = [
    //     affine_map<(i, j) -> (j, i)>,
    //     // affine_map<(i, j) -> (i, j)>,
    //     // affine_map<(i, j) -> (i + 1, j)>,
    //     // affine_map<(i, j) -> (j)>
    //     affine_map<(i, j) -> (i,j)>
    //     // affine_map<(i, j) -> (j,i)>  //not working -- operand rank (1) to match the result rank of indexing_map #1
    //     ]

    //     #attrs = {
    //     indexing_maps = #indexing_maps,
    //     iterator_types = ["parallel", "parallel"]
    //     }

    //     func.func @example(%A: memref<8x?xf32>,
    //                 %B: memref<?x?xf32>) {
    //     linalg.generic #attrs
    //     ins(%A: memref<8x?xf32>)
    //     outs(%B: memref<?x?xf32>) {
    //     ^bb0(%a: f32, %b: f32):
    //         %c = "some_compute"(%a, %b): (f32, f32) -> (f32)
    //         linalg.yield %c: f32
    //     }
    //     return
    //     }



//2) Try affine-map indexing
//  Try affine-map indexing with individual element operation
//3) Try input and output tensors -- new variable for output
//4) Try iterator type as reduction
   #indexing_maps = [
    affine_map<(i, j) -> (j, i)>,
    affine_map<(i, j) -> (i,j)>
     ]

    #attrs = {
    indexing_maps = #indexing_maps,
    iterator_types = ["parallel", "parallel"]
    }

    func.func @example(%A: memref<?x?xf32>,
                %B: memref<?x?xf32>) {
    linalg.generic #attrs
    ins(%A: memref<?x?xf32>)
    outs(%B: memref<?x?xf32>) {
    ^bb0(%a: f32, %b: f32):
        %c = "some_compute"(%a, %b): (f32, f32) -> (f32)
        linalg.yield %c: f32
    }
    return
    }