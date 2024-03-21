

Errors faced :

    1) ./bin/mlir-opt ./TryLinAlg/linalgAdd1.mlir --one-shot-bufferize='bufferize-function-boundaries' -test-lower-to-llvm 
    ./TryLinAlg/linalgAdd1.mlir:16:10: error: 'linalg.generic' op expected the number of tensor results (1) to be equal to the number of output tensors (0)
        %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%0, %1 : f64, f64) outs(%alloc : memref<f64>) {
            ^
    ./TryLinAlg/linalgAdd1.mlir:16:10: note: see current operation: 
    %7 = "linalg.generic"(%5, %6, %2) <{indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = [], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %8 = "arith.addf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%8) : (f64) -> ()
    }) : (f64, f64, memref<f64>) -> tensor<f64>


2) 