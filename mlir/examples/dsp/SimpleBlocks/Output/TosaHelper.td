
Notes: 
    1) Lowering option:
        ./bin/mlir-opt fft_help.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg) ,one-shot-bufferize, func.func(convert-linalg-to-affine-loops))”

        ./bin/mlir-opt ../mlir/examples/dsp/SimpleBlocks/Output/TryTOSA/fft2d_tosa.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg) ,one-shot-bufferize, func.func(convert-linalg-to-affine-loops))”

        ./bin/mlir-opt ../mlir/examples/dsp/SimpleBlocks/Output/TryTOSA/fft2d_tosa.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg) ,one-shot-bufferize, convert-linalg-to-affine-loops,one-shot-bufferize,test-lower-to-llvm)" //Not working

        ./bin/mlir-opt ../mlir/examples/dsp/SimpleBlocks/Output/TryTOSA/fft2d_tosa.mlir --pass-pipeline="builtin.module(func.func(tosa-to-linalg) ,one-shot-bufferize, func.func(convert-linalg-to-affine-loops),test-lower-to-llvm)"