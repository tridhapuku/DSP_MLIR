// module {
//   func.func @test_static_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
//     %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
//     return %output_real, %output_imag : tensor<1x4x8xf32>, tensor<1x4x8xf32>
//   }
// }

//2-D tensors not allowed --
// module {
//   func.func @test_static_fft2d(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
//     %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false}: (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
//     return %output_real, %output_imag : tensor<4x8xf32>, tensor<4x8xf32>
//   }
// }

// module {
//   func.func @test_static_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
//     %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
//     return %output_real, %output_imag : tensor<1x4x8xf32>, tensor<1x4x8xf32>
//   }
// }

module {
  func.func @test_static_fft2d(%arg0: memref<1x4x8xf32>, %arg1: memref<1x4x8xf32>) -> (memref<1x4x8xf32>, memref<1x4x8xf32>) {
    %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (memref<1x4x8xf32>, memref<1x4x8xf32>) -> (memref<1x4x8xf32>, memref<1x4x8xf32>)
    return %output_real, %output_imag : memref<1x4x8xf32>, memref<1x4x8xf32>
  }
}