// module {
//   // Define the main function.
//   dsp.func @main() {
//     // Create a constant tensor with values [1.0, 2.0, ..., 100.0].
//     %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
//     // Create a constant tensor with value 0.5.
//     %1 = dsp.constant dense<5.000000e-01> : tensor<f64>
//     // Call the highPassFilter operation.
//     %2, %3 = dsp.fft1d (%0 : tensor<10xf64>) to tensor<*xf64>, tensor<*xf64>
//     // Print the results.
//     dsp.print %2 : tensor<*xf64>
//     dsp.print %2 : tensor<*xf64>
//     dsp.print %3 : tensor<*xf64>
//     // Return from the function.
//     dsp.return
//   }
// }

// module {
//   dsp.func @main() {
//     %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
//     %1 = dsp.constant dense<5.000000e-01> : tensor<f64>
//     // %real, %img = dsp.fft1d %0 : (tensor<10xf64>) to (tensor<*xf64> , tensor<*xf64>)
//     %real, %img = dsp.fft1d (%0 : tensor<10xf64>) to (tensor<*xf64> , tensor<*xf64>)
//     // %real, %img = dsp.fft1d (%0 : tensor<10xf64>) -> (tensor<*xf64> , tensor<*xf64>)
//     dsp.print %real : tensor<*xf64>
//     // dsp.print %img : tensor<*xf64>
//     dsp.return
//   }
// }

module {
  dsp.func @main() {
    // %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
    %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01]> : tensor<4xf64>
    // %0 = dsp.constant dense<[1.000000e+01, 0.000000e+01, 0.000000e+01, 0.000000e+01]> : tensor<4xf64>
    // %1 = dsp.constant dense<5.000000e-01> : tensor<f64>
    // %real, %img = dsp.fft1d(%0 : tensor<10xf64>) to(tensor<*xf64>, tensor<*xf64>) //working
    %real, %img = dsp.ifft1d(%0 : tensor<4xf64>) to (tensor<*xf64>, tensor<*xf64>)
    // %real, %img = dsp.ifft1d(%0 : tensor<4xf64>) -> (tensor<*xf64>, tensor<*xf64>)
    dsp.print %real : tensor<*xf64>
    dsp.print %img : tensor<*xf64>
    // dsp.print %0 : tensor<10xf64>
    dsp.return
  }
}

