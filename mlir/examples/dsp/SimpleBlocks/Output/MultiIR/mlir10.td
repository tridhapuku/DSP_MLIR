module {
  dsp.func @main() {
    %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
    %1 = dsp.constant dense<2.000000e+00> : tensor<f64>
    %2 = dsp.constant dense<4.000000e+00> : tensor<f64>
    %3 = "dsp.delay"(%0, %1) : (tensor<10xf64>, tensor<f64>) -> tensor<*xf64>
    %4 = "dsp.delay"(%3, %2) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    dsp.print %4 : tensor<*xf64>
    dsp.return
  }
}
