
module {
  dsp.func @main() {
    // %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
    %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
    %gain1 = dsp.constant dense<4.000000e+00> : tensor<f64>
    %k1 = dsp.constant dense<1.000000e+00> : tensor<1xf64>
    %3 = dsp.constant dense<4.000000e+00> : tensor<1xf64>
    // %real = "dsp.fft1dreal"(%0) : (tensor<10xf64>) -> tensor<*xf64>
    // %img = "dsp.fft1dimg"(%0) : (tensor<10xf64>) -> tensor<*xf64>
    %real, %img = dsp.fft1d(%0 : tensor<10xf64>) to (tensor<*xf64>, tensor<*xf64>)
    %6 = "dsp.getElemAtIndx"(%real, %k1) : (tensor<*xf64>, tensor<1xf64>) -> tensor<*xf64>
    %7 = "dsp.getElemAtIndx"(%img, %k1) : (tensor<*xf64>, tensor<1xf64>) -> tensor<*xf64>
    %8 = "dsp.gain"(%6, %gain1) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %9 = "dsp.gain"(%7, %gain1) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    %10 = "dsp.setElemAtIndx"(%real, %k1, %8) : (tensor<*xf64>, tensor<1xf64>, tensor<*xf64>) -> tensor<*xf64>
    %11 = "dsp.setElemAtIndx"(%img, %k1, %9) : (tensor<*xf64>, tensor<1xf64>, tensor<*xf64>) -> tensor<*xf64>
    %12 = "dsp.ifft1d"(%real, %img) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    dsp.print %10 : tensor<*xf64>
    dsp.print %11 : tensor<*xf64>
    dsp.print %12 : tensor<*xf64>
    dsp.print %real : tensor<*xf64>
    dsp.print %img : tensor<*xf64>
    dsp.return
  }
}

