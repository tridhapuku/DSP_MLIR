module {
  func.func @depthwise_conv_1d_nwc_wcm(%arg0: tensor<1x12x8xf32>, %arg1: tensor<3x8x8xf32>) -> tensor<1x10x8x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x10x8x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
    %2 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<1x12x8xf32>, tensor<3x8x8xf32>) outs(%1 : tensor<1x10x8x8xf32>) -> tensor<1x10x8x8xf32>
    return %2 : tensor<1x10x8x8xf32>
  }
}


// -----
module {
  func.func @depthwise_conv_1d_nwc_wc(%arg0: tensor<1x12x8xf32>, %arg1: tensor<3x8xf32>) -> tensor<1x10x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x10x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
    %2 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<1x12x8xf32>, tensor<3x8xf32>) outs(%1 : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
    return %2 : tensor<1x10x8xf32>
  }
}


// -----
module {
  func.func @depthwise_conv_1d_ncw_cw(%arg0: tensor<1x8x12xf32>, %arg1: tensor<8x3xf32>) -> tensor<1x8x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x8x10xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x8x10xf32>) -> tensor<1x8x10xf32>
    %2 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<1x8x12xf32>, tensor<8x3xf32>) outs(%1 : tensor<1x8x10xf32>) -> tensor<1x8x10xf32>
    return %2 : tensor<1x8x10xf32>
  }
}


// -----
module {
  func.func @depthwise_conv_2d_nhwc_hwcm_tensor(%arg0: tensor<2x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>) -> tensor<2x3x4x2x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x3x4x2x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>) outs(%1 : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
    return %2 : tensor<2x3x4x2x3xf32>
  }
  func.func @depthwise_conv_2d_nhwc_hwcm_memref(%arg0: memref<2x4x5x2xf32>, %arg1: memref<2x2x2x3xf32>, %arg2: memref<2x3x4x2x3xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>) outs(%arg2 : memref<2x3x4x2x3xf32>)
    return
  }
  func.func @depthwise_conv_1d_nw_tensor(%arg0: tensor<1x113x96xf32>, %arg1: tensor<3x96xf32>) -> tensor<1x56x96xf32> {
    %0 = tensor.empty() : tensor<1x56x96xf32>
    %1 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%arg0, %arg1 : tensor<1x113x96xf32>, tensor<3x96xf32>) outs(%0 : tensor<1x56x96xf32>) -> tensor<1x56x96xf32>
    return %1 : tensor<1x56x96xf32>
  }
  func.func @depthwise_conv_2d_nhwc_hwc_tensor(%arg0: tensor<1x113x113x96xf32>, %arg1: tensor<3x3x96xf32>) -> tensor<1x56x56x96xf32> {
    %0 = tensor.empty() : tensor<1x56x56x96xf32>
    %1 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %arg1 : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%0 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
    return %1 : tensor<1x56x56x96xf32>
  }
  func.func @depthwise_conv_2d_nhwc_hwc_memref(%arg0: memref<1x113x113x96xf32>, %arg1: memref<3x3x96xf32>, %arg2: memref<1x56x56x96xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %arg1 : memref<1x113x113x96xf32>, memref<3x3x96xf32>) outs(%arg2 : memref<1x56x56x96xf32>)
    return
  }
  func.func @depthwise_conv_2d_nchw_chw_tensor(%arg0: tensor<1x96x113x113xf32>, %arg1: tensor<96x3x3within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:178 offset :5:3: error: invalid properties {dilations = dense<1> : vector<2xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<2.000000e+00> : vector<2xf32>} for op linalg.depthwise_conv_2d_nhwc_hwc: Invalid attribute `strides` in property conversion: dense<2.000000e+00> : vector<2xf32>
  linalg.depthwise_conv_2d_nhwc_hwc <{dilations = dense<1> : vector<2xi64>, strides = dense<2.0> : vector<2xf32>}>
  ^
within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:188 offset :5:37: error: custom op 'linalg.depthwise_conv_2d_nhwc_hwc' 'linalg.depthwise_conv_2d_nhwc_hwc' op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2.0> : vector<2xf32>}
                                    ^
within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:198 offset :5:37: error: custom op 'linalg.depthwise_conv_2d_nhwc_hwc' 'linalg.depthwise_conv_2d_nhwc_hwc' op attribute 'strides' failed to satisfy constraint: 64-bit signless int elements attribute of shape [2]
  linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<3xi64> }
                                    ^
xf32>) -> tensor<1x96x56x56xf32> {
    %0 = tensor.empty() : tensor<1x96x56x56xf32>
    %1 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %arg1 : tensor<1x96x113x113xf32>, tensor<96x3x3xf32>) outs(%0 : tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
    return %1 : tensor<1x96x56x56xf32>
  }
  func.func @depthwise_conv_2d_nchw_chw_memref(%arg0: memref<1x96x113x113xf32>, %arg1: memref<96x3x3xf32>, %arg2: memref<1x96x56x56xf32>) {
    linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %arg1 : memref<1x96x113x113xf32>, memref<96x3x3xf32>) outs(%arg2 : memref<1x96x56x56xf32>)
    return
  }
  func.func @depthwise_conv_2d_nhwc_hwcm_tensor_dilated(%arg0: tensor<2x8x9x2xf32>, %arg1: tensor<2x2x2x3xf32>) -> tensor<2x6x7x2x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x6x7x2x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x6x7x2x3xf32>) -> tensor<2x6x7x2x3xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<2x8x9x2xf32>, tensor<2x2x2x3xf32>) outs(%1 : tensor<2x6x7x2x3xf32>) -> tensor<2x6x7x2x3xf32>
    return %2 : tensor<2x6x7x2x3xf32>
  }
  func.func @depthwise_conv_2d_nhwc_hwcm_memref_dilated(%arg0: memref<2x8x9x2xf32>, %arg1: memref<2x2x2x3xf32>, %arg2: memref<2x6x7x2x3xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<2x8x9x2xf32>, memref<2x2x2x3xf32>) outs(%arg2 : memref<2x6x7x2x3xf32>)
    return
  }
}


// -----
module {
  func.func @depthwise_conv_2d_input_nhwc_filter_default_attributes(%arg0: memref<1x113x113x96xf32>, %arg1: memref<3x3x96xf32>, %arg2: memref<1x56x56x96xf32>) {
    linalg.depthwise_conv_2d_nhwc_hwc ins(%arg0, %arg1 : memref<1x113x113x96xf32>, memref<3x3x96xf32>) outs(%arg2 : memref<1x56x56x96xf32>)
    return
  }
}


// -----

// -----

// -----

// -----
module {
  func.func @depthwise_conv_3d_ndhwc_dhwcm(%arg0: tensor<2x6x13x12x6xf32>, %arg1: tensor<2x1x3x6x6xf32>) -> tensor<2x3x13x4x6x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x3x13x4x6x6xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
    %2 = linalg.depthwise_conv_3d_ndhwc_dhwcm {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>} ins(%arg0, %arg1 : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6x6xf32>) outs(%1 : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
    return %2 : tensor<2x3x13x4x6x6xf32>
  }
}


// -----
module {
  func.func @depthwise_conv_3d_ndhwc_dhwc(%arg0: tensor<2x6x13x12x6xf32>, %arg1: tensor<2x1x3x6xf32>) -> tensor<2x3x13x4x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x3x13x4x6xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
    %2 = linalg.depthwise_conv_3d_ndhwc_dhwc {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>} ins(%arg0, %arg1 : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6xf32>) outs(%1 : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
    return %2 : tensor<2x3x13x4x6xf32>
  }
}


// -----
module {
  func.func @depthwise_conv_3d_ncdhw_cdhw(%arg0: tensor<2x6x6x13x12xf32>, %arg1: tensor<6x2x1x3xf32>) -> tensor<2x6x3x13x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x6x3x13x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x6x3x13x4xf32>) -> tensor<2x6x3x13x4xf32>
    %2 = linalg.depthwise_conv_3d_ncdhw_cdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<[2, 1, 3]> : tensor<3xi64>} ins(%arg0, %arg1 : tensor<2x6x6x13x12xf32>, tensor<6x2x1x3xf32>) outs(%1 : tensor<2x6x3x13x4xf32>) -> tensor<2x6x3x13x4xf32>
    return %2 : tensor<2x6x3x13x4xf32>
  }
}


// -----
module {
  func.func @conv_1d_nwc_wcf(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_1d_nwc_wcf(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @conv_1d_ncw_fcw(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_1d_ncw_fcw(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @conv_2d_nhwc_hwcf(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_2d_ngchw_fgchw(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
    %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
    return %0 : tensor<?x?x?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_2d_nhwc_fhwc(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_2d_nhwc_fhwc_static(%arg0: tensor<?x128x128x32xf32>, %arg1: tensor<64x3x3x32xf32>, %arg2: tensor<?x126x126x64xf32>) -> tensor<?x126x126x64xf32> {
    %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x128x128x32xf32>, tensor<64x3x3x32xf32>) outs(%arg2 : tensor<?x126x126x64xf32>) -> tensor<?x126x126x64xf32>
    return %0 : tensor<?x126x126x64xf32>
  }
}


// -----
module {
  func.func @conv_2d_nhwc_hwcf(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @conv_2d_ngchw_fgchw(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>, %arg2: memref<?x?x?x?x?xf32>) {
    linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @conv_2d_ngchw_fgchw_dimensions(%arg0: tensor<1x5x3x32x32xf32>, %arg1: tensor<2x5x3x3x3xf32>, %arg2: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32> {
    %0 = linalg.conv_2d_ngchw_fgchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x5x3x32x32xf32>, tensor<2x5x3x3x3xf32>) outs(%arg2 : tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
    return %0 : tensor<1x5x2x30x30xf32>
  }
}


// -----
module {
  func.func @conv_2d_ngchw_gfchw(%arg0: tensor<1x5x3x32x32xf32>, %arg1: tensor<5x2x3x3x3xf32>, %arg2: tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32> {
    %0 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x5x3x32x32xf32>, tensor<5x2x3x3x3xf32>) outs(%arg2 : tensor<1x5x2x30x30xf32>) -> tensor<1x5x2x30x30xf32>
    return %0 : tensor<1x5x2x30x30xf32>
  }
}


// -----
module {
  func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
    %0 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
    return %0 : tensor<?x?x?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_3d_ndhwc_dhwcf(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>, %arg2: memref<?x?x?x?x?xf32>) {
    linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @conv_3d_ncdhw_fcdhw(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
    %0 = linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
    return %0 : tensor<?x?x?x?x?xf32>
  }
}


// -----
module {
  func.func @conv_3d_ncdhw_fcdhw(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>, %arg2: memref<?x?x?x?x?xf32>) {
    linalg.conv_3d_ncdhw_fcdhw {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%arg2 : memref<?x?x?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nhwc_sum_tensor(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = tensor.empty() : tensor<1x2x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    %3 = linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    return %3 : tensor<1x2x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_nwc_sum_tensor(%arg0: tensor<1x4x1xf32>) -> tensor<1x2x1xf32> {
    %0 = tensor.empty() : tensor<3xf32>
    %1 = tensor.empty() : tensor<1x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
    %3 = linalg.pooling_nwc_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x4x1xf32>, tensor<3xf32>) outs(%2 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
    return %3 : tensor<1x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_nhwc_sum(%arg0: memref<1x4x4x1xf32>, %arg1: memref<3x3xf32>, %arg2: memref<1x2x2x1xf32>) {
    linalg.pooling_nhwc_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xf32>, memref<3x3xf32>) outs(%arg2 : memref<1x2x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nwc_sum(%arg0: memref<1x4x1xf32>, %arg1: memref<3xf32>, %arg2: memref<1x2x1xf32>) {
    linalg.pooling_nwc_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x4x1xf32>, memref<3xf32>) outs(%arg2 : memref<1x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nchw_sum_tensor(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = tensor.empty() : tensor<1x1x2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %3 = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x1x4x4xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %3 : tensor<1x1x2x2xf32>
  }
}


// -----
module {
  func.func @pooling_ncw_sum_tensor(%arg0: tensor<1x1x4xf32>) -> tensor<1x1x2xf32> {
    %0 = tensor.empty() : tensor<3xf32>
    %1 = tensor.empty() : tensor<1x1x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    %3 = linalg.pooling_ncw_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x1x4xf32>, tensor<3xf32>) outs(%2 : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    return %3 : tensor<1x1x2xf32>
  }
}


// -----
module {
  func.func @pooling_nchw_sum(%arg0: memref<1x1x4x4xf32>, %arg1: memref<3x3xf32>, %arg2: memref<1x1x2x2xf32>) {
    linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x1x4x4xf32>, memref<3x3xf32>) outs(%arg2 : memref<1x1x2x2xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_ncw_sum(%arg0: memref<1x1x4xf32>, %arg1: memref<3xf32>, %arg2: memref<1x1x2xf32>) {
    linalg.pooling_ncw_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x1x4xf32>, memref<3xf32>) outs(%arg2 : memref<1x1x2xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nhwc_max_tensor(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = tensor.empty() : tensor<1x2x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    %3 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    return %3 : tensor<1x2x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_nwc_max_tensor(%arg0: tensor<1x4x1xf32>) -> tensor<1x2x1xf32> {
    %0 = tensor.empty() : tensor<3xf32>
    %1 = tensor.empty() : tensor<1x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
    %3 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x4x1xf32>, tensor<3xf32>) outs(%2 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
    return %3 : tensor<1x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_nchw_max_tensor(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = tensor.empty() : tensor<1x1x2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %3 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x1x4x4xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %3 : tensor<1x1x2x2xf32>
  }
}


// -----
module {
  func.func @pooling_ncw_max_tensor(%arg0: tensor<1x1x4xf32>) -> tensor<1x1x2xf32> {
    %0 = tensor.empty() : tensor<3xf32>
    %1 = tensor.empty() : tensor<1x1x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    %3 = linalg.pooling_ncw_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x1x4xf32>, tensor<3xf32>) outs(%2 : tensor<1x1x2xf32>) -> tensor<1x1x2xf32>
    return %3 : tensor<1x1x2xf32>
  }
}


// -----
module {
  func.func @pooling_nhwc_max(%arg0: memref<1x4x4x1xf32>, %arg1: memref<3x3xf32>, %arg2: memref<1x2x2x1xf32>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xf32>, memref<3x3xf32>) outs(%arg2 : memref<1x2x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nwc_max(%arg0: memref<1x4x1xf32>, %arg1: memref<3xf32>, %arg2: memref<1x2x1xf32>) {
    linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x4x1xf32>, memref<3xf32>) outs(%arg2 : memref<1x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nhwc_i8_max_tensor(%arg0: tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8> {
    %0 = tensor.empty() : tensor<3x3xi8>
    %1 = tensor.empty() : tensor<1x2x2x1xi8>
    %c0_i8 = arith.constant 0 : i8
    %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
    %3 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xi8>, tensor<3x3xi8>) outs(%2 : tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
    return %3 : tensor<1x2x2x1xi8>
  }
}


// -----
module {
  func.func @pooling_nwc_i8_max_tensor(%arg0: tensor<1x4x1xi8>) -> tensor<1x2x1xi8> {
    %0 = tensor.empty() : tensor<3xi8>
    %1 = tensor.empty() : tensor<1x2x1xi8>
    %c0_i8 = arith.constant 0 : i8
    %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<1x2x1xi8>) -> tensor<1x2x1xi8>
    %3 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x4x1xi8>, tensor<3xi8>) outs(%2 : tensor<1x2x1xi8>) -> tensor<1x2x1xi8>
    return %3 : tensor<1x2x1xi8>
  }
}


// -----
module {
  func.func @pooling_nhwc_i8_max(%arg0: memref<1x4x4x1xi8>, %arg1: memref<3x3xi8>, %arg2: memref<1x2x2x1xi8>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xi8>, memref<3x3xi8>) outs(%arg2 : memref<1x2x2x1xi8>)
    return
  }
}


// -----
module {
  func.func @pooling_nwc_i8_max(%arg0: memref<1x4x1xi8>, %arg1: memref<3xi8>, %arg2: memref<1x2x1xi8>) {
    linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x4x1xi8>, memref<3xi8>) outs(%arg2 : memref<1x2x1xi8>)
    return
  }
}


// -----
module {
  func.func @pooling_nhwc_i16_max_tensor(%arg0: tensor<1x4x4x1xi16>) -> tensor<1x2x2x1xi16> {
    %0 = tensor.empty() : tensor<3x3xi16>
    %1 = tensor.empty() : tensor<1x2x2x1xi16>
    %c0_i16 = arith.constant 0 : i16
    %2 = linalg.fill ins(%c0_i16 : i16) outs(%1 : tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
    %3 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xi16>, tensor<3x3xi16>) outs(%2 : tensor<1x2x2x1xi16>) -> tensor<1x2x2x1xi16>
    return %3 : tensor<1x2x2x1xi16>
  }
}


// -----
module {
  func.func @pooling_nwc_i16_max_tensor(%arg0: tensor<1x4x1xi16>) -> tensor<1x2x1xi16> {
    %0 = tensor.empty() : tensor<3xi16>
    %1 = tensor.empty() : tensor<1x2x1xi16>
    %c0_i16 = arith.constant 0 : i16
    %2 = linalg.fill ins(%c0_i16 : i16) outs(%1 : tensor<1x2x1xi16>) -> tensor<1x2x1xi16>
    %3 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x4x1xi16>, tensor<3xi16>) outs(%2 : tensor<1x2x1xi16>) -> tensor<1x2x1xi16>
    return %3 : tensor<1x2x1xi16>
  }
}


// -----
module {
  func.func @pooling_nhwc_i16_max(%arg0: memref<1x4x4x1xi16>, %arg1: memref<3x3xi16>, %arg2: memref<1x2x2x1xi16>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xi16>, memref<3x3xi16>) outs(%arg2 : memref<1x2x2x1xi16>)
    return
  }
}


// -----
module {
  func.func @pooling_nwc_i16_max(%arg0: memref<1x4x1xi16>, %arg1: memref<3xi16>, %arg2: memref<1x2x1xi16>) {
    linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x4x1xi16>, memref<3xi16>) outs(%arg2 : memref<1x2x1xi16>)
    return
  }
}


// -----
module {
  func.func @pooling_nhwc_i32_max_tensor(%arg0: tensor<1x4x4x1xi32>) -> tensor<1x2x2x1xi32> {
    %0 = tensor.empty() : tensor<3x3xi32>
    %1 = tensor.empty() : tensor<1x2x2x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
    %3 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xi32>, tensor<3x3xi32>) outs(%2 : tensor<1x2x2x1xi32>) -> tensor<1x2x2x1xi32>
    return %3 : tensor<1x2x2x1xi32>
  }
}


// -----
module {
  func.func @pooling_nwc_i32_max_tensor(%arg0: tensor<1x4x1xi32>) -> tensor<1x2x1xi32> {
    %0 = tensor.empty() : tensor<3xi32>
    %1 = tensor.empty() : tensor<1x2x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<1x2x1xi32>) -> tensor<1x2x1xi32>
    %3 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x4x1xi32>, tensor<3xi32>) outs(%2 : tensor<1x2x1xi32>) -> tensor<1x2x1xi32>
    return %3 : tensor<1x2x1xi32>
  }
}


// -----
module {
  func.func @pooling_nhwc_i32_max(%arg0: memref<1x4x4x1xi32>, %arg1: memref<3x3xi32>, %arg2: memref<1x2x2x1xi32>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xi32>, memref<3x3xi32>) outs(%arg2 : memref<1x2x2x1xi32>)
    return
  }
}


// -----
module {
  func.func @pooling_nwc_i32_max(%arg0: memref<1x4x1xi32>, %arg1: memref<3xi32>, %arg2: memref<1x2x1xi32>) {
    linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x4x1xi32>, memref<3xi32>) outs(%arg2 : memref<1x2x1xi32>)
    return
  }
}


// -----
module {
  func.func @pooling_nhwc_min_tensor(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = tensor.empty() : tensor<1x2x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    %3 = linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x2x2x1xf32>) -> tensor<1x2x2x1xf32>
    return %3 : tensor<1x2x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_nwc_min_tensor(%arg0: tensor<1x4x1xf32>) -> tensor<1x2x1xf32> {
    %0 = tensor.empty() : tensor<3xf32>
    %1 = tensor.empty() : tensor<1x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
    %3 = linalg.pooling_nwc_min {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %0 : tensor<1x4x1xf32>, tensor<3xf32>) outs(%2 : tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
    return %3 : tensor<1x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_nhwc_min(%arg0: memref<1x4x4x1xf32>, %arg1: memref<3x3xf32>, %arg2: memref<1x2x2x1xf32>) {
    linalg.pooling_nhwc_min {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xf32>, memref<3x3xf32>) outs(%arg2 : memref<1x2x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_nwc_min(%arg0: memref<1x4x1xf32>, %arg1: memref<3xf32>, %arg2: memref<1x2x1xf32>) {
    linalg.pooling_nwc_min {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : memref<1x4x1xf32>, memref<3xf32>) outs(%arg2 : memref<1x2x1xf32>)
    return
  }
}


// -----
module {
 within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:1122 offset :9:8: error: unexpected input index map for convolutions
  %0 = "linalg.conv_2d_nhwc_hwcf"(%arg0, %arg1, %arg2) ({
       ^
within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:1122 offset :9:8: note: see current operation: 
%0 = "linalg.conv_2d_nhwc_hwcf"(%arg0, %arg2, %arg1) <{dilations = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<2> : tensor<2xi64>}> ({
^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  %1 = "arith.mulf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %2 = "arith.addf"(%arg5, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  "linalg.yield"(%2) : (f32) -> ()
}) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2, d2 * 2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:1139 offset :9:8: error: expected output/filter indexing maps to be projected permutations
  %0 = "linalg.conv_2d_nhwc_hwcf"(%arg0, %arg1, %arg2) ({
       ^
within split at ../mlir/examples/dsp/SimpleBlocks/Output/TryLinalg/named-ops.mlir:1139 offset :9:8: note: see current operation: 
%0 = "linalg.conv_2d_nhwc_hwcf"(%arg0, %arg1, %arg2) <{dilations = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>, strides = dense<1> : tensor<2xi64>}> ({
^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  %1 = "arith.mulf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %2 = "arith.addf"(%arg5, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  "linalg.yield"(%2) : (f32) -> ()
}) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3, d5 + 1)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
 func.func @pooling_ndhwc_sum_tensor(%arg0: tensor<1x4x4x4x1xf32>) -> tensor<1x2x2x2x1xf32> {
    %0 = tensor.empty() : tensor<3x3x3xf32>
    %1 = tensor.empty() : tensor<1x2x2x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
    %3 = linalg.pooling_ndhwc_sum {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %0 : tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>) outs(%2 : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
    return %3 : tensor<1x2x2x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_ndhwc_sum(%arg0: memref<1x4x4x4x1xf32>, %arg1: memref<3x3x3xf32>, %arg2: memref<1x2x2x2x1xf32>) {
    linalg.pooling_ndhwc_sum {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : memref<1x4x4x4x1xf32>, memref<3x3x3xf32>) outs(%arg2 : memref<1x2x2x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_ndhwc_max_tensor(%arg0: tensor<1x4x4x4x1xf32>) -> tensor<1x2x2x2x1xf32> {
    %0 = tensor.empty() : tensor<3x3x3xf32>
    %1 = tensor.empty() : tensor<1x2x2x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
    %3 = linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %0 : tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>) outs(%2 : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
    return %3 : tensor<1x2x2x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_ndhwc_max(%arg0: memref<1x4x4x4x1xf32>, %arg1: memref<3x3x3xf32>, %arg2: memref<1x2x2x2x1xf32>) {
    linalg.pooling_ndhwc_max {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : memref<1x4x4x4x1xf32>, memref<3x3x3xf32>) outs(%arg2 : memref<1x2x2x2x1xf32>)
    return
  }
}


// -----
module {
  func.func @pooling_ndhwc_min_tensor(%arg0: tensor<1x4x4x4x1xf32>) -> tensor<1x2x2x2x1xf32> {
    %0 = tensor.empty() : tensor<3x3x3xf32>
    %1 = tensor.empty() : tensor<1x2x2x2x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
    %3 = linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %0 : tensor<1x4x4x4x1xf32>, tensor<3x3x3xf32>) outs(%2 : tensor<1x2x2x2x1xf32>) -> tensor<1x2x2x2x1xf32>
    return %3 : tensor<1x2x2x2x1xf32>
  }
}


// -----
module {
  func.func @pooling_ndhwc_min(%arg0: memref<1x4x4x4x1xf32>, %arg1: memref<3x3x3xf32>, %arg2: memref<1x2x2x2x1xf32>) {
    linalg.pooling_ndhwc_min {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : memref<1x4x4x4x1xf32>, memref<3x3x3xf32>) outs(%arg2 : memref<1x2x2x2x1xf32>)
    return
  }
}


// -----

// -----

// -----
module {
  func.func @batch_reduce_matmul(%arg0: tensor<8x128x256xf32>, %arg1: tensor<8x256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<8x128x256xf32>, tensor<8x256x512xf32>) outs(%arg2 : tensor<128x512xf32>) -> tensor<128x512xf32>
    return %0 : tensor<128x512xf32>
  }
}


// -----
module {
  func.func @batch_reduce_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?xf32>)
    return
  }
}


// -----
module {
  func.func @matmul_transpose_a(%arg0: memref<5x3xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
    linalg.matmul_transpose_a ins(%arg0, %arg1 : memref<5x3xf32>, memref<5x7xf32>) outs(%arg2 : memref<3x7xf32>)
    return
  }
}


// -----
module {
  func.func @matmul_transpose_b(%arg0: memref<3x5xf32>, %arg1: memref<7x5xf32>, %arg2: memref<3x7xf32>) {
    linalg.matmul_transpose_b ins(%arg0, %arg1 : memref<3x5xf32>, memref<7x5xf32>) outs(%arg2 : memref<3x7xf32>)
    return
  }
}


// -----
module {
  func.func @batchmatmul_transpose_a(%arg0: memref<2x5x3xf32>, %arg1: memref<2x5x7xf32>, %arg2: memref<2x3x7xf32>) {
    linalg.batch_matmul_transpose_a ins(%arg0, %arg1 : memref<2x5x3xf32>, memref<2x5x7xf32>) outs(%arg2 : memref<2x3x7xf32>)
    return
  }
}


// -----
module {
  func.func @batchmatmul_transpose_b(%arg0: memref<2x3x5xf32>, %arg1: memref<2x7x5xf32>, %arg2: memref<2x3x7xf32>) {
    linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : memref<2x3x5xf32>, memref<2x7x5xf32>) outs(%arg2 : memref<2x3x7xf32>)
    return
  }
}


// -----
module {
  func.func @mmt4d(%arg0: tensor<10x32x8x1xf32>, %arg1: tensor<80x32x4x1xf32>, %arg2: tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32> {
    %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%arg2 : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
    return %0 : tensor<10x80x8x4xf32>
  }
}


// -----
module {
  func.func @batch_mmt4d(%arg0: tensor<128x10x32x8x1xf32>, %arg1: tensor<128x80x32x4x1xf32>, %arg2: tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32> {
    %0 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<128x10x32x8x1xf32>, tensor<128x80x32x4x1xf32>) outs(%arg2 : tensor<128x10x80x8x4xf32>) -> tensor<128x10x80x8x4xf32>
    return %0 : tensor<128x10x80x8x4xf32>
  }
}


// -----
module {
  func.func @add_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.add ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @add_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
    linalg.add ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @add_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @sub_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.sub ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @sub_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
    linalg.sub ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @sub_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.sub ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @mul_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.mul ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @mul_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
    linalg.mul ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @mul_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.mul ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @div_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.div ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @div_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
    linalg.div ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @div_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.div ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @div_unsigned_dynamic(%arg0: memref<?x?x?xi32>, %arg1: memref<?x?x?xi32>, %arg2: memref<?x?x?xi32>) {
    linalg.div_unsigned ins(%arg0, %arg1 : memref<?x?x?xi32>, memref<?x?x?xi32>) outs(%arg2 : memref<?x?x?xi32>)
    return
  }
}


// -----
module {
  func.func @div_unsigned_static(%arg0: memref<4x8x16xi32>, %arg1: memref<4x8x16xi32>, %arg2: memref<4x8x16xi32>) {
    linalg.div_unsigned ins(%arg0, %arg1 : memref<4x8x16xi32>, memref<4x8x16xi32>) outs(%arg2 : memref<4x8x16xi32>)
    return
  }
}


// -----
module {
  func.func @div_unsigned_tensor(%arg0: tensor<4x8x16xi32>, %arg1: tensor<4x8x16xi32>) -> tensor<4x8x16xi32> {
    %0 = tensor.empty() : tensor<4x8x16xi32>
    %1 = linalg.div_unsigned ins(%arg0, %arg1 : tensor<4x8x16xi32>, tensor<4x8x16xi32>) outs(%0 : tensor<4x8x16xi32>) -> tensor<4x8x16xi32>
    return %1 : tensor<4x8x16xi32>
  }
}


// -----
module {
  func.func @exp_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    linalg.exp ins(%arg0 : memref<?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @exp_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
    linalg.exp ins(%arg0 : memref<4x8x16xf32>) outs(%arg1 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @exp_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.exp ins(%arg0 : tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @log_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    linalg.log ins(%arg0 : memref<?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @log_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
    linalg.log ins(%arg0 : memref<4x8x16xf32>) outs(%arg1 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @log_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.log ins(%arg0 : tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @abs_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    linalg.abs ins(%arg0 : memref<?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @abs_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
    linalg.abs ins(%arg0 : memref<4x8x16xf32>) outs(%arg1 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @abs_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.abs ins(%arg0 : tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @ceil_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    linalg.ceil ins(%arg0 : memref<?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @ceil_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
    linalg.ceil ins(%arg0 : memref<4x8x16xf32>) outs(%arg1 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @ceil_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.ceil ins(%arg0 : tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @floor_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    linalg.floor ins(%arg0 : memref<?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @floor_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
    linalg.floor ins(%arg0 : memref<4x8x16xf32>) outs(%arg1 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @floor_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.floor ins(%arg0 : tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @negf_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) {
    linalg.negf ins(%arg0 : memref<?x?x?xf32>) outs(%arg1 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @negf_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>) {
    linalg.negf ins(%arg0 : memref<4x8x16xf32>) outs(%arg1 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @negf_tensor(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.negf ins(%arg0 : tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @max_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.max ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @max_static(%arg0: memref<4x8x16xf32>, %arg1: memref<4x8x16xf32>, %arg2: memref<4x8x16xf32>) {
    linalg.max ins(%arg0, %arg1 : memref<4x8x16xf32>, memref<4x8x16xf32>) outs(%arg2 : memref<4x8x16xf32>)
    return
  }
}


// -----
module {
  func.func @max_tensor(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32> {
    %0 = tensor.empty() : tensor<4x8x16xf32>
    %1 = linalg.max ins(%arg0, %arg1 : tensor<4x8x16xf32>, tensor<4x8x16xf32>) outs(%0 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    return %1 : tensor<4x8x16xf32>
  }
}


// -----
module {
  func.func @fill_tensor(%arg0: f32, %arg1: vector<2x4xf32>) -> (tensor<f32>, tensor<vector<2x4xf32>>) {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<f32>) -> tensor<f32>
    %2 = tensor.empty() : tensor<vector<2x4xf32>>
    %3 = linalg.fill ins(%arg1 : vector<2x4xf32>) outs(%2 : tensor<vector<2x4xf32>>) -> tensor<vector<2x4xf32>>
    return %1, %3 : tensor<f32>, tensor<vector<2x4xf32>>
  }
}

