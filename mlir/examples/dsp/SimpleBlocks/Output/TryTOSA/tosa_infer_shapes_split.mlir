module {
  func.func @test_return(%arg0: tensor<4xf32>) -> tensor<*xf32> {
    %0 = tosa.log %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
}


// -----
module {
  func.func @test_multiple(%arg0: tensor<4xf32>, %arg1: tensor<1xf32>, %arg2: tensor<f32>) -> tensor<*xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %1 = tosa.log %0 : (tensor<*xf32>) -> tensor<*xf32>
    %2 = tosa.sub %0, %arg2 : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
}


// -----
module {
  func.func @test_unary_f32(%arg0: tensor<4xf32>) {
    %0 = tosa.abs %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %1 = tosa.ceil %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %2 = tosa.clamp %arg0 {max_fp = 1.000000e+01 : f32, max_int = 10 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<4xf32>) -> tensor<*xf32>
    %3 = tosa.exp %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %4 = tosa.floor %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %5 = tosa.log %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %6 = tosa.negate %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %7 = tosa.reciprocal %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %8 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<4xf32>) -> tensor<?xf32>
    %9 = tosa.rsqrt %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %10 = tosa.tanh %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %11 = tosa.sigmoid %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    %12 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<*xi32>
    %13 = tosa.erf %arg0 : (tensor<4xf32>) -> tensor<*xf32>
    return
  }
}


// -----
module {
  func.func @test_unary_i32(%arg0: tensor<4xi32>) {
    %0 = tosa.abs %arg0 : (tensor<4xi32>) -> tensor<*xi32>
    %1 = tosa.bitwise_not %arg0 : (tensor<4xi32>) -> tensor<*xi32>
    %2 = tosa.clamp %arg0 {max_fp = 1.000000e+01 : f32, max_int = 10 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<4xi32>) -> tensor<*xi32>
    %3 = tosa.clz %arg0 : (tensor<4xi32>) -> tensor<*xi32>
    %4 = tosa.negate %arg0 : (tensor<4xi32>) -> tensor<*xi32>
    %5 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<?xi32>
    %6 = tosa.rescale %arg0 {double_round = false, input_zp = 243 : i32, multiplier = array<i32: 42, 43>, output_zp = 252 : i32, per_channel = false, scale32 = false, shift = array<i8: 14, 15>} : (tensor<4xi32>) -> tensor<*xi16>
    %7 = tosa.identity %arg0 : (tensor<4xi32>) -> tensor<?xi32>
    return
  }
}


// -----
module {
  func.func @test_binary_scalar_f32(%arg0: tensor<4xf32>, %arg1: tensor<f32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>
    %1 = tosa.maximum %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>
    %2 = tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>
    %3 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>
    %4 = tosa.pow %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>
    %5 = tosa.sub %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>
    %6 = tosa.equal %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xi1>
    %7 = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xi1>
    %8 = tosa.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<f32>) -> tensor<*xi1>
    return
  }
}


// -----
module {
  func.func @test_binary_broadcast_f32(%arg0: tensor<4xf32>, %arg1: tensor<1xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %1 = tosa.maximum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %2 = tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %3 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %4 = tosa.pow %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %5 = tosa.sub %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>
    %6 = tosa.equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>
    %7 = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>
    %8 = tosa.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>
    return
  }
}


// -----
module {
  func.func @test_binary_i32(%arg0: tensor<4xi32>, %arg1: tensor<i32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %1 = tosa.bitwise_and %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %2 = tosa.bitwise_or %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %3 = tosa.bitwise_xor %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %4 = tosa.equal %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi1>
    %5 = tosa.greater %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi1>
    %6 = tosa.greater_equal %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi1>
    %7 = tosa.logical_left_shift %arg0, %arg1 {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %8 = tosa.logical_right_shift %arg0, %arg1 {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %9 = tosa.maximum %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %10 = tosa.minimum %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %11 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %12 = tosa.pow %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    %13 = tosa.sub %arg0, %arg1 : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>
    return
  }
}


// -----
module {
  func.func @test_binary_i1(%arg0: tensor<4xi1>, %arg1: tensor<i1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<4xi1>, tensor<i1>) -> tensor<*xi1>
    %1 = tosa.logical_or %arg0, %arg1 : (tensor<4xi1>, tensor<i1>) -> tensor<*xi1>
    %2 = tosa.logical_xor %arg0, %arg1 : (tensor<4xi1>, tensor<i1>) -> tensor<*xi1>
    return
  }
}


// -----
module {
  func.func @test_select_i32(%arg0: tensor<4xi1>, %arg1: tensor<i32>, %arg2: tensor<4xi32>) {
    %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<4xi1>, tensor<i32>, tensor<4xi32>) -> tensor<*xi32>
    return
  }
}


// -----
module {
  func.func @test_static_argmax(%arg0: tensor<2x3xi32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<?xi32>
    %1 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<?xi32>
    return
  }
}


// -----
module {
  func.func @test_dynamic_argmax(%arg0: tensor<2x?xi32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<2x?xi32>) -> tensor<?xi32>
    %1 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x?xi32>) -> tensor<?xi32>
    return
  }
}


// -----
module {
  func.func @test_static_fully_connected(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_static_input_fully_connected(%arg0: tensor<3x4xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?xf32>) {
    %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_static_weight_fully_connected(%arg0: tensor<?x?xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<?xf32>) {
    %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_static_bias_fully_connected(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_static_out_fully_connected(%arg0: tensor<3x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_static_matmul(%arg0: tensor<2x3x4xi32>, %arg1: tensor<2x4x5xi32>) {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_dynamic_lhs_matmul(%arg0: tensor<?x?x?xi32>, %arg1: tensor<2x4x5xi32>) {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_dynamic_rhs_matmul(%arg0: tensor<2x3x4xi32>, %arg1: tensor<?x?x?xi32>) {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_dynamic_mixed_matmul(%arg0: tensor<?x3x?xi32>, %arg1: tensor<?x?x5xi32>) {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_table_static(%arg0: tensor<4x5xi16>, %arg1: tensor<513xi16>) {
    %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<?x?xi16>
    return
  }
}


// -----
module {
  func.func @test_table_dynamic(%arg0: tensor<4x?xi16>, %arg1: tensor<513xi16>) {
    %0 = tosa.table %arg0, %arg1 : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<?x?xi16>
    return
  }
}


// -----
module {
  func.func @test_static_reshape(%arg0: tensor<4x4xi32>) {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 16>} : (tensor<4x4xi32>) -> tensor<?xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: -1>} : (tensor<4x4xi32>) -> tensor<?xi32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 2, -1>} : (tensor<4x4xi32>) -> tensor<?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_dynamic_reshape(%arg0: tensor<4x?xi32>) {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 16>} : (tensor<4x?xi32>) -> tensor<?xi32>
    %1 = tosa.reshape %arg0 {new_shape = array<i64: -1>} : (tensor<4x?xi32>) -> tensor<?xi32>
    %2 = tosa.reshape %arg0 {new_shape = array<i64: 2, -1>} : (tensor<4x?xi32>) -> tensor<?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_reduce_binary(%arg0: tensor<2x3x?x?xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>
    %1 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>
    %2 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>
    %3 = tosa.reduce_all %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>
    %4 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>
    return
  }
}


// -----
module {
  func.func @test_reduce_float(%arg0: tensor<2x3x?x?xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    %1 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    %2 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    %3 = tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    %4 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    %5 = tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    %6 = tosa.reduce_prod %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_concat(%arg0: tensor<1x2xf32>, %arg1: tensor<2x2xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_concat_dynamic(%arg0: tensor<1x2xf32>, %arg1: tensor<2x?xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_concat_dynamic_axis(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_concat_axis_1(%arg0: tensor<2x1xf32>, %arg1: tensor<2x2xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_padding_no_const(%arg0: tensor<1x2xf32>, %arg1: tensor<2x2xi32>) {
    %0 = tosa.pad %arg0, %arg1 : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_padding_dynamic_input(%arg0: tensor<1x?xf32>) {
    %cst = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
    %0 = tosa.pad %arg0, %cst : (tensor<1x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_padding_simple(%arg0: tensor<1x2xf32>) {
    %cst = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
    %0 = tosa.pad %arg0, %cst : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_slice(%arg0: tensor<?xi32>) {
    %0 = tosa.slice %arg0 {size = array<i64: 2>, start = array<i64: 1>} : (tensor<?xi32>) -> tensor<?xi32>
    return
  }
}


// -----
module {
  func.func @test_slice_dynamic(%arg0: tensor<10x?x2xf32>) {
    %0 = tosa.slice %arg0 {size = array<i64: 7, -1, 1>, start = array<i64: 1, 0, 0>} : (tensor<10x?x2xf32>) -> tensor<?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_tile(%arg0: tensor<2x3x?xi32>) {
    %0 = tosa.tile %arg0 {multiples = array<i64: 2, 1, 5>} : (tensor<2x3x?xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_transpose_same(%arg0: tensor<4x4x4xi32>, %arg1: tensor<3xi32>) {
    %0 = tosa.transpose %arg0, %arg1 : (tensor<4x4x4xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_transpose_perm_unknown(%arg0: tensor<4x4x5xi32>, %arg1: tensor<3xi32>) {
    %0 = tosa.transpose %arg0, %arg1 : (tensor<4x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_transpose_static(%arg0: tensor<3x4x5xi32>) {
    %cst = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
    %0 = tosa.transpose %arg0, %cst : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @gather_static(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x6xi32>) {
    %0 = tosa.gather %arg0, %arg1 : (tensor<3x4x5xi32>, tensor<3x6xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @gather_dynamic_values(%arg0: tensor<?x?x?xi32>, %arg1: tensor<3x6xi32>) {
    %0 = tosa.gather %arg0, %arg1 : (tensor<?x?x?xi32>, tensor<3x6xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @gather_dynamic_indices(%arg0: tensor<3x4x5xi32>, %arg1: tensor<?x?xi32>) {
    %0 = tosa.gather %arg0, %arg1 : (tensor<3x4x5xi32>, tensor<?x?xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @gather_minimum_info(%arg0: tensor<3x?x5xi32>, %arg1: tensor<?x6xi32>) {
    %0 = tosa.gather %arg0, %arg1 : (tensor<3x?x5xi32>, tensor<?x6xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @scatter_static(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x6xi32>, %arg2: tensor<3x6x5xi32>) {
    %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @scatter_static_values(%arg0: tensor<3x4x5xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?x?xi32>) {
    %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @scatter_static_indices(%arg0: tensor<?x?x?xi32>, %arg1: tensor<3x6xi32>, %arg2: tensor<?x?x?xi32>) {
    %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @scatter_static_input(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<3x6x5xi32>) {
    %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @scatter_minimum_static(%arg0: tensor<?x4x?xi32>, %arg1: tensor<3x?xi32>, %arg2: tensor<?x?x5xi32>) {
    %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @test_pool_static(%arg0: tensor<3x5x6x7xf32>) {
    %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_static(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_pool_dynamic_input(%arg0: tensor<?x?x?x?xf32>) {
    %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_pool_padded(%arg0: tensor<3x5x6x7xf32>) {
    %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_dynamic_weight(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_dynamic_bias(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<?xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @test_pool_stride(%arg0: tensor<3x11x12x7xf32>) {
    %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>
    %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_padded(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_dilated(%arg0: tensor<2x12x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 3, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x12x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv2d_strided(%arg0: tensor<1x13x14x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 2>} : (tensor<1x13x14x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_static(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_dynamic_input(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<?x?x?x?x?xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_dynamic_weight(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<?x?x?x?x?xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_dynamic_bias(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<?xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<?xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_padded(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 2, 3, 4, 5, 6>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_dilated(%arg0: tensor<2x12x14x16x3xf32>, %arg1: tensor<5x3x6x2x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 3, 2, 4>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x12x14x16x3xf32>, tensor<5x3x6x2x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @conv3d_strided(%arg0: tensor<1x13x14x15x1xf32>, %arg1: tensor<1x1x1x1x1xf32>, %arg2: tensor<1xf32>) {
    %0 = tosa.conv3d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 3, 2, 4>} : (tensor<1x13x14x15x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1xf32>) -> tensor<?x?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_static(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x6x4x15xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<?x?x?x15xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_dynamic_weight(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<15xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<15xf32>) -> tensor<2x?x?x15xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_dynamic_bias(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<?xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<?xf32>) -> tensor<2x6x4x15xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_padded(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x9x11x15xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_dilated(%arg0: tensor<2x12x14x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 3, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x12x14x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x6x4x15xf32>
    return
  }
}


// -----
module {
  func.func @depthwise_conv2d_strided(%arg0: tensor<1x13x14x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
    %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 2>} : (tensor<1x13x14x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x5x7x1xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_out_shape(%arg0: tensor<2x?x?x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, 8, 9, -1>, stride = array<i64: 1, 1>} : (tensor<2x?x?x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x8x9x5xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_static(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_static_strided(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 2, 3>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x5xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_dynamic_weights(%arg0: tensor<2x6x4x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x6x4x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_dynamic_bias(%arg0: tensor<2x6x4x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<?xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x6x4x3xf32>, tensor<5x3x6x3xf32>, tensor<?xf32>) -> tensor<2x8x9x5xf32>
    return
  }
}


// -----
module {
  func.func @transpose_conv2d_padded(%arg0: tensor<2x9x11x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 1, 0, 3, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x9x11x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x10x13x5xf32>
    return
  }
  func.func @transpose_conv2d_strided(%arg0: tensor<1x5x7x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 3, 2>} : (tensor<1x5x7x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x13x13x1xf32>
    return
  }
}


// -----
module {
  func.func @resize_int_horizontal(%arg0: tensor<1x15x13x1xi8>) {
    %0 = tosa.resize %arg0 {border = array<i64: 0, 0>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 11, 7, 89, 6>} : (tensor<1x15x13x1xi8>) -> tensor<?x?x?x?xi8>
    return
  }
}


// -----
module {
  func.func @resize_int_vertical(%arg0: tensor<1x49x42x1xi16>) {
    %0 = tosa.resize %arg0 {border = array<i64: 0, 0>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 37, 16, 219, 41>} : (tensor<1x49x42x1xi16>) -> tensor<?x?x?x?xi16>
    return
  }
}


// -----
module {
  func.func @resize_int_power_of_two_upscale(%arg0: tensor<1x23x19x1xi8>) {
    %0 = tosa.resize %arg0 {border = array<i64: 0, 0>, mode = "BILINEAR", offset = array<i64: 0, 0>, scale = array<i64: 16, 1, 16, 1>} : (tensor<1x23x19x1xi8>) -> tensor<?x?x?x?xi32>
    return
  }
}


// -----
module {
  func.func @resize_int_power_of_two_upscale_offsetted(%arg0: tensor<1x41x26x1xi16>) {
    %0 = tosa.resize %arg0 {border = array<i64: 7, 7>, mode = "BILINEAR", offset = array<i64: -7, -7>, scale = array<i64: 16, 2, 16, 2>} : (tensor<1x41x26x1xi16>) -> tensor<?x?x?x?xi48>
    return
  }
}


// -----
module {
  func.func @resize_fp_horizontal(%arg0: tensor<1x50x48x1xf32>) {
    %0 = tosa.resize %arg0 {border = array<i64: 0, 0>, mode = "BILINEAR", offset = array<i64: 0, 0>, scale = array<i64: 15, 7, 84, 47>} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @resize_fp_vertical(%arg0: tensor<1x50x48x1xf32>) {
    %0 = tosa.resize %arg0 {border = array<i64: 0, 0>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 127, 49, 12, 47>} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @resize_fp_power_of_two_upscale(%arg0: tensor<1x23x23x1xf32>) {
    %0 = tosa.resize %arg0 {border = array<i64: 0, 0>, mode = "BILINEAR", offset = array<i64: 0, 0>, scale = array<i64: 4, 1, 4, 1>} : (tensor<1x23x23x1xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @resize_fp_power_of_two_upscale_offsetted(%arg0: tensor<1x50x48x1xf32>) {
    %0 = tosa.resize %arg0 {border = array<i64: 31, 31>, mode = "NEAREST_NEIGHBOR", offset = array<i64: -31, -31>, scale = array<i64: 64, 2, 64, 2>} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
    return
  }
}


// -----
module {
  func.func @if_test_simple(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) {
    %0 = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.log %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.cond_if %arg2 -> (tensor<f32>) {
      tosa.yield %0 : tensor<f32>
    } else {
      tosa.yield %1 : tensor<f32>
    }
    return
  }
}


// -----
module {
  func.func @if_test_dynamic(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<i1>) {
    %0 = tosa.cond_if %arg2 -> (tensor<?xf32>) {
      tosa.yield %arg0 : tensor<2xf32>
    } else {
      tosa.yield %arg1 : tensor<3xf32>
    }
    return
  }
}


// -----
module {
  func.func @if_test_unranked(%arg0: tensor<f32>, %arg1: tensor<3xf32>, %arg2: tensor<i1>) {
    %0 = tosa.cond_if %arg2 -> (tensor<*xf32>) {
      tosa.yield %arg0 : tensor<f32>
    } else {
      tosa.yield %arg1 : tensor<3xf32>
    }
    return
  }
}


// -----
module {
  func.func @if_test_propagate(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) {
    %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
      %1 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      tosa.yield %1 : tensor<f32>
    } else {
      %1 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      tosa.yield %1 : tensor<f32>
    }
    return
  }
}


// -----
module {
  func.func @while_test(%arg0: tensor<i32>) -> tensor<*xi32> {
    %0 = tosa.add %arg0, %arg0 : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
    %1 = tosa.while_loop (%arg1 = %0) : (tensor<*xi32>) -> tensor<*xi32> {
      %2 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
      %3 = tosa.greater_equal %2, %arg1 : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
      tosa.yield %3 : tensor<*xi1>
    } do {
    ^bb0(%arg1: tensor<*xi32>):
      %2 = "tosa.const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
      %3 = tosa.add %arg1, %2 : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      tosa.yield %3 : tensor<*xi32>
    }
    return %1 : tensor<*xi32>
  }
}


// -----
module {
  func.func @while_test(%arg0: tensor<i32>, %arg1: tensor<1xi32>) {
    %0:2 = tosa.while_loop (%arg2 = %arg0, %arg3 = %arg1) : (tensor<i32>, tensor<1xi32>) -> (tensor<i32>, tensor<?xi32>) {
      %1 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
      %2 = tosa.greater_equal %1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      tosa.yield %2 : tensor<i1>
    } do {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<?xi32>):
      %1 = "tosa.const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
      %2 = tosa.add %arg2, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %3 = tosa.concat %arg3, %arg3 {axis = 0 : i32} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
      tosa.yield %2, %3 : tensor<i32>, tensor<?xi32>
    }
    return
  }
}


// -----
module {
  func.func @test_static_rfft2d(%arg0: tensor<5x2x8xf32>) {
    %output_real, %output_imag = tosa.rfft2d %arg0 : (tensor<5x2x8xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @test_dynamic_batch_rfft2d(%arg0: tensor<?x2x4xf32>) {
    %output_real, %output_imag = tosa.rfft2d %arg0 : (tensor<?x2x4xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @test_dynamic_width_rfft2d(%arg0: tensor<5x2x?xf32>) {
    %output_real, %output_imag = tosa.rfft2d %arg0 : (tensor<5x2x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    return
  }
}


// -----
module {
  func.func @test_static_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
    %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
    return %output_real, %output_imag : tensor<1x4x8xf32>, tensor<1x4x8xf32>
  }
}


// -----
module {
  func.func @test_dynamic_batch_fft2d(%arg0: tensor<?x4x8xf32>, %arg1: tensor<?x4x8xf32>) -> (tensor<?x4x8xf32>, tensor<?x4x8xf32>) {
    %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<?x4x8xf32>, tensor<?x4x8xf32>) -> (tensor<?x4x8xf32>, tensor<?x4x8xf32>)
    return %output_real, %output_imag : tensor<?x4x8xf32>, tensor<?x4x8xf32>
  }
}


// -----
module {
  func.func @test_unranked_equal(%arg0: tensor<*xf32>, %arg1: tensor<f32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
    return
  }
}


// -----
module {
  func.func @test_non_tosa_consumer_shape(%arg0: tensor<4x4xf32>) -> !shape.shape {
    %0 = tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<*xf32>
    %1 = shape.shape_of %0 : tensor<*xf32> -> !shape.shape
    return %1 : !shape.shape
  }
}


// -----
module {
  func.func @test_non_tosa_consumer_shape2(%arg0: tensor<4x4xf32>) -> tensor<?xindex> {
    %0 = tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<*xf32>
    %1 = shape.shape_of %0 : tensor<*xf32> -> tensor<?xindex>
    return %1 : tensor<?xindex>
  }
}


// -----
module {
  func.func @test_non_tosa_consumer_extract(%arg0: tensor<4x4xf32>, %arg1: index) -> f32 {
    %0 = tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<?x?xf32>
    %extracted = tensor.extract %0[%arg1, %arg1] : tensor<?x?xf32>
    return %extracted : f32
  }
}


// -----
module {
  func.func @test_non_tosa_consumer_still_propagates(%arg0: tensor<1x1x8xf32>, %arg1: tensor<1x8x1xf32>) -> tensor<?x?xf32> {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<1x1x8xf32>, tensor<1x8x1xf32>) -> tensor<?x1x1xf32>
    %cst = arith.constant dense<1> : tensor<2xindex>
    %reshape = tensor.reshape %0(%cst) : (tensor<?x1x1xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    return %reshape : tensor<?x?xf32>
  }
}


// -----
module {
  func.func @test_tosa_use_def_chain(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<?x16x16x16xf32> {
    %0 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<?x32x32x16xf32>
    %1 = tosa.max_pool2d %0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x32x32x16xf32>) -> tensor<?x16x16x16xf32>
    return %1 : tensor<?x16x16x16xf32>
  }
}


// -----
module {
  func.func @test_rank_size_constant_permutation() {
    %c6 = arith.constant 6 : index
    %cst = arith.constant dense<[0, 2]> : tensor<2xi32>
    %0 = tensor.empty(%c6) : tensor<?x27xi64>
    %1 = tosa.transpose %0, %cst : (tensor<?x27xi64>, tensor<2xi32>) -> tensor<?x27xi64>
    return
  }
}


// -----
module {
  func.func @test_large_constant_permutation() {
    %c6 = arith.constant 6 : index
    %cst = arith.constant dense<[1185677355, 332462212]> : tensor<2xi32>
    %0 = tensor.empty(%c6) : tensor<?x27xi64>
    %1 = tosa.transpose %0, %cst : (tensor<?x27xi64>, tensor<2xi32>) -> tensor<?x27xi64>
    return
  }
}

