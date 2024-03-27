#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module {
  func.func @matmul(%arg0: memref<?xi8>, %arg1: index, %arg2: index, %arg3: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][%arg1, %arg3] : memref<?xi8> to memref<?x?xf32>
    %view_0 = memref.view %arg0[%c0][%arg3, %arg2] : memref<?xi8> to memref<?x?xf32>
    %view_1 = memref.view %arg0[%c0][%arg1, %arg2] : memref<?xi8> to memref<?x?xf32>
    linalg.matmul ins(%view, %view_0 : memref<?x?xf32>, memref<?x?xf32>) outs(%view_1 : memref<?x?xf32>)
    return
  }
  func.func @matvec(%arg0: memref<?xi8>, %arg1: index, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][%arg1, %arg2] : memref<?xi8> to memref<?x?xf32>
    %view_0 = memref.view %arg0[%c0][%arg1] : memref<?xi8> to memref<?xf32>
    %view_1 = memref.view %arg0[%c0][%arg2] : memref<?xi8> to memref<?xf32>
    linalg.matvec ins(%view, %view_0 : memref<?x?xf32>, memref<?xf32>) outs(%view_1 : memref<?xf32>)
    return
  }
  func.func @dot(%arg0: memref<?xi8>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %view = memref.view %arg0[%c0][%arg1] : memref<?xi8> to memref<?xf32>
    %view_0 = memref.view %arg0[%c0][%arg1] : memref<?xi8> to memref<?xf32>
    %view_1 = memref.view %arg0[%c0][] : memref<?xi8> to memref<f32>
    linalg.dot ins(%view, %view_0 : memref<?xf32>, memref<?xf32>) outs(%view_1 : memref<f32>)
    return
  }
  func.func @dot_int(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<i32>) {
    linalg.dot ins(%arg0, %arg1 : memref<?xi32>, memref<?xi32>) outs(%arg2 : memref<i32>)
    return
  }
  func.func @dot_bool(%arg0: memref<?xi1>, %arg1: memref<?xi1>, %arg2: memref<i1>) {
    linalg.dot ins(%arg0, %arg1 : memref<?xi1>, memref<?xi1>) outs(%arg2 : memref<i1>)
    return
  }
  func.func @dot_view(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>, %arg2: memref<f32>) {
    linalg.dot ins(%arg0, %arg1 : memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>) outs(%arg2 : memref<f32>)
    return
  }
  func.func @fill_view(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: f32) {
    linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<?xf32, strided<[1], offset: ?>>)
    return
  }
  func.func @fill_view0(%arg0: memref<f32>, %arg1: f32) {
    linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<f32>)
    return
  }
  func.func @fill_view3(%arg0: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %arg1: f32) {
    linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>)
    return
  }
  func.func @copy_view(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<?xf32, strided<[1], offset: ?>>) {
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf32, strided<[1], offset: ?>>) outs(%arg1 : memref<?xf32, strided<[1], offset: ?>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
  func.func @generic_region(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %arg2: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
    linalg.generic {doc = "B(i,j,k), C(i,k,j) = foo(A(i, j), B(i,j,k), C(i,k,j))", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel"], library_call = "some_external_function_name_2"} ins(%arg0 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%arg1, %arg2 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) attrs =  {args_in = 1 : i64, args_out = 2 : i64} {
    ^bb0(%in: f32, %out: f32, %out_0: f32):
      %0 = arith.mulf %in, %out : f32
      %1 = arith.addf %out_0, %0 : f32
      linalg.yield %0, %1 : f32, f32
    }
    return
  }
  func.func @generic_index_region(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, %arg2: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
    linalg.generic {doc = "B(i,j,k), C(i,k,j) = foo(A(i, j) * B(i,j,k), i * j * k + C(i,k,j))", indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel"], library_call = "some_external_function_name_2"} ins(%arg0 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%arg1, %arg2 : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) attrs =  {args_in = 1 : i64, args_out = 2 : i64} {
    ^bb0(%in: f32, %out: f32, %out_0: f32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = linalg.index 2 : index
      %3 = arith.mulf %in, %out : f32
      %4 = arith.addi %0, %1 : index
      %5 = arith.addi %4, %2 : index
      %6 = arith.index_cast %5 : index to i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.addf %out_0, %7 : f32
      linalg.yield %3, %8 : f32, f32
    }
    return
  }
}


// -----
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
#map4 = affine_map<() -> ()>
module {
  func.func @generic_op_zero_rank(%arg0: memref<f32>, %arg1: memref<3x4xf32>) {
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"], library_call = "some_broadcast_external_fn"} ins(%arg0 : memref<f32>) outs(%arg1 : memref<3x4xf32>) attrs =  {args_in = 1 : i64, args_out = 1 : i64} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
  func.func @generic_op_scalar(%arg0: f32, %arg1: memref<3x4xf32>) {
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"], library_call = "some_broadcast_external_fn"} ins(%arg0 : f32) outs(%arg1 : memref<3x4xf32>) attrs =  {args_in = 1 : i64, args_out = 1 : i64} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
  func.func @generic_index_op_zero_rank(%arg0: memref<i32>, %arg1: memref<3x4xi32>) {
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"], library_call = "some_broadcast_external_fn"} ins(%arg0 : memref<i32>) outs(%arg1 : memref<3x4xi32>) attrs =  {args_in = 1 : i64, args_out = 1 : i64} {
    ^bb0(%in: i32, %out: i32):
      %0 = linalg.index 0 : index
      %1 = linalg.index 1 : index
      %2 = arith.addi %0, %1 : index
      %3 = arith.index_cast %2 : index to i32
      %4 = arith.addi %in, %3 : i32
      linalg.yield %4 : i32
    }
    return
  }
  func.func @generic_op_1D_reduce(%arg0: memref<?xf32>, %arg1: memref<f32>) {
    linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"], library_call = "some_reduce_external_fn"} ins(%arg0 : memref<?xf32>) outs(%arg1 : memref<f32>) attrs =  {args_in = 1 : i64, args_out = 1 : i64} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    }
    return
  }
  func.func @generic_index_op_1D_reduce(%arg0: memref<?xf32>, %arg1: memref<f32>, %arg2: memref<f32>) {
    linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["reduction"], library_call = "some_reduce_external_fn"} ins(%arg0, %arg1 : memref<?xf32>, memref<f32>) outs(%arg2 : memref<f32>) attrs =  {args_in = 2 : i64, args_out = 1 : i64} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = linalg.index 0 : index
      %c0 = arith.constant 0 : index
      %1 = arith.cmpi eq, %c0, %0 : index
      %2 = arith.select %1, %in_0, %out : f32
      %3 = arith.addf %in, %2 : f32
      linalg.yield %3 : f32
    }
    return
  }
  func.func @generic_const_init(%arg0: memref<?xf32>) {
    %cst = arith.constant 1.000000e+00 : f32
    linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"], library_call = "some_external_fn"} outs(%arg0 : memref<?xf32>) attrs =  {args_in = 0 : i64, args_out = 1 : i64} {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    return
  }
  func.func @scalar_code(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: i1) {
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = [], library_call = "some_external_fn"} ins(%arg0, %arg1 : memref<f32>, memref<f32>) outs(%arg2 : memref<f32>) attrs =  {args_in = 2 : i64, args_out = 1 : i64} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = scf.if %arg3 -> (f32) {
        scf.yield %in : f32
      } else {
        scf.yield %in_0 : f32
      }
      linalg.yield %0 : f32
    }
    return
  }
  func.func @named_batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.batch_matmul ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
  func.func @conv1d_no_symbols(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    linalg.conv_1d ins(%arg0, %arg1 : memref<?xf32>, memref<?xf32>) outs(%arg2 : memref<?xf32>)
    return
  }
  func.func @conv2d_no_symbols(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
    return
  }
  func.func @conv3d_no_symbols(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
    linalg.conv_3d ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%arg2 : memref<?x?x?xf32>)
    return
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func @lower_to_loops_with_rank_reducing_subviews(%arg0: memref<?xi32>, %arg1: memref<?x?xi32>, %arg2: index, %arg3: index, %arg4: index) {
    %subview = memref.subview %arg0[%arg2] [%arg3] [1] : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %arg1[0, %arg4] [1, %arg3] [1, 1] : memref<?x?xi32> to memref<?xi32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xi32, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xi32, strided<[1], offset: ?>>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

