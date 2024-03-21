module {
  func.func @nested_loops_both_having_invariant_code() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
      affine.store %0, %alloc[%arg0] : memref<10xf32>
    }
    return
  }
}


// -----
#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @store_affine_apply() -> memref<10xf32> {
    %cst = arith.constant 7.000000e+00 : f32
    %alloc = memref.alloc() : memref<10xf32>
    affine.for %arg0 = 0 to 10 {
      %0 = affine.apply #map(%arg0)
      affine.store %cst, %alloc[%0] : memref<10xf32>
    }
    return %alloc : memref<10xf32>
  }
}


// -----
module {
  func.func @nested_loops_code_invariant_to_both() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
    }
    return
  }
}


// -----
module {
  func.func @nested_loops_inner_loops_invariant_to_outermost_loop(%arg0: memref<10xindex>) {
    affine.for %arg1 = 0 to 30 {
      %0 = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %arg1) -> (index) {
        %1 = affine.load %arg0[%arg2] : memref<10xindex>
        %2 = arith.addi %arg3, %1 : index
        affine.yield %2 : index
      }
    }
    affine.for %arg1 = 0 to 20 {
    }
    return
  }
}


// -----
module {
  func.func @single_loop_nothing_invariant() {
    %alloc = memref.alloc() : memref<10xf32>
    %alloc_0 = memref.alloc() : memref<11xf32>
    affine.for %arg0 = 0 to 10 {
      %0 = affine.load %alloc[%arg0] : memref<10xf32>
      %1 = affine.load %alloc_0[%arg0] : memref<11xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %alloc[%arg0] : memref<10xf32>
    }
    return
  }
}


// -----
#map = affine_map<(d0) -> (d0 + 1)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_code_inside_affine_if() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      %0 = affine.apply #map(%arg0)
      affine.if #set(%arg0, %0) {
        %1 = arith.addf %cst, %cst : f32
        affine.store %1, %alloc[%arg0] : memref<10xf32>
      }
    }
    return
  }
}


// -----
module {
  func.func @dependent_stores() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    %1 = arith.mulf %cst, %cst : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.store %1, %alloc[%arg1] : memref<10xf32>
        affine.store %0, %alloc[%arg0] : memref<10xf32>
      }
    }
    return
  }
}


// -----
module {
  func.func @independent_stores() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    %1 = arith.mulf %cst, %cst : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.store %0, %alloc[%arg0] : memref<10xf32>
        affine.store %1, %alloc[%arg1] : memref<10xf32>
      }
    }
    return
  }
}


// -----
module {
  func.func @load_dependent_store() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    %1 = arith.addf %cst, %cst : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.store %0, %alloc[%arg1] : memref<10xf32>
        %2 = affine.load %alloc[%arg0] : memref<10xf32>
      }
    }
    return
  }
}


// -----
module {
  func.func @load_after_load() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    %1 = arith.addf %cst, %cst : f32
    affine.for %arg0 = 0 to 10 {
      %2 = affine.load %alloc[%arg0] : memref<10xf32>
    }
    affine.for %arg0 = 0 to 10 {
      %2 = affine.load %alloc[%arg0] : memref<10xf32>
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_if() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
      affine.if #set(%arg0, %arg0) {
        %0 = arith.addf %cst, %cst : f32
        affine.store %0, %alloc[%arg0] : memref<10xf32>
      }
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_if2() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.if #set(%arg0, %arg0) {
          %0 = arith.addf %cst, %cst : f32
          affine.store %0, %alloc[%arg1] : memref<10xf32>
        }
      }
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_nested_if() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.if #set(%arg0, %arg0) {
          %0 = arith.addf %cst, %cst : f32
          affine.store %0, %alloc[%arg0] : memref<10xf32>
          affine.if #set(%arg0, %arg0) {
            affine.store %0, %alloc[%arg1] : memref<10xf32>
          }
        }
      }
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_nested_if_else() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.if #set(%arg0, %arg0) {
          %0 = arith.addf %cst, %cst : f32
          affine.store %0, %alloc[%arg0] : memref<10xf32>
          affine.if #set(%arg0, %arg0) {
            affine.store %0, %alloc[%arg0] : memref<10xf32>
          } else {
            affine.store %0, %alloc[%arg1] : memref<10xf32>
          }
        }
      }
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_nested_if_else2() {
    %alloc = memref.alloc() : memref<10xf32>
    %alloc_0 = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
      affine.if #set(%arg0, %arg0) {
        %0 = arith.addf %cst, %cst : f32
        %1 = affine.load %alloc[%arg0] : memref<10xf32>
        affine.if #set(%arg0, %arg0) {
          affine.store %0, %alloc_0[%arg0] : memref<10xf32>
        } else {
          %2 = affine.load %alloc[%arg0] : memref<10xf32>
        }
      }
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_nested_if2() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
      affine.if #set(%arg0, %arg0) {
        %0 = arith.addf %cst, %cst : f32
        %1 = affine.load %alloc[%arg0] : memref<10xf32>
        affine.if #set(%arg0, %arg0) {
          %2 = affine.load %alloc[%arg0] : memref<10xf32>
        }
      }
    }
    return
  }
}


// -----
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module {
  func.func @invariant_affine_for_inside_affine_if() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.if #set(%arg0, %arg0) {
          %0 = arith.addf %cst, %cst : f32
          affine.store %0, %alloc[%arg0] : memref<10xf32>
          affine.for %arg2 = 0 to 10 {
            affine.store %0, %alloc[%arg2] : memref<10xf32>
          }
        }
      }
    }
    return
  }
}


// -----
module {
  func.func @invariant_constant_and_load() {
    %alloc = memref.alloc() : memref<100xf32>
    %alloc_0 = memref.alloc() : memref<100xf32>
    %c0 = arith.constant 0 : index
    %0 = affine.load %alloc_0[%c0] : memref<100xf32>
    affine.for %arg0 = 0 to 5 {
      affine.store %0, %alloc[%arg0] : memref<100xf32>
    }
    return
  }
}


// -----
module {
  func.func @nested_load_store_same_memref() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    %c0 = arith.constant 0 : index
    affine.for %arg0 = 0 to 10 {
      %0 = affine.load %alloc[%c0] : memref<10xf32>
      affine.for %arg1 = 0 to 10 {
        affine.store %cst, %alloc[%arg1] : memref<10xf32>
      }
    }
    return
  }
}


// -----
module {
  func.func @nested_load_store_same_memref2() {
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 8.000000e+00 : f32
    %c0 = arith.constant 0 : index
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
      affine.store %cst, %alloc[%c0] : memref<10xf32>
      %0 = affine.load %alloc[%arg0] : memref<10xf32>
    }
    return
  }
}


// -----
module {
  func.func @do_not_hoist_dependent_side_effect_free_op(%arg0: memref<10x512xf32>) {
    %alloca = memref.alloca() : memref<1xf32>
    %cst = arith.constant 8.000000e+00 : f32
    affine.for %arg1 = 0 to 512 {
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %arg0[%arg1, %arg2] : memref<10x512xf32>
        %3 = affine.load %alloca[0] : memref<1xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %alloca[0] : memref<1xf32>
      }
      %0 = affine.load %alloca[0] : memref<1xf32>
      %1 = arith.mulf %0, %cst : f32
    }
    return
  }
}


// -----
module {
  func.func @vector_loop_nothing_invariant() {
    %alloc = memref.alloc() : memref<40xf32>
    %alloc_0 = memref.alloc() : memref<40xf32>
    affine.for %arg0 = 0 to 10 {
      %0 = affine.vector_load %alloc[%arg0 * 4] : memref<40xf32>, vector<4xf32>
      %1 = affine.vector_load %alloc_0[%arg0 * 4] : memref<40xf32>, vector<4xf32>
      %2 = arith.addf %0, %1 : vector<4xf32>
      affine.vector_store %2, %alloc[%arg0 * 4] : memref<40xf32>, vector<4xf32>
    }
    return
  }
}


// -----
module {
  func.func @vector_loop_all_invariant() {
    %alloc = memref.alloc() : memref<4xf32>
    %alloc_0 = memref.alloc() : memref<4xf32>
    %alloc_1 = memref.alloc() : memref<4xf32>
    %0 = affine.vector_load %alloc[0] : memref<4xf32>, vector<4xf32>
    %1 = affine.vector_load %alloc_0[0] : memref<4xf32>, vector<4xf32>
    %2 = arith.addf %0, %1 : vector<4xf32>
    affine.vector_store %2, %alloc_1[0] : memref<4xf32>, vector<4xf32>
    affine.for %arg0 = 0 to 10 {
    }
    return
  }
}


// -----affine-loop-invariant-code-motion
#set = affine_set<(d0) : (d0 - 10 >= 0)>
module {
  func.func @affine_if_not_invariant(%arg0: memref<1024xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = affine.for %arg1 = 0 to 10 step 2 iter_args(%arg2 = %cst) -> (f32) {
      %1 = affine.load %arg0[%arg1] : memref<1024xf32>
      %2 = affine.if #set(%arg1) -> f32 {
        %4 = arith.addf %arg2, %1 : f32
        affine.yield %4 : f32
      } else {
        affine.yield %arg2 : f32
      }
      %3 = arith.addf %2, %cst_0 : f32
      affine.yield %3 : f32
    }
    return %0 : f32
  }
}


// -----
module {
  func.func @affine_for_not_invariant(%arg0: memref<30x512xf32, 1>, %arg1: memref<30x1xf32, 1>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.100000e+00 : f32
    affine.for %arg2 = 0 to 30 {
      %0 = affine.for %arg3 = 0 to 512 iter_args(%arg4 = %cst) -> (f32) {
        %2 = affine.load %arg0[%arg2, %arg3] : memref<30x512xf32, 1>
        %3 = arith.addf %arg4, %2 : f32
        affine.yield %3 : f32
      }
      %1 = arith.mulf %0, %cst_0 : f32
      affine.store %1, %arg1[%arg2, 0] : memref<30x1xf32, 1>
    }
    return
  }
}


// -----
module {
  func.func @use_of_iter_operands_invariant(%arg0: memref<10xindex>) {
    %c0 = arith.constant 0 : index
    %0 = arith.muli %c0, %c0 : index
    %1 = affine.for %arg1 = 0 to 11 iter_args(%arg2 = %c0) -> (index) {
      %2 = arith.addi %arg2, %0 : index
      affine.yield %2 : index
    }
    return
  }
}


// -----
#map = affine_map<(d0) -> (64, d0 * -64 + 1020)>
module {
  func.func @use_of_iter_args_not_invariant(%arg0: memref<10xindex>) {
    %c0 = arith.constant 0 : index
    %0 = affine.for %arg1 = 0 to 11 iter_args(%arg2 = %c0) -> (index) {
      %1 = arith.addi %arg2, %c0 : index
      affine.yield %1 : index
    }
    return
  }
  func.func @affine_parallel(%arg0: memref<4090x2040xf32>, %arg1: index) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg2) = (0) to (32) {
      affine.for %arg3 = 0 to 16 {
        affine.parallel (%arg4, %arg5) = (0, 0) to (min(128, 122), min(64, %arg2 * -64 + 2040)) {
          affine.store %cst, %arg0[%arg4 + 3968, %arg5 + %arg2 * 64] : memref<4090x2040xf32>
          affine.for %arg6 = 0 to min #map(%arg3) {
          }
        }
      }
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scf.parallel (%arg2) = (%c0) to (%c32) step (%c1) {
      affine.for %arg3 = 0 to 16 {
        affine.parallel (%arg4, %arg5) = (0, 0) to (min(128, 122), min(64, %arg1 * -64 + 2040)) {
          affine.store %cst, %arg0[%arg4 + 3968, %arg5] : memref<4090x2040xf32>
          affine.for %arg6 = 0 to min #map(%arg3) {
          }
        }
      }
      scf.yield
    }
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 16 {
        affine.parallel (%arg4, %arg5) = (0, 0) to (min(128, 122), min(64, %arg2 * -64 + 2040)) {
          scf.for %arg6 = %c0 to %arg1 step %c1 {
            affine.store %cst, %arg0[%arg4 + 3968, %arg5 + %arg2 * 64] : memref<4090x2040xf32>
          }
        }
      }
    }
    return
  }
}


// -----
#map = affine_map<(d0) -> (d0 * 163840)>
module {
  func.func @affine_invariant_use_after_dma(%arg0: memref<10485760xi32>, %arg1: memref<1xi32>, %arg2: memref<10485760xi32>) {
    %c320 = arith.constant 320 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<0xi32, 2>
    %alloc_0 = memref.alloc() : memref<1xi32, 2>
    affine.for %arg3 = 0 to 64 {
      %0 = affine.apply #map(%arg3)
      %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<0xi32, 2>
      %alloc_2 = memref.alloc() : memref<320xi32, 2>
      affine.dma_start %arg0[%0], %alloc_2[%c0], %alloc_1[%c0], %c320 : memref<10485760xi32>, memref<320xi32, 2>, memref<0xi32, 2>
      affine.dma_start %arg1[%c0], %alloc_0[%c0], %alloc[%c0], %c1 : memref<1xi32>, memref<1xi32, 2>, memref<0xi32, 2>
      affine.dma_wait %alloc_1[%c0], %c320 : memref<0xi32, 2>
      affine.dma_wait %alloc[%c0], %c1 : memref<0xi32, 2>
      %1 = affine.apply #map(%arg3)
      %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<0xi32, 2>
      %alloc_4 = memref.alloc() : memref<320xi32, 2>
      affine.for %arg4 = 0 to 320 {
        %2 = affine.load %alloc_2[%arg4] : memref<320xi32, 2>
        %3 = affine.load %alloc_0[0] : memref<1xi32, 2>
        %4 = arith.addi %2, %3 : i32
        %5 = arith.addi %4, %2 : i32
        affine.store %5, %alloc_4[%arg4] : memref<320xi32, 2>
      }
      affine.dma_start %alloc_4[%c0], %arg2[%1], %alloc_3[%c0], %c320 : memref<320xi32, 2>, memref<10485760xi32>, memref<0xi32, 2>
      affine.dma_wait %alloc_3[%c0], %c320 : memref<0xi32, 2>
    }
    return
  }
}


// -----
module {
  func.func @affine_prefetch_invariant() {
    %alloc = memref.alloc() : memref<10x10xf32>
    affine.for %arg0 = 0 to 10 {
      affine.prefetch %alloc[%arg0, %arg0], write, locality<0>, data : memref<10x10xf32>
      affine.for %arg1 = 0 to 10 {
        %0 = affine.load %alloc[%arg0, %arg1] : memref<10x10xf32>
      }
    }
    return
  }
}

