module {
  func.func @main() {
    %alloc = memref.alloc() : memref<8xf64>
    %alloc_0 = memref.alloc() : memref<8xf64>
    %alloc_1 = memref.alloc() : memref<8xf64>
    %alloc_2 = memref.alloc() : memref<8xf64>
    %alloc_3 = memref.alloc() : memref<8xf64>
    %alloc_4 = memref.alloc() : memref<8xf64>
    %alloc_5 = memref.alloc() : memref<8xf64>
    %alloc_6 = memref.alloc() : memref<8xf64>
    %alloc_7 = memref.alloc() : memref<8xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %cst = arith.constant 0.000000e+00 : f64
    affine.store %cst, %alloc_7[%c0] : memref<8xf64>
    %cst_8 = arith.constant 1.000000e+01 : f64
    affine.store %cst_8, %alloc_7[%c1] : memref<8xf64>
    %cst_9 = arith.constant 3.400000e+02 : f64
    affine.store %cst_9, %alloc_7[%c2] : memref<8xf64>
    %cst_10 = arith.constant 3.000000e+01 : f64
    affine.store %cst_10, %alloc_7[%c3] : memref<8xf64>
    %cst_11 = arith.constant 4.000000e+01 : f64
    affine.store %cst_11, %alloc_7[%c4] : memref<8xf64>
    %cst_12 = arith.constant 1.100000e+02 : f64
    affine.store %cst_12, %alloc_7[%c5] : memref<8xf64>
    %cst_13 = arith.constant 6.000000e+01 : f64
    affine.store %cst_13, %alloc_7[%c6] : memref<8xf64>
    %cst_14 = arith.constant 2.500000e+02 : f64
    affine.store %cst_14, %alloc_7[%c7] : memref<8xf64>
    %c0_15 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1_16 = arith.constant 1 : index
    scf.for %arg0 = %c0_15 to %c8 step %c1_16 {
      %0 = memref.load %alloc_7[%arg0] : memref<8xf64>
      %cst_28 = arith.constant 0.000000e+00 : f64
      memref.store %0, %alloc_6[%arg0] : memref<8xf64>
      memref.store %cst_28, %alloc_5[%arg0] : memref<8xf64>
    }
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    scf.for %arg0 = %c0_15 to %c8 step %c1_16 {
      %0 = arith.index_cast %arg0 : index to i64
      %1 = arith.andi %0, %c1_i64 : i64
      %2 = arith.shrui %0, %c1_i64 : i64
      %3 = arith.andi %2, %c1_i64 : i64
      %4 = arith.shrui %0, %c2_i64 : i64
      %5 = arith.andi %4, %c1_i64 : i64
      %6 = arith.shli %1, %c2_i64 : i64
      %7 = arith.shli %3, %c1_i64 : i64
      %8 = arith.ori %6, %7 : i64
      %9 = arith.ori %8, %5 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = memref.load %alloc_6[%arg0] : memref<8xf64>
      %12 = memref.load %alloc_5[%arg0] : memref<8xf64>
      memref.store %11, %alloc_4[%10] : memref<8xf64>
      memref.store %12, %alloc_3[%10] : memref<8xf64>
    }
    %c3_17 = arith.constant 3 : index
    %cst_18 = arith.constant 3.1415926535897931 : f64
    %cst_19 = arith.constant -2.000000e+00 : f64
    scf.for %arg0 = %c0_15 to %c3_17 step %c1_16 {
      %c1_28 = arith.constant 1 : index
      %0 = arith.shli %c1_28, %arg0 : index
      %c1_29 = arith.constant 1 : index
      %1 = arith.shli %0, %c1_29 : index
      scf.for %arg1 = %c0_15 to %c8 step %1 {
        scf.for %arg2 = %c0_15 to %0 step %c1_16 {
          %2 = arith.addi %arg1, %arg2 : index
          %3 = arith.addi %2, %0 : index
          %4 = arith.index_cast %arg2 : index to i64
          %5 = arith.sitofp %4 : i64 to f64
          %6 = arith.index_cast %1 : index to i64
          %7 = arith.sitofp %6 : i64 to f64
          %8 = arith.divf %5, %7 : f64
          %9 = arith.mulf %cst_19, %8 : f64
          %10 = arith.mulf %cst_18, %9 : f64
          %11 = math.cos %10 : f64
          %12 = math.sin %10 : f64
          %13 = memref.load %alloc_4[%3] : memref<8xf64>
          %14 = memref.load %alloc_3[%3] : memref<8xf64>
          %15 = arith.mulf %13, %11 : f64
          %16 = arith.mulf %14, %12 : f64
          %17 = arith.subf %15, %16 : f64
          %18 = arith.mulf %13, %12 : f64
          %19 = arith.mulf %14, %11 : f64
          %20 = arith.addf %18, %19 : f64
          %21 = memref.load %alloc_4[%2] : memref<8xf64>
          %22 = memref.load %alloc_3[%2] : memref<8xf64>
          %23 = arith.addf %21, %17 : f64
          %24 = arith.addf %22, %20 : f64
          %25 = arith.subf %21, %17 : f64
          %26 = arith.subf %22, %20 : f64
          memref.store %23, %alloc_4[%2] : memref<8xf64>
          memref.store %24, %alloc_3[%2] : memref<8xf64>
          memref.store %25, %alloc_4[%3] : memref<8xf64>
          memref.store %26, %alloc_3[%3] : memref<8xf64>
        }
      }
    }
    %c0_20 = arith.constant 0 : index
    %c8_21 = arith.constant 8 : index
    %c1_22 = arith.constant 1 : index
    scf.for %arg0 = %c0_20 to %c8_21 step %c1_22 {
      %0 = memref.load %alloc_7[%arg0] : memref<8xf64>
      %cst_28 = arith.constant 0.000000e+00 : f64
      memref.store %0, %alloc_2[%arg0] : memref<8xf64>
      memref.store %cst_28, %alloc_1[%arg0] : memref<8xf64>
    }
    %c1_i64_23 = arith.constant 1 : i64
    %c2_i64_24 = arith.constant 2 : i64
    scf.for %arg0 = %c0_20 to %c8_21 step %c1_22 {
      %0 = arith.index_cast %arg0 : index to i64
      %1 = arith.andi %0, %c1_i64_23 : i64
      %2 = arith.shrui %0, %c1_i64_23 : i64
      %3 = arith.andi %2, %c1_i64_23 : i64
      %4 = arith.shrui %0, %c2_i64_24 : i64
      %5 = arith.andi %4, %c1_i64_23 : i64
      %6 = arith.shli %1, %c2_i64_24 : i64
      %7 = arith.shli %3, %c1_i64_23 : i64
      %8 = arith.ori %6, %7 : i64
      %9 = arith.ori %8, %5 : i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = memref.load %alloc_2[%arg0] : memref<8xf64>
      %12 = memref.load %alloc_1[%arg0] : memref<8xf64>
      memref.store %11, %alloc_0[%10] : memref<8xf64>
      memref.store %12, %alloc[%10] : memref<8xf64>
    }
    %c3_25 = arith.constant 3 : index
    %cst_26 = arith.constant 3.1415926535897931 : f64
    %cst_27 = arith.constant -2.000000e+00 : f64
    scf.for %arg0 = %c0_20 to %c3_25 step %c1_22 {
      %c1_28 = arith.constant 1 : index
      %0 = arith.shli %c1_28, %arg0 : index
      %c1_29 = arith.constant 1 : index
      %1 = arith.shli %0, %c1_29 : index
      scf.for %arg1 = %c0_20 to %c8_21 step %1 {
        scf.for %arg2 = %c0_20 to %0 step %c1_22 {
          %2 = arith.addi %arg1, %arg2 : index
          %3 = arith.addi %2, %0 : index
          %4 = arith.index_cast %arg2 : index to i64
          %5 = arith.sitofp %4 : i64 to f64
          %6 = arith.index_cast %1 : index to i64
          %7 = arith.sitofp %6 : i64 to f64
          %8 = arith.divf %5, %7 : f64
          %9 = arith.mulf %cst_27, %8 : f64
          %10 = arith.mulf %cst_26, %9 : f64
          %11 = math.cos %10 : f64
          %12 = math.sin %10 : f64
          %13 = memref.load %alloc_0[%3] : memref<8xf64>
          %14 = memref.load %alloc[%3] : memref<8xf64>
          %15 = arith.mulf %13, %11 : f64
          %16 = arith.mulf %14, %12 : f64
          %17 = arith.subf %15, %16 : f64
          %18 = arith.mulf %13, %12 : f64
          %19 = arith.mulf %14, %11 : f64
          %20 = arith.addf %18, %19 : f64
          %21 = memref.load %alloc_0[%2] : memref<8xf64>
          %22 = memref.load %alloc[%2] : memref<8xf64>
          %23 = arith.addf %21, %17 : f64
          %24 = arith.addf %22, %20 : f64
          %25 = arith.subf %21, %17 : f64
          %26 = arith.subf %22, %20 : f64
          memref.store %23, %alloc_0[%2] : memref<8xf64>
          memref.store %24, %alloc[%2] : memref<8xf64>
          memref.store %25, %alloc_0[%3] : memref<8xf64>
          memref.store %26, %alloc[%3] : memref<8xf64>
        }
      }
    }
    dsp.print %alloc_4 : memref<8xf64>
    dsp.print %alloc : memref<8xf64>
    memref.dealloc %alloc_7 : memref<8xf64>
    memref.dealloc %alloc_6 : memref<8xf64>
    memref.dealloc %alloc_5 : memref<8xf64>
    memref.dealloc %alloc_4 : memref<8xf64>
    memref.dealloc %alloc_3 : memref<8xf64>
    memref.dealloc %alloc_2 : memref<8xf64>
    memref.dealloc %alloc_1 : memref<8xf64>
    memref.dealloc %alloc_0 : memref<8xf64>
    memref.dealloc %alloc : memref<8xf64>
    return
  }
}
