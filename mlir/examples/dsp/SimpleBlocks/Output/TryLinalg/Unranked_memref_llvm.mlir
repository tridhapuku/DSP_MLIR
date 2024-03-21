module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.mlir.constant(3 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(10 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(122 : i8) : i8
    %5 = llvm.mlir.constant(1.000000e+01 : f32) : f32
    %6 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(10 : index) : i64
    %9 = llvm.mlir.constant(3 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(30 : index) : i64
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.getelementptr %12[30] : (!llvm.ptr) -> !llvm.ptr, f32
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %15, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %15, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.insertvalue %19, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %8, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %9, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %10, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%25: i64):  // 2 preds: ^bb0, ^bb5
    %26 = llvm.icmp "slt" %25, %2 : i64
    llvm.cond_br %26, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%27: i64):  // 2 preds: ^bb2, ^bb4
    %28 = llvm.icmp "slt" %27, %0 : i64
    llvm.cond_br %28, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %29 = llvm.mlir.constant(3 : index) : i64
    %30 = llvm.mul %25, %29  : i64
    %31 = llvm.add %30, %27  : i64
    %32 = llvm.getelementptr %15[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %5, %32 : f32, !llvm.ptr
    %33 = llvm.add %27, %1  : i64
    llvm.br ^bb3(%33 : i64)
  ^bb5:  // pred: ^bb3
    %34 = llvm.add %25, %1  : i64
    llvm.br ^bb1(%34 : i64)
  ^bb6:  // pred: ^bb1
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.alloca %35 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %24, %36 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %37 = llvm.mlir.constant(2 : index) : i64
    %38 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %39 = llvm.insertvalue %37, %38[0] : !llvm.struct<(i64, ptr)> 
    %40 = llvm.insertvalue %36, %39[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%37, %36) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb7(%3 : i64)
  ^bb7(%41: i64):  // 2 preds: ^bb6, ^bb11
    %42 = llvm.icmp "slt" %41, %2 : i64
    llvm.cond_br %42, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%3 : i64)
  ^bb9(%43: i64):  // 2 preds: ^bb8, ^bb10
    %44 = llvm.icmp "slt" %43, %0 : i64
    llvm.cond_br %44, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %45 = llvm.mlir.constant(3 : index) : i64
    %46 = llvm.mul %41, %45  : i64
    %47 = llvm.add %46, %43  : i64
    %48 = llvm.getelementptr %15[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %6, %48 : f32, !llvm.ptr
    %49 = llvm.add %43, %1  : i64
    llvm.br ^bb9(%49 : i64)
  ^bb11:  // pred: ^bb9
    %50 = llvm.add %41, %1  : i64
    llvm.br ^bb7(%50 : i64)
  ^bb12:  // pred: ^bb7
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.alloca %51 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %24, %52 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %53 = llvm.mlir.constant(2 : index) : i64
    %54 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %55 = llvm.insertvalue %53, %54[0] : !llvm.struct<(i64, ptr)> 
    %56 = llvm.insertvalue %52, %55[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%53, %52) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb13(%3 : i64)
  ^bb13(%57: i64):  // 2 preds: ^bb12, ^bb17
    %58 = llvm.icmp "slt" %57, %2 : i64
    llvm.cond_br %58, ^bb14, ^bb18
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%3 : i64)
  ^bb15(%59: i64):  // 2 preds: ^bb14, ^bb16
    %60 = llvm.icmp "slt" %59, %0 : i64
    llvm.cond_br %60, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %61 = llvm.mlir.constant(3 : index) : i64
    %62 = llvm.mul %57, %61  : i64
    %63 = llvm.add %62, %59  : i64
    %64 = llvm.getelementptr %15[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %7, %64 : f32, !llvm.ptr
    %65 = llvm.add %59, %1  : i64
    llvm.br ^bb15(%65 : i64)
  ^bb17:  // pred: ^bb15
    %66 = llvm.add %57, %1  : i64
    llvm.br ^bb13(%66 : i64)
  ^bb18:  // pred: ^bb13
    %67 = llvm.mlir.constant(1 : index) : i64
    %68 = llvm.alloca %67 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %24, %68 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %69 = llvm.mlir.constant(2 : index) : i64
    %70 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(i64, ptr)> 
    %72 = llvm.insertvalue %68, %71[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%69, %68) : (i64, !llvm.ptr) -> ()
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.mlir.zero : !llvm.ptr
    %75 = llvm.getelementptr %74[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.call @malloc(%76) : (i64) -> !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %79 = llvm.insertvalue %77, %78[0] : !llvm.struct<(ptr, ptr, i64)> 
    %80 = llvm.insertvalue %77, %79[1] : !llvm.struct<(ptr, ptr, i64)> 
    %81 = llvm.mlir.constant(0 : index) : i64
    %82 = llvm.insertvalue %81, %80[2] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %4, %77 : i8, !llvm.ptr
    %83 = llvm.mlir.constant(1 : index) : i64
    %84 = llvm.alloca %83 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %82, %84 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    %85 = llvm.mlir.constant(0 : index) : i64
    %86 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %87 = llvm.insertvalue %85, %86[0] : !llvm.struct<(i64, ptr)> 
    %88 = llvm.insertvalue %84, %87[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefI8(%85, %84) : (i64, !llvm.ptr) -> ()
    llvm.call @free(%77) : (!llvm.ptr) -> ()
    llvm.call @free(%15) : (!llvm.ptr) -> ()
    llvm.call @return_var_memref_caller() : () -> ()
    llvm.call @return_two_var_memref_caller() : () -> ()
    llvm.call @dim_op_of_unranked() : () -> ()
    llvm.return
  }
  llvm.func private @printMemrefI8(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %2, %4 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @_mlir_ciface_printMemrefI8(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_printMemrefI8(!llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @printMemrefF32(%arg0: i64, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %2, %4 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @_mlir_ciface_printMemrefF32(%4) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_printMemrefF32(!llvm.ptr) attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @return_two_var_memref_caller() {
    %0 = llvm.mlir.constant(3 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(12 : index) : i64
    %9 = llvm.alloca %8 x f32 : (i64) -> !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %5, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %6, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %6, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %7, %17[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb5
    %20 = llvm.icmp "slt" %19, %2 : i64
    llvm.cond_br %20, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%21: i64):  // 2 preds: ^bb2, ^bb4
    %22 = llvm.icmp "slt" %21, %0 : i64
    llvm.cond_br %22, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %23 = llvm.mlir.constant(3 : index) : i64
    %24 = llvm.mul %19, %23  : i64
    %25 = llvm.add %24, %21  : i64
    %26 = llvm.getelementptr %9[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %4, %26 : f32, !llvm.ptr
    %27 = llvm.add %21, %1  : i64
    llvm.br ^bb3(%27 : i64)
  ^bb5:  // pred: ^bb3
    %28 = llvm.add %19, %1  : i64
    llvm.br ^bb1(%28 : i64)
  ^bb6:  // pred: ^bb1
    %29 = llvm.call @return_two_var_memref(%9, %9, %13, %5, %6, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)>
    %30 = llvm.extractvalue %29[0] : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)> 
    %31 = llvm.extractvalue %29[1] : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)> 
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.mlir.constant(2 : index) : i64
    %34 = llvm.mlir.constant(8 : index) : i64
    %35 = llvm.mlir.constant(8 : index) : i64
    %36 = llvm.mul %33, %35  : i64
    %37 = llvm.extractvalue %30[0] : !llvm.struct<(i64, ptr)> 
    %38 = llvm.mul %37, %33  : i64
    %39 = llvm.add %38, %32  : i64
    %40 = llvm.mul %39, %34  : i64
    %41 = llvm.add %36, %40  : i64
    %42 = llvm.mlir.constant(8 : index) : i64
    %43 = llvm.mul %33, %42  : i64
    %44 = llvm.extractvalue %31[0] : !llvm.struct<(i64, ptr)> 
    %45 = llvm.mul %44, %33  : i64
    %46 = llvm.add %45, %32  : i64
    %47 = llvm.mul %46, %34  : i64
    %48 = llvm.add %43, %47  : i64
    %49 = llvm.alloca %41 x i8 : (i64) -> !llvm.ptr
    %50 = llvm.extractvalue %30[1] : !llvm.struct<(i64, ptr)> 
    "llvm.intr.memcpy"(%49, %50, %41) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @free(%50) : (!llvm.ptr) -> ()
    %51 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %52 = llvm.extractvalue %30[0] : !llvm.struct<(i64, ptr)> 
    %53 = llvm.insertvalue %52, %51[0] : !llvm.struct<(i64, ptr)> 
    %54 = llvm.insertvalue %49, %53[1] : !llvm.struct<(i64, ptr)> 
    %55 = llvm.alloca %48 x i8 : (i64) -> !llvm.ptr
    %56 = llvm.extractvalue %31[1] : !llvm.struct<(i64, ptr)> 
    "llvm.intr.memcpy"(%55, %56, %48) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @free(%56) : (!llvm.ptr) -> ()
    %57 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %58 = llvm.extractvalue %31[0] : !llvm.struct<(i64, ptr)> 
    %59 = llvm.insertvalue %58, %57[0] : !llvm.struct<(i64, ptr)> 
    %60 = llvm.insertvalue %55, %59[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%52, %49) : (i64, !llvm.ptr) -> ()
    llvm.call @printMemrefF32(%58, %55) : (i64, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @return_two_var_memref(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %9 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %10 = llvm.mlir.constant(2 : index) : i64
    %11 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(i64, ptr)> 
    %13 = llvm.insertvalue %9, %12[1] : !llvm.struct<(i64, ptr)> 
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.constant(8 : index) : i64
    %18 = llvm.mul %15, %17  : i64
    %19 = llvm.mul %10, %15  : i64
    %20 = llvm.add %19, %14  : i64
    %21 = llvm.mul %20, %16  : i64
    %22 = llvm.add %18, %21  : i64
    %23 = llvm.mlir.constant(8 : index) : i64
    %24 = llvm.mul %15, %23  : i64
    %25 = llvm.mul %10, %15  : i64
    %26 = llvm.add %25, %14  : i64
    %27 = llvm.mul %26, %16  : i64
    %28 = llvm.add %24, %27  : i64
    %29 = llvm.call @malloc(%22) : (i64) -> !llvm.ptr
    "llvm.intr.memcpy"(%29, %9, %22) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %30 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %31 = llvm.insertvalue %10, %30[0] : !llvm.struct<(i64, ptr)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(i64, ptr)> 
    %33 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    "llvm.intr.memcpy"(%33, %9, %28) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %34 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %35 = llvm.insertvalue %10, %34[0] : !llvm.struct<(i64, ptr)> 
    %36 = llvm.insertvalue %33, %35[1] : !llvm.struct<(i64, ptr)> 
    %37 = llvm.mlir.undef : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)>
    %38 = llvm.insertvalue %32, %37[0] : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)> 
    llvm.return %39 : !llvm.struct<(struct<(i64, ptr)>, struct<(i64, ptr)>)>
  }
  llvm.func @return_var_memref_caller() {
    %0 = llvm.mlir.constant(3 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(12 : index) : i64
    %9 = llvm.alloca %8 x f32 : (i64) -> !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %5, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %6, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %6, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %7, %17[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%19: i64):  // 2 preds: ^bb0, ^bb5
    %20 = llvm.icmp "slt" %19, %2 : i64
    llvm.cond_br %20, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%21: i64):  // 2 preds: ^bb2, ^bb4
    %22 = llvm.icmp "slt" %21, %0 : i64
    llvm.cond_br %22, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %23 = llvm.mlir.constant(3 : index) : i64
    %24 = llvm.mul %19, %23  : i64
    %25 = llvm.add %24, %21  : i64
    %26 = llvm.getelementptr %9[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %4, %26 : f32, !llvm.ptr
    %27 = llvm.add %21, %1  : i64
    llvm.br ^bb3(%27 : i64)
  ^bb5:  // pred: ^bb3
    %28 = llvm.add %19, %1  : i64
    llvm.br ^bb1(%28 : i64)
  ^bb6:  // pred: ^bb1
    %29 = llvm.call @return_var_memref(%9, %9, %13, %5, %6, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(i64, ptr)>
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(2 : index) : i64
    %32 = llvm.mlir.constant(8 : index) : i64
    %33 = llvm.mlir.constant(8 : index) : i64
    %34 = llvm.mul %31, %33  : i64
    %35 = llvm.extractvalue %29[0] : !llvm.struct<(i64, ptr)> 
    %36 = llvm.mul %35, %31  : i64
    %37 = llvm.add %36, %30  : i64
    %38 = llvm.mul %37, %32  : i64
    %39 = llvm.add %34, %38  : i64
    %40 = llvm.alloca %39 x i8 : (i64) -> !llvm.ptr
    %41 = llvm.extractvalue %29[1] : !llvm.struct<(i64, ptr)> 
    "llvm.intr.memcpy"(%40, %41, %39) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.call @free(%41) : (!llvm.ptr) -> ()
    %42 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %43 = llvm.extractvalue %29[0] : !llvm.struct<(i64, ptr)> 
    %44 = llvm.insertvalue %43, %42[0] : !llvm.struct<(i64, ptr)> 
    %45 = llvm.insertvalue %40, %44[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%43, %40) : (i64, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @return_var_memref(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(i64, ptr)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %9 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %10 = llvm.mlir.constant(2 : index) : i64
    %11 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(i64, ptr)> 
    %13 = llvm.insertvalue %9, %12[1] : !llvm.struct<(i64, ptr)> 
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.constant(8 : index) : i64
    %18 = llvm.mul %15, %17  : i64
    %19 = llvm.mul %10, %15  : i64
    %20 = llvm.add %19, %14  : i64
    %21 = llvm.mul %20, %16  : i64
    %22 = llvm.add %18, %21  : i64
    %23 = llvm.call @malloc(%22) : (i64) -> !llvm.ptr
    "llvm.intr.memcpy"(%23, %9, %22) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %24 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %25 = llvm.insertvalue %10, %24[0] : !llvm.struct<(i64, ptr)> 
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(i64, ptr)> 
    llvm.return %26 : !llvm.struct<(i64, ptr)>
  }
  llvm.func @printU64(i64) attributes {sym_visibility = "private"}
  llvm.func @printNewline() attributes {sym_visibility = "private"}
  llvm.func @dim_op_of_unranked() {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = llvm.mlir.constant(3 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(12 : index) : i64
    %6 = llvm.alloca %5 x f32 : (i64) -> !llvm.ptr
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.insertvalue %10, %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %2, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %3, %12[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %3, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %4, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.alloca %16 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %15, %17 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %18 = llvm.mlir.constant(2 : index) : i64
    %19 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(i64, ptr)> 
    %21 = llvm.insertvalue %17, %20[1] : !llvm.struct<(i64, ptr)> 
    %22 = llvm.getelementptr %17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64)>
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.add %23, %1  : i64
    %25 = llvm.getelementptr %22[%24] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %26 = llvm.load %25 : !llvm.ptr -> i64
    llvm.call @printU64(%26) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %27 = llvm.getelementptr %17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64)>
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.add %28, %0  : i64
    %30 = llvm.getelementptr %27[%29] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %31 = llvm.load %30 : !llvm.ptr -> i64
    llvm.call @printU64(%31) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return
  }
}

