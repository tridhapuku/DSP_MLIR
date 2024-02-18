module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(2 : index) : i64
    %7 = llvm.mlir.constant(3 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(6 : index) : i64
    %10 = llvm.mlir.null : !llvm.ptr
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr
    %14 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %6, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %7, %19[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %7, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %8, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(3 : index) : i64
    %27 = llvm.mul %23, %26  : i64
    %28 = llvm.add %27, %24  : i64
    %29 = llvm.getelementptr %25[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %5, %29 : f64, !llvm.ptr
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(3 : index) : i64
    %34 = llvm.mul %30, %33  : i64
    %35 = llvm.add %34, %31  : i64
    %36 = llvm.getelementptr %32[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %4, %36 : f64, !llvm.ptr
    %37 = llvm.mlir.constant(0 : index) : i64
    %38 = llvm.mlir.constant(2 : index) : i64
    %39 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(3 : index) : i64
    %41 = llvm.mul %37, %40  : i64
    %42 = llvm.add %41, %38  : i64
    %43 = llvm.getelementptr %39[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %3, %43 : f64, !llvm.ptr
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.mlir.constant(0 : index) : i64
    %46 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.mlir.constant(3 : index) : i64
    %48 = llvm.mul %44, %47  : i64
    %49 = llvm.add %48, %45  : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %2, %50 : f64, !llvm.ptr
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mlir.constant(3 : index) : i64
    %55 = llvm.mul %51, %54  : i64
    %56 = llvm.add %55, %52  : i64
    %57 = llvm.getelementptr %53[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %1, %57 : f64, !llvm.ptr
    %58 = llvm.mlir.constant(1 : index) : i64
    %59 = llvm.mlir.constant(2 : index) : i64
    %60 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.mlir.constant(3 : index) : i64
    %62 = llvm.mul %58, %61  : i64
    %63 = llvm.add %62, %59  : i64
    %64 = llvm.getelementptr %60[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %0, %64 : f64, !llvm.ptr
    %65 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.getelementptr %65[%66, %66] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %68 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
    %69 = llvm.mlir.constant(0 : index) : i64
    %70 = llvm.getelementptr %68[%69, %69] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %71 = llvm.mlir.constant(0 : index) : i64
    %72 = llvm.mlir.constant(2 : index) : i64
    %73 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%71 : i64)
  ^bb1(%74: i64):  // 2 preds: ^bb0, ^bb5
    %75 = llvm.icmp "slt" %74, %72 : i64
    llvm.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.mlir.constant(0 : index) : i64
    %77 = llvm.mlir.constant(3 : index) : i64
    %78 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%76 : i64)
  ^bb3(%79: i64):  // 2 preds: ^bb2, ^bb4
    %80 = llvm.icmp "slt" %79, %77 : i64
    llvm.cond_br %80, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %81 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(3 : index) : i64
    %83 = llvm.mul %74, %82  : i64
    %84 = llvm.add %83, %79  : i64
    %85 = llvm.getelementptr %81[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %86 = llvm.load %85 : !llvm.ptr -> f64
    %87 = llvm.call @printf(%67, %86) : (!llvm.ptr<i8>, f64) -> i32
    %88 = llvm.add %79, %78  : i64
    llvm.br ^bb3(%88 : i64)
  ^bb5:  // pred: ^bb3
    %89 = llvm.call @printf(%70) : (!llvm.ptr<i8>) -> i32
    %90 = llvm.add %74, %73  : i64
    llvm.br ^bb1(%90 : i64)
  ^bb6:  // pred: ^bb1
    %91 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%91) : (!llvm.ptr) -> ()
    llvm.return
  }
}