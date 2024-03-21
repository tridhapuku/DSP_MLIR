Enabling Delay Optimization
Enabling Delay Optimization
Enabling Delay Optimization
delaySecondArg.getDefiningOp()= 0x558751438000
Defining Opp is not constant so no lowering for now
*****SecondValueInt = 2 ***
tensorType->getRank = 1
tensorType->getNumElements = 10
lowerBounds.size() = 1
steps.size() = 1
tensorType.getShape() = 10
254
264
delaySecondArg.getDefiningOp()= 0x55875144fb10
Defining Opp is not constant so no lowering for now
*****SecondValueInt = 4 ***
tensorType->getRank = 1
tensorType->getNumElements = 10
lowerBounds.size() = 1
steps.size() = 1
tensorType.getShape() = 10
254
264
module {
  llvm.func @free(!llvm.ptr)
  llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(1.000000e+02 : f64) : f64
    %4 = llvm.mlir.constant(9.000000e+01 : f64) : f64
    %5 = llvm.mlir.constant(8.000000e+01 : f64) : f64
    %6 = llvm.mlir.constant(7.000000e+01 : f64) : f64
    %7 = llvm.mlir.constant(6.000000e+01 : f64) : f64
    %8 = llvm.mlir.constant(5.000000e+01 : f64) : f64
    %9 = llvm.mlir.constant(4.000000e+01 : f64) : f64
    %10 = llvm.mlir.constant(3.000000e+01 : f64) : f64
    %11 = llvm.mlir.constant(2.000000e+01 : f64) : f64
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %13 = llvm.mlir.constant(10 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.null : !llvm.ptr
    %16 = llvm.getelementptr %15[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %13, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %14, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.mlir.constant(10 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.null : !llvm.ptr
    %29 = llvm.getelementptr %28[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %26, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.insertvalue %27, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.mlir.null : !llvm.ptr
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.call @malloc(%42) : (i64) -> !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<(ptr, ptr, i64)> 
    %46 = llvm.insertvalue %43, %45[1] : !llvm.struct<(ptr, ptr, i64)> 
    %47 = llvm.mlir.constant(0 : index) : i64
    %48 = llvm.insertvalue %47, %46[2] : !llvm.struct<(ptr, ptr, i64)> 
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.mlir.null : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.call @malloc(%52) : (i64) -> !llvm.ptr
    %54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %55 = llvm.insertvalue %53, %54[0] : !llvm.struct<(ptr, ptr, i64)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(ptr, ptr, i64)> 
    %57 = llvm.mlir.constant(0 : index) : i64
    %58 = llvm.insertvalue %57, %56[2] : !llvm.struct<(ptr, ptr, i64)> 
    %59 = llvm.mlir.constant(10 : index) : i64
    %60 = llvm.mlir.constant(1 : index) : i64
    %61 = llvm.mlir.null : !llvm.ptr
    %62 = llvm.getelementptr %61[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.call @malloc(%63) : (i64) -> !llvm.ptr
    %65 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %66 = llvm.insertvalue %64, %65[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.insertvalue %59, %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %71 = llvm.insertvalue %60, %70[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.getelementptr %73[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %12, %74 : f64, !llvm.ptr
    %75 = llvm.mlir.constant(1 : index) : i64
    %76 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.getelementptr %76[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %11, %77 : f64, !llvm.ptr
    %78 = llvm.mlir.constant(2 : index) : i64
    %79 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %80 = llvm.getelementptr %79[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %10, %80 : f64, !llvm.ptr
    %81 = llvm.mlir.constant(3 : index) : i64
    %82 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %83 = llvm.getelementptr %82[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %9, %83 : f64, !llvm.ptr
    %84 = llvm.mlir.constant(4 : index) : i64
    %85 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %86 = llvm.getelementptr %85[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %8, %86 : f64, !llvm.ptr
    %87 = llvm.mlir.constant(5 : index) : i64
    %88 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %89 = llvm.getelementptr %88[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %7, %89 : f64, !llvm.ptr
    %90 = llvm.mlir.constant(6 : index) : i64
    %91 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %92 = llvm.getelementptr %91[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %6, %92 : f64, !llvm.ptr
    %93 = llvm.mlir.constant(7 : index) : i64
    %94 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %5, %95 : f64, !llvm.ptr
    %96 = llvm.mlir.constant(8 : index) : i64
    %97 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %98 = llvm.getelementptr %97[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %4, %98 : f64, !llvm.ptr
    %99 = llvm.mlir.constant(9 : index) : i64
    %100 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %101 = llvm.getelementptr %100[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %3, %101 : f64, !llvm.ptr
    %102 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %2, %102 : f64, !llvm.ptr
    %103 = llvm.extractvalue %48[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %1, %103 : f64, !llvm.ptr
    %104 = llvm.mlir.constant(0 : index) : i64
    %105 = llvm.mlir.constant(2 : index) : i64
    %106 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%104 : i64)
  ^bb1(%107: i64):  // 2 preds: ^bb0, ^bb2
    %108 = llvm.icmp "slt" %107, %105 : i64
    llvm.cond_br %108, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %109 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %110 = llvm.getelementptr %109[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %0, %110 : f64, !llvm.ptr
    %111 = llvm.add %107, %106  : i64
    llvm.br ^bb1(%111 : i64)
  ^bb3:  // pred: ^bb1
    %112 = llvm.mlir.constant(0 : index) : i64
    %113 = llvm.mlir.constant(4 : index) : i64
    %114 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb4(%112 : i64)
  ^bb4(%115: i64):  // 2 preds: ^bb3, ^bb5
    %116 = llvm.icmp "slt" %115, %113 : i64
    llvm.cond_br %116, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %117 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %118 = llvm.getelementptr %117[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    llvm.store %0, %118 : f64, !llvm.ptr
    %119 = llvm.add %115, %114  : i64
    llvm.br ^bb4(%119 : i64)
  ^bb6:  // pred: ^bb4
    %120 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
    %121 = llvm.mlir.constant(0 : index) : i64
    %122 = llvm.getelementptr %120[%121, %121] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %123 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
    %124 = llvm.mlir.constant(0 : index) : i64
    %125 = llvm.getelementptr %123[%124, %124] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %126 = llvm.mlir.constant(0 : index) : i64
    %127 = llvm.mlir.constant(10 : index) : i64
    %128 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%126 : i64)
  ^bb7(%129: i64):  // 2 preds: ^bb6, ^bb8
    %130 = llvm.icmp "slt" %129, %127 : i64
    llvm.cond_br %130, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %131 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %132 = llvm.getelementptr %131[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f64
    %133 = llvm.load %132 : !llvm.ptr -> f64
    %134 = llvm.call @printf(%122, %133) : (!llvm.ptr<i8>, f64) -> i32
    %135 = llvm.add %129, %128  : i64
    llvm.br ^bb7(%135 : i64)
  ^bb9:  // pred: ^bb7
    %136 = llvm.extractvalue %71[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%136) : (!llvm.ptr) -> ()
    %137 = llvm.extractvalue %58[0] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @free(%137) : (!llvm.ptr) -> ()
    %138 = llvm.extractvalue %48[0] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @free(%138) : (!llvm.ptr) -> ()
    %139 = llvm.extractvalue %38[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%139) : (!llvm.ptr) -> ()
    %140 = llvm.extractvalue %25[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%140) : (!llvm.ptr) -> ()
    llvm.return
  }
}
