; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define void @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 30) to i64))
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 10, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 3, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 3, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %21, %0
  %10 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %11 = icmp slt i64 %10, 10
  br i1 %11, label %12, label %23

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %20, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 3
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = mul i64 %10, 3
  %18 = add i64 %17, %14
  %19 = getelementptr float, ptr %1, i64 %18
  store float 1.000000e+01, ptr %19, align 4
  %20 = add i64 %14, 1
  br label %13

21:                                               ; preds = %13
  %22 = add i64 %10, 1
  br label %9

23:                                               ; preds = %9
  %24 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %24, align 8
  %25 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %24, 1
  call void @printMemrefF32(i64 2, ptr %24)
  br label %26

26:                                               ; preds = %38, %23
  %27 = phi i64 [ %39, %38 ], [ 0, %23 ]
  %28 = icmp slt i64 %27, 10
  br i1 %28, label %29, label %40

29:                                               ; preds = %26
  br label %30

30:                                               ; preds = %33, %29
  %31 = phi i64 [ %37, %33 ], [ 0, %29 ]
  %32 = icmp slt i64 %31, 3
  br i1 %32, label %33, label %38

33:                                               ; preds = %30
  %34 = mul i64 %27, 3
  %35 = add i64 %34, %31
  %36 = getelementptr float, ptr %1, i64 %35
  store float 5.000000e+00, ptr %36, align 4
  %37 = add i64 %31, 1
  br label %30

38:                                               ; preds = %30
  %39 = add i64 %27, 1
  br label %26

40:                                               ; preds = %26
  %41 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %41, align 8
  %42 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %41, 1
  call void @printMemrefF32(i64 2, ptr %41)
  br label %43

43:                                               ; preds = %55, %40
  %44 = phi i64 [ %56, %55 ], [ 0, %40 ]
  %45 = icmp slt i64 %44, 10
  br i1 %45, label %46, label %57

46:                                               ; preds = %43
  br label %47

47:                                               ; preds = %50, %46
  %48 = phi i64 [ %54, %50 ], [ 0, %46 ]
  %49 = icmp slt i64 %48, 3
  br i1 %49, label %50, label %55

50:                                               ; preds = %47
  %51 = mul i64 %44, 3
  %52 = add i64 %51, %48
  %53 = getelementptr float, ptr %1, i64 %52
  store float 2.000000e+00, ptr %53, align 4
  %54 = add i64 %48, 1
  br label %47

55:                                               ; preds = %47
  %56 = add i64 %44, 1
  br label %43

57:                                               ; preds = %43
  %58 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %58, align 8
  %59 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %58, 1
  call void @printMemrefF32(i64 2, ptr %58)
  %60 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64))
  %61 = insertvalue { ptr, ptr, i64 } undef, ptr %60, 0
  %62 = insertvalue { ptr, ptr, i64 } %61, ptr %60, 1
  %63 = insertvalue { ptr, ptr, i64 } %62, i64 0, 2
  store i8 122, ptr %60, align 1
  %64 = alloca { ptr, ptr, i64 }, i64 1, align 8
  store { ptr, ptr, i64 } %63, ptr %64, align 8
  %65 = insertvalue { i64, ptr } { i64 0, ptr undef }, ptr %64, 1
  call void @printMemrefI8(i64 0, ptr %64)
  call void @free(ptr %60)
  call void @free(ptr %1)
  call void @return_var_memref_caller()
  call void @return_two_var_memref_caller()
  call void @dim_op_of_unranked()
  ret void
}

define private void @printMemrefI8(i64 %0, ptr %1) {
  %3 = insertvalue { i64, ptr } undef, i64 %0, 0
  %4 = insertvalue { i64, ptr } %3, ptr %1, 1
  %5 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %4, ptr %5, align 8
  call void @_mlir_ciface_printMemrefI8(ptr %5)
  ret void
}

declare void @_mlir_ciface_printMemrefI8(ptr)

define private void @printMemrefF32(i64 %0, ptr %1) {
  %3 = insertvalue { i64, ptr } undef, i64 %0, 0
  %4 = insertvalue { i64, ptr } %3, ptr %1, 1
  %5 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %4, ptr %5, align 8
  call void @_mlir_ciface_printMemrefF32(ptr %5)
  ret void
}

declare void @_mlir_ciface_printMemrefF32(ptr)

define void @return_two_var_memref_caller() {
  %1 = alloca float, i64 12, align 4
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 4, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 3, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 3, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %21, %0
  %10 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %11 = icmp slt i64 %10, 4
  br i1 %11, label %12, label %23

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %20, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 3
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = mul i64 %10, 3
  %18 = add i64 %17, %14
  %19 = getelementptr float, ptr %1, i64 %18
  store float 1.000000e+00, ptr %19, align 4
  %20 = add i64 %14, 1
  br label %13

21:                                               ; preds = %13
  %22 = add i64 %10, 1
  br label %9

23:                                               ; preds = %9
  %24 = call { { i64, ptr }, { i64, ptr } } @return_two_var_memref(ptr %1, ptr %1, i64 0, i64 4, i64 3, i64 3, i64 1)
  %25 = extractvalue { { i64, ptr }, { i64, ptr } } %24, 0
  %26 = extractvalue { { i64, ptr }, { i64, ptr } } %24, 1
  %27 = extractvalue { i64, ptr } %25, 0
  %28 = mul i64 %27, 2
  %29 = add i64 %28, 1
  %30 = mul i64 %29, 8
  %31 = add i64 16, %30
  %32 = extractvalue { i64, ptr } %26, 0
  %33 = mul i64 %32, 2
  %34 = add i64 %33, 1
  %35 = mul i64 %34, 8
  %36 = add i64 16, %35
  %37 = alloca i8, i64 %31, align 1
  %38 = extractvalue { i64, ptr } %25, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %37, ptr %38, i64 %31, i1 false)
  call void @free(ptr %38)
  %39 = extractvalue { i64, ptr } %25, 0
  %40 = insertvalue { i64, ptr } undef, i64 %39, 0
  %41 = insertvalue { i64, ptr } %40, ptr %37, 1
  %42 = alloca i8, i64 %36, align 1
  %43 = extractvalue { i64, ptr } %26, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %42, ptr %43, i64 %36, i1 false)
  call void @free(ptr %43)
  %44 = extractvalue { i64, ptr } %26, 0
  %45 = insertvalue { i64, ptr } undef, i64 %44, 0
  %46 = insertvalue { i64, ptr } %45, ptr %42, 1
  call void @printMemrefF32(i64 %39, ptr %37)
  call void @printMemrefF32(i64 %44, ptr %42)
  ret void
}

define { { i64, ptr }, { i64, ptr } } @return_two_var_memref(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %1, 1
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %5, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %4, 3, 1
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %6, 4, 1
  %15 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, ptr %15, align 8
  %16 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %15, 1
  %17 = call ptr @malloc(i64 56)
  call void @llvm.memcpy.p0.p0.i64(ptr %17, ptr %15, i64 56, i1 false)
  %18 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %17, 1
  %19 = call ptr @malloc(i64 56)
  call void @llvm.memcpy.p0.p0.i64(ptr %19, ptr %15, i64 56, i1 false)
  %20 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %19, 1
  %21 = insertvalue { { i64, ptr }, { i64, ptr } } undef, { i64, ptr } %18, 0
  %22 = insertvalue { { i64, ptr }, { i64, ptr } } %21, { i64, ptr } %20, 1
  ret { { i64, ptr }, { i64, ptr } } %22
}

define void @return_var_memref_caller() {
  %1 = alloca float, i64 12, align 4
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 4, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 3, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 3, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  br label %9

9:                                                ; preds = %21, %0
  %10 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %11 = icmp slt i64 %10, 4
  br i1 %11, label %12, label %23

12:                                               ; preds = %9
  br label %13

13:                                               ; preds = %16, %12
  %14 = phi i64 [ %20, %16 ], [ 0, %12 ]
  %15 = icmp slt i64 %14, 3
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = mul i64 %10, 3
  %18 = add i64 %17, %14
  %19 = getelementptr float, ptr %1, i64 %18
  store float 1.000000e+00, ptr %19, align 4
  %20 = add i64 %14, 1
  br label %13

21:                                               ; preds = %13
  %22 = add i64 %10, 1
  br label %9

23:                                               ; preds = %9
  %24 = call { i64, ptr } @return_var_memref(ptr %1, ptr %1, i64 0, i64 4, i64 3, i64 3, i64 1)
  %25 = extractvalue { i64, ptr } %24, 0
  %26 = mul i64 %25, 2
  %27 = add i64 %26, 1
  %28 = mul i64 %27, 8
  %29 = add i64 16, %28
  %30 = alloca i8, i64 %29, align 1
  %31 = extractvalue { i64, ptr } %24, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %30, ptr %31, i64 %29, i1 false)
  call void @free(ptr %31)
  %32 = extractvalue { i64, ptr } %24, 0
  %33 = insertvalue { i64, ptr } undef, i64 %32, 0
  %34 = insertvalue { i64, ptr } %33, ptr %30, 1
  call void @printMemrefF32(i64 %32, ptr %30)
  ret void
}

define { i64, ptr } @return_var_memref(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %1, 1
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %5, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %4, 3, 1
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %6, 4, 1
  %15 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, ptr %15, align 8
  %16 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %15, 1
  %17 = call ptr @malloc(i64 56)
  call void @llvm.memcpy.p0.p0.i64(ptr %17, ptr %15, i64 56, i1 false)
  %18 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %17, 1
  ret { i64, ptr } %18
}

declare void @printU64(i64)

declare void @printNewline()

define void @dim_op_of_unranked() {
  %1 = alloca float, i64 12, align 4
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 4, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 3, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 3, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %9, align 8
  %10 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %9, 1
  %11 = getelementptr { ptr, ptr, i64 }, ptr %9, i32 0, i32 2
  %12 = getelementptr i64, ptr %11, i64 1
  %13 = load i64, ptr %12, align 4
  call void @printU64(i64 %13)
  call void @printNewline()
  %14 = getelementptr { ptr, ptr, i64 }, ptr %9, i32 0, i32 2
  %15 = getelementptr i64, ptr %14, i64 2
  %16 = load i64, ptr %15, align 4
  call void @printU64(i64 %16)
  call void @printNewline()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
