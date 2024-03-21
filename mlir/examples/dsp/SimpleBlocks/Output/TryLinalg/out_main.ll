; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define { ptr, ptr, i64 } @main(ptr %0, ptr %1, i64 %2, ptr %3, ptr %4, i64 %5) {
  %7 = insertvalue { ptr, ptr, i64 } undef, ptr %3, 0
  %8 = insertvalue { ptr, ptr, i64 } %7, ptr %4, 1
  %9 = insertvalue { ptr, ptr, i64 } %8, i64 %5, 2
  %10 = insertvalue { ptr, ptr, i64 } undef, ptr %0, 0
  %11 = insertvalue { ptr, ptr, i64 } %10, ptr %1, 1
  %12 = insertvalue { ptr, ptr, i64 } %11, i64 %2, 2
  %13 = getelementptr float, ptr %1, i64 %2
  %14 = load float, ptr %13, align 4
  %15 = getelementptr float, ptr %4, i64 %5
  %16 = load float, ptr %15, align 4
  %17 = fadd float %14, %16
  %18 = getelementptr float, ptr %1, i64 %2
  store float %17, ptr %18, align 4
  ret { ptr, ptr, i64 } %12
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
