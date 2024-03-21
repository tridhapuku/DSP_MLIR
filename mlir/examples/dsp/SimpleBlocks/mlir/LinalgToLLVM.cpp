//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'dsp.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'dsp.print' --> Loop (SCF) --

// This file implements full lowering of Linalg operations to LLVM MLIR dialect.
// 'dsp.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'dsp.print' --> Loop (SCF) --
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

//From LinalgToStd 
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyLinalgToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace{
  struct ToyLinalgToStdLoweringPass : 
      public PassWrapper<ToyLinalgToStdLoweringPass , OperationPass<ModuleOp>> {
         MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyLinalgToStdLoweringPass)
      
      void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<affine::AffineDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();  //LLVM::LLVMDialect,
      }
     void runOnOperation() final; 
 };
}

void ToyLinalgToStdLoweringPass::runOnOperation(){
  //define conversion target
  ConversionTarget target(getContext());

  //add the legal dialects 
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  //also dsp dialect will be illegal now except the PrintOp
  target.addIllegalDialect<dsp::DspDialect>();
  target.addDynamicallyLegalOp<dsp::PrintOp>([](dsp::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  //Now, we have to provide the set of patterns
  RewritePatternSet patterns(&getContext());
  mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);
  
  //apply PartialConversion
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, and
/// Linalg and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::dsp::createLinalgToStdPass() {
  return std::make_unique<ToyLinalgToStdLoweringPass>();
}