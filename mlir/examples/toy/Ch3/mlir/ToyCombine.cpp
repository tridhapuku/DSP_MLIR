//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/Dialect.h"
#include <numeric>

//Added by ABhinav
#include <iostream>
#include <typeinfo>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
//Abhinav end

using namespace mlir;
using namespace toy;
using namespace std;


namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}



//Added by Abhinav 
bool isSSAValueF64TensorWithValueOne(Value value) {
  // Check if the value has type F64Tensor
  if (auto tensorType = value.getType().dyn_cast<RankedTensorType>()) {
    if (tensorType.getElementType().isF64()) {
      // Check if the value is a constant operation
      if (auto constOp = value.getDefiningOp<ConstantOp>()) {
        // Check if the constant value is a floating-point attribute and its value is 1.0
        if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>()) {
          return floatAttr.getValueAsDouble() == 1.0;
        }
      }
    }
  }
  return false;
}

bool isValueConstantOne(Value value) {
  // Check if the value is a constant operation
  if (auto constOp = value.getDefiningOp<ConstantOp>()) {
    // Check if the constant value is an integer and its value is 1
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
      return intAttr.getInt() == 1;
    }
  }
  return false;
}


 bool CheckIfSecondOperandis1(MulOp op, mlir::PatternRewriter &rewriter){
    auto constOp1 = op.getOperand(1).getDefiningOp<mlir::toy::ConstantOp>();
    // llvm::errs() << "Func = " << __func__ << "\n";
    llvm::errs() << "constOp.getValue().isa<mlir::IntegerAttr>(): " << 
          constOp1.getValue().isa<mlir::IntegerAttr>() << "\n";
    llvm::errs() << "constOp = op.getOperand(1).getDefiningOp<mlir::toy::ConstantOp>(): " << \
        constOp1 << "\n";

    //this line gives assertion error:
    //constOp.getValue().cast<mlir::IntegerAttr>().getInt() == 1: toyc-ch3: /home/abhinav/ForMLIR/SourceCode/llvm-project/llvm/include/llvm/Support/Casting.h:566: decltype(auto) llvm::cast(const From&) [with To = mlir::IntegerAttr; From = mlir::Attribute]: Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.

    // llvm::errs() << "constOp.getValue().cast<mlir::IntegerAttr>().getInt() == 1: " << \
    //     constOp1.getValue().cast<mlir::IntegerAttr>().getInt() << "\n"; //thi

    llvm::errs() << "Line : " << __LINE__ << "\n";
    // Check if second operand is a constant 1
    if(auto constOp = op.getOperand(1).getDefiningOp<mlir::toy::ConstantOp>()){
      llvm::errs() << "Line : " << __LINE__ << "\n";
      if(constOp.getValue().isa<mlir::IntegerAttr>() || 
          constOp.getValue().cast<mlir::IntegerAttr>().getInt() == 1){
            //replace multiplication with other operand
            cout << "replacing op with operand0\n"; 
            rewriter.replaceOp(op, op.getOperand(0));
            return true;
          }
    }

    return false;
 }

/// This is an example of a c++ rewrite pattern for the Multiply ie, MulOp. It
/// optimizes the following scenario: x * 1 -> x
struct SimplifyMultiplyBy1 : public mlir::OpRewritePattern<MulOp> {
  //Init constructor with respective Base rewrite pattern
  SimplifyMultiplyBy1(mlir::MLIRContext *context)
      : OpRewritePattern<MulOp>(context, 1) {}

  // llvm::errs() << "Debugging SimplifyMultiplyBy1\n";
  // llvm::errs() << "Error can't load file " << inputFilename << "\n";
  // Match a pattern & rewrite
  mlir::LogicalResult
  matchAndRewrite(MulOp op, mlir::PatternRewriter &rewriter) const override {
    // check if one of the operands is 1, then return the other operand
    //  auto inputs = op.inputs();
    //  auto inputs = op.getOperands();
    mlir::Value operandA = op.getOperand(0);
    mlir::Value operandB = op.getOperand(1);

    // cout << "type(operandA)" << typeid(operandA).name() << "\n";
    llvm::errs() << "Debugging SimplifyMultiplyBy1\n";
    // Check if either operand is constant with a value of 1
    // Attribute oneAttr = rewriter.getIntegerAttr(op.getType(), 1);

    // if( (operandA.getDefiningOp() &&
    // operandA.getDefiningOp()->getAttr("value") == oneAttr) ||
    //     (operandB.getDefiningOp() &&
    //     operandB.getDefiningOp()->getAttr("value") == oneAttr)) {

    //       // Replace the MulOp with the non-1 operand
    //       Value nonOneOperand = (operandA.getDefiningOp() &&
    //       operandA.getDefiningOp()->getAttr("value") == oneAttr) ? operandB :
    //       operandA; rewriter.replaceOp(op, nonOneOperand); return success();
    //     }

    // check if 2nd operand is 1 then replace result with first operand
    //  if(isSSAValueF64TensorWithValueOne(operandB) ){
    //    rewriter.replaceOp(op, operandA);
    //    return success();
    //  }
    // if(CheckIfSecondOperandis1(op, rewriter) == mlir::success()){
    if(CheckIfSecondOperandis1(op, rewriter) ){
      llvm::errs() << "SecondOperand is 1\n";
      return mlir::success();
    }
    // Create two DenseElementsAttrs with the same values
    // Type floatType = FloatType::getF64(&context);
    // ArrayRef<int64_t> shape = {1};
    // DenseElementsAttr attr2 = DenseElementsAttr::get<floatType>(shape, {1.0});

    // // Check if the second operand is a constant with value 1.0
    // if (auto constOp = operandB.getDefiningOp<toy::ConstantOp>()) {
    //   // if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>()) {
    //     if (constOp.getValue() == attr2) {
    //       rewriter.replaceOp(op, operandA);
    //       return success();
    //     }
    //   // }
    //   // else{
    //   //   llvm::errs() << "operandB is not float\n";
    //   // }
    // }
    // else{
    //   llvm::errs() << "operandB is not ConstantOp\n"; 
    // }

    return failure();
  }
};

/// Register our patterns as "canonicalization" patterns on the MulOp so
/// that they can be picked up by the Canonicalization framework.
void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // results.add<SimplifyRedundantTranspose>(context);
  results.add<SimplifyMultiplyBy1>(context);
}