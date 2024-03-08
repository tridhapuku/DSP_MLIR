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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Dialect.h"
#include <numeric>
using namespace mlir;
using namespace dsp;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every dsp.transpose in the IR.
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


//Pseudo-Code
//Find back to back gain operation
    // result1 = gain(input1, gain1)
    // result2 = gain(result1, gain2)
// if result1 is coming from another delay operation
  // result2 will be now delay(input1, gain1 + gain2)
  // replaceOp 
struct SimplifyBack2BackGain: public mlir::OpRewritePattern<GainOp>{
  //
  SimplifyBack2BackGain(mlir::MLIRContext *context) 
    : OpRewritePattern<GainOp>(context, 1) {}

    mlir::LogicalResult matchAndRewrite(GainOp op, 
                        mlir::PatternRewriter &rewriter) const override {
     
     //
     mlir::Value gainOp_operand0 = op.getOperand(0);
     
     //check if this is coming from another gain operation
     GainOp prev_gainOp = gainOp_operand0.getDefiningOp<GainOp>();

     if(!prev_gainOp)
        return failure();

     mlir::Value gainOp_operand1 = op.getOperand(1);
     mlir::Value prev_gainOp_operand0 = prev_gainOp.getOperand(0);
     mlir::Value prev_gainOp_operand1 = prev_gainOp.getOperand(1);

     //create add op 
     auto addOp = rewriter.create<MulOp>(op.getLoc(), prev_gainOp_operand1, gainOp_operand1);
     auto newGainOp = rewriter.create<GainOp>(op.getLoc(),
                          prev_gainOp_operand0 , addOp.getResult());
    
    //Repalce the use of original gain operation with this newGainOp
    rewriter.replaceOp(op, newGainOp.getResult());
    return mlir::success();

    }
};

//Pseudo-Code
//Find back to back delay operation
    // result1 = delay(input1, unit1)
    // result2 = delay(result1, unit2)
// if result1 is coming from another delay operation
  // result2 will be now delay(input1, unit1 + unit2)
  // replaceOp 
struct SimplifyBack2BackDelay: public mlir::OpRewritePattern<DelayOp>{
  //
  SimplifyBack2BackDelay(mlir::MLIRContext *context) 
    : OpRewritePattern<DelayOp>(context, 1) {}

    mlir::LogicalResult matchAndRewrite(DelayOp op, 
                        mlir::PatternRewriter &rewriter) const override {
     
     //
     mlir::Value delayOp_operand0 = op.getOperand(0);
     
     //check if this is coming from another delay operation
     DelayOp prev_delayOp = delayOp_operand0.getDefiningOp<DelayOp>();

     if(!prev_delayOp)
        return failure();

     mlir::Value delayOp_operand1 = op.getOperand(1);
     mlir::Value prev_delayOp_operand0 = prev_delayOp.getOperand(0);
     mlir::Value prev_delayOp_operand1 = prev_delayOp.getOperand(1);

     //create add op 
     auto addOp = rewriter.create<AddOp>(op.getLoc(), prev_delayOp_operand1, delayOp_operand1);
     auto newDelayOp = rewriter.create<DelayOp>(op.getLoc(),
                          prev_delayOp_operand0 , addOp.getResult());
    
    //Repalce the use of original delay operation with this newDelayOp
    rewriter.replaceOp(op, newDelayOp.getResult());
    return mlir::success();

    }
};


/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

void DelayOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
    // llvm::errs() << "Enabling Delay Optimization\n";
    results.add<SimplifyBack2BackDelay>(context);  
}

void GainOp::getCanonicalizationPatterns(RewritePatternSet &results, 
                                              MLIRContext *context) {
  results.add<SimplifyBack2BackGain>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
