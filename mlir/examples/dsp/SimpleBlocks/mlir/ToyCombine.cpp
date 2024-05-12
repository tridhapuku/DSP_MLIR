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
#include "toy/DebugConfig.h"
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
    // result1 = upsampling(input1, rate1)
    // result2 = downsampling(result1, rate2)
// if rate1 == rate2 then result2 = input1
  // result2 will be now delay(input1, gain1 + gain2)
  // replaceOp 
struct SimplifyUpsamplingDownsampling : public mlir::OpRewritePattern<DownsamplingOp> {
  /// We register this pattern to match every dsp.downsampling in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyUpsamplingDownsampling(mlir::MLIRContext *context)
      : OpRewritePattern<DownsamplingOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(DownsamplingOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current downsampling.
    mlir::Value downsamplingOperand1_Rate = op.getOperand(1);
    mlir::Value downsamplingOperand0_input = op.getOperand(0);
    dsp::UpsamplingOp prev_UpSamplingOp = downsamplingOperand0_input.getDefiningOp<UpsamplingOp>();

    // Input defined by another downsampling? If not, no match.
    if (!prev_UpSamplingOp)
      return failure();

    //Get operands for UpSamplingOp
    mlir::Value UpsamplingOperand1_Rate = prev_UpSamplingOp.getOperand(1);
    mlir::Value UpsamplingOperand0_input = prev_UpSamplingOp.getOperand(0);

    //get constant value from the downsamplingOp -- operand1
    dsp::ConstantOp constant_Op1_downsamplingOp = downsamplingOperand1_Rate.getDefiningOp<dsp::ConstantOp>();
  	// llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    DenseElementsAttr DenseValueFrmDownsampling = constant_Op1_downsamplingOp.getValue();
  	// llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    auto elements = DenseValueFrmDownsampling.getValues<FloatAttr>();
    float FirstValue = elements[0].getValueAsDouble();
    int64_t DownsamplingRate = (int64_t) FirstValue;

    //Get constant value from upsampling: -- operand1
    dsp::ConstantOp constant_Op1_upSamplingOp = UpsamplingOperand1_Rate.getDefiningOp<dsp::ConstantOp>();
  	// llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    DenseElementsAttr DenseValueFrmUpsampling = constant_Op1_upSamplingOp.getValue();
  	// llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    elements = DenseValueFrmUpsampling.getValues<FloatAttr>();
    FirstValue = elements[0].getValueAsDouble();
    int64_t UpsamplingRate = (int64_t) FirstValue;

    llvm::errs() << "DownsamplingRate = " << DownsamplingRate << " UpsamplingRate" << UpsamplingRate << "\n";
    if(DownsamplingRate == UpsamplingRate)
    {
	    // Otherwise, we have a redundant downsampling. Use the rewriter.
	    // rewriter.replaceOp(op, {downsamplingInputOp.getOperand()}); //downsamplingOperand0_input
      llvm::errs() << "Going for Downsampling pass\n";
      rewriter.replaceOp(op, UpsamplingOperand0_input);
	    return success();

    }
    else if(UpsamplingRate > DownsamplingRate)
    {
      //check if UpSamplingRate is a multiple of DownsamplingRate
      //if yes, final result should be UpSampling with SamplingRate as division 
      if(UpsamplingRate % DownsamplingRate != 0)
      {
        return failure();
      }

      //
      if(DownsamplingRate == 0)
      {
        llvm::errs() << "DownSamplingRate= 0 Not allowed" << "\n"; 
        return failure();
      }
      double finalUpSamplingRate = (double) UpsamplingRate / DownsamplingRate;

      auto constOp_finalSamplingRate = rewriter.create<ConstantOp>(op.getLoc(), finalUpSamplingRate);

      auto finalUpSamplingOp = rewriter.create<UpsamplingOp>(op.getLoc(),
                          UpsamplingOperand0_input , constOp_finalSamplingRate);

      llvm::errs() << "Going for Downsampling pass\n";
      rewriter.replaceOp(op, finalUpSamplingOp);

    }
    return failure();

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

void DownsamplingOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context){
  // results.add<SimplifyUpsamplingDownsampling>(context);
}

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
