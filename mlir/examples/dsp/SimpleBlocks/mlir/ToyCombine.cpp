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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/DebugConfig.h"
#include "toy/Dialect.h"
#include <numeric>
using namespace mlir;
using namespace dsp;
using namespace std;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

// Declare the function to get the option value
extern bool getEnableCanonicalOpt();

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

// Pseudo-Code
// Find back to back gain operation
//  result1 = upsampling(input1, rate1)
//  result2 = downsampling(result1, rate2)
// if rate1 == rate2 then result2 = input1
// result2 will be now delay(input1, gain1 + gain2)
// replaceOp
struct SimplifyUpsamplingDownsampling
    : public mlir::OpRewritePattern<DownsamplingOp> {
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
    dsp::UpsamplingOp prev_UpSamplingOp =
        downsamplingOperand0_input.getDefiningOp<UpsamplingOp>();

    // Input defined by another downsampling? If not, no match.
    if (!prev_UpSamplingOp)
      return failure();

    // Get operands for UpSamplingOp
    mlir::Value UpsamplingOperand1_Rate = prev_UpSamplingOp.getOperand(1);
    mlir::Value UpsamplingOperand0_input = prev_UpSamplingOp.getOperand(0);

    // get constant value from the downsamplingOp -- operand1
    dsp::ConstantOp constant_Op1_downsamplingOp =
        downsamplingOperand1_Rate.getDefiningOp<dsp::ConstantOp>();
    // DEBUG_PRINT_NO_ARGS();
    DenseElementsAttr DenseValueFrmDownsampling =
        constant_Op1_downsamplingOp.getValue();
    // DEBUG_PRINT_NO_ARGS();
    auto elements = DenseValueFrmDownsampling.getValues<FloatAttr>();
    float FirstValue = elements[0].getValueAsDouble();
    int64_t DownsamplingRate = (int64_t)FirstValue;

    // Get constant value from upsampling: -- operand1
    dsp::ConstantOp constant_Op1_upSamplingOp =
        UpsamplingOperand1_Rate.getDefiningOp<dsp::ConstantOp>();
    // DEBUG_PRINT_NO_ARGS();
    DenseElementsAttr DenseValueFrmUpsampling =
        constant_Op1_upSamplingOp.getValue();
    // DEBUG_PRINT_NO_ARGS();
    elements = DenseValueFrmUpsampling.getValues<FloatAttr>();
    FirstValue = elements[0].getValueAsDouble();
    int64_t UpsamplingRate = (int64_t)FirstValue;

    llvm::errs() << "DownsamplingRate = " << DownsamplingRate
                 << " UpsamplingRate" << UpsamplingRate << "\n";
    if (DownsamplingRate == UpsamplingRate) {
      // Otherwise, we have a redundant downsampling. Use the rewriter.
      // rewriter.replaceOp(op, {downsamplingInputOp.getOperand()});
      // //downsamplingOperand0_input
      llvm::errs() << "Going for Downsampling pass\n";
      rewriter.replaceOp(op, UpsamplingOperand0_input);
      return success();

    } else if (UpsamplingRate > DownsamplingRate) {
      // check if UpSamplingRate is a multiple of DownsamplingRate
      // if yes, final result should be UpSampling with SamplingRate as division
      if (UpsamplingRate % DownsamplingRate != 0) {
        return failure();
      }

      //
      if (DownsamplingRate == 0) {
        llvm::errs() << "DownSamplingRate= 0 Not allowed" << "\n";
        return failure();
      }
      double finalUpSamplingRate = (double)UpsamplingRate / DownsamplingRate;

      auto constOp_finalSamplingRate =
          rewriter.create<ConstantOp>(op.getLoc(), finalUpSamplingRate);

      auto finalUpSamplingOp = rewriter.create<UpsamplingOp>(
          op.getLoc(), UpsamplingOperand0_input, constOp_finalSamplingRate);

      llvm::errs() << "Going for Downsampling pass\n";
      rewriter.replaceOp(op, finalUpSamplingOp);
    }
    return failure();
  }
};

// Pseudo-Code
// Find back to back gain operation
//  result1 = gain(input1, gain1)
//  result2 = gain(result1, gain2)
// if result1 is coming from another delay operation
// result2 will be now delay(input1, gain1 + gain2)
// replaceOp
struct SimplifyBack2BackGain : public mlir::OpRewritePattern<GainOp> {
  //
  SimplifyBack2BackGain(mlir::MLIRContext *context)
      : OpRewritePattern<GainOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(GainOp op, mlir::PatternRewriter &rewriter) const override {

    //
    mlir::Value gainOp_operand0 = op.getOperand(0);

    // check if this is coming from another gain operation
    GainOp prev_gainOp = gainOp_operand0.getDefiningOp<GainOp>();

    if (!prev_gainOp)
      return failure();

    mlir::Value gainOp_operand1 = op.getOperand(1);
    mlir::Value prev_gainOp_operand0 = prev_gainOp.getOperand(0);
    mlir::Value prev_gainOp_operand1 = prev_gainOp.getOperand(1);

    // create add op
    auto addOp = rewriter.create<MulOp>(op.getLoc(), prev_gainOp_operand1,
                                        gainOp_operand1);
    auto newGainOp = rewriter.create<GainOp>(op.getLoc(), prev_gainOp_operand0,
                                             addOp.getResult());

    // Repalce the use of original gain operation with this newGainOp
    rewriter.replaceOp(op, newGainOp.getResult());
    return mlir::success();
  }
};

// Pseudo-Code
//  Mean of diff is equal to (input[-1] - input[0])/len(input).
//  For example, for array (a, b, c, d, e)
//  diff(array) = (b-a, c-b, d-c, e-d)
//  mean(diff(array)) = ((b-a) + (c-b) + (d-c) + (e-d))/4 = (e-a)/4
//  result1 = diff(input1, diff_length) //NOTE: len(result1) == diff_length-1
//  virtually (tensor size is fixed as len(input)-1). result2 = mean(result1,
//  mean_length)
// if mean_length <= (diff_length-1),
// result2 will be now (input1[mean_length] - input[0])/mean_length
// replaceOp
struct SimplifyDiff2Mean : public mlir::OpRewritePattern<MeanOp> {
  //
  SimplifyDiff2Mean(mlir::MLIRContext *context)
      : OpRewritePattern<MeanOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(MeanOp op, mlir::PatternRewriter &rewriter) const override {

    //
    mlir::Value meanOp_operand0 = op.getOperand(0);

    // check if this is coming from diff operation.
    DiffOp prev_diffOp = meanOp_operand0.getDefiningOp<DiffOp>();

    if (!prev_diffOp)
      return failure();

    mlir::Value meanOp_operand1 = op.getOperand(1);
    mlir::Value prev_diffOp_operand0 = prev_diffOp.getOperand(0);

    auto optimizedOp = rewriter.create<dsp::Diff2MeanOptimizedOp>(
        op.getLoc(), prev_diffOp_operand0, meanOp_operand1);

    // Repalce the use of original diff operation with this operation
    rewriter.replaceOp(op, optimizedOp.getResult());
    return mlir::success();
  }
};

struct SimplifyLMS2FindPeaks : public mlir::OpRewritePattern<FindPeaksOp> {
  //
  SimplifyLMS2FindPeaks(mlir::MLIRContext *context)
      : OpRewritePattern<FindPeaksOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(FindPeaksOp op,
                  mlir::PatternRewriter &rewriter) const override {
    //
    mlir::Value findPeaksOp_operand0 = op.getOperand(0);

    // check if this is coming from diff operation.
    LMSFilterResponseOp prev_lmsFilterResponseOp =
        findPeaksOp_operand0.getDefiningOp<LMSFilterResponseOp>();

    if (!prev_lmsFilterResponseOp)
      return failure();

    mlir::Value findPeaksOp_operand1 = op.getOperand(1);
    mlir::Value findPeaksOp_operand2 = op.getOperand(2);
    mlir::Value prev_lmsFilterResponseOp_operand0 =
        prev_lmsFilterResponseOp.getOperand(0);
    mlir::Value prev_lmsFilterResponseOp_operand1 =
        prev_lmsFilterResponseOp.getOperand(1);
    mlir::Value prev_lmsFilterResponseOp_operand2 =
        prev_lmsFilterResponseOp.getOperand(2);
    mlir::Value prev_lmsFilterResponseOp_operand3 =
        prev_lmsFilterResponseOp.getOperand(3);

    auto optimizedOp = rewriter.create<dsp::LMS2FindPeaksOptimizedOp>(
        op.getLoc(), prev_lmsFilterResponseOp_operand0,
        prev_lmsFilterResponseOp_operand1, prev_lmsFilterResponseOp_operand2,
        prev_lmsFilterResponseOp_operand3, findPeaksOp_operand1,
        findPeaksOp_operand2);

    // Repalce the use of original diff operation with this operation
    rewriter.replaceOp(op, optimizedOp.getResult());
    return mlir::success();
  }
};

struct SimplifyFindPeaks2Diff2Mean : public mlir::OpRewritePattern<MeanOp> {
  //
  SimplifyFindPeaks2Diff2Mean(mlir::MLIRContext *context)
      : OpRewritePattern<MeanOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(MeanOp op, mlir::PatternRewriter &rewriter) const override {

    //
    mlir::Value meanOp_operand0 = op.getOperand(0);

    // check if this is coming from diff operation.
    DiffOp prev_diffOp = meanOp_operand0.getDefiningOp<DiffOp>();

    if (!prev_diffOp)
      return failure();

    mlir::Value prev_diffOp_operand0 = prev_diffOp.getOperand(0);
    FindPeaksOp prev_findPeaksOp =
        prev_diffOp_operand0.getDefiningOp<FindPeaksOp>();

    if (!prev_findPeaksOp)
      return failure();

    mlir::Value prev_findPeaksOp_operand0 = prev_findPeaksOp.getOperand(0);
    mlir::Value prev_findPeaksOp_operand1 = prev_findPeaksOp.getOperand(1);
    mlir::Value prev_findPeaksOp_operand2 = prev_findPeaksOp.getOperand(2);

    auto optimizedOp = rewriter.create<dsp::FindPeaks2Diff2MeanOptimizedOp>(
        op.getLoc(), prev_findPeaksOp_operand0, prev_findPeaksOp_operand1,
        prev_findPeaksOp_operand2);

    // Repalce the use of original diff operation with this operation
    rewriter.replaceOp(op, optimizedOp.getResult());
    return mlir::success();
  }
};

struct SimplifyMedian2Sliding
    : public mlir::OpRewritePattern<SlidingWindowAvgOp> {
  //
  SimplifyMedian2Sliding(mlir::MLIRContext *context)
      : OpRewritePattern<SlidingWindowAvgOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(SlidingWindowAvgOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Value slidingOp_operand0 = op.getOperand();

    // check if this is coming from medianFilter operation.
    MedianFilterOp prev_medianFilterOp =
        slidingOp_operand0.getDefiningOp<MedianFilterOp>();

    if (!prev_medianFilterOp)
      return failure();

    mlir::Value prev_medianFilterOp_operand0 = prev_medianFilterOp.getOperand();

    auto optimizedOp = rewriter.create<dsp::Median2SlidingOptimizedOp>(
        op.getLoc(), prev_medianFilterOp_operand0);

    rewriter.replaceOp(op, optimizedOp.getResult());
    return mlir::success();
  }
};

struct SimplifyBack2BackDelay : public mlir::OpRewritePattern<DelayOp> {
  //
  SimplifyBack2BackDelay(mlir::MLIRContext *context)
      : OpRewritePattern<DelayOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(DelayOp op, mlir::PatternRewriter &rewriter) const override {

    //
    mlir::Value delayOp_operand0 = op.getOperand(0);

    // check if this is coming from another delay operation
    DelayOp prev_delayOp = delayOp_operand0.getDefiningOp<DelayOp>();

    if (!prev_delayOp)
      return failure();

    mlir::Value delayOp_operand1 = op.getOperand(1);
    mlir::Value prev_delayOp_operand0 = prev_delayOp.getOperand(0);
    mlir::Value prev_delayOp_operand1 = prev_delayOp.getOperand(1);

    // create add op
    auto addOp = rewriter.create<AddOp>(op.getLoc(), prev_delayOp_operand1,
                                        delayOp_operand1);
    auto newDelayOp = rewriter.create<DelayOp>(
        op.getLoc(), prev_delayOp_operand0, addOp.getResult());

    // Repalce the use of original delay operation with this newDelayOp
    rewriter.replaceOp(op, newDelayOp.getResult());
    return mlir::success();
  }
};

// Pseudo-code
// if operand of square is coming from real part of fft1d
// replace fft1d with fft1dreal
// still squareOp will remain same
struct SimplifyFFTSquare : public mlir::OpRewritePattern<SquareOp> {
  /// We register this pattern to match every dsp.downsampling in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyFFTSquare(mlir::MLIRContext *context)
      : OpRewritePattern<SquareOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(SquareOp op, mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current downsampling.
    // mlir::Value squareOperand1_Rate = op.getOperand(1);
    mlir::Value squareOperand0_input = op.getInput();
    dsp::FFT1DOp prev_FFT1DOp = squareOperand0_input.getDefiningOp<FFT1DOp>();
    // DEBUG_PRINT_NO_ARGS();
    // Input defined by another FFT1D? If not, no match.
    if (!prev_FFT1DOp)
      return failure();

    // Replace fft1d with fft1dreal
    DEBUG_PRINT_WITH_ARGS(squareOperand0_input);
    DEBUG_PRINT_WITH_ARGS("Going fr some");
    DEBUG_PRINT_NO_ARGS();
    mlir::Value prev_FFT1DOp_Operand = prev_FFT1DOp.getInput();
    auto fft1drealOp1 =
        rewriter.create<FFT1DRealOp>(op.getLoc(), prev_FFT1DOp_Operand);
    // DEBUG_PRINT_NO_ARGS();
    auto SquareOp1 = rewriter.create<SquareOp>(op.getLoc(), fft1drealOp1);

    rewriter.replaceOp(op, SquareOp1);
    return mlir::success();
  }
};

struct SimplifyGainwZero : public mlir::OpRewritePattern<GainOp> {
  SimplifyGainwZero(mlir::MLIRContext *context)
      : OpRewritePattern<GainOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(GainOp op, mlir::PatternRewriter &rewriter) const override {

    //
    mlir::Value gainOp_operand1 = op.getOperand(1);

    // check if the value is zero
    DEBUG_PRINT_NO_ARGS();
    dsp::ConstantOp constant_Op1 =
        gainOp_operand1.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr DenseValueFrmgainOp = constant_Op1.getValue();
    auto elements = DenseValueFrmgainOp.getValues<FloatAttr>();
    float FirstValue = elements[0].getValueAsDouble();
    int64_t GainRate = (int64_t)FirstValue;

    if (!GainRate == 0)
      return failure();

    mlir::Value gainOp_operand0 = op.getOperand(0);
    dsp::ConstantOp constant_Op0 =
        gainOp_operand0.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr InputValueFrmgainOp = constant_Op0.getValue();
    int64_t inputSize = InputValueFrmgainOp.size();

    // Define the type of the tensor (tensor<f64>).
    RankedTensorType tensorType =
        RankedTensorType::get({inputSize}, rewriter.getF64Type());

    // Create a constant operation with the specified value and type.
    DenseElementsAttr zerovalue = DenseElementsAttr::get(tensorType, 0.0);
    Operation *constantOp = rewriter.create<ConstantOp>(op.getLoc(), zerovalue);

    rewriter.replaceOp(op, constantOp);
    return mlir::success();
  }
};

// Pseudo-code
// if operands of MulOp are coming from lowPassFIRFilter & hamming
// then replace the MulOp with the symmetrical operation
struct SimplifyFilterMulHamming : public mlir::OpRewritePattern<MulOp> {
  /// We register this pattern to match every dsp.downsampling in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyFilterMulHamming(mlir::MLIRContext *context)
      : OpRewritePattern<MulOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(MulOp op, mlir::PatternRewriter &rewriter) const override {
    // Get the operands operation from MulFOp
    // check if op0 is Low/HighPassFIRFilterOp & op1 is HammingWindowOp
    // if this true then get the operands of op0 ie, Low/HighPassFIRFilterOp
    // use these operands to form FIRHammingOptimizedOp
    // mlir::Value squareOperand1_Rate = op.getOperand(1);
    mlir::Value mulOperand0_Lhs = op.getLhs();
    mlir::Value mulOperand1_Rhs = op.getRhs();
    dsp::LowPassFIRFilterOp op_LowPassFIRFilterOp =
        mulOperand0_Lhs.getDefiningOp<LowPassFIRFilterOp>();
    dsp::HammingWindowOp op_HammingWindowOp =
        mulOperand1_Rhs.getDefiningOp<HammingWindowOp>();

    DEBUG_PRINT_NO_ARGS();
    // Inputs are LowPassFIRFilterOp && HammingWindowOp => If not, no match.
    if (!op_LowPassFIRFilterOp || !op_HammingWindowOp)
      return failure();

    // Replace fft1d with fft1dreal
    DEBUG_PRINT_WITH_ARGS(mulOperand0_Lhs);
    DEBUG_PRINT_WITH_ARGS("SimplifyFilterMulHamming - ConditionMet");
    DEBUG_PRINT_NO_ARGS();
    mlir::Value LowPassFIRFilterOperand_wc = op_LowPassFIRFilterOp.getWc();
    mlir::Value LowPassFIRFilterOperand_N = op_LowPassFIRFilterOp.getN();

    auto firFilterHammingOptimized =
        rewriter.create<FIRFilterHammingOptimizedOp>(
            op.getLoc(), LowPassFIRFilterOperand_wc, LowPassFIRFilterOperand_N);
    DEBUG_PRINT_NO_ARGS();

    rewriter.replaceOp(op, firFilterHammingOptimized);
    return mlir::success();
  }
};

// Pseudo-code
// if operands of MulOp are coming from highPassFIRFilter & hamming
// then replace the MulOp with the symmetrical operation
struct SimplifyHighPassFIRHamming : public mlir::OpRewritePattern<MulOp> {
  /// We register this pattern to match every dsp.downsampling in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyHighPassFIRHamming(mlir::MLIRContext *context)
      : OpRewritePattern<MulOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(MulOp op, mlir::PatternRewriter &rewriter) const override {
    // Get the operands operation from MulFOp
    // check if op0 is Low/HighPassFIRFilterOp & op1 is HammingWindowOp
    // if this true then get the operands of op0 ie, Low/HighPassFIRFilterOp
    // use these operands to form FIRHammingOptimizedOp
    // mlir::Value squareOperand1_Rate = op.getOperand(1);
    mlir::Value mulOperand0_Lhs = op.getLhs();
    mlir::Value mulOperand1_Rhs = op.getRhs();
    dsp::HighPassFIRFilterOp op_HighPassFIRFilterOp =
        mulOperand0_Lhs.getDefiningOp<HighPassFIRFilterOp>();
    dsp::HammingWindowOp op_HammingWindowOp =
        mulOperand1_Rhs.getDefiningOp<HammingWindowOp>();

    DEBUG_PRINT_NO_ARGS();
    // Inputs are HighPassFIRFilterOp && HammingWindowOp => If not, no match.
    if (!op_HighPassFIRFilterOp || !op_HammingWindowOp)
      return failure();

    // Replace fft1d with fft1dreal
    DEBUG_PRINT_WITH_ARGS(mulOperand0_Lhs);
    DEBUG_PRINT_WITH_ARGS("SimplifyHighPassFIRHamming - ConditionMet");
    DEBUG_PRINT_NO_ARGS();
    mlir::Value HighPassFIRFilterOperand_wc = op_HighPassFIRFilterOp.getWc();
    mlir::Value HighPassFIRFilterOperand_N = op_HighPassFIRFilterOp.getN();

    auto highPassFIRHammingOptimized =
        rewriter.create<HighPassFIRHammingOptimizedOp>(
            op.getLoc(), HighPassFIRFilterOperand_wc,
            HighPassFIRFilterOperand_N);
    DEBUG_PRINT_NO_ARGS();

    rewriter.replaceOp(op, highPassFIRHammingOptimized);
    return mlir::success();
  }
};

// Pseudo-Code
// Find FIRFilterResponse & FIRFilterHammingOptimized &  operation
//  result1 = dsp.FIRFilterHammingOptimized(input1, rate1) //filter and hamming
//  result2 = dsp.FIRFilterResponse(result1, rate2) //FilterResponse
// For above pattern , replace dsp.FIRFilterResponse with
// FIRFilterResSymmOptimized result1 = dsp.FIRFilterHammingOptimized(input1,
// rate1) result2 = dsp.FIRFilterResSymmOptimized(result1, rate2)
struct SimplifyFIRFilterRespnseWithSymmFilter
    : public mlir::OpRewritePattern<FIRFilterResponseOp> {
  /// We register this pattern to match every dsp.downsampling in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyFIRFilterRespnseWithSymmFilter(mlir::MLIRContext *context)
      : OpRewritePattern<FIRFilterResponseOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(FIRFilterResponseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current downsampling.
    // if 1 of the operands is FIRFilterHammingOptimized then go for rewrite
    // ie, if
    mlir::Value Operand1_forFIRFilterResp = op.getOperand(1);
    mlir::Value Operand0_forFIRFilterResp = op.getOperand(0);
    dsp::FIRFilterHammingOptimizedOp prev_FIRFilterSymmOp =
        Operand1_forFIRFilterResp.getDefiningOp<FIRFilterHammingOptimizedOp>();

    // Input defined by another downsampling? If not, no match.
    if (!prev_FIRFilterSymmOp) {
      return failure();
    }

    // create FIRFilterHammingOptimizedOp with current operands
    DEBUG_PRINT_WITH_ARGS("Going for FIRFilterresponse Opt when the operand1 "
                          "is a symmetric filter");

    auto firFilterResSymmOptimizedOp =
        rewriter.create<FIRFilterResSymmOptimizedOp>(
            op.getLoc(), Operand0_forFIRFilterResp, Operand1_forFIRFilterResp);

    DEBUG_PRINT_NO_ARGS();
    rewriter.replaceOp(op, firFilterResSymmOptimizedOp);

    return mlir::success();
  }
};

// label: pass 1st
// Pseudo code:
//  if the FFT1DRealOp & FFT1DImgOp has same input then replace them with single
//  %4 = "dsp.fft1dreal"(%3) : (tensor<10xf64>) -> tensor<10xf64>
//  %5 = "dsp.fft1dimg"(%3) : (tensor<10xf64>) -> tensor<10xf64>
//  replace with %4, %5 = "dsp.fft1d"(%3) : (tensor<10xf64>) -> (tensor<10xf64 ,
//  tensor<10xf64)>
//
//  Define the canonicalization pattern.
struct SimplifyFFTRealAndImg : public OpRewritePattern<FFT1DRealOp> {
  SimplifyFFTRealAndImg(MLIRContext *context)
      : OpRewritePattern<FFT1DRealOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(FFT1DRealOp realOp,
                                PatternRewriter &rewriter) const override {
    // Check if there is a corresponding FFT1DImgOp with the same input.
    Operation *nextOp = realOp->getNextNode();
    if (!nextOp || !isa<FFT1DImgOp>(nextOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto imgOp = cast<FFT1DImgOp>(nextOp);
    if (realOp.getInput() != imgOp.getInput())
      return failure();

    // Replace the two operations with the combined FFT1D operation.
    DEBUG_PRINT_NO_ARGS();
    auto combinedOp =
        rewriter.create<FFT1DOp>(realOp.getLoc(), realOp.getInput());
    rewriter.replaceOp(realOp, combinedOp.getResult(0));
    rewriter.replaceOp(imgOp, combinedOp.getResult(1));

    return success();
  }
};

// Pseudo-Code
// Find FIRFilterResponse & reverseInput
//  %1 = "dsp.reverseInput"(%0) : (tensor<4xf64>) -> tensor<*xf64>
//  %2 = "dsp.FIRFilterResponse"(%0, %1) : (tensor<4xf64>, tensor<*xf64>) ->
//  tensor<*xf64>
// For above pattern , replace dsp.FIRFilterResponse with
// FIRFilterYSymmOptimized %1 = "dsp.reverseInput"(%0) result2 =
// dsp.FIRFilterYSymmOptimized(result1, rate2)
struct SimplifyFilterRespX_ReverseXYSymmFilter
    : public mlir::OpRewritePattern<FIRFilterResponseOp> {
  /// We register this pattern to match every dsp.downsampling in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyFilterRespX_ReverseXYSymmFilter(mlir::MLIRContext *context)
      : OpRewritePattern<FIRFilterResponseOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(FIRFilterResponseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current downsampling.
    // if 1 of the operands is FIRFilterHammingOptimized then go for rewrite
    // ie, if
    mlir::Value Operand1_forFIRFilterResp = op.getOperand(1);
    mlir::Value Operand0_forFIRFilterResp = op.getOperand(0);
    dsp::ReverseInputOp prev_ReverseOp =
        Operand1_forFIRFilterResp.getDefiningOp<ReverseInputOp>();

    // Operand1 defined by another ReverseOp? If not, no match.
    if (!prev_ReverseOp) {
      return failure();
    }

    // create FIRFilterYSymmOptimizedOp with current operands
    DEBUG_PRINT_WITH_ARGS("Going for FIRFilterResponse Opt when the operand1 "
                          "is a ReverseInputOp");

    auto firFilterResYSymmOptimizedOp =
        rewriter.create<FIRFilterYSymmOptimizedOp>(
            op.getLoc(), Operand0_forFIRFilterResp, Operand1_forFIRFilterResp);

    DEBUG_PRINT_NO_ARGS();
    rewriter.replaceOp(op, firFilterResYSymmOptimizedOp);

    return mlir::success();
  }
};

// Pseudo code:
//  if the  input of FFT1DRealOp = FIRFilterYSymmOptimizedOp then replace it
//  with FFT1DRealSymmOp Define the canonicalization pattern.
struct SimplifyFFTRealAtInputRealSymm : public OpRewritePattern<FFT1DRealOp> {
  SimplifyFFTRealAtInputRealSymm(MLIRContext *context)
      : OpRewritePattern<FFT1DRealOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(FFT1DRealOp Op,
                                PatternRewriter &rewriter) const override {
    // Check if there is a corresponding FFT1DImgOp with the same input.
    mlir::Value fftOperand_input = Op.getInput();
    dsp::FIRFilterYSymmOptimizedOp op_FIRFilterYSymmOptimizedOp =
        fftOperand_input.getDefiningOp<FIRFilterYSymmOptimizedOp>();

    if (!op_FIRFilterYSymmOptimizedOp)
      return failure();

    DEBUG_PRINT_NO_ARGS();

    // Replace the two operations with the combined FFT1D operation.
    auto fft1dRealSymmOp =
        rewriter.create<FFT1DRealSymmOp>(Op.getLoc(), Op.getInput());
    DEBUG_PRINT_NO_ARGS();
    // rewriter.replaceOp(Op, fft1dRealSymmOp.getResult());
    rewriter.replaceOp(Op, fft1dRealSymmOp);
    DEBUG_PRINT_NO_ARGS();
    return success();
  }
};

// Pseudo code:
//  if the  input of FFT1DImgOp = FIRFilterYSymmOptimizedOp then replace it with
//  FFT1DImgConjSymmOp Define the canonicalization pattern.
struct SimplifyFFTImgAtInputRealSymm : public OpRewritePattern<FFT1DImgOp> {
  SimplifyFFTImgAtInputRealSymm(MLIRContext *context)
      : OpRewritePattern<FFT1DImgOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(FFT1DImgOp Op,
                                PatternRewriter &rewriter) const override {
    // Check if there is a corresponding FFT1DImgOp with the same input.
    mlir::Value fftOperand_input = Op.getInput();
    dsp::FIRFilterYSymmOptimizedOp op_FIRFilterYSymmOptimizedOp =
        fftOperand_input.getDefiningOp<FIRFilterYSymmOptimizedOp>();

    if (!op_FIRFilterYSymmOptimizedOp)
      return failure();

    DEBUG_PRINT_NO_ARGS();

    // Replace the two operations with the combined FFT1D operation.

    auto fft1dImgConjSymmOp =
        rewriter.create<FFT1DImgConjSymmOp>(Op.getLoc(), Op.getInput());
    DEBUG_PRINT_NO_ARGS();
    // rewriter.replaceOp(Op, fft1dImgConjSymmOp.getResult());
    rewriter.replaceOp(Op, fft1dImgConjSymmOp);
    DEBUG_PRINT_NO_ARGS();
    return success();
  }
};

// Pseudo-Code
// Find lmsFIlter with gain operation
//  result1 = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize);
//  result2 = gain(result1, G1)
// result2 will be now lmsFilterResponse(noisy_sig, clean_sig, mu*g1,
// filterSize); replaceOp
struct SimplifyLMSFilterResponsewithGain
    : public mlir::OpRewritePattern<GainOp> {
  SimplifyLMSFilterResponsewithGain(mlir::MLIRContext *context)
      : OpRewritePattern<GainOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(GainOp op, mlir::PatternRewriter &rewriter) const override {

    mlir::Value gainOp_operand0 = op.getOperand(0);

    LMSFilterResponseOp prev_LMSFilterResponseOp =
        gainOp_operand0.getDefiningOp<LMSFilterResponseOp>();

    if (!prev_LMSFilterResponseOp)
      return failure();

    mlir::Value gainOp_operand1 = op.getOperand(1);
    mlir::Value prev_LMSFilterResponseOp_0 =
        prev_LMSFilterResponseOp.getOperand(0);
    mlir::Value prev_LMSFilterResponseOp_1 =
        prev_LMSFilterResponseOp.getOperand(1);
    mlir::Value prev_LMSFilterResponseOp_mu =
        prev_LMSFilterResponseOp.getOperand(2);
    mlir::Value prev_LMSFilterResponseOp_3 =
        prev_LMSFilterResponseOp.getOperand(3);

    // create mul op
    auto mulOp = rewriter.create<MulOp>(
        op.getLoc(), prev_LMSFilterResponseOp_mu, gainOp_operand1);
    auto newLMSFilterResponseOp = rewriter.create<LMSFilterResponseOp>(
        op.getLoc(), prev_LMSFilterResponseOp_0, prev_LMSFilterResponseOp_1,
        mulOp.getResult(), prev_LMSFilterResponseOp_3);

    // Repalce the use of original gain operation with this newGainOp
    rewriter.replaceOp(op, newLMSFilterResponseOp.getResult());
    return mlir::success();
  }
};

struct SimplifySpaceModDemodulate
    : public mlir::OpRewritePattern<SpaceDemodulateOp> {
  SimplifySpaceModDemodulate(mlir::MLIRContext *context)
      : OpRewritePattern<SpaceDemodulateOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(SpaceDemodulateOp op,
                  mlir::PatternRewriter &rewriter) const override {

    // a flag checking if the define operation chain of demod op contains mod op
    bool opt = false;
    SpaceModulateOp prev_mod;
    auto iter = op.getOperand();
    while (iter.getDefiningOp()) {
      auto pred = iter.getDefiningOp();
      // llvm::errs() << pred->getName().getStringRef() << "\n";
      if (llvm::dyn_cast<SpaceModulateOp>(*pred)) {
        opt = true;
        prev_mod = llvm::dyn_cast<SpaceModulateOp>(*pred);
        break;
      }
      iter = (*pred).getOperand(0);
    }

    if (!opt)
      return failure();

    auto constVal = prev_mod.getOperand().getDefiningOp();
    rewriter.replaceOp(op, constVal);
    return mlir::success();
  }
};

struct SimplifyNormLMSFilterResponse
    : public mlir::OpRewritePattern<NormalizeOp> {
  SimplifyNormLMSFilterResponse(mlir::MLIRContext *ctx)
      : OpRewritePattern<NormalizeOp>(ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(NormalizeOp op,
                  mlir::PatternRewriter &rewriter) const override {

    Value signal = op.getOperand();
    Operation *filterOp = signal.getDefiningOp<LMSFilterResponseOp>();

    if (!filterOp)
      return failure();

    Value filterOp_operand0 = filterOp->getOperand(0);
    Value filterOp_operand1 = filterOp->getOperand(1);
    Value filterOp_operand2 = filterOp->getOperand(2);
    Value filterOp_operand3 = filterOp->getOperand(3);

    auto normLMSfilterOpt = rewriter.create<NormLMSFilterResponseOptimizeOp>(
        op.getLoc(), filterOp_operand0, filterOp_operand1, filterOp_operand2,
        filterOp_operand3);

    rewriter.replaceOp(op, normLMSfilterOpt);
    if (filterOp->use_empty()) {
      rewriter.eraseOp(filterOp);
    }

    return mlir::success();
  }
};

struct SimplifyDSSDPass : public mlir::OpRewritePattern<DivOp> {
  SimplifyDSSDPass(mlir::MLIRContext *ctx) : OpRewritePattern<DivOp>(ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(DivOp op, mlir::PatternRewriter &rewriter) const override {

#define CHECK(x)                                                               \
  if (!x)                                                                      \
    return failure();
#define REMOVE(x)                                                              \
  if (x->use_empty())                                                          \
    rewriter.eraseOp(x);
#define DEBUG(x)                                                               \
  { llvm::errs() << "check for " << x << "\n"; }
#define PASS llvm::errs() << "pass\n";

    auto loc = op.getLoc();

    // pattern -> CHECK()
    Operation *sumOp = op.getOperand(0).getDefiningOp<SumOp>();
    CHECK(sumOp);

    Operation *addOp = sumOp->getOperand(0).getDefiningOp<AddOp>();
    CHECK(addOp);

    Operation *sqrtOp0 = addOp->getOperand(0).getDefiningOp<SquareOp>();
    CHECK(sqrtOp0);

    Operation *sqrtOp1 = addOp->getOperand(1).getDefiningOp<SquareOp>();
    CHECK(sqrtOp1);

    Operation *fftRealOp = sqrtOp0->getOperand(0).getDefiningOp<FFT1DRealOp>();
    CHECK(fftRealOp);

    // See defining op: suppose to be fftImg, but modified beforhand by <label>
    // pass 1st
    Operation *fftImgOp = sqrtOp1->getOperand(0).getDefiningOp<FFT1DRealOp>();
    CHECK(fftImgOp);

    // check if come from same input
    Value input1 = fftRealOp->getOperand(0);
    Value input2 = fftImgOp->getOperand(0);
    CHECK((input1 == input2));

    auto newSqrt = rewriter.create<SquareOp>(loc, input1);
    auto newResult = rewriter.create<SumOp>(loc, newSqrt);

    rewriter.replaceOp(op, newResult);

    REMOVE(fftImgOp);
    REMOVE(fftRealOp);
    REMOVE(sqrtOp1);
    REMOVE(sqrtOp0);
    REMOVE(addOp);
    REMOVE(sumOp);

    return mlir::success();
  }
};

struct SimplifyFIRFilterHammingThreholdUpOptimized
    : public mlir::OpRewritePattern<ThresholdUpOp> {
  SimplifyFIRFilterHammingThreholdUpOptimized(mlir::MLIRContext *context)
      : OpRewritePattern<ThresholdUpOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ThresholdUpOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value Operand0_threshold = op.getOperand(0);
    mlir::Value Operand1_threshold = op.getOperand(1);
    mlir::Value Operand2_threshold = op.getOperand(2);
    dsp::FIRFilterResSymmOptimizedOp prev_FIRFilterSymmOp =
        Operand0_threshold.getDefiningOp<FIRFilterResSymmOptimizedOp>();

    if (!prev_FIRFilterSymmOp) {
      return failure();
    }
    Value input1 = prev_FIRFilterSymmOp->getOperand(0);
    Value input2 = prev_FIRFilterSymmOp->getOperand(1);
    auto fIRFilterResSymmThresholdUpOptimizedOp =
        rewriter.create<FIRFilterResSymmThresholdUpOptimizedOp>(
            op.getLoc(), input1, input2, Operand1_threshold,
            Operand2_threshold);

    DEBUG_PRINT_NO_ARGS();
    rewriter.replaceOp(op, fIRFilterResSymmThresholdUpOptimizedOp);

    return mlir::success();
  }
};

//  Define the canonicalization pattern.
struct SimplifyFFTAbs : public OpRewritePattern<FFTRealOp> {
  SimplifyFFTAbs(MLIRContext *context)
      : OpRewritePattern<FFTRealOp>(context, 1) {}

  LogicalResult matchAndRewrite(FFTRealOp realOp,
                                PatternRewriter &rewriter) const override {
    // Check if there is a corresponding FFT1DImgOp with the same input.
    Operation *nextofFFTRealOp = realOp->getNextNode();
    if (!nextofFFTRealOp || !isa<FFTImagOp>(nextofFFTRealOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto fftImagOp = cast<FFTImagOp>(nextofFFTRealOp);
    if (realOp.getLhs() != fftImagOp.getLhs())
      return failure();

    Operation *nextofFFTImagOp = fftImagOp->getNextNode();
    if (!nextofFFTImagOp || !isa<SquareOp>(nextofFFTImagOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto square1Op = cast<SquareOp>(nextofFFTImagOp);
    if (realOp.getResult() != square1Op.getInput())
      return failure();

    Operation *nextofSquare1Op = square1Op->getNextNode();
    if (!nextofSquare1Op || !isa<SquareOp>(nextofSquare1Op))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto square2Op = cast<SquareOp>(nextofSquare1Op);
    if (fftImagOp.getResult() != square2Op.getInput())
      return failure();

    Operation *nextofSquare2Op = square2Op->getNextNode();
    if (!nextofSquare2Op || !isa<AddOp>(nextofSquare2Op))
      return failure();
    // (addOp.getLhs() != squareOp.getResult()) || (addOp.getRhs() !=
    // square2Op.getResult())   &&  (addOp.getRhs() != squareOp.getResult()) &&
    // (addOp.getLhs() != square2Op.getResult())
    DEBUG_PRINT_NO_ARGS();
    auto addOp = cast<AddOp>(nextofSquare2Op);
    if ((addOp.getLhs() != square1Op.getResult() ||
         addOp.getRhs() != square2Op.getResult()) &&
        (addOp.getRhs() != square1Op.getResult() ||
         addOp.getLhs() != square2Op.getResult()))
      return failure();

    Operation *nextofAddOp = addOp->getNextNode();
    if (!nextofAddOp || !isa<SqrtOp>(nextofAddOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto sqrtOp = cast<SqrtOp>(nextofAddOp);
    if (sqrtOp.getInput() != addOp.getResult())
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto combinedOp =
        rewriter.create<FFTAbsOp>(realOp.getLoc(), realOp.getLhs());
    rewriter.replaceOp(sqrtOp, combinedOp.getAmplitude());

    rewriter.eraseOp(addOp);
    rewriter.eraseOp(square2Op);
    rewriter.eraseOp(square1Op);
    rewriter.eraseOp(fftImagOp);
    rewriter.eraseOp(realOp);

    return success();
  }
};

struct SimplifyDFTAbs : public OpRewritePattern<FFT1DRealOp> {
  SimplifyDFTAbs(MLIRContext *context)
      : OpRewritePattern<FFT1DRealOp>(context, 1) {}

  LogicalResult matchAndRewrite(FFT1DRealOp realOp,
                                PatternRewriter &rewriter) const override {
    // Check if there is a corresponding FFT1DImgOp with the same input.
    Operation *nextofFFTRealOp = realOp->getNextNode();
    if (!nextofFFTRealOp || !isa<FFT1DImgOp>(nextofFFTRealOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto fftImagOp = cast<FFT1DImgOp>(nextofFFTRealOp);
    if (realOp.getInput() != fftImagOp.getInput())
      return failure();

    Operation *nextofFFTImagOp = fftImagOp->getNextNode();
    if (!nextofFFTImagOp || !isa<SquareOp>(nextofFFTImagOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto square1Op = cast<SquareOp>(nextofFFTImagOp);
    if (realOp.getResult() != square1Op.getInput())
      return failure();

    Operation *nextofSquare1Op = square1Op->getNextNode();
    if (!nextofSquare1Op || !isa<SquareOp>(nextofSquare1Op))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto square2Op = cast<SquareOp>(nextofSquare1Op);
    if (fftImagOp.getResult() != square2Op.getInput())
      return failure();

    Operation *nextofSquare2Op = square2Op->getNextNode();
    if (!nextofSquare2Op || !isa<AddOp>(nextofSquare2Op))
      return failure();
    // (addOp.getLhs() != squareOp.getResult()) || (addOp.getRhs() !=
    // square2Op.getResult())   &&  (addOp.getRhs() != squareOp.getResult()) &&
    // (addOp.getLhs() != square2Op.getResult())
    DEBUG_PRINT_NO_ARGS();
    auto addOp = cast<AddOp>(nextofSquare2Op);
    if ((addOp.getLhs() != square1Op.getResult() ||
         addOp.getRhs() != square2Op.getResult()) &&
        (addOp.getRhs() != square1Op.getResult() ||
         addOp.getLhs() != square2Op.getResult()))
      return failure();

    Operation *nextofAddOp = addOp->getNextNode();
    if (!nextofAddOp || !isa<SqrtOp>(nextofAddOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto sqrtOp = cast<SqrtOp>(nextofAddOp);
    if (sqrtOp.getInput() != addOp.getResult())
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto combinedOp =
        rewriter.create<DFTAbsOp>(realOp.getLoc(), realOp.getInput());
    rewriter.replaceOp(sqrtOp, combinedOp.getAmplitude());

    rewriter.eraseOp(addOp);
    rewriter.eraseOp(square2Op);
    rewriter.eraseOp(square1Op);
    rewriter.eraseOp(fftImagOp);
    rewriter.eraseOp(realOp);

    return success();
  }
};

struct SimplifyDFTAbsThreshold : public mlir::OpRewritePattern<ThresholdUpOp> {
  SimplifyDFTAbsThreshold(mlir::MLIRContext *context)
      : OpRewritePattern<ThresholdUpOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ThresholdUpOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value Operand0_threshold = op.getOperand(0);
    mlir::Value Operand1_threshold = op.getOperand(1);
    mlir::Value Operand2_threshold = op.getOperand(2);
    dsp::DFTAbsOp prev_dftAbsOp = Operand0_threshold.getDefiningOp<DFTAbsOp>();

    if (!prev_dftAbsOp) {
      return failure();
    }
    Value input1 = prev_dftAbsOp->getOperand(0);

    auto combinedOp = rewriter.create<DFTAbsThresholdUpOp>(
        op.getLoc(), input1, Operand1_threshold, Operand2_threshold);

    DEBUG_PRINT_NO_ARGS();
    rewriter.replaceOp(op, combinedOp);

    return mlir::success();
  }
};

//  Define the canonicalization pattern.
struct SimplifyFFTRealAndImagToFFT : public OpRewritePattern<FFTRealOp> {
  SimplifyFFTRealAndImagToFFT(MLIRContext *context)
      : OpRewritePattern<FFTRealOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(FFTRealOp realOp,
                                PatternRewriter &rewriter) const override {
    // Check if there is a corresponding FFT1DImgOp with the same input.
    Operation *nextOp = realOp->getNextNode();
    if (!nextOp || !isa<FFTImagOp>(nextOp))
      return failure();

    DEBUG_PRINT_NO_ARGS();
    auto imgOp = cast<FFTImagOp>(nextOp);
    if (realOp.getLhs() != imgOp.getLhs())
      return failure();

    // Replace the two operations with the combined FFT1D operation.
    DEBUG_PRINT_NO_ARGS();
    auto combinedOp = rewriter.create<FFTOp>(realOp.getLoc(), realOp.getLhs());
    rewriter.replaceOp(realOp, combinedOp.getResult(0));
    rewriter.replaceOp(imgOp, combinedOp.getResult(1));

    return success();
  }
};


struct SimplifyCorrel2Max : public mlir::OpRewritePattern<MaxOp> {
  SimplifyCorrel2Max(mlir::MLIRContext *context)
      : OpRewritePattern<MaxOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(MaxOp op, mlir::PatternRewriter &rewriter) const override {

    mlir::Value maxOp_operand0 = op.getOperand();

    CorrelateOp prev_correlateOp = maxOp_operand0.getDefiningOp<CorrelateOp>();

    if (!prev_correlateOp)
      return failure();

    mlir::Value prev_correlateOp_operand1 = prev_correlateOp.getOperand(0);
	  mlir::Value prev_correlateOp_operand2 = prev_correlateOp.getOperand(1);

    auto optimizedOp = rewriter.create<dsp::Correl2MaxOptimizedOp>(
        op.getLoc(), prev_correlateOp_operand1, prev_correlateOp_operand2);

    // Repalce the use of original diff operation with this operation
    rewriter.replaceOp(op, optimizedOp.getResult());
    return mlir::success();
  }
};

// Pseudo-Code
// Find pattern on DivOp
//  %3 = "dsp.getRangeOfVector"(%0, %1, %2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
//  %4 = "dsp.fft1dreal"(%3) : (tensor<*xf64>) -> tensor<*xf64>
//  %5 = "dsp.fft1dimg"(%3) : (tensor<*xf64>) -> tensor<*xf64>
//  %6 = dsp.square(%4 : tensor<*xf64>) to tensor<*xf64>
//  %7 = dsp.square(%5 : tensor<*xf64>) to tensor<*xf64>
//  %8 = dsp.add %6, %7 : tensor<*xf64>
//  %9 = dsp.sum(%8 : tensor<*xf64>) to tensor<*xf64>
//  %10 = "dsp.len"(%3) : (tensor<*xf64>) -> tensor<*xf64>
//  %11 = dsp.div %9, %10 : tensor<*xf64> 
//  fft_real = fft1dreal(input)
//  sq1 = square(fft_real)
//  sq_abs = AddOp (sq1, square(fft_img)) // this is actually + sign
//  result1 = sum(sq_abs)
//  len1  = len(result1)
//  result2 = DivOp(sum1, len1)
//  
// if result2 is coming from DivOp operation
// output pattern is sq= square(input)
// ans = sum(sq)

struct SimplifyEnergyOfSignal : public mlir::OpRewritePattern<DivOp> {
  //
  SimplifyEnergyOfSignal(mlir::MLIRContext *context)
      : OpRewritePattern<DivOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(DivOp op, mlir::PatternRewriter &rewriter) const override {

    //Get operands of DivOp
    mlir::Value divOp_operand0 = op.getOperand(0);
    mlir::Value divOp_operand1 = op.getOperand(1);

    //check if FirstOperand is coming from sumOp & 2nd operand from LenOp
    SumOp prev_SumOp = divOp_operand0.getDefiningOp<SumOp>();
    LengthOp prev_LenOp = divOp_operand1.getDefiningOp<LengthOp>();
    if (!prev_SumOp || !prev_LenOp)
      return failure();

    //check if sumOp operand is coming from AddOp
    mlir::Value sumOp_operand0 = prev_SumOp.getOperand();
    AddOp prev_AddOp = sumOp_operand0.getDefiningOp<AddOp>();
    if (!prev_AddOp )
      return failure();

    //check if addOp opernad is coming from squareOp
    mlir::Value addOp_operand0 = prev_AddOp.getOperand(0);
    mlir::Value addOp_operand1 = prev_AddOp.getOperand(1);
    SquareOp prev_SqOp0 = addOp_operand0.getDefiningOp<SquareOp>();
    SquareOp prev_SqOp1 = addOp_operand1.getDefiningOp<SquareOp>();
    if (!prev_SqOp0 || !prev_SqOp1)
      return failure();

    //check if squareOp is coming from fft1dreal & other from fft1dImg 
    mlir::Value sqOp_operand0 = prev_SqOp0.getOperand();
    mlir::Value sqOp_operand1 = prev_SqOp1.getOperand();
    FFT1DRealOp prev_fftRealOp = sqOp_operand0.getDefiningOp<FFT1DRealOp>();
    FFT1DImgOp prev_fftImgOp = sqOp_operand1.getDefiningOp<FFT1DImgOp>();

    if (!prev_fftRealOp || !prev_fftImgOp)
      return failure();

    // get the opernad of fftReal 
    mlir::Value input = prev_fftRealOp.getOperand();

    // if result2 is coming from DivOp operation
    // output pattern is sq= square(input)
    // ans = sum(sq)
    auto ansSqOp = rewriter.create<SquareOp>(op.getLoc(), input);
    auto ansSumOp = rewriter.create<SumOp>(op.getLoc(), ansSqOp.getResult());

    // Repalce the use of original gain operation with this newGainOp
    rewriter.replaceOp(op, ansSumOp.getResult());
    return mlir::success();
  }
};

// ===================================
// ===================================
// ===================================
// ===================================
// =====Registration of Patterns =====
// ===================================
// ===================================
// ===================================
// ===================================
// ===================================
/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void FFT1DImgOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyFFTImgAtInputRealSymm>(context);
  }
}

void FFT1DRealOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyDFTAbs, SimplifyFFTRealAndImg,
                SimplifyFFTRealAtInputRealSymm>(context);
  }
}

void FIRFilterResponseOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyFIRFilterRespnseWithSymmFilter,
                SimplifyFilterRespX_ReverseXYSymmFilter>(context);
  }
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyFilterMulHamming, SimplifyHighPassFIRHamming>(context);
  }
}

void SquareOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyFFTSquare>(context);
  }
}

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.

void DownsamplingOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyUpsamplingDownsampling>(context);
  }
}

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyRedundantTranspose>(context);
  }
}

void DelayOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  // llvm::errs() << "Enabling Delay Optimization\n";

  if (getEnableCanonicalOpt()) {
    DEBUG_PRINT_WITH_ARGS("Enabling Delay Optimization\n");
    results.add<SimplifyBack2BackDelay>(context);
  }
}

void GainOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  // results.add<SimplifyBack2BackGain, SimplifyGainwZero>(context);
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyBack2BackGain, SimplifyLMSFilterResponsewithGain>(
        context);
  }
}

void MeanOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    // results.add<SimplifyDiff2Mean>(context);
    results.add<SimplifyFindPeaks2Diff2Mean>(context);
  }
}

void FindPeaksOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyLMS2FindPeaks>(context);
  }
}

void SlidingWindowAvgOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyMedian2Sliding>(context);
  }
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                FoldConstantReshapeOptPattern>(context);
  }
}

void SpaceDemodulateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifySpaceModDemodulate>(context);
  }
}

void NormalizeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *ctx) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyNormLMSFilterResponse>(ctx);
  }
}

void DivOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *ctx) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyDSSDPass
    // SimplifyEnergyOfSignal
    >(ctx);
  }
}

void ThresholdUpOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *ctx) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyFIRFilterHammingThreholdUpOptimized,
                SimplifyDFTAbsThreshold>(ctx);
  }
}

void FFTRealOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyFFTAbs, SimplifyFFTRealAndImagToFFT>(context);
  }
}

void MaxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  if (getEnableCanonicalOpt()) {
    results.add<SimplifyCorrel2Max>(context);
  }
}


