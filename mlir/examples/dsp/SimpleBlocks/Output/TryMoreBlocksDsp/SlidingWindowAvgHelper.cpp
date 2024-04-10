//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct SlidingWindowAvgOpLowering : public ConversionPattern {
  SlidingWindowAvgOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoopsSlidingWindow(op, operands, rewriter, loc );
    return success();
  }
};


struct FIRFilterOpLowering: public ConversionPattern {
      FIRFilterOpLowering(MLIRContext *ctx)
        : ConversionPattern(dsp::FIRFilterOp::getOperationName(), 1 , ctx) {}

    LogicalResult 
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
              ConversionPatternRewriter &rewriter) const final {
      //dsp.FIRFilterOp has 2 operands -- both of type tensor f64 

      //Get the location of FIRFilterOp
      auto loc = op->getLoc();
      
      //Pseudo-Code
      // y[n] = sum( h[k] * x[n-k]) k = 0 to lenOfh 

      //Range for each element of the output tensor -- i = %arg0
      //  Create a tempValue = 0
        //  Range for each of the elements of filter len -- k = %arg1
        //  check for the condition that %arg0  - %arg1 >= 0 && < inputLen
          //  get elem1 = filter[k] , elem2 = x[i-k]
          // use affine-map expression for calculating i-k
          //  tempValue = tempValue + elem1 * elem2
      // y[i] = tempValue
        
        lowerOpToLoopsSlidingWindow(op, operands, rewriter, 
            [loc, op ] (OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) );

      return success();
    }


};



