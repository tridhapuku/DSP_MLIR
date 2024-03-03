struct DelayOpLowering : public ConversionPattern {
  DelayOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::DelayOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Get the location of DelayOp
    auto loc = op->getLoc();

    // Get the second_arg attribute from toy::DelayOp
    int64_t secondArg = op->getAttrOfType<IntegerAttr>("second_arg").getInt();

    // Create arith.const operation with value 0
    auto zeroValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

    // Call lowerOpToLoops with a custom function to handle DelayOp
    lowerOpToLoops(op, operands, rewriter,
                   [loc, secondArg, zeroValue](OpBuilder &builder, ValueRange memRefOperands,
                                               ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the DelayOp.
                     typename toy::DelayOp::Adaptor delayAdaptor(memRefOperands);

                     // Create an affine expression for the index adjusted by secondArg
                     Value adjustedIndex = builder.create<arith::AddIOp>(loc, loopIvs[0], builder.create<arith::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getIntegerAttr(rewriter.getIntegerType(64), secondArg)));

                     // Create the affine load operation using the adjusted index
                     auto loadedValue = builder.create<affine::AffineLoadOp>(loc, delayAdaptor.getOperand(), adjustedIndex);

                     // Return the loaded value
                     return loadedValue;
                   });

    return success();
  }
};



struct DelayOpLowering : public ConversionPattern {
  DelayOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::DelayOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Get the location of DelayOp
    auto loc = op->getLoc();

    // Create arith.const operation with value 0 & type=f64
    auto zeroValue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                          rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));

    // Get the second operand of the DelayOp (f64) & convert it to int
    auto delaySecondArg = rewriter.create<arith::TruncateIOp>(
        loc, rewriter.getIntegerType(64), operands[1]);

    // Add check for delay_2ndArg shouldn't exceed lengthOfOperand0
    auto lengthOfOperand0 =
        rewriter.create<toy::LengthOp>(loc, rewriter.getIntegerType(64),
                                       operands[0]);

    rewriter.create<AssertOp>(
        loc, lengthOfOperand0, "Delay second operand should not exceed the length of the first operand");

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(operands[0].getType());
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Create a nest of affine loops
    affine::buildAffineLoopNest(
        rewriter, loc, /*lowerBounds=*/ArrayRef<int64_t>({}),
        /*upperBounds=*/ArrayRef<Value>({delaySecondArg}),
        /*steps=*/ArrayRef<int64_t>({1}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Inside the first loop (0 to delay_2ndArg)
          Value valueToStore = nestedBuilder.create<affine::AffineLoadOp>(
              loc, zeroValue, ArrayRef<Value>({ivs[0]}));
          nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                      ArrayRef<Value>({ivs[0]}));
        });

    // Create another loop from delay_2ndArg to lengthOfOperand0 of DelayOp
    affine::buildAffineLoopNest(
        rewriter, loc, /*lowerBounds=*/ArrayRef<int64_t>({0}),
        /*upperBounds=*/ArrayRef<Value>({lengthOfOperand0}),
        /*steps=*/ArrayRef<int64_t>({1}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Inside the second loop (delay_2ndArg to lengthOfOperand0)
          Value adjustedIndex = nestedBuilder.create<arith::AddIOp>(
              loc, ivs[0], delaySecondArg);
          Value valueToStore = nestedBuilder.create<affine::AffineLoadOp>(
              loc, operands[0], ArrayRef<Value>({adjustedIndex}));
          nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                      ArrayRef<Value>({ivs[0]}));
        });

    // Replace this operation with the generated alloc
    rewriter.replaceOp(op, alloc);

    return success();
  }
};
