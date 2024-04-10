
#define TryFIRFilter 1 

static void lowerOpToLoopsFIR(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
      
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

      //inside the forOp body --> create the operations & then close the body

              // #set2 = affine_set<(d0, d1)[]: (d0 - 5 >= 0, d1- 5 >= 0 ) >
          // affine.for %arg0 = 0 to 10 {
          //     %N = len(output)
          //   %4 =  affine.for %arg1 = 0 to 10 {
          //         affine.if #set2(%arg0 , %arg1 )[%N] {
          //             %1 = const 5
          //             %2 = const 3
          //             %3 = arith.mulf %1 , %2
          //             affine.yield %3 
          //         }
          //     }
          //   affine.store %4, alloc[%arg0]                
          // }

#if TryFIRFilter

    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    Value sum0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), 
                                                rewriter.getF64FloatAttr(0));
    //get filter len
    // auto tensorTypeFilter = llvm::cast<RankedTensorType>((*op->getOperand(1))); //operand_type_end
    // auto tensorTypeFilter = llvm::cast<RankedTensorType>((*op->operand_type_begin()));
    auto operandIt = op->operand_type_begin();
    auto tensorTypeInput = llvm::cast<RankedTensorType>(*operandIt);
    int64_t ubForInput = tensorTypeInput.getShape()[0];
    //get second operand
    operandIt = operandIt + 1;

    // auto tensorTypeFilter = llvm::cast<RankedTensorType>((*op->operand_type_begin())); //operandIt
    auto tensorTypeFilter = llvm::cast<RankedTensorType>(*operandIt);
    int64_t ubForFilter = tensorTypeFilter.getShape()[0];

    llvm::errs() << "ubForFilter= " << ubForFilter << "\n";
    //create a constant for sum
    Value constant0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ubForFilter, step , ValueRange{constant0});
    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();

    auto getIterArg =  forOp2.getBody()->getArgument(1);       //forOp1.getIterOperands();
    
    // AffineExpr dimExpr = rewriter.getAffineDimExpr(0);
    AffineExpr dimExpr2 = rewriter.getAffineDimExpr(0) - rewriter.getAffineDimExpr(1);
    //n-k <= inputLen -1 or, k-n >= 1 - inputLen ie, k - n + inputLen - 1 >= 0
    AffineExpr ExprForUpperBoundCheck = rewriter.getAffineConstantExpr(ubForInput) + rewriter.getAffineDimExpr(1)
                     - rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(1)  ;
    IntegerSet set2 = IntegerSet::get(2, 0, {dimExpr2,ExprForUpperBoundCheck}, {false, false});
    
    //use typeRange too:
    Type floatType = rewriter.getF64Type();
    auto ifOp = rewriter.create<affine::AffineIfOp>( loc, TypeRange{floatType}, set2 , ValueRange{iv,iv2} , true /*else*/ );
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    
    AffineMap addMap = AffineMap::get(2, 0, dimExpr2);
    // auto inputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap , ValueRange{iv,iv2});

    FIRFilterOpAdaptor firOpAdaptor(operands);
    Value loadInput = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getLhs(), addMap , ValueRange{iv,iv2});

    rewriter.create<AffineYieldOp>(loc, ValueRange{loadInput});
    //else block
    rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    Value const0ForElse = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    rewriter.create<AffineYieldOp>(loc, ValueRange{const0ForElse});
    rewriter.setInsertionPointAfter(ifOp);

    //load filter and then mult and then sum
    Value loadFilter = rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getRhs() ,  iv2);
    // Value constant25 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(25));
    Value filterMulInput = rewriter.create<arith::MulFOp>(loc, ifOp.getResult(0) , loadFilter);
    Value sumNext = rewriter.create<arith::AddFOp>(loc, filterMulInput, getIterArg);
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    // rewriter.setInsertionPointToEnd(forOp2->getBlock());
    rewriter.setInsertionPointAfter(forOp2);
    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0) , alloc, iv);
    rewriter.setInsertionPointAfter(forOp1);

    ifOp->dump();
    

    //FIRFilter code
    //iterate for output
        //start with sum=0
        //iterate for filter len
            //check for input_indx must be within bounds
            //load filter and input[indx]
            //multiply them
            //add this to sum
    //update output with sum

    




    
    // rewriter.create<affine::AffineYieldOp>(loc, constant25);
    llvm::errs() << "LINE = " << __LINE__ << "\n";
    //Back to parentOp -- ifOp stops here
    // rewriter.setInsertionPointAfter(ifOp);
    
    llvm::errs() << "LINE = " << __LINE__ << "  xx\n";



#endif

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}
