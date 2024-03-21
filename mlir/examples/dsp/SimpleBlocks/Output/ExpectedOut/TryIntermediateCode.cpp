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

  // llvm::errs() << "tensorType.getRank() " << tensorType.getRank() << "\n";
  // cout << "tensorType.getRank() .. " << tensorType.getRank() << "\n";
  // for (auto i : tensorType.getRank())
  // {
  //   llvm::errs() << "tensorType.getRank() = " << i << "\n";
  // }
  // for (auto i : tensorType.getShape())
  // {
  //   llvm::errs() << "tensorType.getShape() = " << i << "\n";
  // }
  // llvm::errs() << "tensorType.getShape() " << tensorType.getShape() << "\n";
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

// Define an affine map: #map2 = affine_map<(d0) -> (d0 + 2)>
      AffineExpr indx = nestedBuilder.getAffineDimExpr(0);
      AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
      AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
      auto outputIndex = nestedBuilder.create<affine::AffineApplyOp>(loc, addMap , ivs);


// Code 
 std::pair<affine::AffineForOp, mlir::Value>
  positiveConstantStep(fir::DoLoopOp op, int64_t step,
                       mlir::PatternRewriter &rewriter) const {
    auto affineFor = rewriter.create<affine::AffineForOp>(
        op.getLoc(), ValueRange(op.getLowerBound()),
        mlir::AffineMap::get(0, 1,
                             mlir::getAffineSymbolExpr(0, op.getContext())),
        ValueRange(op.getUpperBound()),
        mlir::AffineMap::get(0, 1,
                             1 + mlir::getAffineSymbolExpr(0, op.getContext())),
        step);
    return std::make_pair(affineFor, affineFor.getInductionVar());
  }

  std::pair<affine::AffineForOp, mlir::Value>
  genericBounds(fir::DoLoopOp op, mlir::PatternRewriter &rewriter) const {
    auto lowerBound = mlir::getAffineSymbolExpr(0, op.getContext());
    auto upperBound = mlir::getAffineSymbolExpr(1, op.getContext());
    auto step = mlir::getAffineSymbolExpr(2, op.getContext());
    mlir::AffineMap upperBoundMap = mlir::AffineMap::get(
        0, 3, (upperBound - lowerBound + step).floorDiv(step));
    auto genericUpperBound = rewriter.create<affine::AffineApplyOp>(
        op.getLoc(), upperBoundMap,
        ValueRange({op.getLowerBound(), op.getUpperBound(), op.getStep()}));
    auto actualIndexMap = mlir::AffineMap::get(
        1, 2,
        (lowerBound + mlir::getAffineDimExpr(0, op.getContext())) *
            mlir::getAffineSymbolExpr(1, op.getContext()));

    auto affineFor = rewriter.create<affine::AffineForOp>(
        op.getLoc(), ValueRange(),
        AffineMap::getConstantMap(0, op.getContext()),
        genericUpperBound.getResult(),
        mlir::AffineMap::get(0, 1,
                             1 + mlir::getAffineSymbolExpr(0, op.getContext())),
        1);
    rewriter.setInsertionPointToStart(affineFor.getBody());
    auto actualIndex = rewriter.create<affine::AffineApplyOp>(
        op.getLoc(), actualIndexMap,
        ValueRange(
            {affineFor.getInductionVar(), op.getLowerBound(), op.getStep()}));
    return std::make_pair(affineFor, actualIndex.getResult());
  }

    AffineFunctionAnalysis &functionAnalysis;
  };

//code helper 

AffineForOp mlir::affine::replaceForOpWithNewYields(OpBuilder &b,
                                                    AffineForOp loop,
                                                    ValueRange newIterOperands,
                                                    ValueRange newYieldedValues,
                                                    ValueRange newIterArgs,
                                                    bool replaceLoopResults) {
  assert(newIterOperands.size() == newYieldedValues.size() &&
         "newIterOperands must be of the same size as newYieldedValues");
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getIterOperands());
  operands.append(newIterOperands.begin(), newIterOperands.end());
  SmallVector<Value, 4> lbOperands(loop.getLowerBoundOperands());
  SmallVector<Value, 4> ubOperands(loop.getUpperBoundOperands());
  SmallVector<Value, 4> steps(loop.getStep());
  auto lbMap = loop.getLowerBoundMap();
  auto ubMap = loop.getUpperBoundMap();
  AffineForOp newLoop =
      b.create<AffineForOp>(loop.getLoc(), lbOperands, lbMap, ubOperands, ubMap,
                            loop.getStep(), operands);
  // Take the body of the original parent loop.
  newLoop.getLoopBody().takeBody(loop.getLoopBody());
  for (Value val : newIterArgs)
    newLoop.getLoopBody().addArgument(val.getType(), val.getLoc());

  // Update yield operation with new values to be added.
  if (!newYieldedValues.empty()) {
    auto yield = cast<AffineYieldOp>(newLoop.getBody()->getTerminator());
    b.setInsertionPoint(yield);
    auto yieldOperands = llvm::to_vector<4>(yield.getOperands());
    yieldOperands.append(newYieldedValues.begin(), newYieldedValues.end());
    b.create<AffineYieldOp>(yield.getLoc(), yieldOperands);
    yield.erase();
  }
  if (replaceLoopResults) {
    for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                    loop.getNumResults()))) {
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
    }
  }
  return newLoop;
}


class AffineIfConversion : public mlir::OpRewritePattern<fir::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AffineIfConversion(mlir::MLIRContext *context, AffineFunctionAnalysis &afa)
      : OpRewritePattern(context) {}
  mlir::LogicalResult
  matchAndRewrite(fir::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: rewriting if:\n";
               op.dump(););
    auto &ifOps = op.getThenRegion().front().getOperations();
    auto affineCondition = AffineIfCondition(op.getCondition());
    if (!affineCondition.hasIntegerSet()) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "AffineIfConversion: couldn't calculate affine condition\n";);
      return failure();
    }
    auto affineIf = rewriter.create<affine::AffineIfOp>(
        op.getLoc(), affineCondition.getIntegerSet(),
        affineCondition.getAffineArgs(), !op.getElseRegion().empty());
    rewriter.startRootUpdate(affineIf);
    affineIf.getThenBlock()->getOperations().splice(
        std::prev(affineIf.getThenBlock()->end()), ifOps, ifOps.begin(),
        std::prev(ifOps.end()));
    if (!op.getElseRegion().empty()) {
      auto &otherOps = op.getElseRegion().front().getOperations();
      affineIf.getElseBlock()->getOperations().splice(
          std::prev(affineIf.getElseBlock()->end()), otherOps, otherOps.begin(),
          std::prev(otherOps.end()));
    }
    rewriter.finalizeRootUpdate(affineIf);
    rewriteMemoryOps(affineIf.getBody(), rewriter);

    LLVM_DEBUG(llvm::dbgs() << "AffineIfConversion: if converted to:\n";
               affineIf.dump(););
    rewriter.replaceOp(op, affineIf.getOperation()->getResults());
    return success();
  }
};
explain what does the affineIf is created without affineFor -- 
//Create IntegerSet
        d0 = AffineDimExpr.get(0)
        d1 = AffineDimExpr.get(1)
        s0 = AffineSymbolExpr.get(0)
        c42 = AffineConstantExpr.get(42)

        set0 = IntegerSet.get(2, 1, [d0 - d1, s0 - c42, s0 - d0], [True, False, False])
        # CHECK: 2
        print(set0.n_dims)
        # CHECK: 1
        print(set0.n_symbols)
        # CHECK: 3
        print(set0.n_inputs)
        # CHECK: 1
        print(set0.n_equalities)
        # CHECK: 2
        print(set0.n_inequalities)

        # CHECK: 3
        print(len(set0.constraints))

        # CHECK-DAG: d0 - d1 == 0
        # CHECK-DAG: s0 - 42 >= 0
        # CHECK-DAG: -d0 + s0 >= 0
