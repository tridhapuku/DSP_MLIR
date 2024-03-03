affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.

        //Get the input allocated space for the load
        toy::DelayOpAdaptor delayAdaptor(operands);
        auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, delayAdaptor.getLhs(),
                          ivs);

        Value secondValueMLIR = nestedBuilder.create<arith::ConstantOp>(
        loc, nestedBuilder.getIntegerAttr(nestedBuilder.getIntegerType(64), SecondValueInt));

        Value secondValueIndex = nestedBuilder.create<arith::IndexCastOp>(
        loc, nestedBuilder.getIndexType(), secondValueMLIR);

        Value outputIndex = nestedBuilder.create<arith::AddIOp>(loc, ivs[0], secondValueIndex);

        nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
                                                ArrayRef<Value>({outputIndex}) );
      }

    this is output desired --
    affine.for %arg0 = %0 to 3 {
      %0 = affine.load %alloc_6[%arg0] : memref<3xf64>
      %1 = arith.constant 1 : i64
      %2 = arith.IndexCast %1 : index
      %3 = arith.addi %arg0 , %2 : index
      affine.store %0, %alloc[%3] : memref<3xf64>
    }

    but I am getting the error : error: 'affine.store' op index must be a dimension or symbol identifier
    %2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
         ^
note: see current operation: "affine.store"(%12, %0, %15) {map = affine_map<(d0) -> (d0)>} : (f64, memref<3xf64>, index) -> ()