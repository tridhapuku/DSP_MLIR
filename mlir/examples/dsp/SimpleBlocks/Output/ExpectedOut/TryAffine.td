

#set affine_set<(d0) : (d0 - 5 <= 0)>
affine.for %arg0 = 0 to 10 {

    affine.if #set (%arg0) {
        affine.store %cst, alloc[%arg0]
    }
    else{
        %1 = affine.load input[%arg0]
        %2 = arith.mulf %1, %1
        affine.store %2,alloc[%arg0]
    }
}

#set affine_set<(d0) : (d0 - 5 <= 0)>
affine.for %arg0 = 0 to 10 {
   %3 = affine.if #set (%arg0) {
        %1 = arith.const 22
        affine.yield %1
    }
    affine.store %3, alloc[%arg0]
} 

I am still not getting -- 
help me in creating the AffineIfOp below according to the given IR --
AffineExpr dimExpr = rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {true});
affine::AffineForOp forOp = rewriter.create<affine::AffineForOp>( loc, lb, ub , step, ValueRange{}, [&](OpBuilder &nestedBuilder, Location loc, Value val, ValueRange valRange) {

    auto ifOp = nestedBuilder.create<affine::AffineIfOp>(loc, set1, ??, false );

}

I am trying to implement the follwoing:
#set affine_set<(d0) : (d0 - 5 <= 0)>
affine.for %arg0 = 0 to 10 {
   %3 = affine.if #set (%arg0) {
        %1 = arith.const 22
        affine.yield %1
    }
    affine.store %3, alloc[%arg0]
} 
//Steps:
    i) Create AffineForOp
    2) SetOperation point to the body of For loop
    3) Create AffineIf 
    4) Start adding operations to the ifThen block of AffineIf
    5) Create a constant val whose value is to be stored 
    6) Return this constant value from the if block using ??
    7) Create affineStore for saving the returned constant value into someOutput memref Location
    8) Close the forloop 


// mlir c++ code

affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, lb, ub, step);
rewriter.setInsertionPointToStart(forOp1.getBody());

auto ifOp = rewriter.create<affine::AffineIfOp>(loc, set1, ValueRange{forOp1.getInductionVar()}, false);

rewriter.setInsertionPointToStart(ifOp.getThenBlock());
Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                     rewriter.getF64FloatAttr(15));
rewriter.create<affine::AffineYieldOp>(loc, ValueRange{constant15});

rewriter.setInsertionPointAfter(ifOp);

auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, ifOp.getResult(0), alloc, forOp1.getInductionVar());
rewriter.setInsertionPointAfter(storeOp);

rewriter.setInsertionPointAfter(forOp1);

but the error is : error: 'affine.yield' op must be the last operation in the parent block


#

// Initial sum set to 0.
    // %sum_0 = arith.constant 0.0 : f32
    // // iter_args binds initial values to the loop's region arguments.
    %sum = affine.for %i = 0 to 10 step 1
        iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = affine.load %buffer[%i] : memref<10xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        affine.yield %sum_next : f32
        }
    return %sum : f32
}

affine.for %arg0 = 0 to 10 {
   %1 = affine.load input[%arg0]
  %4 = affine.for %arg1 = 0 to 10 step 1 
    iter_args(%sum_iter = %1) {
      %2 = affine.load filter[%arg1]
      %3 = arith.add sum_iter , %2
        affine.yield %3 : f64
   }
  affine.store %4, output[%arg0]
}

// Try affine.if  with chatgpt
#set affine_set<(d0) : (d0 - 5 <= 0)>
affine.for %arg0 = 0 to 10{
%3 = affine.if #set (%arg0) {
        %1 = arith.const 25
        affine.yield %1
    }
    else{
        %2 = arith.const 15
        affine.yield %2
    }
    affine.store %3, alloc[%arg0]
} 


    // here, we have to use iter
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0] ;
    int64_t step = 1;

    Value constant0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step , ValueRange{constant0} );

    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    //try replace constant15 ie, with input & filter
    FIRFilterOpAdaptor firOpAdaptor(operands);

    Value inputForFilter = rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs() , iv);

    //Get iter_arg 
    auto getIterArg =  forOp1.getBody()->getArgument(1);       //forOp1.getIterOperands();
    Value sumNext = rewriter.create<arith::AddFOp>(loc, inputForFilter, getIterArg);
    // Value sumNext = rewriter.create<arith::AddFOp>(loc, inputForFilter, constant0);

    //here, at indx 0 , o/p = in[0]
    // at indx 1 , o/p = in[0] + in[1] & so on
    //at indx last o/p[9] = sum of all input elements
    auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, sumNext,  alloc,ValueRange{iv});
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext} );
    // rewriter.create<AffineYieldOp>(loc);
    // auto result = forOp1.getResult(0);
    llvm::errs() << "LINE = " << __LINE__ << "\n";
