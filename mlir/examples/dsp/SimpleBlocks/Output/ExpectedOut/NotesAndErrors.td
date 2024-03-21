
Errors:

1) Error: error: 'affine.yield' op must be the last operation in the parent block

Trying to implement the following 
//init code
module {
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+02 : f64
    %cst_2 = arith.constant 9.000000e+01 : f64
    ..
    %alloc = memref.alloc() : memref<10xf64>
    %alloc_11 = memref.alloc() : memref<f64>
    %alloc_12 = memref.alloc() : memref<10xf64>
    affine.store %cst_10, %alloc_12[0] : memref<10xf64>
    affine.store %cst_9, %alloc_12[1] : memref<10xf64>
    affine.store %cst_8, %alloc_12[2] : memref<10xf64>
    ..

#set affine_set<(d0) : (d0 - 5 <= 0)>
affine.for %arg0 = 0 to 10 {
   %3 = affine.if #set (%arg0) {
        %1 = arith.const 15.0
        affine.yield %1
    }
    affine.store %3, alloc[%arg0]
}

dsp.print %alloc : memref<10xf64>
    memref.dealloc %alloc_12 : memref<10xf64>
    memref.dealloc %alloc_11 : memref<f64>
    memref.dealloc %alloc : memref<10xf64>
    return

//Not sure of the code --
//Steps:
    i) Create AffineForOp
    2) SetOperation point to the body of For loop
    3) Create AffineIf 
    4) SetOperation point to the start of ifOp then block 
    5) Create a constant val whose value is to be stored 
    6) Return this constant value from the if block
    7) Create affineStore for saving the returned constant value into someOutput memref Location
    8) Close the forloop by setting operation 
    9) 

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

rewriter.replaceOp(op, alloc);