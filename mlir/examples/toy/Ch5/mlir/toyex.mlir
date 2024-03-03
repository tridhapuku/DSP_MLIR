toy.func @main() {
    %0 = toy.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01]> : tensor<3xf64>
    %1 = toy.constant dense<2.000000e+00> : tensor<f64>
    %2 = "toy.delay"(%0, %1) : (tensor<6xf64>, tensor<f64>) -> tensor<3xf64>
    toy.print %2 : tensor<3xf64>
    toy.return
}


  func.func @main() {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 6.000000e+01 : f64
    %cst_1 = arith.constant 5.000000e+01 : f64
    %cst_2 = arith.constant 4.000000e+01 : f64
    %cst_3 = arith.constant 3.000000e+01 : f64
    %cst_4 = arith.constant 2.000000e+01 : f64
    %cst_5 = arith.constant 1.000000e+01 : f64


    %alloc = memref.alloc() : memref<f64>
    %alloc_6 = memref.alloc() : memref<6xf64>
    %out_alloc_6 = memref.alloc() : memref<6xf64>

    affine.store %cst_5, %alloc_6[0] : memref<6xf64>
    affine.store %cst_4, %alloc_6[1] : memref<6xf64>
    affine.store %cst_3, %alloc_6[2] : memref<6xf64>
    affine.store %cst_2, %alloc_6[3] : memref<6xf64>
    affine.store %cst_1, %alloc_6[4] : memref<6xf64>
    affine.store %cst_0, %alloc_6[5] : memref<6xf64>
    affine.store %cst, %alloc[] : memref<f64>

    //from 0 to delayOp 2nd argumentt 
    affine.for %arg0 = 0 to %cst ie 2 {
        %0 = affine.load %alloc[] : memref<6xf64>
        affine.store %0, %out_alloc_6[%arg0] : memref<6x1xf64>
    }
    //rom 0 to delayOp 2nd argument to last index 
    affine.for %arg0 = 2 to 6 {
        %0 = affine.load %alloc_6[%arg0] : memref<6xf64>
        %1 = arith.sub %arg0 , %cst : f64
        %2 = arith.const %1 : i64 //index
        affine.store %0, %out_alloc_6[%2] : memref<6xf64>
    }

    toy.print %alloc_6 : memref<6xf64>
    toy.print %alloc : memref<f64>
    memref.dealloc %alloc_6 : memref<6xf64>
    memref.dealloc %alloc : memref<f64>
    return
  }


  func.func @main() {

    %cst = arith.constant 5.300000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64

    %cst_2 = arith.constant 3.000000e+01 : f64
    %cst_3 = arith.constant 2.000000e+01 : f64
    %cst_4 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<3xf64>
    %alloc_5 = memref.alloc() : memref<f64>
    %alloc_6 = memref.alloc() : memref<3xf64>
    affine.store %cst_4, %alloc_6[0] : memref<3xf64>
    affine.store %cst_3, %alloc_6[1] : memref<3xf64>
    affine.store %cst_2, %alloc_6[2] : memref<3xf64>
    affine.store %cst_1, %alloc_5[] : memref<f64>

    //here, 2 should be from 2nd opernad ie %1 from "toy.delay"(%0, %1) : (tensor<6xf64>, tensor<f64>)
    affine.for %arg0 = 0 to %cst_1 {
      affine.store %cst_0, %alloc[%arg0] : memref<3xf64>
    }

    //here, 2 is coming from 2nd operand of toy.delayOP
    //alloc_6 is location for input array 
    affine.for %arg0 = %0 to 3 {
      %0 = affine.load %alloc_6[%arg0] : memref<3xf64>
      %1 = arith.constant 1 : i64
      %2 = arith.IndexCast %1 : index
      %3 = arith.addi %arg0 , %2 : index
      affine.store %0, %alloc[%3] : memref<3xf64>
    }

    toy.print %alloc : memref<3xf64>
    memref.dealloc %alloc_6 : memref<3xf64>
    memref.dealloc %alloc_5 : memref<f64>
    memref.dealloc %alloc : memref<3xf64>
    return
  }


      If this is an input mlir --
    %0 = toy.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01]> : tensor<3xf64>
    %1 = toy.constant dense<2.000000e+00> : tensor<f64>
    %2 = "toy.delay"(%0, %1) : (tensor<6xf64>, tensor<f64>) -> tensor<3xf64>
    
    this is an output mlir --
    //here, 2 should be from 2nd opernad ie %1 from "toy.delay"(%0, %1) : (tensor<6xf64>, tensor<f64>)
    %1 = arith.const
    affine.for %arg0 = 0 to 2 {
      affine.store %cst_0, %alloc[%arg0] : memref<3xf64>
    }

    //here, 2 is coming from 2nd operand of toy.delayOP
    //alloc_6 is location for input array 
    affine.for %arg0 = 2 to 3 {
      %0 = affine.load %alloc_6[%arg0 - 2] : memref<3xf64>
      affine.store %0, %alloc[%arg0] : memref<3xf64>
    }

    what should be the C++ code for this conversion?

    Pseudo code is:
    1) create arith.const operation with value 0 & type=f64 --
    2) get second operand of the DelayOp f64 & process it so that we can define affine loop index from this second operand
      2.a) create arith.constOp for the 2nd operand
    3) Build two affine loop nests
    4) First 1 will iterate from 0 to delayOP second operand value
            --inside the loop 
            -- we will store arith.constOp with value zero at output memref 
    5) 2nd loop will iterate from  delayOP second operand value to length of 1st operand of delayOp 
        --inside the loop
        -- load the values from input allocated memref and store to output memref : the input allocated memref will start from 0 while output memref will be at continuing from delayOP second operand value
        --


  

  --------------------------
// Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs)
  --what does ivs represent here ? Give 2 examples

what does loop induction variable mean here -- for this output 
    affine.for %arg0 = 0 to 2 {
      %0 = affine.load %alloc_5[%arg0] : memref<3xf64>
      affine.store %0, %alloc[%arg0] : memref<3xf64>
    }
    the code being --
        auto zeroValue = nestedBuilder.create<arith::ConstantOp>(loc, nestedBuilder.getF64Type(),
                        nestedBuilder.getFloatAttr(nestedBuilder.getF64Type(), 5.3) );
        Value valueToStore = zeroValue;

        nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
                                                    ivs);    

  --does ivs[0] mean %arg0 -- explain more in detail? 



Can you help in completing this function when the output looks like the below?
  //affine loop nest should look like this --
      affine.for %arg0 = 0 to 3 {
      %0 = affine.load %alloc_5[%arg0] : memref<3xf64>  
      %1 = arith.const 2 : i64
      %2 = arith.add %arg0 , %1 : i64
      affine.store %0, %alloc[%2] : memref<3xf64>
    }


  what should be the code for manipulating loop induction variable such that for storing, the output index would be some index ahead of the loop induction variable --
  : hint: we can arith , affine or any other builtin dialect for this 
    affine.for %arg0 = 0 to 3 {
      %0 = affine.load %alloc_5[%arg0] : memref<3xf64>  
      affine.store %0, %alloc[%arg0 + 2] : memref<3xf64>
    }

     Value outputIndex = nestedBuilder.create<arith::AddIOp>(loc, ivs[0], SecondValueInt);
      getting this error : error: invalid conversion from ‘long int’ to ‘mlir::detail::ValueImpl*’ [-fpermissive]

      here, SecondValueInt is coming from f64 tenosr type having just 1 element ie,
      %2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

      Value delaySecondArg = op->getOperand(1); 
      DenseElementsAttr constantValue = constantOp2ndArg.getValue(); 
      auto elements = constantValue.getValues<FloatAttr>();
      int64_t SecondValueInt = (int64_t) SecondValue;

      can you help in rectifying this error?

  func.func @main() {
    %cst = arith.constant 5.300000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+01 : f64
    %cst_3 = arith.constant 2.000000e+01 : f64
    %cst_4 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<3xf64>
    %alloc_5 = memref.alloc() : memref<f64>
    %alloc_6 = memref.alloc() : memref<3xf64>
    affine.store %cst_4, %alloc_6[0] : memref<3xf64>
    affine.store %cst_3, %alloc_6[1] : memref<3xf64>
    affine.store %cst_2, %alloc_6[2] : memref<3xf64>
    affine.store %cst_1, %alloc_5[] : memref<f64>
    affine.for %arg0 = 0 to %cst_1 {
      affine.store %cst_0, %alloc[%arg0] : memref<3xf64>
    }
    affine.for %arg0 = %cst_1 to 3 {
      affine.store %cst, %alloc[%arg0] : memref<3xf64>
    }
    toy.print %alloc : memref<3xf64>
    memref.dealloc %alloc_6 : memref<3xf64>
    memref.dealloc %alloc_5 : memref<f64>
    memref.dealloc %alloc : memref<3xf64>
    return
  }

   func.func @main() {
    %cst = arith.constant 5.300000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+01 : f64
    %cst_3 = arith.constant 2.000000e+01 : f64
    %cst_4 = arith.constant 1.000000e+01 : f64
    %alloc = memref.alloc() : memref<3xf64>
    %alloc_5 = memref.alloc() : memref<f64>
    %alloc_6 = memref.alloc() : memref<3xf64>
    affine.store %cst_4, %alloc_6[0] : memref<3xf64>
    affine.store %cst_3, %alloc_6[1] : memref<3xf64>
    affine.store %cst_2, %alloc_6[2] : memref<3xf64>
    affine.store %cst_1, %alloc_5[] : memref<f64>
    affine.for %arg0 = 0 to %cst_1 {
      affine.store %cst_0, %alloc[%arg0] : memref<3xf64>
    }
    affine.for %arg0 = %cst_1 to 3 {
      affine.store %cst, %alloc[%arg0] : memref<3xf64>
    }
    toy.print %alloc : memref<3xf64>
    memref.dealloc %alloc_6 : memref<3xf64>
    memref.dealloc %alloc_5 : memref<f64>
    memref.dealloc %alloc : memref<3xf64>
    return
  }




%2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<*xf64>
how to cast tensor<f64> of single element to ConstantIndexOp .. ie,  %1 is to be casted to ConstantIndexOp ?

%2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<*xf64>
Value delaySecondArg = op->getOperand(1); how to convert tensor<f64> to int ie, Value is of type tensor<f64> 
and I want to convert it to Value of type int? 

mlir::Value integerValue = rewriter.create<mlir::arith::ExtUIOp>(
      loc, elementValue, rewriter.getIntegerType(64)); // convert mlir::Value integerValue to 

  OpBuilder b(forOp);
  Location loc(forOp.getLoc());
  AffineExpr lhs, rhs;
  bindSymbols(forOp.getContext(), lhs, rhs);
  auto mulMap = AffineMap::get(0, 2, lhs * rhs);
  auto addMap = AffineMap::get(0, 2, lhs + rhs);

  Value linearIndex = processorId.front();
  for (unsigned i = 1, e = processorId.size(); i < e; ++i) {
    auto mulApplyOp = b.create<AffineApplyOp>(
        loc, mulMap, ValueRange{linearIndex, numProcessors[i]});
    linearIndex = b.create<AffineApplyOp>(
        loc, addMap, ValueRange{mulApplyOp, processorId[i]});
  }


    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.

        //
        AffineExpr indx;
        AffineExpr constantExpr = nestedBuilder.getAffineConstantExpr(SecondValueInt);
        AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
        //Get the input allocated space for the load
        toy::DelayOpAdaptor delayAdaptor(operands);
        auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, delayAdaptor.getLhs(),ivs);
        auto outputIndex = nestedBuilder.create<affine::AffineApplyOp>(loc, addMap , ivs);

        nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
                    ValueRange{outputIndex});
      });


SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
mlir::ValueRange lbs1 = mlir::ValueRange(lowerBounds);
Getting the error -- error: no matching function for call to ‘mlir::ValueRange::ValueRange(llvm::SmallVector<long int, 4>&)’
  194 |   mlir::ValueRange lbs1 = mlir::ValueRange(lowerBounds);


SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
mlir::ValueRange lbsRange(llvm::ArrayRef<mlir::Value>(lowerBounds.data(), lowerBounds.size()));

error: no matching function for call to ‘llvm::ArrayRef<mlir::Value>::ArrayRef(llvm::SmallVectorTemplateCommon<long int, void>::pointer, size_t)’
  199 |   ValueRange lbsrange = ValueRange(ArrayRef<mlir::Value>(lowerBounds.data(), lowerBounds.size()));



 SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);

 LowerToAffineLoops.cpp:201:69: error: no matching function for call to ‘mlir::ValueRange::ValueRange(<brace-enclosed initializer list>)’
  201 |   mlir::ValueRange lbsRange({lowerBounds.begin(), lowerBounds.end()});
      |                                                                     ^
In file included from /home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/include/mlir/IR/TypeRange.h:18,
                 from /home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/include/mlir/IR/OperationSupport.h:23,
                 from /home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/include/mlir/IR/Dialect.h:17,
                 from /home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/include/mlir/IR/BuiltinDialect.h:17,
                 from /home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/examples/toy/Ch6/mlir/LowerToAffineLoops.cpp:15:
/home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/include/mlir/IR/ValueRange.h:383:3: note: candidate: ‘template<class Arg, class> mlir::ValueRange::ValueRange(Arg&&)’
  383 |   ValueRange(Arg &&arg) : ValueRange(ArrayRef<Value>(std::forward<Arg>(arg))) {}
      |   ^~~~~~~~~~
/home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/include/mlir/IR/ValueRange.h:383:3: note:   template argument deduction/substitution failed:
/home/abhinav/ForMLIR/SourceCode/llvm-project/mlir/examples/toy/Ch6/mlir/LowerToAffineLoops.cpp:201:69: note:   couldn’t deduce template parameter ‘Arg’
  201 |   mlir::ValueRange lbsRange({lowerBounds.begin(), lowerBounds.end()});
------------------------------
  toy.func @main() {
   .................
    %3 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<*xf64>
    %4 = "toy.delay"(%3, %2) : (tensor<*xf64>, tensor<f64>) -> tensor<*xf64>
    toy.print %4 : tensor<*xf64>
    toy.return
  }

  toy.func @main() {
   ........................
    %3 = toy.add %1, %2 : tensor<f64>
    %4 = "toy.delay"(%0, %3) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    toy.print %4 : tensor<3xf64>
    toy.return
  }


%3 = toy.add %1, %2 : tensor<f64>
%4 = arith.fptoui %3 : ui
%5 = arith.ConstanIndex %4 : index
affine.for %arg0 = 0 to %5 {
  affine.store %cst,%alloc[%arg0]:memref<3xf64>  
}

%6 = arith.sub %8, %5: 
affine.for %arg0 = 0 to %6 {
  %0 = affine.load %alloc_3[%arg0] : memref<3xf64> 
  #map2 = affine_map<(d0) -> (d0 + %4)>
  %1 = affine.apply #map2(%arg0)
  affine.store %0, %alloc[%1]: memref<3xf64>
}

affine.for %arg0 = 0 to %cst {
    %0 = affine.load %alloc[%arg0] : memref<6xf64>
    %1 = affine.apply #map2(%arg0)
    affine.store %0, %out_alloc_6[%1] : memref<6xf64>
}




----------------------------------
       //create a affine map : addMap(d0) -> (d0 + 2)
        //apply it on loop variable
        //o/p of the applied map will be the index for StoreOp
        //from 0 to delayOp 2nd argumentt 

        #map2 = affine_map<(d0) -> (d0 + 2)>
        affine.for %arg0 = 0 to %cst {
            %0 = affine.load %alloc[%arg0] : memref<6xf64>
            %1 = affine.apply #map2(%arg0)
            affine.store %0, %out_alloc_6[%1] : memref<6xf64>
        }

        
    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        //Get the input allocated space for the load
        toy::DelayOpAdaptor delayAdaptor(operands);
        auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, delayAdaptor.getLhs(),ivs);

 
        AffineExpr indx; 
        AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
        //error in next line -- segmentation fault 
        AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
        auto outputIndex = nestedBuilder.create<affine::AffineApplyOp>(loc, addMap , ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
                    ValueRange{outputIndex});


FirstMLIR: 
toy.func @main() {
    %0 = toy.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01]> : tensor<3xf64>
    %1 = toy.constant dense<2.000000e+00> : tensor<f64>
    %2 = "toy.delay"(%0, %1) : (tensor<6xf64>, tensor<f64>) -> tensor<3xf64>
    toy.print %2 : tensor<3xf64>
    toy.return
}


SecondMLIR: 
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01, 4.000000e+01, 5.000000e+01, 6.000000e+01, 7.000000e+01, 8.000000e+01, 9.000000e+01, 1.000000e+02]> : tensor<10xf64>
    %1 = toy.constant dense<2.000000e+00> : tensor<f64>
    %2 = toy.constant dense<4.000000e+00> : tensor<f64>
    %3 = toy.add %1, %2 : tensor<f64>
    %4 = "toy.delay"(%0, %3) : (tensor<10xf64>, tensor<f64>) -> tensor<10xf64>
    toy.print %4 : tensor<10xf64>
    toy.return
  }

I can extract the value of "toy.delay" in FirstMLIR using the below code:
Value delaySecondArg = op->getOperand(1);
toy::ConstantOp constantOp2ndArg = delaySecondArg.getDefiningOp<toy::ConstantOp>();
DenseElementsAttr constantValue = constantOp2ndArg.getValue();
auto elements = constantValue.getValues<FloatAttr>();
float SecondValue = elements[0].getValueAsDouble();
int64_t SecondValueInt = (int64_t) SecondValue;

//

Now, how can I do this in general to extract Integer from second argument of toy.delay operation ?
I am going to use this SecondValueInt in Affine constantExpr snd also for affine::buildAffineLoopNest lowerBounds? Can you suggest some solutions?

But for the below mlir --
 %3 = toy.add %1, %2 : tensor<f64>
 %4 = "toy.delay"(%0, %3) : (tensor<10xf64>, tensor<f64>) -> tensor<10xf64>

toy::ConstantOp constantOp2ndArg = delaySecondArg.getDefiningOp<toy::ConstantOp>();
we can see the 2nd argument is not coming from toy.constant operation but it is coming from toy.add operation -- can you help now to extract the integer value?

AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
Here, in the above -- can I use Value type argument instead of integer type argument? If yes, can u give a suitable example.


Value mlir::getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                            OpFoldResult ofr) {
  if (auto value = llvm::dyn_cast_if_present<Value>(ofr))
    return value;
  auto attr = dyn_cast<IntegerAttr>(llvm::dyn_cast_if_present<Attribute>(ofr));
  assert(attr && "expect the op fold result casts to an integer attribute");
  return b.create<arith::ConstantIndexOp>(loc, attr.getValue().getSExtValue());
}

-----------------------------------------------

3) error: 
       'arith.addi' op requires the same type for all operands and results
    %2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

          note: see current operation: %14 = "arith.addi"(%arg0, %13) : (index, i64) -> index

          which is coming from the line : Value outputIndex = nestedBuilder.create<arith::AddIOp>(loc, ivs[0], secondValueMLIR); -- can you help further on this?

4) Runtime error for arith.AddIOp -- 
    --error: 'arith.addi' op requires the same type for all operands and results
    %2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>

      --Sol: use arith::IndexCastOp

    Value secondValueIndex = nestedBuilder.create<arith::IndexCastOp>(
        loc, secondValueMLIR, nestedBuilder.getIndexType()); 
    error: no matching function for call to ‘mlir::arith::IndexCastOp::build(mlir::OpBuilder&, mlir::OperationState&, mlir::Value&, mlir::IndexType)’
      490 |     OpTy::build(*this, state, std::forward<Args>(args)...); can u help with this error?


      error: no matching function for call to ‘mlir::arith::IndexCastOp::build(mlir::OpBuilder&, mlir::OperationState&, mlir::Value&)’


5) Affine.StoreOp 
    -- nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
                                                ArrayRef<Value>({outputIndex}) )

  error: 'affine.store' op index must be a dimension or symbol identifier
    %2 = "toy.delay"(%0, %1) : (tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
         ^
         : note: see current operation: "affine.store"(%12, %0, %15) {map = affine_map<(d0) -> (d0)>} : (f64, memref<3xf64>, index) -> ()
         can u help with this error? Also, explain what is affine_map with suitable example


6) Segmentation fault with affine.apply


7) Segmentation fault for back to back delayOp 
    --Reason: we are converting second operand to toy::constantOp and extracting value from it

8)  affine::buildAffineLoopNest -- 
error: failed to materialize conversion for result #0 of operation 'toy.constant' that remained live after conversion
