//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wall"

#include "mlir/IR/BuiltinDialect.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include <iostream>
using namespace mlir;
using namespace std;
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//


// #pragma warning(push, 0)
/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as dsp functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    
    // for (auto i : tensorType.getShape())
    // {
    //    llvm::errs() << "tensortype =" << i << "\n" ;
    // }
    // llvm::errs() << "tensortype.getElementType =" << tensorType.getElementType() << "\n" ;
    // llvm::errs() << "op->getLoc = " << op->getLoc() << "\n"; //getDialect
    // llvm::errs() << "op->getDialect = " << op->getDialect() << "\n";
    // llvm::errs() << "op->getName = " << op->getName() << "\n";
    // // llvm::errs() << "op->getType = " << op->getType() << "\n";
    // llvm::errs() << "op->getParentRegion = " << op->getParentRegion() << "\n";
    // llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName() << "\n";
    
    // llvm::errs() << "op->getNumOperands = " << op->getNumOperands() << "\n";
    // for (auto i : op->getOperands())
    // {
    //   llvm::errs() << "op->Operand = " << i << "\n";
    // }
    
    // llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName() << "\n";
    // llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName() << "\n";
    // llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName() << "\n";
  
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


// static void lowerOpToLoops3Debug(Operation *op, ValueRange operands,
//                            PatternRewriter &rewriter,
//                            LoopIterationFn processIteration) {
//   auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
//   // llvm::errs() << "tensorType= " << tensorType.getTypeID() << "\n";
//   auto loc = op->getLoc();

//   // Insert an allocation and deallocation for the result of this operation.
//   auto memRefType = convertTensorToMemRef(tensorType);
//   auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
//   //get the 2nd operand of delayOp & convert it into int
//   Value delaySecondArg = op->getOperand(1);

//   //Get convert this tensor<f64> to Integer using arith::FPtoUIOp 
//   //Also, convert this integer to constantIndexOp 
//   // Use this for forming ubsRange 
//   //Build loopNest from this 
//   // auto result= rewriter.create<arith::FPToUIOp>(loc,  rewriter.getIntegerType(64), delaySecondArg);
//   //gives error
//   // auto indxFor2ndArg = rewriter.create<arith::ConstantIndexOp>(loc, result.getResult());

//   // Get the defining operation of the second operand
//   // Operation *definingOp = delaySecondArg.getDefiningOp();
//   //extract integer value form constant attribute of 2nd operand
//   llvm::errs() << "delaySecondArg.getDefiningOp()= " << delaySecondArg.getDefiningOp() << "\n";
//   // int64_t delayValue = delaySecondArg.getDefiningOp()->getAttrOfType<IntegerAttr>("value").getInt();
    
//   //Pseudo-Code  
//   //Get 2nd argument -- check if it is coming from constantOp -- 
//   // if yes, get int attr
//   //else
//   //get definingOp & also, get the constant values from definingOp->operands
//   // add those constant values 
//   // use this sum for the de
//   // auto constantOp = dyn_cast_or_null<dsp::ConstantOp>(definingOp);
//   dsp::ConstantOp constantOp2ndArg = delaySecondArg.getDefiningOp<dsp::ConstantOp>();
//   dsp::AddOp addOp2ndArg = delaySecondArg.getDefiningOp<dsp::AddOp>();
  
//   int64_t SecondValueInt = 0;
//   if(constantOp2ndArg)
//   {
//     llvm::errs() << "Defining Opp is not constant so no lowering for now";

//     DenseElementsAttr constantValue = constantOp2ndArg.getValue();
//     auto elements = constantValue.getValues<FloatAttr>();
//     float SecondValue = elements[0].getValueAsDouble();
//     SecondValueInt = (int64_t) SecondValue;
//   }
//   else if(addOp2ndArg)
//   {
//     Value lhs = addOp2ndArg.getLhs();
//     Value rhs = addOp2ndArg.getRhs();

//     dsp::ConstantOp constantAdd1arg = lhs.getDefiningOp<dsp::ConstantOp>();
//     dsp::ConstantOp constantAdd2arg = rhs.getDefiningOp<dsp::ConstantOp>();

//     if(!constantAdd1arg || !constantAdd2arg)
//     {
//       llvm::errs() << "No support when add operation is not coming from constants\n";
//       return;
//     }
//     DenseElementsAttr constant1 = constantAdd1arg.getValue();
//     DenseElementsAttr constant2 = constantAdd2arg.getValue();

//     auto elements1 = constant1.getValues<FloatAttr>();
//     float Val1 = elements1[0].getValueAsDouble();

//     auto elements2 = constant2.getValues<FloatAttr>();
//     float Val2 = elements2[0].getValueAsDouble();

//     SecondValueInt = (int64_t) (Val1 + Val2);
//   }
//   else{
//     llvm::errs() << "delay operation with this sequence not supported !!\n";
//     return;
//   }

//   llvm::errs() << "\n*****SecondValueInt = " << SecondValueInt << " ***\n"; 
//   // Create a nest of affine loops, with one loop per dimension of the shape.
//   // The buildAffineLoopNest function takes a callback that is used to construct
//   // the body of the innermost loop given a builder, a location and a range of
//   // loop induction variables.
//   llvm::errs() << "tensorType->getRank = " << tensorType.getRank() << "\n";
//   llvm::errs() << "tensorType->getNumElements = " << tensorType.getNumElements() << "\n";
//   SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
//   SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

//   llvm::errs() << "lowerBounds.size() = " << lowerBounds.size() << "\n";
//   llvm::errs() << "steps.size() = " << steps.size() << "\n";
  
//   // SmallVector<Value, 4> mlirValues;
//   // for (auto intValue: lowerBounds)
//   // {
//   //    Value constantValue = rewriter.create<arith::ConstantIndexOp>(loc, intValue);
//   //    mlirValues.push_back(constantValue);
//   // }
  
//   // // Print the values from mlirValues
//   // llvm::errs() << "Printing mlirValues for lbsrange\n";
//   // for (mlir::Value value : mlirValues) {
//   //   value.print(llvm::errs());
//   //   llvm::errs() << "\n";
//   // }
//   // mlir::ValueRange lbsrange(mlirValues);

//   for (auto i : tensorType.getShape())
//   {
//     llvm::errs() << "tensorType.getShape() = " << i << "\n";
//   }
  
//   // llvm::errs() << "tensorType.getShape() = " << tensorType.getShape() << "\n";
//   // SmallVector<int64_t, 4> upperBounds(tensorType.getRank(), /*Value=*/0);
//   // llvm::ArrayRef<int64_t> upperBounds = tensorType.getShape();
//   std::vector<int64_t> upperBounds = tensorType.getShape();
//   //delaySecondArg should be of type tensor of rank 1 & shape 1 
//   // Value delaySecondArg = op->getOperand(1);
//   // SmallVector<Value, 4> mlirValues2;

//   // auto delay2ndArgTensor = llvm::cast<RankedTensorType>(op->getOperand(1));
//   // Convert the single element to arith::ConstantIndexOp
//   // mlir::Value index = rewriter.create<mlir::arith::ExtUIOp>(
//   //       loc,  rewriter.getIndexType(), delaySecondArg);
//   // mlir::Value index = rewriter.create<mlir::arith::ExtUIOp>(
//   //       loc,  rewriter.getIndexType(), delaySecondArg);
//   // mlirValues2.push_back(index); //try with direct constant integer
//   // mlir::ValueRange ubsRange(mlirValues2);

//   //change upperBounds with 2nd operand value
//   // llvm::errs() << "Printing mlirValues for ubsrange\n";
//   // for (mlir::Value value : mlirValues2) {
//   //   value.print(llvm::errs());
//   //   llvm::errs() << "\n";
//   // }
//   llvm::errs() << __LINE__ << "\n";

//   for(auto& shape: upperBounds)
//   {
//     shape = SecondValueInt;
//   }
//   // llvm::errs() << __LINE__ << "\n";
//   for(auto shape: upperBounds)
//   {
//     // llvm::errs() << "shape= " << shape << "\n";
//     llvm::errs() << __LINE__ << "\n";
//   }

//   // affine::buildAffineLoopNest(
//   //     rewriter, loc, lbsrange, ubsRange, steps,
//   //     [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//   //       // Call the processing function with the rewriter, the memref operands,
//   //       // and the loop induction variables. This function will return the value
//   //       // to store at the current index.
//   //       llvm::errs() << __LINE__ << "\n";
//   //       auto zeroValue = nestedBuilder.create<arith::ConstantOp>(loc, nestedBuilder.getF64Type(),
//   //                       nestedBuilder.getFloatAttr(nestedBuilder.getF64Type(), 0.0) );
        
//   //       llvm::errs() << __LINE__ << "\n";
//   //       Value valueToStore = zeroValue;

//   //       llvm::errs() << __LINE__ << "\n";
//   //        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
//   //                                                   ivs);

//   //       llvm::errs() << __LINE__ << "\n";
//   //     });

//   //working
//   affine::buildAffineLoopNest(
//       rewriter, loc, lowerBounds, upperBounds, steps,
//       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//         // Call the processing function with the rewriter, the memref operands,
//         // and the loop induction variables. This function will return the value
//         // to store at the current index.
//         auto zeroValue = nestedBuilder.create<arith::ConstantOp>(loc, nestedBuilder.getF64Type(),
//                         nestedBuilder.getFloatAttr(nestedBuilder.getF64Type(), 0.0) );
//         Value valueToStore = zeroValue;
//         // Value valueToStore = processIteration(nestedBuilder, operands, ivs);
//         //  [loc ] (OpBuilder &builder, ValueRange memRefOperands,
//         //           ValueRange loopIvs) {
//         //             //
//         //             dsp::DelayOpAdaptor delayAdaptor(memRefOperands);
//         //             Value input0 = delayAdaptor.getLhs();

//         //             auto zeroValue = builder.create<arith::ConstantOp>(loc, builder.getF64Type(),
//         //                 builder.getFloatAttr(builder.getF64Type(), 0.0) );
//         nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
//                                                     ivs);
//       });


//            //change lower bounds and also change upper bounds 
//       upperBounds = tensorType.getShape();
//       for(auto& shape: upperBounds)
//       {
//         shape = shape - SecondValueInt; //replace 4 by 2ndOperand
//       } 

      
//      auto intDelaySSAValue = rewriter.create<arith::ConstantOp>(loc, 
//                           IntegerAttr::get(rewriter.getIntegerType(64), SecondValueInt));

//     // Define an affine map: #map2 = affine_map<(d0) -> (d0 + 2)>
//     // AffineMap affineMap = AffineMap::get(1, 0, {AffineExpr::getAddExpr(AffineDimExpr::get(0, context), 2)});
            

//     // llvm::errs() << __LINE__ << "\n";
//     // AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
//     // llvm::errs() << __LINE__ << "\n";
    
//     // llvm::errs() << __LINE__ << "\n";

    
    

//     affine::buildAffineLoopNest(
//       rewriter, loc, lowerBounds, upperBounds, steps,
//       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
//         // Call the processing function with the rewriter, the memref operands,
//         // and the loop induction variables. This function will return the value
//         // to store at the current index.

//         //Get the input allocated space for the load
//         dsp::DelayOpAdaptor delayAdaptor(operands);
//         auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, delayAdaptor.getLhs(),ivs);
//         // llvm::errs() << __LINE__ << "\n";
//         // AffineExpr indx; 
//         // AffineExpr indx = ivs[0];
//         AffineExpr indx = nestedBuilder.getAffineDimExpr(0);
//         AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
//         AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
//         auto outputIndex = nestedBuilder.create<affine::AffineApplyOp>(loc, addMap , ivs);
//         nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
//                     ValueRange{outputIndex});
//         // llvm::errs() << __LINE__ << "\n";
                                       
//       });


//   // Replace this operation with the generated alloc.
//   rewriter.replaceOp(op, alloc);
// }

static void lowerOpToLoops3(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  // llvm::errs() << "tensorType= " << tensorType.getTypeID() << "\n";
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
  //get the 2nd operand of delayOp & convert it into int
  Value delaySecondArg = op->getOperand(1);

  // Get the defining operation of the second operand
  // Operation *definingOp = delaySecondArg.getDefiningOp();

  //Pseudo-Code  
  //Get 2nd argument -- check if it is coming from constantOp -- 
  // if yes, get int attr
  //else
  //get definingOp & also, get the constant values from definingOp->operands
  // add those constant values 
  // use this sum for the de
  // auto constantOp = dyn_cast_or_null<dsp::ConstantOp>(definingOp);
  dsp::ConstantOp constantOp2ndArg = delaySecondArg.getDefiningOp<dsp::ConstantOp>();
  dsp::AddOp addOp2ndArg = delaySecondArg.getDefiningOp<dsp::AddOp>();
  
  int64_t SecondValueInt = 0;
  if(constantOp2ndArg)
  {
    // llvm::errs() << "Defining Opp is not constant so no lowering for now";

    DenseElementsAttr constantValue = constantOp2ndArg.getValue();
    auto elements = constantValue.getValues<FloatAttr>();
    float SecondValue = elements[0].getValueAsDouble();
    SecondValueInt = (int64_t) SecondValue;
  }
  else if(addOp2ndArg)
  {
    Value lhs = addOp2ndArg.getLhs();
    Value rhs = addOp2ndArg.getRhs();

    dsp::ConstantOp constantAdd1arg = lhs.getDefiningOp<dsp::ConstantOp>();
    dsp::ConstantOp constantAdd2arg = rhs.getDefiningOp<dsp::ConstantOp>();

    if(!constantAdd1arg || !constantAdd2arg)
    {
      llvm::errs() << "No support when add operation is not coming from constants\n";
      return;
    }
    DenseElementsAttr constant1 = constantAdd1arg.getValue();
    DenseElementsAttr constant2 = constantAdd2arg.getValue();

    auto elements1 = constant1.getValues<FloatAttr>();
    float Val1 = elements1[0].getValueAsDouble();

    auto elements2 = constant2.getValues<FloatAttr>();
    float Val2 = elements2[0].getValueAsDouble();

    SecondValueInt = (int64_t) (Val1 + Val2);
  }
  else{
    llvm::errs() << "delay operation with this sequence not supported !!\n";
    return;
  }

  //   llvm::errs() << "\n*****SecondValueInt = " << SecondValueInt << " ***\n"; 
  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  //   llvm::errs() << "tensorType->getRank = " << tensorType.getRank() << "\n";
  //   llvm::errs() << "tensorType->getNumElements = " << tensorType.getNumElements() << "\n";
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  std::vector<int64_t> upperBounds = tensorType.getShape();


  for(auto& shape: upperBounds)
  {
    shape = SecondValueInt;
  }

  //working
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        auto zeroValue = nestedBuilder.create<arith::ConstantOp>(loc, nestedBuilder.getF64Type(),
                        nestedBuilder.getFloatAttr(nestedBuilder.getF64Type(), 0.0) );
        Value valueToStore = zeroValue;
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });


           //change lower bounds and also change upper bounds 
      upperBounds = tensorType.getShape();
      for(auto& shape: upperBounds)
      {
        shape = shape - SecondValueInt; //replace 4 by 2ndOperand
      } 

      
    //  auto intDelaySSAValue = rewriter.create<arith::ConstantOp>(loc, 
    //                       IntegerAttr::get(rewriter.getIntegerType(64), SecondValueInt));

    // Define an affine map: #map2 = affine_map<(d0) -> (d0 + 2)>
    

    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.

        //Get the input allocated space for the load
        dsp::DelayOpAdaptor delayAdaptor(operands);
        auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, delayAdaptor.getLhs(),ivs);

        AffineExpr indx = nestedBuilder.getAffineDimExpr(0);
        AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
        AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
        auto outputIndex = nestedBuilder.create<affine::AffineApplyOp>(loc, addMap , ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
                    ValueRange{outputIndex});
     
                                       
      });


  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

// static void lowerOpToLoopsGain1(Operation *op, ValueRange operands,
//                            PatternRewriter &rewriter,
//                            LoopIterationFn processIteration) {
  //     auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  //     // llvm::errs() << "tensorType= " << tensorType.getTypeID() << "\n";
  //     auto loc = op->getLoc();

  //     // Insert an allocation and deallocation for the result of this operation.
  //     auto memRefType = convertTensorToMemRef(tensorType);
  //     auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);


  //     // Create a nest of affine loops, with one loop per dimension of the shape.
  //     // The buildAffineLoopNest function takes a callback that is used to construct
  //     // the body of the innermost loop given a builder, a location and a range of
  //     // loop induction variables.
  //     llvm::errs() << "tensorType->getRank = " << tensorType.getRank() << "\n";
  //     llvm::errs() << "tensorType->getNumElements = " << tensorType.getNumElements() << "\n";
  //     SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  //     SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

  //     llvm::errs() << "lowerBounds.size() = " << lowerBounds.size() << "\n";
  //     llvm::errs() << "steps.size() = " << steps.size() << "\n";
      

  //     for (auto i : tensorType.getShape())
  //     {
  //         llvm::errs() << "tensorType.getShape() = " << i << "\n";
  //     }
      
  //     std::vector<int64_t> upperBounds = tensorType.getShape();

  //     // llvm::errs() << __LINE__ << "\n";
  //     for(auto shape: upperBounds)
  //     {
  //         llvm::errs() << "upperBounds shape= " << shape << "\n";
  //         llvm::errs() << __LINE__ << "\n";
  //     }

    
  //     // llvm::errs() << __LINE__ << "\n";
  //     //get the 2nd operand of delayOp & convert it into int
  //     Value gainSecondArg = op->getOperand(1);
  //     affine::buildAffineLoopNest(
  //       rewriter, loc, lowerBounds, upperBounds, steps,
  //       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
  //         // Call the processing function with the rewriter, the memref operands,
  //         // and the loop induction variables. This function will return the value
  //         // to store at the current index.

  //         //Get the input allocated space for the load
  //         dsp::GainOpAdaptor firFilterAdaptor(operands);
  //         auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, firFilterAdaptor.getLhs(),ivs);
  //         auto loadFromRhs = nestedBuilder.create<affine::AffineLoadOp>(loc, firFilterAdaptor.getRhs());
  //         // auto loadFromRhs = nestedBuilder.create<affine::AffineLoadOp>(loc, gainAdaptor.getRhs(), ivs);
  //         // llvm::errs() << __LINE__ << "\n";
  //         auto mulOp = nestedBuilder.create<arith::MulFOp>(loc, loadFromRhs, loadFromIP);
  //         nestedBuilder.create<affine::AffineStoreOp>(loc, mulOp, alloc,
  //                     ivs);
  //         // nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
  //         //             ivs); //working
  //         // llvm::errs() << __LINE__ << "\n";
                                        
  //       });


  //   // Replace this operation with the generated alloc.
  //   rewriter.replaceOp(op, alloc);
  // }


  // static void lowerOpToLoops2(Operation *op, ValueRange operands,
  //                            PatternRewriter &rewriter,
  //                            LoopIterationFn processIteration) {
  //   auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  //   // llvm::errs() << "tensorType= " << tensorType.getTypeID() << "\n";
  //   auto loc = op->getLoc();

  //   // Insert an allocation and deallocation for the result of this operation.
  //   auto memRefType = convertTensorToMemRef(tensorType);
  //   auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
  //   //get the 2nd operand of delayOp & convert it into int
  //   Value delaySecondArg = op->getOperand(1);
  //   // DenseElementsAttr constantValue = op->getOperand(1); 

  //   // Get the defining operation of the second operand
  //   Operation *definingOp = delaySecondArg.getDefiningOp();
  //   //extract integer value form constant attribute of 2nd operand
  //   llvm::errs() << "delaySecondArg.getDefiningOp()= " << delaySecondArg.getDefiningOp() << "\n";
  //   // int64_t delayValue = delaySecondArg.getDefiningOp()->getAttrOfType<IntegerAttr>("value").getInt();

  //   // auto constantOp = dyn_cast_or_null<dsp::ConstantOp>(definingOp);
  //   dsp::ConstantOp constantOp2ndArg = delaySecondArg.getDefiningOp<dsp::ConstantOp>();
  //   if(!constantOp2ndArg)
  //   {
  //     llvm::errs() << "Defining Opp is not constant so no lowering for now";
  //     return;
  //   }
  //   DenseElementsAttr constantValue = constantOp2ndArg.getValue(); 
  //   RankedTensorType tensorType2 = constantValue.getType().cast<RankedTensorType>();
  //   ArrayRef<int64_t> shape = tensorType.getShape();
  //   llvm::errs() << "tensorType2.rank" << tensorType2.getRank() << "\n";
  //   auto elements = constantValue.getValues<FloatAttr>();

  //   float SecondValue = elements[0].getValueAsDouble();
  //   int64_t SecondValueInt = (int64_t) SecondValue;


  //   for (auto i : tensorType2.getShape())
  //   {
  //     llvm::errs() << "tensorType2.getShape() = " << i << "\n";
  //   }

  //   if(!constantOp2ndArg)
  //   {
  //     llvm::errs() << "delay 2nd arg is not coming from constantOp -faliure \n";
  //     // return mlir::failure(); 
  //   }
  //   // llvm::errs() << "constantOp=" << constantOp << "\n";

  //   // Create a nest of affine loops, with one loop per dimension of the shape.
  //   // The buildAffineLoopNest function takes a callback that is used to construct
  //   // the body of the innermost loop given a builder, a location and a range of
  //   // loop induction variables.
  //   llvm::errs() << "tensorType->getRank = " << tensorType.getRank() << "\n";
  //   llvm::errs() << "tensorType->getNumElements = " << tensorType.getNumElements() << "\n";
  //   SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  //   SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

  //   llvm::errs() << "lowerBounds.size() = " << lowerBounds.size() << "\n";
  //   llvm::errs() << "steps.size() = " << steps.size() << "\n";

  //   //Use SSA values instead of int values
  //   // TestMLIRClass();
  //   // mlir::ValueRange lbs1 = mlir::ValueRange(lowerBounds);
  //   // ValueRange lbsrange = ValueRange(ArrayRef<mlir::Value>(lowerBounds.data(), lowerBounds.size()));
  //   // ValueRange lbsrange = ValueRange(ArrayRef<Value>(lowerBounds));
  //   // mlir::ValueRange lbsRange({lowerBounds.begin(), lowerBounds.end()});
    
  //   // SmallVector<Value, 4> mlirValues;
  //   // for (auto intValue: lowerBounds)
  //   // {
  //   //    Value constantValue = rewriter.create<arith::ConstantIndexOp>(loc, intValue);
  //   //    mlirValues.push_back(constantValue);
  //   // }
    
  //   // // Print the values from mlirValues
  //   // llvm::errs() << "Printing mlirValues for lbsrange\n";
  //   // for (mlir::Value value : mlirValues) {
  //   //   value.print(llvm::errs());
  //   //   llvm::errs() << "\n";
  //   // }
  //   // mlir::ValueRange lbsrange(mlirValues);

  //   for (auto i : tensorType.getShape())
  //   {
  //     llvm::errs() << "tensorType.getShape() = " << i << "\n";
  //   }
    
  //   // llvm::errs() << "tensorType.getShape() = " << tensorType.getShape() << "\n";
  //   // SmallVector<int64_t, 4> upperBounds(tensorType.getRank(), /*Value=*/0);
  //   // llvm::ArrayRef<int64_t> upperBounds = tensorType.getShape();
  //   std::vector<int64_t> upperBounds = tensorType.getShape();
  //   //delaySecondArg should be of type tensor of rank 1 & shape 1 
  //   // Value delaySecondArg = op->getOperand(1);
  //   // SmallVector<Value, 4> mlirValues2;

  //   // // auto delay2ndArgTensor = llvm::cast<RankedTensorType>(op->getOperand(1));
  //   // // Convert the single element to arith::ConstantIndexOp
  //   // mlir::Value index = rewriter.create<mlir::arith::ExtUIOp>(
  //   //       loc,  rewriter.getIndexType(), delaySecondArg);
  //   // mlirValues2.push_back(index);
  //   // mlir::ValueRange ubsRange(mlirValues2);

  //   //change upperBounds with 2nd operand value
  //   llvm::errs() << "Printing mlirValues for ubsrange\n";
  //   // for (mlir::Value value : mlirValues2) {
  //   //   value.print(llvm::errs());
  //   //   llvm::errs() << "\n";
  //   // }
  //   llvm::errs() << __LINE__ << "\n";

  //   for(auto& shape: upperBounds)
  //   {
  //     shape = SecondValueInt;
  //   }
  //   // llvm::errs() << __LINE__ << "\n";
  //   for(auto shape: upperBounds)
  //   {
  //     // llvm::errs() << "shape= " << shape << "\n";
  //     llvm::errs() << __LINE__ << "\n";
  //   }

  //   // affine::buildAffineLoopNest(
  //   //     rewriter, loc, lbsrange, ubsRange, steps,
  //   //     [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
  //   //       // Call the processing function with the rewriter, the memref operands,
  //   //       // and the loop induction variables. This function will return the value
  //   //       // to store at the current index.
  //   //       llvm::errs() << __LINE__ << "\n";
  //   //       auto zeroValue = nestedBuilder.create<arith::ConstantOp>(loc, nestedBuilder.getF64Type(),
  //   //                       nestedBuilder.getFloatAttr(nestedBuilder.getF64Type(), 0.0) );
          
  //   //       llvm::errs() << __LINE__ << "\n";
  //   //       Value valueToStore = zeroValue;

  //   //       llvm::errs() << __LINE__ << "\n";
  //   //        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
  //   //                                                   ivs);

  //   //       llvm::errs() << __LINE__ << "\n";
  //   //     });

  //   //working
  //   affine::buildAffineLoopNest(
  //       rewriter, loc, lowerBounds, upperBounds, steps,
  //       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
  //         // Call the processing function with the rewriter, the memref operands,
  //         // and the loop induction variables. This function will return the value
  //         // to store at the current index.
  //         auto zeroValue = nestedBuilder.create<arith::ConstantOp>(loc, nestedBuilder.getF64Type(),
  //                         nestedBuilder.getFloatAttr(nestedBuilder.getF64Type(), 0.0) );
  //         // auto result= rewriter.create<arith::FPToUIOp>(loc,  rewriter.getIntegerType(64), delaySecondArg);
  //         Value valueToStore = zeroValue;
  //         // Value valueToStore = processIteration(nestedBuilder, operands, ivs);
  //         //  [loc ] (OpBuilder &builder, ValueRange memRefOperands,
  //         //           ValueRange loopIvs) {
  //         //             //
  //         //             dsp::DelayOpAdaptor delayAdaptor(memRefOperands);
  //         //             Value input0 = delayAdaptor.getLhs();

  //         //             auto zeroValue = builder.create<arith::ConstantOp>(loc, builder.getF64Type(),
  //         //                 builder.getFloatAttr(builder.getF64Type(), 0.0) );
  //         nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
  //                                                     ivs);
  //       });

  //       //change lower bounds and also change upper bounds 
  //       upperBounds = tensorType.getShape();
  //       for(auto& shape: upperBounds)
  //       {
  //         shape = shape - SecondValueInt; //replace 4 by 2ndOperand
  //       } 

        
  //      auto intDelaySSAValue = rewriter.create<arith::ConstantOp>(loc, 
  //                           IntegerAttr::get(rewriter.getIntegerType(64), SecondValueInt));

  //     // Define an affine map: #map2 = affine_map<(d0) -> (d0 + 2)>
  //     // AffineMap affineMap = AffineMap::get(1, 0, {AffineExpr::getAddExpr(AffineDimExpr::get(0, context), 2)});
              

  //     // llvm::errs() << __LINE__ << "\n";
  //     // AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
  //     // llvm::errs() << __LINE__ << "\n";
      
  //     // llvm::errs() << __LINE__ << "\n";

      
      

  //     affine::buildAffineLoopNest(
  //       rewriter, loc, lowerBounds, upperBounds, steps,
  //       [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
  //         // Call the processing function with the rewriter, the memref operands,
  //         // and the loop induction variables. This function will return the value
  //         // to store at the current index.

  //         //Get the input allocated space for the load
  //         dsp::DelayOpAdaptor delayAdaptor(operands);
  //         auto loadFromIP = nestedBuilder.create<affine::AffineLoadOp>(loc, delayAdaptor.getLhs(),ivs);
  //         // llvm::errs() << __LINE__ << "\n";
  //         // AffineExpr indx; 
  //         // AffineExpr indx = ivs[0];
  //         AffineExpr indx = nestedBuilder.getAffineDimExpr(0);
  //         AffineExpr constantExpr = rewriter.getAffineConstantExpr(SecondValueInt );
  //         AffineMap addMap = AffineMap::get(1, 0, indx + constantExpr);
  //         auto outputIndex = nestedBuilder.create<affine::AffineApplyOp>(loc, addMap , ivs);
  //         nestedBuilder.create<affine::AffineStoreOp>(loc, loadFromIP, alloc,
  //                     ValueRange{outputIndex});
  //         // llvm::errs() << __LINE__ << "\n";
                                        
  //       });

  //   // Replace this operation with the generated alloc.
  //   rewriter.replaceOp(op, alloc);
  // }



namespace {


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FIRFilter operations
//===----------------------------------------------------------------------===//
struct FIRFilterOpLowering: public ConversionPattern {
      FIRFilterOpLowering(MLIRContext *ctx)
        : ConversionPattern(dsp::FIRFilterOp::getOperationName(), 1 , ctx) {}

    LogicalResult 
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
              ConversionPatternRewriter &rewriter) const final {
      //dsp.FIRFilterOp has 2 operands -- both of type tensor f64 , 2ndOperand should have only 1 element

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
        
        lowerOpToLoops(op, operands, rewriter, 
            [loc, op ] (OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                    dsp::FIRFilterOpAdaptor firFilterAdaptor(memRefOperands);

                    // Generate loads for the element of 'lhs' and 'rhs' at the
                    // inner loop.
                    // auto lhsTensor = delayAdaptor.getLhs();
                    auto lhsTensor = builder.create<affine::AffineLoadOp>(
                         loc, firFilterAdaptor.getLhs(), loopIvs);

                    // auto rhsScalar = op->getOperand(1);     
                    auto rhsScalar = builder.create<affine::AffineLoadOp>(
                         loc, firFilterAdaptor.getRhs());

                    auto resultMulOp = builder.create<arith::MulFOp>(loc, lhsTensor,
                                                            rhsScalar);

                    return resultMulOp;

        });

      // lowerOpToLoopsFIRFilter1(op, operands, rewriter, 
      //       [loc ] (OpBuilder &builder, ValueRange memRefOperands,
      //             ValueRange loopIvs) {
      //               //
      //               dsp::FIRFilterOpAdaptor gainAdaptor(memRefOperands);
      //               Value input0 = gainAdaptor.getLhs();

      //               // auto zeroValue = builder.create<arith::ConstantOp>(loc, builder.getF64Type(),
      //               //     builder.getFloatAttr(builder.getF64Type(), 0.0) );

      //               return input0;

      //   });

      return success();
    }


};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Delay operations
//===----------------------------------------------------------------------===//
struct DelayOpLowering: public ConversionPattern {
      DelayOpLowering(MLIRContext *ctx)
        : ConversionPattern(dsp::DelayOp::getOperationName(), 1 , ctx) {}

    LogicalResult 
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
              ConversionPatternRewriter &rewriter) const final {
      //dsp.DelayOp has 2 operands -- both of type tensor f64

      //Get the location of delayop
      auto loc = op->getLoc();
      
      //create arith.const operation with value 0 & type=f64 --
      // auto zeroValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
      //                   rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
      
      // llvm::errs() << "zeroValue() " << zeroValue.getType() << "\n";
      //get second operand of the DelayOp f64 & convert it to int
      //delay_2ndArg 
      // Value delay_2ndArg = operands[1];
      // Value delay_firstArg = operands[0];


      // auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
     
      // if(tensorType.getRank() > 1){
      //   llvm::errs() << "Only Vectors are supported -- not higher ranks\n";
      //   return mlir::failure();
      // }
      //Add check for delay_2ndArg shouldn't exceed lengthOfOperand0 
     
      // Insert an allocation and deallocation for the result of this operation.
      // auto memRefType = convertTensorToMemRef(tensorType);

      // auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

      //Create a nest of affine loops --
      //2 affine loops -- 
      //first from 0 to delay_2ndArg
      //          here, inside AffineNest
      //          create affine:load from the arith.const operation with value 0
      //          use affine:store to store at result_op at indx
      // 
      //2nd from delay_2ndArg to lengthOfOperand0 of delayOp 
      //          here, inside AffineNest
      //          create affine:load from input memref & indx = indx - delay_2ndArg 
      //          create affine:store at result_op indx

      //replace this operation with generate alloc

      // lowerOpToLoops2(op, operands, rewriter, 
      //       [loc ] (OpBuilder &builder, ValueRange memRefOperands,
      //             ValueRange loopIvs) {
      //               //
      //               dsp::DelayOpAdaptor delayAdaptor(memRefOperands);
      //               Value input0 = delayAdaptor.getLhs();

      //               auto zeroValue = builder.create<arith::ConstantOp>(loc, builder.getF64Type(),
      //                   builder.getFloatAttr(builder.getF64Type(), 0.0) );

      //               return zeroValue;

      //   });

        lowerOpToLoops3(op, operands, rewriter, 
            [loc ] (OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) {
                    //
                    dsp::DelayOpAdaptor delayAdaptor(memRefOperands);
                    // Value input0 = delayAdaptor.getLhs();

                    auto zeroValue = builder.create<arith::ConstantOp>(loc, builder.getF64Type(),
                        builder.getFloatAttr(builder.getF64Type(), 0.0) );

                    return zeroValue;

        });

      // auto processIteration = [loc](OpBuilder &builder, ValueRange memRefOperands,
      //                    ValueRange loopIvs) {
      //                // Generate an adaptor for the remapped operands of the
      //                // BinaryOp. This allows for using the nice named accessors
      //                // that are generated by the ODS.
      //                typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

      //                // Generate loads for the element of 'lhs' and 'rhs' at the
      //                // inner loop.
      //                auto loadedLhs = builder.create<affine::AffineLoadOp>(
      //                    loc, binaryAdaptor.getLhs(), loopIvs);
      //                auto loadedRhs = builder.create<affine::AffineLoadOp>(
      //                    loc, binaryAdaptor.getRhs(), loopIvs);

      //                // Create the binary operation performed on the loaded
      //                // values.
      //                return builder.create<LoweredBinaryOp>(loc, loadedLhs,
      //                                                       loadedRhs);
      //              }
      // affine::buildAffineLoopNest(
      //     rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      //   [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
      //     // Call the processing function with the rewriter, the memref operands,
      //     // and the loop induction variables. This function will return the value
      //     // to store at the current index.
      //     Value valueToStore = processIteration(nestedBuilder, operands, ivs);
      //     nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
      //                                                 ivs);
      //   });
      // rewriter.replaceOp(op, alloc);
      return success();
    }


};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Gain operations
//===----------------------------------------------------------------------===//
struct GainOpLowering: public ConversionPattern {
      GainOpLowering(MLIRContext *ctx)
        : ConversionPattern(dsp::GainOp::getOperationName(), 1 , ctx) {}

    LogicalResult 
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
              ConversionPatternRewriter &rewriter) const final {
      //dsp.GainOp has 2 operands -- both of type tensor f64 , 2ndOperand should have only 1 element

      //Get the location of GainOp
      auto loc = op->getLoc();
      
      //Pseudo-Code
      //Range for each element of the tensor
      //Load the element from the first argument
      // multiply with the 2nd argument
      // return the result to store into the respective index
        lowerOpToLoops(op, operands, rewriter, 
            [loc, op ] (OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                    dsp::GainOpAdaptor gainAdaptor(memRefOperands);

                    // Generate loads for the element of 'lhs' and 'rhs' at the
                    // inner loop.
                    // auto lhsTensor = delayAdaptor.getLhs();
                    auto lhsTensor = builder.create<affine::AffineLoadOp>(
                         loc, gainAdaptor.getLhs(), loopIvs);

                    // auto rhsScalar = op->getOperand(1);     
                    auto rhsScalar = builder.create<affine::AffineLoadOp>(
                         loc, gainAdaptor.getRhs());

                    auto resultMulOp = builder.create<arith::MulFOp>(loc, lhsTensor,
                                                            rhsScalar);

                    return resultMulOp;

        });

      // lowerOpToLoopsGain1(op, operands, rewriter, 
      //       [loc ] (OpBuilder &builder, ValueRange memRefOperands,
      //             ValueRange loopIvs) {
      //               //
      //               dsp::GainOpAdaptor gainAdaptor(memRefOperands);
      //               Value input0 = gainAdaptor.getLhs();

      //               // auto zeroValue = builder.create<arith::ConstantOp>(loc, builder.getF64Type(),
      //               //     builder.getFloatAttr(builder.getF64Type(), 0.0) );

      //               return input0;

      //   });

      return success();
    }


};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     // Generate loads for the element of 'lhs' and 'rhs' at the
                     // inner loop.
                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     // Create the binary operation performed on the loaded
                     // values.
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<dsp::AddOp, arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<dsp::SubOp, arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<dsp::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<dsp::ConstantOp> {
  using OpRewritePattern<dsp::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dsp::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<dsp::FuncOp> {
  using OpConversionPattern<dsp::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dsp::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-dsp function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<dsp::PrintOp> {
  using OpConversionPattern<dsp::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dsp::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "dsp.print" in this pass, but we need to update its
    // operands.
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<dsp::ReturnOp> {
  using OpRewritePattern<dsp::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dsp::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "dsp.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // TransposeOp. This allows for using the nice named
                     // accessors that are generated by the ODS.
                     dsp::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();

                     // Transpose the elements by generating a load from the
                     // reverse indices.
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the dsp operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `dsp.print`, as `legal`. `dsp.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<dsp::DspDialect>();
  target.addDynamicallyLegalOp<dsp::PrintOp>([](dsp::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering ,
               DelayOpLowering, GainOpLowering, SubOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::dsp::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

#pragma GCC diagnostic pop
// #pragma warning(pop)