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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

//For IntegerSet
#include "mlir/IR/IntegerSet.h"
#include <iostream>
using namespace mlir;
using namespace std;
using namespace affine;
using namespace dsp;
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
  alloc->moveBefore(&parentBlock->front()); //Abhinav-- move allock->block->front before alloc operation??

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as dsp functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back()); //move alloc->block->back before dealloc
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

        // Define an affine map: #map2 = affine_map<(d0) -> (d0 + 2)>
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

#define TryJustAffineLoop 0  //working
#define TryAffineForAndAffineIf 0  // working 
#define TryAffineIf2 0
#define TryAffineMap  0   //working basic -- TO do --try with symbols
#define TrySumOfVector 0  //Working
#define TryMultiDimLoop 0  //Working
#define TryFIRFilter 1 
#define TryMultiDimForAndIf 0 //
#define TryMultiDimLoopAndAffineMap 0  //Working
#define TryMultiDimLoopAndAffineSet 0  //Working
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

  //working
  // affine::buildAffineLoopNest(
  //     rewriter, loc, lowerBounds, tensorType.getShape(), steps,
  //     [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
  //       // Call the processing function with the rewriter, the memref operands,
  //       // and the loop induction variables. This function will return the value
  //       // to store at the current index.
  //       Value valueToStore = processIteration(nestedBuilder, operands, ivs);
  //       nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
  //                                                   ivs);
  //     });

    // affine::AffineForOp forOp = rewriter.create<affine::AffineForOp>(
      //   loc, lowerBounds, tensorType.getShape() , steps, ValueRange());
      // mlir::IntegerSet set1 = mlir::IntegerSet::get(1, 0, map, {true});

      //create an affineFor
      // affineFor It has one region containing its body & the region must contain a block terminating with affine.yield
      //block has argument of index type
      //

#if TryJustAffineLoop
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    //create AffineMap and set
    // %1 = affine.load 
    //  if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
    // AffineMap map = AffineMap::get(1, 0, dimExpr);
    // AffineMap map = AffineMap::get(1, 0 , rewriter.getAffineDimExpr(0) - 5);
    IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});
    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );

    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp1.getBody());

    //start adding operations like a arith::constant = 100.0 to the body of forOp1
      // Inside the loop body:
    
    Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                         rewriter.getF64FloatAttr(15));
    
    llvm::errs() << "LINE = " << __LINE__ << "\n";
    auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, constant15, alloc, forOp1.getInductionVar());

#endif 

#if TryAffineForAndAffineIf
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    //create AffineMap and set
    // %1 = affine.load 
    //  if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
    // AffineExpr dimExpr2 = rewriter
    // AffineMap map = AffineMap::get(1, 0, dimExpr);
    // AffineMap map = AffineMap::get(1, 0 , rewriter.getAffineDimExpr(0) - 5);
    IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});

     //affine.if %arg1 >= 0 and %5 <= %1 - 1
     // n-k >= 0 && n-k <= len -1 //n = %arg0 , k = %arg1
     // %arg0 >= 0 and %arg0 - %arg1 - %sym1 + 1 <= 0

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );

    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();
    //start adding operations like a arith::constant = 100.0 to the body of forOp1
      // Inside the loop body:

    // #set affine_set<(d0) : (d0 - 5 <= 0)>
    // affine.for %arg0 = 0 to 10 {
    //   %3 = affine.if #set (%arg0) {
    //         %1 = arith.const 25
    //         affine.yield %1
    //     }
    // else{
    //       %2 = arith.const 15
    //       affine.yield %2
    //   }
    //     affine.store %3, alloc[%arg0]
    // } 

    // auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv} , false /*no else*/ );
    // auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv} , true /*no else*/ );
    
    //use typeRange too:
    Type floatType = rewriter.getF64Type();
    auto ifOp = rewriter.create<affine::AffineIfOp>( loc, TypeRange{ floatType },set1 , ValueRange{iv} , true /*no else*/ );

    rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    
    FIRFilterOpAdaptor firFilterOperands(operands);

    //load from the input
    Value loadInput = rewriter.create<AffineLoadOp>(loc, firFilterOperands.getLhs(), iv);
    Value constant25 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                         rewriter.getF64FloatAttr(25));
    Value constsq25 = rewriter.create<arith::MulFOp>(loc, loadInput, constant25)  ;                                                   
    
    rewriter.create<AffineStoreOp>(loc, constsq25 , alloc, iv);
    rewriter.create<AffineYieldOp>(loc, ValueRange{constsq25});
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());

    rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    Value loadInput2 = rewriter.create<AffineLoadOp>(loc, firFilterOperands.getRhs(), iv);
    Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                         rewriter.getF64FloatAttr(15));
    Value elseResult = rewriter.create<arith::MulFOp>(loc, loadInput2, constant15)  ; 
    rewriter.create<AffineStoreOp>(loc, elseResult , alloc, iv);
    rewriter.create<AffineYieldOp>(loc, ValueRange{elseResult});
    // rewriter.setInsertionPointToEnd(ifOp.getElseBlock());
    rewriter.setInsertionPointAfter(ifOp);
    ifOp->dump();
    // forOp1->dump();
    rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);
    //getParentBlock then use 
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock()->getParentOp());
    // rewriter.setInsertionPointToEnd(ifOp->getBlock());
    // rewriter.setInsertionPoint(ifOp->getParentOp());
    // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());
    
    // rewriter.setInsertionPointAfter(ifOp);
    // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);
    
    //try to add the affine.If condition 
    //create affine.If , 
    // use integer set to represent the condition 
    //check the AffineArgs 
    // affine.if operation contains two regions for the “then” and “else” clauses
      //each region of affine.if must contain a single block with no args and terminated by affine.yield op
      // if affine.if defines no values --> no need for affine.yield
    
    // affineIf.setConditional(set1, forOp1.getInductionVar());
    //start then "block"
    // "then" block
    
    // Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(15));
     
    //  rewriter.create<affine::AffineYieldOp>(loc, ValueRange{constant15});
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());
    //else block
    // rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    
    // Set insertion point to the end of the "then" block
    // rewriter.setInsertionPointAfter(ifOp.getThenBlock()->getTerminator());
   

    // rewriter.create<affine::AffineYieldOp>(loc, constant25);
    llvm::errs() << "LINE = " << __LINE__ << "\n";
    //Back to parentOp -- ifOp stops here
    // rewriter.setInsertionPointAfter(ifOp);
    

    //also use affine::AffineStore to store at the loop induction variable
    // auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, ifOp.getResult(0), alloc, forOp1.getInductionVar());
    // auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, constant25, alloc, forOp1.getInductionVar());
    // Back to parentOp -- forOp1
    // rewriter.setInsertionPointAfter(storeOp);

    llvm::errs() << "LINE = " << __LINE__ << "  xx\n";
    //create affine yield for the loop
    // rewriter.create<affine::AffineYieldOp>(loc);  

#endif

#if TryAffineIf2

    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    //create AffineMap and set
    // %1 = affine.load 
    //  if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
    // AffineExpr dimExpr2 = rewriter
    // AffineMap map = AffineMap::get(1, 0, dimExpr);
    // AffineMap map = AffineMap::get(1, 0 , rewriter.getAffineDimExpr(0) - 5);
    IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});

     //affine.if %arg1 >= 0 and %5 <= %1 - 1
     // n-k >= 0 && n-k <= len -1 //n = %arg0 , k = %arg1
     // %arg0 >= 0 and %arg0 - %arg1 - %sym1 + 1 <= 0

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );

    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();
    //start adding operations like a arith::constant = 100.0 to the body of forOp1
      // Inside the loop body:

    // #set affine_set<(d0) : (d0 - 5 <= 0)>
    // affine.for %arg0 = 0 to 10 {
    //   %3 = affine.if #set (%arg0) {
    //         %1 = arith.const 25
    //         affine.yield %1
    //     }
    //     affine.store %3, alloc[%arg0]
    // } 

    // auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv} , false /*no else*/ );
    auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv} , true /*no else*/ );
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());
    Value constant25 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                         rewriter.getF64FloatAttr(25));
    Value constsq25 = rewriter.create<arith::MulFOp>(loc, constant25, constant25)  ;                                                   
    
    // ifOp.setR
    // rewriter.create<AffineStoreOp>(loc, constant25 , alloc, iv);
    // rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                         rewriter.getF64FloatAttr(15));
    rewriter.create<AffineStoreOp>(loc, constsq25 , alloc, iv);


    //getParentBlock then use 
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock()->getParentOp());
    // rewriter.setInsertionPointToEnd(ifOp->getBlock());
    rewriter.setInsertionPoint(ifOp->getParentOp());
    // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
    // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());
    
    // rewriter.setInsertionPointAfter(ifOp);
    // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);
    // rewriter.cre
    
#endif

#if TryAffineMap
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0] - 2;
    int64_t step = 1;

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );

    
    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();
    //start adding operations like a arith::constant = 100.0 to the body of forOp1
      // Inside the loop body:
        //create affine for
    // use affine-map expression for dimension then symbol then combination
    // affine-map expression for dimension: affine_map<d0, d1)[s0] -> (d0 , d1 + s0, d1 - s0)
    // use affine map 
    // Define an affine map: #map2 = affine_map<(d0) -> (d0 + 2)>
    auto symbol1 = tensorType.getShape()[0];
    AffineExpr indx = rewriter.getAffineDimExpr(0);
    AffineExpr constantExpr = rewriter.getAffineConstantExpr(2);
    AffineMap addMap = AffineMap::get(1, 0, symbol1 - indx);
    auto outputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap , iv);

    // Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(15));
    
    
    //try replace constant15 ie, with input & filter
    FIRFilterOpAdaptor firOpAdaptor(operands);

    Value inputForFilter = rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs() , iv);
    // Value inputForFilterMapped = rewriter.create<affine::AffineLoadOp>(loc,  firOpAdaptor.getLhs() , addMap, iv);

    Value impulseFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , iv);

    auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, inputForFilter,      alloc,ValueRange{outputIndex});

    
    llvm::errs() << "LINE = " << __LINE__ << "\n";

#endif

#if TrySumOfVector
    // here, we have to use iter
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0] ;
    int64_t step = 1;

    Value constant0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step , ValueRange{constant0} );

    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();
    

    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    // Initial sum set to 0.
        // %sum_0 = arith.constant 0.0 : f32
        // // iter_args binds initial values to the loop's region arguments.
        // %sum = affine.for %i = 0 to 10 step 1
        //     iter_args(%sum_iter = %sum_0) -> (f32) {
        //   %t = affine.load %buffer[%i] : memref<10xf32>
        //   %sum_next = arith.addf %sum_iter, %t : f32
        //   // Yield current iteration sum to next iteration %sum_iter or to %sum
        //   // if final iteration.
        //   affine.yield %sum_next : f32
        // }
        // return %sum : f32
        // }


      // Inside the loop body:

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

#endif

#if TryMultiDimLoop
    // here, we have to use iter
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0] ;
    int64_t step = 1;

    Value constant0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step  );

    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    //create loadOp
    FIRFilterOpAdaptor firOpAdaptor(operands);

    Value loadInput = rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs() , iv);

    //create another loop --
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step , ValueRange{loadInput} );

    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();
    Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , iv2);
    
    // get iterArg
    auto getIterArg =  forOp2.getBody()->getArgument(1);
    auto sumNext = rewriter.create<arith::AddFOp>(loc, loadInput, loadFilter);

    

    //store the result to output
    // rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    rewriter.setInsertionPointAfter(forOp2);
    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv );
    //
    //yield the 
    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    // Initial sum set to 0.
        // affine.for %arg0 = 0 to 10 {
        //   %1 = affine.load input[%arg0]
        //   %4 = affine.for %arg1 = 0 to 10 step 1 
        //     iter_args(%sum_iter = %1) {
        //       %2 = affine.load filter[%arg1]
        //       %3 = arith.add sum_iter , %2
        //         affine.yield %3 : f64
        //   }
        //   affine.store %4, output[%arg0]
        // }


      // Inside the loop body:

    
    llvm::errs() << "LINE = " << __LINE__ << "\n";


#endif

#if TryMultiDimForAndIf
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    //create AffineMap and set
    // %1 = affine.load 
    //  if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
  
     //affine.if %arg1 >= 0 and %5 <= %1 - 1
     // n-k >= 0 && n-k <= len -1 //n = %arg0 , k = %arg1
     // %arg0 >= 0 and %arg0 - %arg1 - %sym1 + 1 <= 0

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );

    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();
    //start adding operations like a arith::constant = 100.0 to the body of forOp1
      // Inside the loop body:

    AffineExpr dimExpr = rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
    IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});


    // create 2nd loop
    // use loop inductn variable for 2nd loop
    // use if condition on 2nd loop inductn variable
    // get the result of inner for loop and store at output 

    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step );
    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();
    AffineExpr dimExpr2 = rewriter.getAffineDimExpr(1) - rewriter.getAffineConstantExpr(6);
    IntegerSet set2 = IntegerSet::get(1, 0, {dimExpr,dimExpr2}, {false});

    auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set2 , ValueRange{iv} , false /*no else*/ );
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    Value constant25 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                         rewriter.getF64FloatAttr(25));
    Value resultFromInnerLoop = rewriter.create<arith::MulFOp>(loc, constant25 , constant25);

    // rewriter.setInsertionPointAfter(forOp2);
    // rewriter.setInsertionPointToEnd(forOp2->getBlock());
    // rewriter.create<AffineStoreOp>(loc, constant25 , alloc, iv2);
    // rewriter.create<AffineYieldOp>(loc, ValueRange{resultFromInnerLoop});
    // rewriter.setInsertionPointAfter(ifOp);
    // rewriter.create<AffineYieldOp>(loc, ValueRange{resultFromInnerLoop});
    // rewriter.setInsertionPointAfter(forOp2);
    rewriter.create<AffineStoreOp>(loc, constant25 , alloc, iv);
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

   

    // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
    // rewriter.setInsertionPointAfter(ifOp);
    // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);
    
    //try to add the affine.If condition 
    //create affine.If , 
    // use integer set to represent the condition 
    //check the AffineArgs 
    // affine.if operation contains two regions for the “then” and “else” clauses
      //each region of affine.if must contain a single block with no args and terminated by affine.yield op
      // if affine.if defines no values --> no need for affine.yield
    
    // affineIf.setConditional(set1, forOp1.getInductionVar());
    //start then "block"
    // "then" block
    
    // rewriter.create<affine::AffineYieldOp>(loc, constant25);
    llvm::errs() << "LINE = " << __LINE__ << "\n";
    //Back to parentOp -- ifOp stops here
    // rewriter.setInsertionPointAfter(ifOp);
    
    llvm::errs() << "LINE = " << __LINE__ << "  xx\n";

#endif

#if TryMultiDimLoopAndAffineMap
    // here, we have to use iter
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0] ;
    int64_t step = 1;

    Value constant0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step  );

    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    //create loadOp
    FIRFilterOpAdaptor firOpAdaptor(operands);

    Value loadInput = rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs() , iv);

    //create another loop --
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step , ValueRange{loadInput} );

    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();

    //Use AffineMap for affine.load alloc_9[%arg0 - %arg1]
    AffineExpr OuterIndx = rewriter.getAffineDimExpr(0);
    AffineExpr InnerIndx = rewriter.getAffineDimExpr(1);
    AffineMap addMap = AffineMap::get(2, 0, OuterIndx - InnerIndx);
    // auto outputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap , ValueRange{iv,iv2});

    // Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(15));
    

    // Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , addMap, ValueRange{iv2,iv});
    Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , addMap, ValueRange{iv,iv2});
    // Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , outputIndex);
    // get iterArg
    auto getIterArg =  forOp2.getBody()->getArgument(1);
    auto sumNext = rewriter.create<arith::AddFOp>(loc, getIterArg, loadFilter);

    

    //store the result to output
    // rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    rewriter.setInsertionPointAfter(forOp2);
    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv );
    //
    //yield the 
    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    // Initial sum set to 0.
        // affine.for %arg0 = 0 to 10 {
        //   %1 = affine.load input[%arg0]
        //   %4 = affine.for %arg1 = 0 to 10 step 1 
        //     iter_args(%sum_iter = %1) {
        //       %2 = affine.load filter[%arg1]
        //       %3 = arith.add sum_iter , %2
        //         affine.yield %3 : f64
        //   }
        //   affine.store %4, output[%arg0]
        // }


      // Inside the loop body:

    
    llvm::errs() << "LINE = " << __LINE__ << "\n";


#endif

#if TryMultiDimLoopAndAffineSet
    // here, we have to use iter
    int64_t lb = 0 ;
    int64_t ub = tensorType.getShape()[0] ;
    int64_t step = 1;

    Value constant0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step  );

    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    //create loadOp
    FIRFilterOpAdaptor firOpAdaptor(operands);

    Value loadInput = rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs() , iv);

    //create another loop --
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(loc, 
                lb, ub, step , ValueRange{loadInput} );

    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();

    //Use AffineMap for affine.load alloc_9[%arg0 - %arg1]
    AffineExpr OuterIndx = rewriter.getAffineDimExpr(0);
    AffineExpr InnerIndx = rewriter.getAffineDimExpr(1);
    AffineMap addMap = AffineMap::get(2, 0, OuterIndx - InnerIndx);
    auto outputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap , ValueRange{iv,iv2});

    // Value constant15 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(15));
    AffineExpr dimExpr = OuterIndx - InnerIndx;
    IntegerSet set1 = IntegerSet::get(2, 0, {dimExpr}, {false});

    auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv,iv2} , false /*no else*/ );
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    // Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , addMap, ValueRange{iv2,iv});
    Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs() , addMap, ValueRange{iv,iv2});
    // get iterArg
    auto getIterArg =  forOp2.getBody()->getArgument(1);
    auto sumNext = rewriter.create<arith::AddFOp>(loc, loadFilter, loadFilter);
    // rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});

    //store the result to output
    // rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    rewriter.setInsertionPointAfter(forOp2);
    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv );
    //
    //yield the 
    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    // Initial sum set to 0.
        // affine.for %arg0 = 0 to 10 {
        //   %1 = affine.load input[%arg0]
        //   %4 = affine.for %arg1 = 0 to 10 step 1 
        //     iter_args(%sum_iter = %1) {
        //       %2 = affine.load filter[%arg1]
        //       %3 = arith.add sum_iter , %2
        //         affine.yield %3 : f64
        //   }
        //   affine.store %4, output[%arg0]
        // }


      // Inside the loop body:

    
    llvm::errs() << "LINE = " << __LINE__ << "\n";


#endif

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

    

    //inside the forOp body --> create the operations & then close the body
    // OpBuilder::InsertionGuard guard(rewriter);
    
    //start adding operations like a arith::constant = 100.0 to the body of forOp1
      // Inside the loop body:


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

   

    // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
    // rewriter.setInsertionPointAfter(ifOp);
    // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);
    
    //try to add the affine.If condition 
    //create affine.If , 
    // use integer set to represent the condition 
    //check the AffineArgs 
    // affine.if operation contains two regions for the “then” and “else” clauses
      //each region of affine.if must contain a single block with no args and terminated by affine.yield op
      // if affine.if defines no values --> no need for affine.yield
    
    // affineIf.setConditional(set1, forOp1.getInductionVar());
    //start then "block"
    // "then" block
    
    // rewriter.create<affine::AffineYieldOp>(loc, constant25);
    llvm::errs() << "LINE = " << __LINE__ << "\n";
    //Back to parentOp -- ifOp stops here
    // rewriter.setInsertionPointAfter(ifOp);
    
    llvm::errs() << "LINE = " << __LINE__ << "  xx\n";



#endif
    // Terminate the loop body with affine.yield.
    // rewriter.create<affine::AffineYieldOp>(loc);


  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}



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
        
        lowerOpToLoopsFIR(op, operands, rewriter, 
            [loc, op ] (OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) {
                  // ValueRange loopIvs) {
                     
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
                         loc, firFilterAdaptor.getRhs(), loopIvs);

                    auto resultMulOp = builder.create<arith::MulFOp>(loc, lhsTensor,
                                                            rhsScalar);

                    return resultMulOp;

        });

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
    rewriter.modifyOpInPlace(op,
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
               DelayOpLowering, GainOpLowering, SubOpLowering, FIRFilterOpLowering>(
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
