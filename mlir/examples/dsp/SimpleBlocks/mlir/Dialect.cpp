//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "toy/Dialect.h"
#include "toy/DebugConfig.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::dsp;
using namespace std;

#include "toy/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within dsp can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within dsp can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within dsp can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(dsp.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// DspDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void DspDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

// void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                        int value) {
//   auto dataType = RankedTensorType::get({}, builder.getI64Type());
//   auto dataAttribute = DenseElementsAttr::get(dataType, value);
//   ConstantOp::build(builder, state, dataType, dataAttribute);
// }

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
mlir::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult DivOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void DivOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the DivOp, this is required by the shape inference
/// interface.
void DivOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

mlir::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//
// void DelayOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                          mlir::Value lhs, unsigned rhs){
void DelayOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value lhs, mlir::Value rhs){    
    //
    // state.addTypes(UnrankedTensorType::get(builder.getF64Type()), builder.getI32Type());
    state.addTypes(UnrankedTensorType::get(builder.getF64Type())); //working
    state.addOperands({lhs, rhs});
    // state.addOperands(value);

 }

 mlir::LogicalResult DelayOp::verify(){
    // auto inputType1 = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
    // auto inputType2 = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
    // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
    // if(!inputType || !resultType)
    //   return mlir::success();

    return mlir::success();
 }

// void DelayOp::inferShapes() { getResult().setType(getOperand(0).getType()) ;}
//getLHS defined with Operation as :
//   fro addOp 
//     ::mlir::TypedValue<::mlir::TensorType> AddOp::getLhs() {
//   return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
// }
void DelayOp::inferShapes() { getResult().setType(getLhs().getType()) ;}


//===----------------------------------------------------------------------===//
// GainOp
//===----------------------------------------------------------------------===//
// void GainOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                          mlir::Value lhs, unsigned rhs){
// void GainOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                          mlir::Value lhs, mlir::Float64Type rhs){    
void GainOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value lhs, mlir::Value rhs){ 
    // state.addTypes(UnrankedTensorType::get(builder.getF64Type()), builder.getI32Type());
    // state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    // state.addTypes({UnrankedTensorType::get(builder.getF64Type()), builder.getF64Type()}); //working
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
    // state.addOperands({rhs});
    // state.addTypes();
    // state.addAttribute("rhs", rhs);
    // state.addAttribute("rhs", builder.getF64FloatAttr(builder.getF64Type()));
    // state.addAttribute("rhs", builder.getF64Type());
    // state.addAttribute("rhs", builder.getFloatAttr(builder.getF64Type() , rhs));
    // state.addOperands(value);
 }

//  mlir::LogicalResult GainOp::verify(){
//     auto inputType1 = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
//     auto inputType2 = llvm::dyn_cast<Float64Type>(getOperand(1).getType());
//     // auto inputType2 = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
//     // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
//     // if(!inputType || !resultType)
//     //   return mlir::success();

//     return mlir::success();
//  }

// void GainOp::inferShapes() { getResult().setType(getOperand(0).getType()) ;}
//getLHS defined with Operation as :
//   fro addOp 
//     ::mlir::TypedValue<::mlir::TensorType> AddOp::getLhs() {
//   return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
// }
void GainOp::inferShapes() { getResult().setType(getLhs().getType()) ;}

//===----------------------------------------------------------------------===//
 // SubOp
 //===----------------------------------------------------------------------===//

 void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value lhs, mlir::Value rhs) {
   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
   state.addOperands({lhs, rhs});
 }

 // mlir::ParseResult SubOp::parse(mlir::OpAsmParser &parser,
 //                                mlir::OperationState &result) {
 //   return parseBinaryOp(parser, result);
 // }

 // void SubOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

 /// Infer the output shape of the SubOp, this is required by the shape inference
 /// interface.
 void SubOp::inferShapes() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// zeroCrossCountOp
//===----------------------------------------------------------------------===//

void zeroCrossCountOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  // state.addTypes(builder.getF64Type()));
  // state.addTypes(builder.getI64Type());
  state.addOperands({lhs});
}

/// Infer the output shape of the zeroCrossCountOp, this is required by the shape inference
 /// interface.
 void zeroCrossCountOp::inferShapes() { getResult().setType(getLhs().getType()); }


//===----------------------------------------------------------------------===//
// FIRFilterResponseOp
//===----------------------------------------------------------------------===//

void FIRFilterResponseOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}



/// Infer the output shape of the FIRFilterResponseOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of Lhs + Rhs - 1
void FIRFilterResponseOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getLhs().getType();
  auto shapeOfInput = tensorInput.getShape();

  auto tensorFilter = getRhs().getType();
  auto shapeOfFilter = tensorFilter.getShape();
  std::vector<int64_t> shapeForOutput ;

  for(size_t i=0; i < shapeOfInput.size() ; i++){
    shapeForOutput.push_back(shapeOfInput[i] + shapeOfFilter[i] - 1);
  }
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getLhs().getType().getElementType());

  // getResult().setType(getLhs().getType()); 
  getResult().setType(manipulatedType);
  }

//get rank of Input & Filter -- make sure it is of rank 1 
mlir::LogicalResult FIRFilterResponseOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto filterType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto filterRank = filterType.getRank();

  // if( inputRank != 1 || filterRank != 1)
  // {
  //   return emitError()
  //          << "expected rank of input & filter is 1";
  // }

  return mlir::success();
} 


//===----------------------------------------------------------------------===//
// SlidingWindowAvgOp
//===----------------------------------------------------------------------===//

void SlidingWindowAvgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void SlidingWindowAvgOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size - 2
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());

  auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput;

  //Iterate for each rank : tensor<1x2x3x2> = rank 4
  for(size_t i=0; i < shapeOfInput.size() ; i++){
    shapeForOutput.push_back(shapeOfInput[i] - 2);
  }

  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getInput().getType().getElementType());
    // getOperand().getType());
    // getOperand().getType().getElementType());

  getResult().setType(outputType);

}

mlir::LogicalResult SlidingWindowAvgOp::verify() {
  // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // if(!inputType)
  // {
  //   llvm::errs() << "SlidingWindowAvgOp failed --\n";
  //   return failure();
  // }
  // auto shapeOfInput = inputType.getShape();

  // for(size_t i=0; i < shapeOfInput.size() ; i++){
  //   if(shapeOfInput[i] < 3){
  //     llvm::errs() << "Warning:SlidingWindowAvgOp = Input size < 3 " << "size= " << shapeOfInput[i] << "\n"  ;
  //   }
  // }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DownsamplingOp
//===----------------------------------------------------------------------===//

void DownsamplingOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}



/// Infer the output shape of the DownsamplingOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of Lhs + Rhs - 1
void DownsamplingOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getLhs().getType();
  auto shapeOfInput = tensorInput.getShape();

  // auto tensorDownsampling = getRhs().getType(); 
  // auto shapeOfDownsampling = tensorDownsampling.getShape(); //shape is the dimension
  

  std::vector<int64_t> shapeForOutput ;

  int64_t SecondValueInt = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  Value downsampling2ndArg = getOperand(1);
  dsp::ConstantOp constantOp2ndArg = downsampling2ndArg.getDefiningOp<dsp::ConstantOp>();
  DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();;
  auto elements = constantRhsValue.getValues<FloatAttr>();
  float SecondValue = elements[0].getValueAsDouble();
  SecondValueInt = (int64_t) SecondValue;
  // llvm::errs() << "Downsampling: SamplingRate: " << SecondValueInt << " \n"; //downsamplingRate
    

  for(size_t i=0; i < shapeOfInput.size() ; i++){
    double GetLenForOutput  = static_cast<double>(shapeOfInput[i] )/ SecondValueInt ;
    if(fmod(GetLenForOutput, 1.0) != 0) {
      //if remainder remains
      GetLenForOutput = ceil(GetLenForOutput);
    }
    int64_t OutlenInt = static_cast<int64_t> (GetLenForOutput);
    llvm::errs() << "Downsampling: OutlenInt: " << OutlenInt << " \n";
    shapeForOutput.push_back(OutlenInt);
  }
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getLhs().getType().getElementType());

  // getResult().setType(getLhs().getType()); 
  getResult().setType(manipulatedType);
  }

//get rank of Input & Downsampling -- make sure it is of rank 1 
mlir::LogicalResult DownsamplingOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto samplingRateType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto samplingRateRank = samplingRateType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " samplingRateRank: " << samplingRateRank << "\n";
  // //once ensured only 1 rank from above -- also make sure there is just 1 elem  
  // if( inputRank != 1 || samplingRateRank != 0 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank << " samplingRateRank: " << samplingRateRank << "\n";
  //   return emitError()
  //          << "expected rank of input & Downsampling is 1";
  // }
  return mlir::success();
} 

//===----------------------------------------------------------------------===//
// UpsamplingOp
//===----------------------------------------------------------------------===//

void UpsamplingOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}



/// Infer the output shape of the UpsamplingOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of input * UpsamplingRate ie, Rhs
void UpsamplingOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getLhs().getType();
  auto shapeOfInput = tensorInput.getShape();

  // auto tensorUpsampling = getRhs().getType(); 
  // auto shapeOfUpsampling = tensorUpsampling.getShape(); //shape is the length
  

  std::vector<int64_t> shapeForOutput ;

  int64_t SecondValueInt = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  Value upsampling2ndArg = getOperand(1);
  dsp::ConstantOp constantOp2ndArg = upsampling2ndArg.getDefiningOp<dsp::ConstantOp>();
  DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();;
  auto elements = constantRhsValue.getValues<FloatAttr>();
  float SecondValue = elements[0].getValueAsDouble();
  SecondValueInt = (int64_t) SecondValue;
  // llvm::errs() << "Upsampling: SamplingRate: " << SecondValueInt << " \n"; //downsamplingRate
    

  for(size_t i=0; i < shapeOfInput.size() ; i++){
    double GetLenForOutput  = static_cast<double>(shapeOfInput[i] ) * SecondValueInt ;
    int64_t OutlenInt = static_cast<int64_t> (GetLenForOutput);
    llvm::errs() << "Upsampling: OutlenInt: " << OutlenInt << " \n";
    shapeForOutput.push_back(OutlenInt);
  }
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getLhs().getType().getElementType());

  // getResult().setType(getLhs().getType()); 
  getResult().setType(manipulatedType);
  }

//get rank of Input & Upsampling -- make sure it is of rank 1 
mlir::LogicalResult UpsamplingOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto samplingRateType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto samplingRateRank = samplingRateType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " samplingRateRank: " << samplingRateRank << "\n";
  // //once ensured only 1 rank from above -- also make sure there is just 1 elem  
  // if( inputRank != 1 || samplingRateRank != 0 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank << " samplingRateRank: " << samplingRateRank << "\n";
  //   return emitError()
  //          << "expected rank of input is 1 & Upsampling is 0";
  // }
  return mlir::success();
} 


//===----------------------------------------------------------------------===//
// LowPassFilter1stOrderOp
//===----------------------------------------------------------------------===//

void LowPassFilter1stOrderOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}



/// Infer the output shape of the LowPassFilter1stOrderOp, this is required by the shape inference
/// interface.
void LowPassFilter1stOrderOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getLhs().getType(); 
  getResult().setType(tensorInput);
}

//get rank of Input & alphaValue -- make sure it is of rank 1 
mlir::LogicalResult LowPassFilter1stOrderOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto alphaValueType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto alphaValueRank = alphaValueType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above -- also make sure there is just 1 elem  
  // if( inputRank != 1 || alphaValueRank != 0 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  //   return emitError()
  //          << "expected rank of input & Upsampling is 1";
  // }
  return mlir::success();
} 

//===----------------------------------------------------------------------===//
// HighPassFilterOp
//===----------------------------------------------------------------------===//

void HighPassFilterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void HighPassFilterOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  getResult().setType(tensorInput);

}

mlir::LogicalResult HighPassFilterOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // llvm::errs() << "inputRank: " << inputRank <<  "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// FFT1DOp
//===----------------------------------------------------------------------===//

void FFT1DOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type()), 
                UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands(value);
  DEBUG_PRINT_NO_ARGS() ;
}

void FFT1DOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  // getResult().setType(tensorInput);
  getResult(0).setType(tensorInput);
  getResult(1).setType(tensorInput);
  // getResult(2).setType(tensorInput);
}

mlir::LogicalResult FFT1DOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// IFFT1DOp
//===----------------------------------------------------------------------===//

void IFFT1DOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value real, mlir::Value img) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({real , img});
  DEBUG_PRINT_NO_ARGS() ;
}

void IFFT1DOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getReal().getType(); 
  getResult().setType(tensorInput);
  // getResult(0).setType(tensorInput);
  // getResult(1).setType(tensorInput);
}

mlir::LogicalResult IFFT1DOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// HammingWindowOp
//===----------------------------------------------------------------------===//

void HammingWindowOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void HammingWindowOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  int64_t FirstOpInt = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
  Value hammingLen = getOperand();
  dsp::ConstantOp constantOp1stArg = hammingLen.getDefiningOp<dsp::ConstantOp>();
  // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float FirstValue = elements[0].getValueAsDouble();
  FirstOpInt = (int64_t) FirstValue;
  // llvm::errs() << "FirstOpInt " << FirstOpInt << "\n" ;
  // llvm::errs() << "shapeOfInput.size() " << shapeOfInput.size() << "\n" ;

  // for(size_t i=0; i < shapeOfInput.size() ; i++){
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    shapeForOutput.push_back(FirstOpInt);
  // }

  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getInput().getType().getElementType());
    // getOperand().getType());
    // getOperand().getType().getElementType());

  getResult().setType(outputType);
  // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
  

}

mlir::LogicalResult HammingWindowOp::verify() {
  // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // if(!inputType)
  // {
  //   llvm::errs() << "HammingWindowOp failed --\n";
  //   return failure();
  // }
  // auto shapeOfInput = inputType.getShape();

  // for(size_t i=0; i < shapeOfInput.size() ; i++){
  //   if(shapeOfInput[i] < 3){
  //     llvm::errs() << "Warning:HammingWindowOp = Input size < 3 " << "size= " << shapeOfInput[i] << "\n"  ;
  //   }
  // }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DCTOp
//===----------------------------------------------------------------------===//

void DCTOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  // DEBUG_PRINT_NO_ARGS() ;
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
  // DEBUG_PRINT_NO_ARGS() ;
}

void DCTOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  getResult().setType(tensorInput);
  // getResult(0).setType(tensorInput);
  // getResult(1).setType(tensorInput);
}

mlir::LogicalResult DCTOp::verify() {
  // DEBUG_PRINT_NO_ARGS() ;
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto inputRank = inputType.getRank();

  // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  //once ensured only 1 rank from above --   
  if( inputRank != 1 )
  {
    llvm::errs() << "inputRank: " << inputRank <<  "\n";
    return emitError()
           << "expected rank of input  is 1";
  }
  return mlir::success();
}



//===----------------------------------------------------------------------===//
// filterOp
//===----------------------------------------------------------------------===//

void filterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value b, mlir::Value a, mlir::Value x) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({b, a, x});
}



/// Infer the output shape of the filterOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of Lhs + Rhs - 1
void filterOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getX().getType();
  getResult().setType(tensorInput );
  }

//get rank of Input & Filter -- make sure it is of rank 1 
mlir::LogicalResult filterOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto filterType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto filterRank = filterType.getRank();

  // if( inputRank != 1 || filterRank != 1)
  // {
  //   return emitError()
  //          << "expected rank of input & filter is 1";
  // }

  return mlir::success();
} 


//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

void SumOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void SumOp::inferShapes() {
  // auto tensorInput =  getInput().getType();
  // auto shapeOfInput = tensorInput.getShape();
  std::vector<int64_t> shapeForOutput;

  shapeForOutput.push_back(1);

  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getInput().getType().getElementType());
  getResult().setType(manipulatedType);
}

mlir::LogicalResult SumOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  // if (!inputType || !resultType)
  //   return mlir::success();

  // auto inputShape = inputType.getShape();
  // if (!std::equal(inputShape.begin(), inputShape.end(),
  //                 resultType.getShape().rbegin())) {
  //   return emitError()
  //          << "expected result shape to be a transpose of the input";
  // }
  return mlir::success();
}

 //===----------------------------------------------------------------------===//
 // CosOp
 //===----------------------------------------------------------------------===//

 void CosOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value value) {
   // DEBUG_PRINT_NO_ARGS() ;
   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
   state.addOperands(value);
   // DEBUG_PRINT_NO_ARGS() ;
 }

 void CosOp::inferShapes() {
   //for each rank
   //Get the shape/size of input 
   //output size = input_size 
   auto tensorInput =  getInput().getType(); 
   getResult().setType(tensorInput);
   // getResult(0).setType(tensorInput);
   // getResult(1).setType(tensorInput);
 }

 mlir::LogicalResult CosOp::verify() {
   // DEBUG_PRINT_NO_ARGS() ;
  //  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  //  auto inputRank = inputType.getRank();

  //  // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  //  //once ensured only 1 rank from above --   
  //  if( inputRank != 1 )
  //  {
  //    llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //    return emitError()
  //           << "expected rank of input  is 1";
  //  }
   return mlir::success();
 }

 //===----------------------------------------------------------------------===//
 // SinOp
 //===----------------------------------------------------------------------===//

 void SinOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value value) {
   // DEBUG_PRINT_NO_ARGS() ;
   state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
   state.addOperands(value);
   // DEBUG_PRINT_NO_ARGS() ;
 }

 void SinOp::inferShapes() {
   //for each rank
   //Get the shape/size of input 
   //output size = input_size 
   auto tensorInput =  getInput().getType(); 
   getResult().setType(tensorInput);
   // getResult(0).setType(tensorInput);
   // getResult(1).setType(tensorInput);
 }

 mlir::LogicalResult SinOp::verify() {
   // DEBUG_PRINT_NO_ARGS() ;
  //  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  //  auto inputRank = inputType.getRank();

  //  // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  //  //once ensured only 1 rank from above --   
  //  if( inputRank != 1 )
  //  {
  //    llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //    return emitError()
  //           << "expected rank of input  is 1";
  //  }
   return mlir::success();
 }

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

void SquareOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void SquareOp::inferShapes() {
  auto tensorInput =  getInput().getType();
  // mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          // getInput().getType().getElementType());
  getResult().setType(tensorInput);
}

mlir::LogicalResult SquareOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  // if (!inputType || !resultType)
  //   return mlir::success();

  // auto inputShape = inputType.getShape();
  // if (!std::equal(inputShape.begin(), inputShape.end(),
  //                 resultType.getShape().rbegin())) {
  //   return emitError()
  //          << "expected result shape to be a transpose of the input";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FFT1DRealOp
//===----------------------------------------------------------------------===//

void FFT1DRealOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands(value);
  DEBUG_PRINT_NO_ARGS() ;
}

void FFT1DRealOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  // getResult().setType(tensorInput);
  getResult().setType(tensorInput);
  // getResult(2).setType(tensorInput);
}

mlir::LogicalResult FFT1DRealOp::verify() {
  DEBUG_PRINT_NO_ARGS();
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FFT1DImgOp
//===----------------------------------------------------------------------===//

void FFT1DImgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands(value);
  DEBUG_PRINT_NO_ARGS() ;
}

void FFT1DImgOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  // getResult().setType(tensorInput);
  getResult().setType(tensorInput);
  // getResult(2).setType(tensorInput);
}

mlir::LogicalResult FFT1DImgOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SincOp
//===----------------------------------------------------------------------===//

void SincOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value wc, mlir::Value n) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({wc, n});
  DEBUG_PRINT_NO_ARGS() ;
}

void SincOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getN().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  int64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (int64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);

  shapeForOutput.push_back(GetLen);
  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getWc().getType().getElementType());


  getResult().setType(outputType);

}

mlir::LogicalResult SincOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetElemAtIndxOp
//===----------------------------------------------------------------------===//

void GetElemAtIndxOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input, mlir::Value indx) {
  DEBUG_PRINT_NO_ARGS();
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({input, indx} );
  DEBUG_PRINT_NO_ARGS();
}

void GetElemAtIndxOp::inferShapes() {
  // auto tensorInput =  getInput().getType();
  // auto shapeOfInput = tensorInput.getShape();
  std::vector<int64_t> shapeForOutput;
  DEBUG_PRINT_NO_ARGS();
  shapeForOutput.push_back(1);

  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getInput().getType().getElementType());
  getResult().setType(manipulatedType);
  DEBUG_PRINT_NO_ARGS();
}

mlir::LogicalResult GetElemAtIndxOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  // if (!inputType || !resultType)
  //   return mlir::success();

  // auto inputShape = inputType.getShape();
  // if (!std::equal(inputShape.begin(), inputShape.end(),
  //                 resultType.getShape().rbegin())) {
  //   return emitError()
  //          << "expected result shape to be a transpose of the input";
  // }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// SetElemAtIndxOp
//===----------------------------------------------------------------------===//

void SetElemAtIndxOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input, mlir::Value indx, mlir::Value val) {
  DEBUG_PRINT_NO_ARGS();
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({input, indx, val} );
  DEBUG_PRINT_NO_ARGS();
}

void SetElemAtIndxOp::inferShapes() {
  // auto tensorInput =  getInput().getType();
  // auto shapeOfInput = tensorInput.getShape();
  std::vector<int64_t> shapeForOutput;
  DEBUG_PRINT_NO_ARGS();
  shapeForOutput.push_back(1);

  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getInput().getType().getElementType());
  getResult().setType(manipulatedType);
  DEBUG_PRINT_NO_ARGS();
}

mlir::LogicalResult SetElemAtIndxOp::verify() {
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LowPassFIRFilterOp
//===----------------------------------------------------------------------===//

void LowPassFIRFilterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value wc, mlir::Value n) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({wc, n});
  DEBUG_PRINT_NO_ARGS() ;
}

void LowPassFIRFilterOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getN().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  uint64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (uint64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);
  
  //int64_t N = tensorType.getShape()[0];  


  shapeForOutput.push_back(GetLen);
  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getWc().getType().getElementType());


  getResult().setType(outputType);

}

mlir::LogicalResult LowPassFIRFilterOp::verify() {
  uint64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (uint64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);
  
  //filter-order even not supported -- so making it odd
  if(GetLen % 2 == 0 )
  {
    // GetLen = GetLen + 1;
    llvm::errs() << "N for lowPassFilter must be odd but is " << GetLen << "\n";
    // DEBUG_PRINT_WITH_ARGS("Making LowPassFilterLen Odd= " , GetLen); 
    return mlir::failure(); 
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LMSFilterOp
//===----------------------------------------------------------------------===//

void LMSFilterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value lhs, mlir::Value rhs, mlir::Value mu, mlir::Value filterLen, mlir::Value iters) {
  
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
   state.addOperands({lhs, rhs, mu, filterLen, iters});
}


void LMSFilterOp::inferShapes() { getResult().setType(getLhs().getType()); }

mlir::LogicalResult LMSFilterOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto filterType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto filterRank = filterType.getRank();

  // if( inputRank != 1 || filterRank != 1)
  // {
  //   return emitError()
  //          << "expected rank of input & filter is 1";
  // }

  return mlir::success();
}


//===----------------------------------------------------------------------===//
// HighPassFIRFilterOp
//===----------------------------------------------------------------------===//

void HighPassFIRFilterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value wc, mlir::Value n) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({wc, n});
  DEBUG_PRINT_NO_ARGS() ;
}

void HighPassFIRFilterOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getN().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  int64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (int64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);

  shapeForOutput.push_back(GetLen);
  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getWc().getType().getElementType());


  getResult().setType(outputType);

}

mlir::LogicalResult HighPassFIRFilterOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GetRangeOfVectorOp
//===----------------------------------------------------------------------===//

void GetRangeOfVectorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value first, mlir::Value N, mlir::Value step) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({first, N, step});
  DEBUG_PRINT_NO_ARGS() ;
}

void GetRangeOfVectorOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getN().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  int64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (int64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);

  shapeForOutput.push_back(GetLen);
  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getFirst().getType().getElementType());


  getResult().setType(outputType);

}

mlir::LogicalResult GetRangeOfVectorOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FIRFilterHammingOptimizedOp
//===----------------------------------------------------------------------===//

void FIRFilterHammingOptimizedOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value wc, mlir::Value n) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({wc, n});
  DEBUG_PRINT_NO_ARGS() ;
}

void FIRFilterHammingOptimizedOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getN().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  uint64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (uint64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);
  
  //int64_t N = tensorType.getShape()[0];  


  shapeForOutput.push_back(GetLen);
  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getWc().getType().getElementType());


  getResult().setType(outputType);

}

mlir::LogicalResult FIRFilterHammingOptimizedOp::verify() {
  uint64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (uint64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);
  
  //filter-order even not supported -- so making it odd
  if(GetLen % 2 == 0 )
  {
    // GetLen = GetLen + 1;
    llvm::errs() << "N for lowPassFilter must be odd but is " << GetLen << "\n";
    // DEBUG_PRINT_WITH_ARGS("Making LowPassFilterLen Odd= " , GetLen); 
    return mlir::failure(); 
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// HighPassFIRHammingOptimizedOp
//===----------------------------------------------------------------------===//

void HighPassFIRHammingOptimizedOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value wc, mlir::Value n) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({wc, n});
  DEBUG_PRINT_NO_ARGS() ;
}

void HighPassFIRHammingOptimizedOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getN().getType());

  // auto shapeOfInput = inputType.getShape();

  std::vector<int64_t> shapeForOutput ;

  uint64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (uint64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);
  
  //int64_t N = tensorType.getShape()[0];  


  shapeForOutput.push_back(GetLen);
  mlir::TensorType outputType = mlir::RankedTensorType::get(shapeForOutput, 
    getWc().getType().getElementType());


  getResult().setType(outputType);

}

mlir::LogicalResult HighPassFIRHammingOptimizedOp::verify() {
  uint64_t GetLen = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value inputLen = getOperand(1);
  dsp::ConstantOp constantOp1stArg = inputLen.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float LenN = elements[0].getValueAsDouble();
  GetLen = (uint64_t) LenN;
  DEBUG_PRINT_WITH_ARGS(GetLen);
  DEBUG_PRINT_WITH_ARGS("GetLen= " , GetLen);
  
  //filter-order even not supported -- so making it odd
  if(GetLen % 2 == 0 )
  {
    // GetLen = GetLen + 1;
    llvm::errs() << "N for lowPassFilter must be odd but is " << GetLen << "\n";
    // DEBUG_PRINT_WITH_ARGS("Making LowPassFilterLen Odd= " , GetLen); 
    return mlir::failure(); 
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// ThresholdOp
//===----------------------------------------------------------------------===//

void ThresholdOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input, mlir::Value threshld) {
  DEBUG_PRINT_NO_ARGS();
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({input, threshld} );
  DEBUG_PRINT_NO_ARGS();
}

void ThresholdOp::inferShapes() {
  DEBUG_PRINT_NO_ARGS();
  auto tensorInput =  getInput().getType();
  getResult().setType(tensorInput);
  DEBUG_PRINT_NO_ARGS();
}

mlir::LogicalResult ThresholdOp::verify() {
  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value threshold = getOperand(1);
  dsp::ConstantOp constantOp1stArg = threshold.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float GetThresholdVal = elements[0].getValueAsDouble();
  
  DEBUG_PRINT_WITH_ARGS(GetThresholdVal);
  DEBUG_PRINT_WITH_ARGS("GetThresholdVal= " , GetThresholdVal);
  
  //filter-order even not supported -- so making it odd
  if(GetThresholdVal <= 0 )
  {
    // GetThresholdVal = GetThresholdVal + 1;
    llvm::errs() << "threshold value must be >= 0 but got: " << GetThresholdVal << "\n";
    // DEBUG_PRINT_WITH_ARGS("Making LowPassFilterLen Odd= " , GetThresholdVal); 
    return mlir::failure(); 
  }
  return mlir::success();

}

//===----------------------------------------------------------------------===//
// QuantizationOp
//===----------------------------------------------------------------------===//

void QuantizationOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input, mlir::Value nLevels, mlir::Value max, mlir::Value min) {
  DEBUG_PRINT_NO_ARGS();
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({input, nLevels, max, min} );
  DEBUG_PRINT_NO_ARGS();
}

void QuantizationOp::inferShapes() {
  DEBUG_PRINT_NO_ARGS();
  auto tensorInput =  getInput().getType();
  getResult().setType(tensorInput);
  DEBUG_PRINT_NO_ARGS();
}

mlir::LogicalResult QuantizationOp::verify() {
  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  // DEBUG_PRINT_NO_ARGS();
  // check max > min && NoOfLevels = powerOf2 

  Value maxOperand = getOperand(2);
  dsp::ConstantOp constantOp1stArg = maxOperand.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantLhsValue = constantOp1stArg.getValue();
  auto elements = constantLhsValue.getValues<FloatAttr>();
  float getMax = elements[0].getValueAsDouble();

  Value minOperand = getOperand(3);
  constantOp1stArg = minOperand.getDefiningOp<dsp::ConstantOp>();

  if(!constantOp1stArg){
    llvm::errs() << "QuantizationOp: unable to get Constant for minOp -- 4th opernad " << "\n";
    return mlir::failure(); 
  }
  DEBUG_PRINT_NO_ARGS();
  constantLhsValue = constantOp1stArg.getValue();
  elements = constantLhsValue.getValues<FloatAttr>();
  float getMin = elements[0].getValueAsDouble();

  if(getMax < getMin){
    llvm::errs() << "QuantizatnOp : Max < Min --" << " Max: " << getMax ;
    llvm::errs() << " Min: " << getMin ;
    return mlir::failure();
  }
  

  return mlir::success();

}


//===----------------------------------------------------------------------===//
// LMSFilterResponseOp
//===----------------------------------------------------------------------===//

void LMSFilterResponseOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value lhs, mlir::Value rhs, mlir::Value mu, mlir::Value filterLen) {
  
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
   state.addOperands({lhs, rhs, mu, filterLen});
}


void LMSFilterResponseOp::inferShapes() { getResult().setType(getLhs().getType()); }

mlir::LogicalResult LMSFilterResponseOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto filterType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto filterRank = filterType.getRank();

  // if( inputRank != 1 || filterRank != 1)
  // {
  //   return emitError()
  //          << "expected rank of input & filter is 1";
  // }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RunLenEncodingOp
//===----------------------------------------------------------------------===//

void RunLenEncodingOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input) {
  DEBUG_PRINT_NO_ARGS();
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({input} );
  DEBUG_PRINT_NO_ARGS();
}

void RunLenEncodingOp::inferShapes() {
  DEBUG_PRINT_NO_ARGS();
  auto tensorInput =  getInput().getType();
  auto shapeOfInput = tensorInput.getShape();

  // auto tensorUpsampling = getRhs().getType(); 
  // auto shapeOfUpsampling = tensorUpsampling.getShape(); //shape is the length
  //Assume rank is 1 , then get the shape of output
  // shapeOfInput

  std::vector<int64_t> shapeForOutput ;

  int64_t LengthOfInput = shapeOfInput[0];
  int64_t lenOfOutput = 2 * LengthOfInput;
  shapeForOutput.push_back(lenOfOutput);
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getInput().getType().getElementType());

  getResult().setType(manipulatedType);
  DEBUG_PRINT_NO_ARGS();
}

mlir::LogicalResult RunLenEncodingOp::verify() {
  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  // DEBUG_PRINT_NO_ARGS();
  
  return mlir::success();

}

//===----------------------------------------------------------------------===//
// FIRFilterResSymmOptimizedOp
//===----------------------------------------------------------------------===//

void FIRFilterResSymmOptimizedOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}



/// Infer the output shape of the FIRFilterResSymmOptimizedOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of Lhs + Rhs - 1
void FIRFilterResSymmOptimizedOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getLhs().getType();
  auto shapeOfInput = tensorInput.getShape();

  auto tensorFilter = getRhs().getType();
  auto shapeOfFilter = tensorFilter.getShape();
  std::vector<int64_t> shapeForOutput ;

  for(size_t i=0; i < shapeOfInput.size() ; i++){
    shapeForOutput.push_back(shapeOfInput[i] + shapeOfFilter[i] - 1);
  }
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getLhs().getType().getElementType());

  // getResult().setType(getLhs().getType()); 
  getResult().setType(manipulatedType);
}

//get rank of Input & Filter -- make sure it is of rank 1 
mlir::LogicalResult FIRFilterResSymmOptimizedOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto filterType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto filterRank = filterType.getRank();

  // if( inputRank != 1 || filterRank != 1)
  // {
  //   return emitError()
  //          << "expected rank of input & filter is 1";
  // }

  return mlir::success();
}


//===----------------------------------------------------------------------===//
// LengthOp
//===----------------------------------------------------------------------===//

void LengthOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input) {
  DEBUG_PRINT_NO_ARGS();
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands({input} );
  DEBUG_PRINT_NO_ARGS();
}

void LengthOp::inferShapes() {
  // auto tensorInput =  getInput().getType();
  // auto shapeOfInput = tensorInput.getShape();
  std::vector<int64_t> shapeForOutput;
  DEBUG_PRINT_NO_ARGS();
  shapeForOutput.push_back(1);

  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getInput().getType().getElementType());
  getResult().setType(manipulatedType);
  DEBUG_PRINT_NO_ARGS();
}

mlir::LogicalResult LengthOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  // if (!inputType || !resultType)
  //   return mlir::success();

  // auto inputShape = inputType.getShape();
  // if (!std::equal(inputShape.begin(), inputShape.end(),
  //                 resultType.getShape().rbegin())) {
  //   return emitError()
  //          << "expected result shape to be a transpose of the input";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ReverseInputOp
//===----------------------------------------------------------------------===//

void ReverseInputOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(input);
}

void ReverseInputOp::inferShapes() {
  auto tensorInput =  getInput().getType();
  // mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          // getInput().getType().getElementType());
  getResult().setType(tensorInput);
}

mlir::LogicalResult ReverseInputOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  // if (!inputType || !resultType)
  //   return mlir::success();

  // auto inputShape = inputType.getShape();
  // if (!std::equal(inputShape.begin(), inputShape.end(),
  //                 resultType.getShape().rbegin())) {
  //   return emitError()
  //          << "expected result shape to be a transpose of the input";
  // }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// PaddingOp
//===----------------------------------------------------------------------===//

void PaddingOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value input, mlir::Value PadValue, mlir::Value PadLen) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({input, PadValue, PadLen});
}



/// Infer the output shape of the PaddingOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of input * UpsamplingRate ie, Rhs
void PaddingOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getInput().getType();
  auto shapeOfInput = tensorInput.getShape();

  // auto tensorUpsampling = getRhs().getType(); 
  // auto shapeOfUpsampling = tensorUpsampling.getShape(); //shape is the length
  

  std::vector<int64_t> shapeForOutput ;

  int64_t SecondValueInt = 1;

  //To extract value from the SSA value:
    //get the Operand 
    //convert it to ConstantOp
    //convert it to corresponding elements attribute
    //extract the value as float then convert to int
  DEBUG_PRINT_NO_ARGS();
  Value padding3rdArg = getOperand(2);
  dsp::ConstantOp constantOp2ndArg = padding3rdArg.getDefiningOp<dsp::ConstantOp>();
  DEBUG_PRINT_NO_ARGS();
  DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();;
  auto elements = constantRhsValue.getValues<FloatAttr>();
  float SecondValue = elements[0].getValueAsDouble();
  SecondValueInt = (int64_t) SecondValue;
  // llvm::errs() << "Upsampling: SamplingRate: " << SecondValueInt << " \n"; //downsamplingRate
    
  DEBUG_PRINT_NO_ARGS();
  for(size_t i=0; i < shapeOfInput.size() ; i++){
    double GetLenForOutput  = static_cast<double>(shapeOfInput[i] ) + SecondValueInt ;
    int64_t OutlenInt = static_cast<int64_t> (GetLenForOutput);
    DEBUG_PRINT_WITH_ARGS("PaddingLen= " , OutlenInt);
    shapeForOutput.push_back(OutlenInt);
  }
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getInput().getType().getElementType());

  // getResult().setType(getLhs().getType()); 
  getResult().setType(manipulatedType);
  }

//get rank of Input & Upsampling -- make sure it is of rank 1 
mlir::LogicalResult PaddingOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto samplingRateType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto samplingRateRank = samplingRateType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " samplingRateRank: " << samplingRateRank << "\n";
  // //once ensured only 1 rank from above -- also make sure there is just 1 elem  
  // if( inputRank != 1 || samplingRateRank != 0 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank << " samplingRateRank: " << samplingRateRank << "\n";
  //   return emitError()
  //          << "expected rank of input is 1 & Upsampling is 0";
  // }
  return mlir::success();
} 


//===----------------------------------------------------------------------===//
// FIRFilterYSymmOptimizedOp
//===----------------------------------------------------------------------===//

void FIRFilterYSymmOptimizedOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}



/// Infer the output shape of the FIRFilterYSymmOptimizedOp, this is required by the shape inference
/// interface.
//ToDo -- shape should be the length of Lhs + Rhs - 1
void FIRFilterYSymmOptimizedOp::inferShapes() { 
  //get the shape of Lhs & rhs 
  //add the shape for each dimension
  // auto tensorInput =  llvm::cast<RankedTensorType>(getLhs().getType());
  auto tensorInput =  getLhs().getType();
  auto shapeOfInput = tensorInput.getShape();

  auto tensorFilter = getRhs().getType();
  auto shapeOfFilter = tensorFilter.getShape();
  std::vector<int64_t> shapeForOutput ;

  for(size_t i=0; i < shapeOfInput.size() ; i++){
    shapeForOutput.push_back(shapeOfInput[i] + shapeOfFilter[i] - 1);
  }
  
  mlir::TensorType manipulatedType = mlir::RankedTensorType::get(shapeForOutput, 
          getLhs().getType().getElementType());

  // getResult().setType(getLhs().getType()); 
  getResult().setType(manipulatedType);
}

//get rank of Input & Filter -- make sure it is of rank 1 
mlir::LogicalResult FIRFilterYSymmOptimizedOp::verify() {
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand(0).getType());
  // auto filterType = llvm::dyn_cast<RankedTensorType>(getOperand(1).getType());
  // // auto resultType = llvm::dyn_cast<RankedTensorType>(getType());

  // auto inputRank = inputType.getRank();
  // auto filterRank = filterType.getRank();

  // if( inputRank != 1 || filterRank != 1)
  // {
  //   return emitError()
  //          << "expected rank of input & filter is 1";
  // }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FFT1DRealSymmOp
//===----------------------------------------------------------------------===//

void FFT1DRealSymmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands(value);
  DEBUG_PRINT_NO_ARGS() ;
}

void FFT1DRealSymmOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  // getResult().setType(tensorInput);
  getResult().setType(tensorInput);
  // getResult(2).setType(tensorInput);
}

mlir::LogicalResult FFT1DRealSymmOp::verify() {
  DEBUG_PRINT_NO_ARGS();
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FFT1DImgConjSymmOp
//===----------------------------------------------------------------------===//

void FFT1DImgConjSymmOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  DEBUG_PRINT_NO_ARGS() ;
  state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
  state.addOperands(value);
  DEBUG_PRINT_NO_ARGS() ;
}

void FFT1DImgConjSymmOp::inferShapes() {
  //for each rank
  //Get the shape/size of input 
  //output size = input_size 
  auto tensorInput =  getInput().getType(); 
  // getResult().setType(tensorInput);
  getResult().setType(tensorInput);
  // getResult(2).setType(tensorInput);
}

mlir::LogicalResult FFT1DImgConjSymmOp::verify() {
  DEBUG_PRINT_NO_ARGS() ;
  // auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  // auto inputRank = inputType.getRank();

  // // llvm::errs() << "inputRank: " << inputRank << " alphaValueRank: " << alphaValueRank << "\n";
  // //once ensured only 1 rank from above --   
  // if( inputRank != 1 )
  // {
  //   llvm::errs() << "inputRank: " << inputRank <<  "\n";
  //   return emitError()
  //          << "expected rank of input  is 1";
  // }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

void Conv2DOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, 
        mlir::Value input, mlir::Value weight, mlir::Value bias) {
    state.addTypes({UnrankedTensorType::get(builder.getF64Type())});
    state.addOperands({input, weight, bias});
}
void Conv2DOp::inferShapes() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
    auto kernelType = llvm::dyn_cast<RankedTensorType>(getKernel().getType());

    int64_t IH = inputType.getShape()[0];
    int64_t IW = inputType.getShape()[1];
    int64_t KH = kernelType.getShape()[0];
    int64_t KW = kernelType.getShape()[1];
    int64_t OH = IH-KH+1, OW=IW-KW+1;

    SmallVector<int64_t, 2> dims = {OH, OW};
    getResult().setType(RankedTensorType::get(dims, inputType.getElementType()));
}

mlir::LogicalResult Conv2DOp::verify() {

    auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
    auto kernelType = llvm::dyn_cast<RankedTensorType>(getKernel().getType());
    auto biasType = llvm::dyn_cast<RankedTensorType>(getBias().getType());

    if(!inputType) {
        llvm::errs() << "expect a ranked tensor for input, get " << getInput();
        return mlir::failure();
    }
    if(!kernelType) {
        llvm::errs() << "expect a ranked tensor for kernel, get " << getKernel();
        return mlir::failure();
    }
    if(!biasType) {
        llvm::errs() << "expect a one dimensional ranked tensor for bias, get " << getBias();
        return mlir::failure();
    }

    auto inputRank = inputType.getRank();
    auto kernelRank = kernelType.getRank();

    if(inputRank != 2 ) {
        llvm::errs() << "expect 2 dimensional input, format N IH IW IC, get " << inputRank;
        return mlir::failure();
    }
    if(kernelRank != 2 ) {
        llvm::errs() << "expect 2 dimensional kernel, format OC KH KW IC.";
        return mlir::failure();
    }

    if(inputType.getShape()[0] < kernelType.getShape()[0]) {
        llvm::errs() << "input shape < kernel shape at 1st dimension";
        return mlir::failure();
    }

    if(inputType.getShape()[1] < kernelType.getShape()[1]) {
        llvm::errs() << "input shape < kernel shape at 2nd dimension";
        return mlir::failure();
    }
    
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
