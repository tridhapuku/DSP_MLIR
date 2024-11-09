//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/MLIRGen.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <bitset>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir::dsp;
using namespace dsp;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      llvm::errs() << "Line : " << __LINE__ << " func:" << __FILE__ << " \n";
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::dsp::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(VarType{}));
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<mlir::dsp::FuncOp>(location, proto.getName(),
                                             funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::dsp::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::dsp::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();

    return function;
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    case '/':
      return builder.create<DivOp>(location, lhs, rhs);
    case '-':
      return builder.create<SubOp>(location, lhs, rhs);
    case '^':
      return builder.create<PowOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location,
                             expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `dsp.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "dsp.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    // Build the MLIR op `dsp.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.

    if (callee == "bitwiseand") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.bitwiseand "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<BitwiseAndOp>(location, operands[0], operands[1]);
    }

    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return builder.create<TransposeOp>(location, operands[0]);
    }

    //
    if (callee == "delay") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.delay "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<DelayOp>(location, operands[0], operands[1]);
    }

    if (callee == "gain") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.gain "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<GainOp>(location, operands[0], operands[1]);
    }

    // Sub Op
    if (callee == "sub") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.sub "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<SubOp>(location, operands[0], operands[1]);
    }
    if(callee == "pow"){
       if(call.getArgs().size() != 2){
         emitError(location, "MLIR codegen encountered an error: dsp.pow "
                             "accepts only 2 arguments");
         return nullptr;
       }
       return builder.create<PowOp>(location, operands[0], operands[1]);
    }


    // Modulo Op
    if (callee == "modulo") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.modulo "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<ModuloOp>(location, operands[0], operands[1]);
    }

    if (callee == "fftReal") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.zeroCrossCount "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<FFTRealOp>(location, operands[0]);
    }

    if (callee == "fftImag") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.zeroCrossCount "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<FFTImagOp>(location, operands[0]);
    }

    // FindPeaks Op
    if (callee == "find_peaks") {
      if (call.getArgs().size() != 3) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.find_peaks "
                  "accepts only 3 arguments: signal, height, and distance");
        return nullptr;
      }
      return builder.create<FindPeaksOp>(location, operands[0], operands[1],
                                         operands[2]);
    }

    // Max Op
    if (callee == "max") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.max "
                            "accepts only 1 argument.");
        return nullptr;
      }
      return builder.create<MaxOp>(location, operands[0]);
    }

    // Mean Op
    if (callee == "mean") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.mean "
                            "accepts only 2 arguments: input tensor, length");
        return nullptr;
      }
      return builder.create<MeanOp>(location, operands[0], operands[1]);
    }

    // Diff Op
    if (callee == "diff") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.diff "
                            "accepts only 2 arguments: input tensor, legnth");
        return nullptr;
      }
      return builder.create<DiffOp>(location, operands[0], operands[1]);
    }
       
    // Abs Op
    if(callee == "abs") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.abs "
                            "accepts only 1 arguments: input tensor.");
        return nullptr;
      }
      return builder.create<AbsOp>(location, operands[0]);
    }

    // ArgMax Op
    if(callee == "argmax") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.argmax "
                            "accepts only 2 arguments: input tensor, axis.");
        return nullptr;
      }

      auto axisOp = operands[1].getDefiningOp<mlir::dsp::ConstantOp>();
      auto axisVal = axisOp.getValue().getValues<mlir::FloatAttr>();
      double axis = axisVal[0].getValueAsDouble();

      return builder.create<ArgMaxOp>(location, operands[0], axis);
    }

    // Normalize Op
    if (callee == "normalize") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.normalize "
                            "accepts only 1 arguments: input tensor");
        return nullptr;
      }
      return builder.create<NormalizeOp>(location, operands[0]);
    }
   
    // Normalize LMS filter Op
    if (callee == "norm_LMSFilterResponse_opt") {
      if (call.getArgs().size() != 4) {
        emitError(location, "MLIR codegen encountered an error: dsp.norm_LMSFilterResponse_opt "
                            "accepts 4 arguments ");
        return nullptr;
      }
      return builder.create<NormLMSFilterResponseOptimizeOp>(location, operands[0], operands[1], operands[2], operands[3]);
    }

    // Shift right Op
    if (callee == "shiftRight") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.shiftRight "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<ShiftRightOp>(location, operands[0], operands[1]);
    }

    // Matmul Op
    if (callee == "matmul") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.matmul "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<MatmulOp>(location, operands[0], operands[1]);
    }

    if (callee == "zeroCrossCount") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.zeroCrossCount "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<zeroCrossCountOp>(location, operands[0]);
    }

    if (callee == "FIRFilterResponse") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.FIRFilterResponse "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<FIRFilterResponseOp>(location, operands[0],
                                                 operands[1]);
    }

    if (callee == "medianFilter") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.medianFilter "
                  "accepts only 1 argument");
        return nullptr;
      }
      return builder.create<MedianFilterOp>(location, operands[0]);
    }

    if (callee == "slidingWindowAvg") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.slidingWindowAvg "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<SlidingWindowAvgOp>(location, operands[0]);
    }

    if (callee == "downsampling") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.downsampling "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<DownsamplingOp>(location, operands[0], operands[1]);
    }

    if (callee == "upsampling") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.upsampling "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<UpsamplingOp>(location, operands[0], operands[1]);
    }

    if (callee == "lowPassFilter") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.lowPassFilter "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<LowPassFilter1stOrderOp>(location, operands[0],
                                                     operands[1]);
    }

    if (callee == "highPassFilter") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.highPassFilter "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<HighPassFilterOp>(location, operands[0]);
    }

    if (callee == "fft1d") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.fft1d "
                            "accepts only 1 arguments");
        return nullptr;
      }
      // return builder.create<FFT1DOp>(location, operands[0] );
    }

    if (callee == "fft1dreal") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.fft1dreal "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<FFT1DRealOp>(location, operands[0]);
    }

    if (callee == "fft1dimg") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.fft1dimg "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<FFT1DImgOp>(location, operands[0]);
    }

    if (callee == "ifft1d") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.ifft1d "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<IFFT1DOp>(location, operands[0], operands[1]);
    }

    if (callee == "hamming") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.hamming "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<HammingWindowOp>(location, operands[0]);
    }

    if (callee == "dct") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.dct "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<DCTOp>(location, operands[0]);
    }

    if (callee == "filter") {
      if (call.getArgs().size() != 3) {
        emitError(location, "MLIR codegen encountered an error: dsp.filter "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<filterOp>(location, operands[0], operands[1],
                                      operands[2]);
    }

    if (callee == "div") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.div "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<DivOp>(location, operands[0], operands[1]);
    }

    if (callee == "sum") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.sum "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<SumOp>(location, operands[0]);
    }

    if (callee == "sin") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.sin "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<SinOp>(location, operands[0]);
    }

    if (callee == "cos") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.cos "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<CosOp>(location, operands[0]);
    }

    if (callee == "square") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.square "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<SquareOp>(location, operands[0]);
    }

    // Sinc Op
    if (callee == "sinc") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: dsp.sinc "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<SincOp>(location, operands[0], operands[1]);
    }

    // Get Elem At Op
    if (callee == "getElemAtIndx") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.getElemAtIndx "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<GetElemAtIndxOp>(location, operands[0],
                                             operands[1]);
    }

    // Get Single Element At Op
    if (callee == "getSingleElemAtIndx") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.getSingleElemAtIndx "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<GetSingleElemAtIdxOp>(location, operands[0],
                                                  operands[1]);
    }

    // Diff2MeanOptimized Op
    if (callee == "diff2meanOpt") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.diff2meanOpt "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<Diff2MeanOptimizedOp>(location, operands[0],
                                                  operands[1]);
    }
	
    // FindPeaksDiff2MeanOptimized Op
    if (callee == "findpeaks2diff2meanOpt") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.findpeaks2diff2meanOpt "
                  "accepts only 3 arguments.");
        return nullptr;
      }
      return builder.create<FindPeaks2Diff2MeanOptimizedOp>(location, operands[0], operands[1], operands[2]);
    }

    // LMS2FindPeaksOptimizedOp Op
    if (callee == "lms2findPeaks") {
      if (call.getArgs().size() != 6) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.lmsFilterResponse2findPeaks "
                  "accepts only 6 arguments");
        return nullptr;
      }
      return builder.create<LMS2FindPeaksOptimizedOp>(location, operands[0],
                                                  operands[1], operands[2], operands[3], operands[4], operands[5]);
    }

    // Median2SlidingOptimized Op
    if (callee == "median2slidingOp") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.median2slidingOp"
                  "accepts only 1 argument.");
        return nullptr;
      }
      return builder.create<Median2SlidingOptimizedOp>(location, operands[0]);
    }


    // Set Elem At Indx
    if (callee == "setElemAtIndx") {
      if (call.getArgs().size() != 3) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.setElemAtIndx "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<SetElemAtIndxOp>(location, operands[0], operands[1],
                                             operands[2]);
    }

    // lowPassFilter Op
    if (callee == "lowPassFIRFilter") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.lowPassFilter "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<LowPassFIRFilterOp>(location, operands[0],
                                                operands[1]);
    }

    // highPassFilter Op
    if (callee == "highPassFIRFilter") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.highPassFilter "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<HighPassFIRFilterOp>(location, operands[0],
                                                 operands[1]);
    }

    if (callee == "getRangeOfVector") {
      if (call.getArgs().size() != 3) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.getRangeOfVector "
                  "accepts only 3 arguments");
        return nullptr;
      }
      return builder.create<GetRangeOfVectorOp>(location, operands[0],
                                                operands[1], operands[2]);
    }

    // FIRHammingOptimizedOp
    if (callee == "FIRFilterHammingOptimized") {
      if (call.getArgs().size() != 2) {
        emitError(
            location,
            "MLIR codegen encountered an error: dsp.FIRFilterHammingOptimized "
            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<FIRFilterHammingOptimizedOp>(location, operands[0],
                                                         operands[1]);
    }

    // HighPassFIRHammingOptimizedOp
    if (callee == "highPassFIRHammingOptimized") {
      if (call.getArgs().size() != 2) {
        emitError(location, "MLIR codegen encountered an error: "
                            "dsp.HighPassFIRHammingOptimizedOp "
                            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<HighPassFIRHammingOptimizedOp>(
          location, operands[0], operands[1]);
    }

    // LMS FILTER
    if (callee == "lmsFilter") {
      if (call.getArgs().size() != 5) {
        emitError(location, "MLIR codegen encountered an error: dsp.lmsFilter"
                            "accepts only 5 arguments");
        return nullptr;
      }
      return builder.create<LMSFilterOp>(location, operands[0], operands[1],
                                         operands[2], operands[3], operands[4]);
    }

    if (callee == "threshold") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.ThresholdOp "
                  "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<ThresholdOp>(location, operands[0], operands[1]);
    }

    if (callee == "quantization") {
      if (call.getArgs().size() != 4) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.quantization "
                  "accepts only 4 arguments");
        return nullptr;
      }
      return builder.create<QuantizationOp>(location, operands[0], operands[1],
                                            operands[2], operands[3]);
    }

    if (callee == "lmsFilterResponse") {
      if (call.getArgs().size() != 4) {
        emitError(location, "MLIR codegen encountered an error: dsp.lmsFilter"
                            "accepts only 4 arguments");
        return nullptr;
      }
      return builder.create<LMSFilterResponseOp>(
          location, operands[0], operands[1], operands[2], operands[3]);
    }

    if (callee == "runLenEncoding") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.runLenEncoding "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<RunLenEncodingOp>(location, operands[0]);
    }

    if (callee == "FIRFilterResSymmOptimized") {
      if (call.getArgs().size() != 2) {
        emitError(
            location,
            "MLIR codegen encountered an error: dsp.FIRFilterResSymmOptimized "
            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<FIRFilterResSymmOptimizedOp>(location, operands[0],
                                                         operands[1]);
    }

    if (callee == "len") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: dsp.len "
                            "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<LengthOp>(location, operands[0]);
    }

    if (callee == "reverseInput") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.reverseInput "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<ReverseInputOp>(location, operands[0]);
    }

    if (callee == "padding") {
      if (call.getArgs().size() != 3) {
        emitError(location, "MLIR codegen encountered an error: dsp.padding "
                            "accepts only 3 arguments");
        return nullptr;
      }
      return builder.create<PaddingOp>(location, operands[0], operands[1],
                                       operands[2]);
    }

    if (callee == "FIRFilterYSymmOptimized") {
      if (call.getArgs().size() != 2) {
        emitError(
            location,
            "MLIR codegen encountered an error: dsp.FIRFilterYSymmOptimizedOp "
            "accepts only 2 arguments");
        return nullptr;
      }
      return builder.create<FIRFilterYSymmOptimizedOp>(location, operands[0],
                                                       operands[1]);
    }
    if (callee == "fft1DRealSymm") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.FFT1DRealSymmOp "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<FFT1DRealSymmOp>(location, operands[0]);
    } // FFT1DImgConjSymmOpLowering
    if (callee == "fft1DimgConjSymm") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.FFT1DImgConjSymmOp "
                  "accepts only 1 arguments");
        return nullptr;
      }
      return builder.create<FFT1DImgConjSymmOp>(location, operands[0]);
    }

    if (callee == "conv2d") {
      if (call.getArgs().size() != 3) {
        emitError(location, "MLIR codegen encountered an error: dsp.Conv2DOp "
                            "accepts 3 arguments");
        return nullptr;
      }
      return builder.create<Conv2DOp>(location, operands[0], operands[1],
                                      operands[2]);
    }

    if (callee == "thresholdUp") {
      if (call.getArgs().size() != 3) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.thresholdUp "
                  "accepts 3 arguments");
        return nullptr;
      }
      return builder.create<ThresholdUpOp>(location, operands[0], operands[1],
                                           operands[2]);
    }

    if (callee == "generateDtmf") {
      if (call.getArgs().size() != 3) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.GenerateDTMFOp "
                  "accepts 3 argument");
        return nullptr;
      }
      return builder.create<GenerateDTMFOp>(location, operands[0], operands[1],
                                            operands[2]);
    }
    // beam form
    if (callee == "beam_form") {
      if (call.getArgs().size() != 4) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.GenerateDTMFOp "
                  "accepts 4 argument");
        return nullptr;
      }
      auto antennaConst = operands[0].getDefiningOp<mlir::dsp::ConstantOp>();
      auto freqConst = operands[1].getDefiningOp<mlir::dsp::ConstantOp>();
      auto antennaVal = antennaConst.getValue().getValues<mlir::FloatAttr>();
      auto freqVal = freqConst.getValue().getValues<mlir::FloatAttr>();

      double antenna = antennaVal[0].getValueAsDouble();
      double freq = freqVal[0].getValueAsDouble();

      return builder.create<BeamFormOp>(location, antenna, freq, operands[2],
                                        operands[3]);
    }
    // qam modulate op
    if (callee == "qam_modulate_real") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.QamModulateRealOp "
                  "accepts 1 arguments");
        return nullptr;
      }

      return builder.create<QamModulateRealOp>(location, operands[0]);
    }

    if (callee == "qam_modulate_imagine") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.QamModualteImgOp "
                  "accepts 1 arguments");
        return nullptr;
      }

      return builder.create<QamModulateImgOp>(location, operands[0]);
    }
    // qam_demodulate
    if (callee == "qam_demodulate") {
      if (call.getArgs().size() != 2) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.QamDemodulateOp"
                  "accepts 2 arguments");
        return nullptr;
      }
      return builder.create<QamDemodulateOp>(location, operands[0],
                                             operands[1]);
    }
    // space_demodulate
    if (callee == "space_demodulate") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.SpaceDemodulateOp"
                  "accepts 1 arguments");
        return nullptr;
      }
      return builder.create<SpaceDemodulateOp>(location, operands[0]);
    }
    // space_modulate
    if (callee == "space_modulate") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.SpaceModulateOp"
                  "accepts 1 arguments");
        return nullptr;
      }
      return builder.create<SpaceModulateOp>(location, operands[0]);
    }
    // space_err_correction
    if (callee == "space_err_correction") {
      if (call.getArgs().size() != 1) {
        emitError(location,
                  "MLIR codegen encountered an error: dsp.SpaceErrCorrectionOp"
                  "accepts 1 arguments");
        return nullptr;
      }
      return builder.create<SpaceErrCorrectionOp>(location, operands[0]);
    }
    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.
    // if(callee == "delay"){
    //   if(call.getArgs().size() != 1){
    //     emitError(location, "MLIR codegen encountered an error: dsp.delay "
    //                         "does not accept multiple arguments");
    //     return nullptr;
    //   }
    //   return builder.create<DelayOp>(location, operands[0]);
    // }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return builder.create<GenericCallOp>(location, callee, operands);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    builder.create<PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &num) {
    return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
  }

  /// Emit a string exression
  mlir::Value mlirGen(StringExprAST &expr) {
    auto string_val = expr.getStringVal();

    std::vector<double> signals;
    for (char ch : string_val) {
      std::bitset<8> bits(static_cast<unsigned char>(ch)), reversed;
      int n = 8;
      for (int i = 0; i < n; ++i)
        reversed[i] = bits[n - i - 1];
      for (int i = 0; i < n; ++i)
        signals.push_back(reversed[i]);
    }

    mlir::Type eleType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(signals.size(), eleType);

    auto dataAttr =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(signals));

    auto type = getType(signals.size());

    return builder.create<ConstantOp>(loc(expr.loc()), type, dataAttr);
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case dsp::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case dsp::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case dsp::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case dsp::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case dsp::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    case dsp::ExprAST::Expr_String:
      return mlirGen(cast<StringExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value;
    // Register the value in the symbol table.
    value = mlirGen(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                        getType(vardecl.getType()), value);
    }
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace

namespace dsp {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace dsp
