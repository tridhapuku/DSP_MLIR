class PrintOpLowering : public ConversionPattern
FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter, ModuleOp module)
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name, StringRef value, ModuleOp module)

struct ToyToLLVMLoweringPass : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>>
void getDependentDialects(DialectRegistry &registry) const override
void runOnOperation() final

std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass()
