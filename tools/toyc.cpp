#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "toy/context/ToyContext.h"
#include "toy/mlir/Dialect/ToyDialect.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/mlir/LLVMIR/LLVMDumper.h"
#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "toy/mlir/Pass/Passes.h"
#include "toy/parser/Parser.h"
#include "toy/reporter/DiagnosticReporter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace ::toy;

static llvm::cl::opt<std::string> InputFile(llvm::cl::Positional,
                                            llvm::cl::desc("Toy input file"),
                                            llvm::cl::value_desc("filename"),
                                            llvm::cl::init("-"));

static llvm::cl::opt<std::string> OutputFile("output",
                                             llvm::cl::desc("Toy output File"),
                                             llvm::cl::value_desc("filename"),
                                             llvm::cl::init("-"));

namespace {
enum Action {
  DumpAST,
  DumpToyDialect,
  DumpAffineDialect,
  DumpLLVMDialect,
  DumpLLVMIR,
  RunJIT
};

enum InputKind {
  Unknown,
  Toy,
  MLIR,
};
} // namespace

static llvm::cl::opt<Action> ActionFlag(
    "action", llvm::cl::desc("select kind of behavior of this compiler"),
    llvm::cl::values(
        clEnumValN(DumpAST, "ast", "Print the AST"),
        clEnumValN(DumpToyDialect, "toy", "Print the Toy dialect"),
        clEnumValN(DumpAffineDialect, "affine", "Print the Affine dialect"),
        clEnumValN(DumpLLVMDialect, "llvm", "Print the LLVM dialect"),
        clEnumValN(DumpLLVMIR, "llvm-ir", "Print the LLVM IR"),
        clEnumValN(RunJIT, "jit", "Run the JIT")));

static llvm::cl::opt<InputKind>
    InputType("ext", llvm::cl::desc("Specify the input type"),
              llvm::cl::values(clEnumValN(Toy, "toy", "Toy input file"),
                               clEnumValN(MLIR, "mlir", "MLIR input file")),
              llvm::cl::init(Unknown));

Module parse(ToyContext &Ctx, llvm::SourceMgr &SM,
             DiagnosticReporter &Reporter) {
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(InputFile);
  if (buffer.getError()) {
    llvm::errs() << "read file error: " << buffer.getError().message() << "\n";
    return nullptr;
  }

  auto bufID = SM.AddNewSourceBuffer(std::move(*buffer), SMLoc());
  auto bufferRef = SM.getMemoryBuffer(bufID)->getBuffer();
  Lexer L(bufferRef, &Ctx, SM, Reporter);
  Parser P(L, SM);

  return P.Parse();
}

OwningOpRef<ModuleOp> irGen(Module M, ToyContext &Ctx,
                            DiagnosticReporter &Reporter) {
  IRGenerator G(&Ctx, Reporter.getSourceMgr(), Reporter);
  M.accept(G);
  if (Reporter.getErrorCount() > 0)
    return nullptr;
  return G.getModuleOp();
}

bool dump(llvm::function_ref<void(raw_ostream &)> outFn) {
  if (OutputFile == "-") {
    outFn(llvm::outs());
    return 0;
  }

  std::error_code EC;
  llvm::raw_fd_ostream os(OutputFile, EC);
  if (EC) {
    llvm::errs() << "error opening file: " << EC.message() << "\n";
    return 1;
  }
  outFn(os);
  return 0;
}

bool runPM(PassManager &PM, ModuleOp moduleOp) {
  PM.addPass(createCSEPass());
  return failed(PM.run(moduleOp));
}

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);

  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "Toy MLIR Compiler\n");

  ToyContext toyContext;
  if (InputType == InputKind::Unknown) {
    if (InputFile == "-") {
      llvm::errs() << "Unknown input file type\n";
      return 1;
    }
    InputType = InputFile.ends_with(".toy") ? Toy : MLIR;
  }

  llvm::SourceMgr srcMgr;
  DiagnosticReporter reporter(srcMgr, llvm::errs());

  auto module = parse(toyContext, srcMgr, reporter);
  if (!module)
    return 1;

  if (ActionFlag == DumpAST)
    return dump([&module](raw_ostream &os) { os << module.toString(); });

  auto moduleOp = irGen(module, toyContext, reporter);
  if (!moduleOp)
    return 1;

  auto moduleDumpFn = [&moduleOp](raw_ostream &os) { moduleOp->print(os); };

  if (ActionFlag == DumpToyDialect)
    return dump(moduleDumpFn);

  PassManager PM = PassManager(&toyContext);
  PM.addPass(createInlinerPass());

  auto &fnPM = PM.nest<mlir::toy::FuncOp>();
  fnPM.addPass(mlir::toy::createShapeInferencePass());
  fnPM.addPass(createCanonicalizerPass());

  PM.addPass(mlir::toy::createToyToAffineLoweringPass());

  if (runPM(PM, moduleOp.get()))
    return 1;
  if (ActionFlag == DumpAffineDialect)
    return dump(moduleDumpFn);

  PM.addPass(mlir::toy::createToyToLLVMLoweringPass());

  if (runPM(PM, moduleOp.get()))
    return 1;
  if (ActionFlag == DumpLLVMDialect)
    return dump([&moduleOp](raw_ostream &os) { moduleOp->print(os); });

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}
