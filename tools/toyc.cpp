#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "toy/context/ToyContext.h"
#include "toy/mlir/Dialect/ToyOp.h"
#include "toy/mlir/LLVMIR/LLVMDumper.h"
#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "toy/mlir/Pass/Passes.h"
#include "toy/parser/Parser.h"
#include "toy/reporter/DiagnosticReporter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

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
static llvm::cl::alias OutputFileAlias("o", llvm::cl::desc("Alias for -output"),
                                       llvm::cl::aliasopt(OutputFile));

namespace {
enum Action {
  DumpToken = 0,
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

enum OptLevel { O0 = 0, O1, O2, O3 };

} // namespace

static llvm::cl::opt<Action> ActionFlag(
    "action", llvm::cl::desc("select kind of behavior of this compiler"),
    llvm::cl::values(
        clEnumValN(DumpToken, "token", "Print the token stream"),
        clEnumValN(DumpAST, "ast", "Print the AST"),
        clEnumValN(DumpToyDialect, "toy", "Print the Toy dialect"),
        clEnumValN(DumpAffineDialect, "affine", "Print the Affine dialect"),
        clEnumValN(DumpLLVMDialect, "llvm", "Print the LLVM dialect"),
        clEnumValN(DumpLLVMIR, "llvm-ir", "Print the LLVM IR"),
        clEnumValN(RunJIT, "jit", "Run the JIT")),
    llvm::cl::init(DumpToyDialect));

static llvm::cl::opt<InputKind>
    InputType("ext", llvm::cl::desc("Specify the input type"),
              llvm::cl::values(clEnumValN(Toy, "toy", "Toy input file"),
                               clEnumValN(MLIR, "mlir", "MLIR input file")),
              llvm::cl::init(Unknown));

static llvm::cl::opt<OptLevel> OptLevelFlag(
    llvm::cl::desc("choose the optimization level"),
    llvm::cl::values(clEnumValN(O0, "g", "No optimization"),
                     clEnumValN(O1, "O1", "Optimization level 1"),
                     clEnumValN(O2, "O2", "Optimization level 2"),
                     clEnumValN(O3, "O3", "Optimization level 3")));

class Compilation {
public:
  Compilation() : SM(), Reporter(SM, llvm::errs()) {}

  Compilation &LoadFileOrStdIn(llvm::StringRef filename) {
    auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (buffer.getError()) {
      Reporter.Report("read file error: " + buffer.getError().message());
      return *this;
    }

    BufID = SM.AddNewSourceBuffer(std::move(*buffer), SMLoc());
    return *this;
  }

  Compilation &Parse() {
    if (Reporter.getErrorCount())
      return *this;
    auto bufferRef = SM.getMemoryBuffer(BufID)->getBuffer();
    L = std::make_unique<Lexer>(bufferRef, &Ctx, SM, Reporter);
    P = std::make_unique<Parser>(*L, SM);

    module = P->Parse();
    return *this;
  }

  Compilation &ParseFromMLIR() {
    if (Reporter.getErrorCount())
      return *this;
    moduleOp = mlir::parseSourceFile<ModuleOp>(
        SM.getMemoryBuffer(BufID)->getBuffer(), &Ctx);
    if (!moduleOp)
      Reporter.Report("failed to parse MLIR input");
    return *this;
  }

  Compilation &GenIR() {
    if (Reporter.getErrorCount())
      return *this;
    G = std::make_unique<IRGenerator>(&Ctx, SM, Reporter);
    module.accept(*G);
    moduleOp = G->getModuleOp();
    return *this;
  }

  Lexer *getLexer() { return L.get(); }
  Parser *getParser() { return P.get(); }
  IRGenerator *getIRGenerator() { return G.get(); }
  Module getModule() { return module; }
  ModuleOp getModuleOp() { return moduleOp.get(); }
  ToyContext *getContext() { return &Ctx; }
  DiagnosticReporter &getReporter() { return Reporter; }

private:
  llvm::SourceMgr SM;
  DiagnosticReporter Reporter;
  std::size_t BufID = -1;
  ToyContext Ctx;

  std::unique_ptr<Lexer> L;
  std::unique_ptr<Parser> P;
  std::unique_ptr<IRGenerator> G;

  Module module;
  OwningOpRef<ModuleOp> moduleOp;
};

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

bool dump(llvm::function_ref<bool(raw_ostream &)> outFn) {
  if (OutputFile == "-")
    return outFn(llvm::outs());

  std::error_code EC;
  llvm::raw_fd_ostream os(OutputFile, EC);
  if (EC) {
    llvm::errs() << "error opening file: " << EC.message() << "\n";
    return 1;
  }
  return outFn(os);
}

bool runPM(PassManager &PM, ModuleOp moduleOp) {
  PM.addPass(createCSEPass());
  return failed(PM.run(moduleOp));
}

int main(int argc, const char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "Toy MLIR Compiler\n");

  if (InputType == InputKind::Unknown) {
    if (InputFile == "-") {
      llvm::errs() << "Unknown input file type\n";
      return 1;
    }
    InputType = InputFile.ends_with(".toy") ? Toy : MLIR;
  }

  Compilation C;
  if (InputType == InputKind::MLIR &&
      ActionFlag < static_cast<unsigned>(InputKind::Toy)) {
    C.getReporter().Report("MLIR input is not supported for this action");
    return 1;
  }

  C.LoadFileOrStdIn(InputFile);
  if (C.getReporter().getErrorCount() > 0)
    return 1;

  if (InputType == InputKind::Toy)
    C.Parse();
  else
    C.ParseFromMLIR();
  if (C.getReporter().getErrorCount() > 0)
    return 1;

  if (ActionFlag == DumpToken)
    return dump([&C](raw_ostream &os) {
      for (auto *token : C.getLexer()->getTokenStream())
        os << token->toString() << '\n';
      return false; // no error
    });

  if (ActionFlag == DumpAST)
    return dump([&C](raw_ostream &os) {
      os << C.getModule().toString();
      return false; // no error
    });

  C.GenIR();
  if (C.getReporter().getErrorCount() > 0)
    return 1;

  auto moduleDumpFn = [&C](raw_ostream &os) {
    C.getModuleOp()->print(os);
    return false; // no error
  };

  if (ActionFlag == DumpToyDialect)
    return dump(moduleDumpFn);

  PassManager PM = PassManager(C.getContext());
  PM.addPass(createInlinerPass());

  auto &fnPM = PM.nest<mlir::toy::FuncOp>();
  fnPM.addPass(mlir::toy::createShapeInferencePass());
  fnPM.addPass(createCanonicalizerPass());

  PM.addPass(mlir::toy::createToyToAffineLoweringPass());
  if (ActionFlag > static_cast<unsigned>(DumpAffineDialect))
    PM.addPass(mlir::toy::createAffineToLLVMLoweringPass());

  if (runPM(PM, C.getModuleOp()))
    return 1;
  if (ActionFlag <= static_cast<unsigned>(DumpLLVMDialect))
    return dump(moduleDumpFn);

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::LLVMContext llvmContext;

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return 1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return 1;
  }

  return dump([&C, tm = tmOrError->get()](raw_ostream &os) {
    mlir::toy::LLVMDumper dumper(C.getContext(), *tm, OptLevelFlag, os);
    if (ActionFlag == DumpLLVMIR)
      return failed(dumper.Dump(C.getModuleOp()));
    assert(ActionFlag == RunJIT);
    return failed(dumper.RunJIT(C.getModuleOp()));
  });
}
