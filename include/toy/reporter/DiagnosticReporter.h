#ifndef TOY_DIAGNOSTIC_REPORTER_H
#define TOY_DIAGNOSTIC_REPORTER_H

#include "llvm/Support/SourceMgr.h"
#include <llvm-18/llvm/ADT/StringRef.h>
#include <llvm-18/llvm/Support/raw_ostream.h>

namespace toy {

class DiagnosticReporter {
public:
  DiagnosticReporter(llvm::SourceMgr &SM, llvm::raw_ostream &OS)
      : SM(SM), OS(OS) {}

  void Report(llvm::SMRange loc, llvm::SourceMgr::DiagKind diagKind,
              llvm::StringRef msg) {
    if (diagKind == llvm::SourceMgr::DK_Error)
      ++ErrorCount;
    SM.PrintMessage(OS, loc.Start, diagKind, msg, {loc});
  }

  std::size_t getErrorCount() const { return ErrorCount; }

private:
  llvm::SourceMgr &SM;
  llvm::raw_ostream &OS;
  std::size_t ErrorCount = 0;
};

} // namespace toy

#endif // TOY_DIAGNOSTIC_REPORTER_H
