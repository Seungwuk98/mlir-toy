#ifndef TEST_MLIR_TEST_UTILS_H
#define TEST_MLIR_TEST_UTILS_H

#include "TestExtras.h"
#include "doctest/doctest.h"
#include "toy/mlir/MLIRGen/IRGenerator.h"
#include "toy/parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

namespace toy::test {

bool MLIRTest(llvm::StringRef Program,
              std::function<bool(mlir::ModuleOp)> Check);

bool MLIRGenTest(llvm::StringRef Program, llvm::StringRef Expected);

bool MLIRGenCanonicalizerPassTest(llvm::StringRef Program,
                                  llvm::StringRef Expected);

} // namespace toy::test

#define MLIR_GEN_TEST(TestName, Program, Expected)                             \
  SUBCASE(TestName) CHECK(::toy::test::MLIRGenTest(Program, Expected))
#define MLIR_PASS_TEST(TestName, Program, Expected)                            \
  SUBCASE(TestName)                                                            \
  CHECK(::toy::test::MLIRGenCanonicalizerPassTest(Program, Expected))

#endif // TEST_MLIR_TEST_UTILS_H
