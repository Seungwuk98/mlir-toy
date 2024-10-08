#include "toy/context/ToyContext.h"
#include "toy/parser/Lexer.h"
#include "toy/parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "TestExtras.h"
#include "doctest/doctest.h"

namespace toy::test {

static void ParserTest(llvm::StringRef Program, llvm::StringRef Expected) {
  llvm::SourceMgr SM;
  ToyContext ctx;
  DiagnosticReporter Reporter(SM, llvm::errs());

  auto memBuf = llvm::MemoryBuffer::getMemBuffer(Program);
  auto bufID = SM.AddNewSourceBuffer(std::move(memBuf), {});
  auto bufferRef = SM.getMemoryBuffer(bufID);

  Lexer Lexer(bufferRef->getBuffer(), &ctx, SM, Reporter);
  Parser Parser(Lexer, SM);

  auto module = Parser.Parse();
  if (!module) {
    FAIL("Failed to parse the program");
    return;
  }
  auto emittedProgram = module.toString();

  STR_EQ(Expected, emittedProgram);
}

#define PARSER_TEST(CaseName, Program, Expected)                               \
  SUBCASE(CaseName) ParserTest(Program, Expected)

TEST_CASE("Parser Tests") {
  PARSER_TEST("Toy Simple Program 1", R"(
def main() {
  var a = [1, 2, 3, 4, 5, 6];
  return;
}
)",
              R"(
def main() {
  var a = [1, 2, 3, 4, 5, 6];
  return;
}
)");

  PARSER_TEST("Toy Simple Program 2", R"(
def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  var b<1, 1> = [3];
  var c = a * b;
  return c;
}
)",
              R"(
def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  var b<1, 1> = [3];
  var c = a * b;
  return c;
}
)");

  PARSER_TEST("Toy Simple Program 3", R"(
def transpose_transpose(a) {
  return transpose(transpose(a));
}

def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  return transpose_transpose(a);
}
)",
              R"(
def transpose_transpose(a) {
  return transpose(transpose(a));
}
def main() {
  var a<2, 3> = [1, 2, 3, 4, 5, 6];
  return transpose_transpose(a);
}
)");

  PARSER_TEST("Toy Simple Program 4", R"(
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
)",
              R"(
def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  print(transpose(a) * transpose(b));
}
)");

  PARSER_TEST("Toy Simple Program 5", R"(
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);
}
)",
              R"(
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  var e = multiply_transpose(c, d);
  var f = multiply_transpose(a, c);
}
)");
}

} // namespace toy::test