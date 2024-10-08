#include "TestExtras.h"
#include "doctest/doctest.h"
#include "toy/context/ToyContext.h"
#include "toy/parser/Lexer.h"

namespace toy::test {

static void LexTest(llvm::StringRef Program, llvm::StringRef Expected) {
  llvm::SourceMgr SM;
  ToyContext ctx;
  Lexer Lexer(Program, &ctx, SM);

  std::string LexResult;
  llvm::raw_string_ostream SS(LexResult);

  Token *Tok;
  while (!Tok->is<Token::Tok_EOF>()) {
    Tok = Lexer.GetNextToken();
    SS << Tok->toString() << '\n';
  }

  STR_EQ(Expected, SS.str());
}

#define LEX_TEST(CaseName, Program, Expected)                                  \
  SUBCASE(CaseName) LexTest(Program, Expected)

TEST_CASE("Lexer tests") {
  LEX_TEST("Toy Simple Program 1", R"(
def main() {
  var a = [1, 2, 3, 4, 5, 6]
}
)",
           R"(
def `def` at 1:0
identifier `main` at 1:4
lparen `(` at 1:8
rparen `)` at 1:9
lbrace `{` at 1:11
var `var` at 2:2
identifier `a` at 2:6
equal `=` at 2:8
lbracket `[` at 2:10
number `1` at 2:11
comma `,` at 2:12
number `2` at 2:14
comma `,` at 2:15
number `3` at 2:17
comma `,` at 2:18
number `4` at 2:20
comma `,` at 2:21
number `5` at 2:23
comma `,` at 2:24
number `6` at 2:26
rbracket `]` at 2:27
rbrace `}` at 3:0
EOF `` at 4:0
)");

  LEX_TEST("Toy Simple Program 2", R"(
def main() {
  var a = [1, 2, 3, 4, 5, 6];
  var b<2, 3> = a;
  return b;
}           
)",
           R"(
def `def` at 1:0
identifier `main` at 1:4
lparen `(` at 1:8
rparen `)` at 1:9
lbrace `{` at 1:11
var `var` at 2:2
identifier `a` at 2:6
equal `=` at 2:8
lbracket `[` at 2:10
number `1` at 2:11
comma `,` at 2:12
number `2` at 2:14
comma `,` at 2:15
number `3` at 2:17
comma `,` at 2:18
number `4` at 2:20
comma `,` at 2:21
number `5` at 2:23
comma `,` at 2:24
number `6` at 2:26
rbracket `]` at 2:27
semicolon `;` at 2:28
var `var` at 3:2
identifier `b` at 3:6
lt `<` at 3:7
number `2` at 3:8
comma `,` at 3:9
number `3` at 3:11
gt `>` at 3:12
equal `=` at 3:14
identifier `a` at 3:16
semicolon `;` at 3:17
return `return` at 4:2
identifier `b` at 4:9
semicolon `;` at 4:10
rbrace `}` at 5:0
EOF `` at 6:0
)");
}

} // namespace toy::test
