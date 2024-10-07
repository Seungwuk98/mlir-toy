add_library(ToyParser
  STATIC 
  Token.cpp
  Lexer.cpp
  Parser.cpp
)

link_libraries(ToyParser PRIVATE ToyAST)
