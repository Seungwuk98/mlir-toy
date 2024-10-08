add_library(ToyParser
  STATIC 
  Token.cpp
  Lexer.cpp
  Parser.cpp
)

target_link_libraries(ToyParser PRIVATE ToyAST)

