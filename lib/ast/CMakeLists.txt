add_library(ToyAST
  STATIC
  ToyASTSet.cpp
)


add_dependencies(ToyAST ToyExprGen ToyStmtGen)
