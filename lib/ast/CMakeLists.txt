add_library(ToyAST
  STATIC
  ToyASTSet.cpp
  ToyExpr.cpp
  ToyStmt.cpp
)

target_link_libraries(ToyAST 
  PRIVATE
  AST
)

add_dependencies(ToyAST ToyExprGen ToyStmtGen)
