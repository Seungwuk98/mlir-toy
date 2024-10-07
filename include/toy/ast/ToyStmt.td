#ifndef TOY_STMT_TD
#define TOY_STMT_TD

include "toy/ast/ToyAST.td"

class ToyStmt : ToyAST { let parent = "::toy::Stmt"; }

def ToyAST_Module : ToyAST { let treeMember = (ins Vector<Stmt> : $stmts); }

def ToyAST_BlockStmt : ToyStmt { let treeMember = (ins Vector<Stmt> : $stmts); }

def ToyAST_FuncDecl : ToyStmt {
  let treeMember = (ins String : $name, SmallVector<String> : $params, "::toy::BlockStmt" : $body);
}

def ToyAST_VarDecl : ToyStmt {
  let treeMember = (ins String : $name, Optional<ShapeInfo>: $shape, Expr : $init);
}

def ToyAST_ExprStmt : ToyStmt { let treeMember = (ins Expr : $expr); }

def ToyAST_ReturnStmt : ToyStmt {
  let treeMember = (ins Optional<Expr> : $expr);
}

#endif // TOY_STMT_TD
