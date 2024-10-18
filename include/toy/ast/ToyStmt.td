#ifndef TOY_STMT_TD
#define TOY_STMT_TD

include "toy/ast/ToyAST.td"

class ToyStmt : ToyAST { let parent = "::toy::Stmt"; }

def ToyAST_Module : ToyAST { let treeMember = (ins Vector<Stmt> : $stmts); }

def ToyAST_BlockStmt : ToyStmt { let treeMember = (ins Vector<Stmt> : $stmts); }

def ToyAST_FuncDecl : ToyStmt {
  let treeMember = (ins String : $name, SmallVector<Pair<Optional<String>, String>> : $params, "::toy::BlockStmt" : $body);
  let tag = (ins Map<SizeT, UserDefineType<"::toy::StructDecl">> : $paramStructDecls);
}

def ToyAST_StructDecl : ToyStmt {
  let treeMember = (ins String : $name, SmallVector<String> : $fields);
}

def ToyAST_VarDecl : ToyStmt {
  let treeMember = (ins String : $name, Optional<ShapeInfo>: $shape, Expr : $init);
}

def ToyAST_StructVarDecl : ToyStmt {
  let treeMember = (ins String : $structName, String : $name, SmallVector<Expr> : $init); 
  let tag = (ins "::toy::StructDecl" : $structDecl);
}

def ToyAST_ExprStmt : ToyStmt { let treeMember = (ins Expr : $expr); }

def ToyAST_ReturnStmt : ToyStmt {
  let treeMember = (ins Optional<Expr> : $expr);
}

#endif // TOY_STMT_TD
