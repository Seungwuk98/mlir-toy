#ifndef TOY_EXPR_TD
#define TOY_EXPR_TD

include "toy/ast/ToyAST.td"

class ToyExpr : ToyAST { let parent = "::toy::Expr"; }

def ToyAST_Number : ToyExpr { let treeMember = (ins String : $value); }

def ToyAST_Tensor : ToyExpr {
  let treeMember = (ins Vector<Expr> : $elements);
  let tag = (ins ShapeInfo : $shape);
}

def BinaryOpKind : UserDefineType<"::toy::BinaryOpKind">;

def ToyAST_BinaryOp : ToyExpr {
  let treeMember = (ins Expr : $lhs, Expr : $rhs, BinaryOpKind : $opKind);
}

def ToyAST_FunctionCall : ToyExpr {
  let treeMember = (ins String : $callee, SmallVector<Expr> : $args);
}

def ToyAST_Identifier : ToyExpr { let treeMember = (ins String : $name); }

def ToyAST_Transpose : ToyExpr { let treeMember = (ins Expr : $target); }

def ToyAST_Print : ToyExpr { let treeMember = (ins Expr : $target); }

#endif // TOY_EXPR_TD
