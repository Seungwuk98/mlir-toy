#ifndef TOY_AST_TD
#define TOY_AST_TD

include "ast/AST.td"

def Stmt : UserDefineType<"::toy::Stmt">;
def Expr : UserDefineType<"::toy::Expr">;
def ShapeInfo : UserDefineType<"::toy::ShapeInfo">;

class ToyAST : AST { let namespace = "toy"; }

#endif // TOY_AST_TD
