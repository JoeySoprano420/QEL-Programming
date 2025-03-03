%{
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ast.hpp"  // Define AST node classes here
%}

%union {
    int int_val;
    std::string *str_val;
    ExprNode *expr;  // Pointer to abstract syntax tree node
}

%token IF THEN ELSE WHILE TRUE FALSE IDENTIFIER INTEGER
%token ADD SUBTRACT MULTIPLY DIVIDE EQUAL LPAREN RPAREN SEMICOLON

%%

program:
    statements
;

statements:
    statement
    | statements statement
;

statement:
    if_statement
    | assignment_statement
    | expression_statement
;

if_statement:
    IF expression THEN statement ELSE statement {
        $$ = new IfExprNode($2, $4, $6);  // Create a conditional expression node
    }
;

assignment_statement:
    IDENTIFIER EQUAL expression SEMICOLON {
        $$ = new AssignmentNode(*$1, $3);  // Assign a value to a variable
    }
;

expression_statement:
    expression SEMICOLON {
        $$ = $1;
    }
;

expression:
    INTEGER {
        $$ = new IntegerNode($1);
    }
    | IDENTIFIER {
        $$ = new VariableNode(*$1);
    }
    | expression ADD expression {
        $$ = new AddNode($1, $3);
    }
    | expression SUBTRACT expression {
        $$ = new SubtractNode($1, $3);
    }
    | LPAREN expression RPAREN {
        $$ = $2;
    }
;

%%

int main() {
    std::cout << "Enter your QEL code:\n";
    yyparse();
    return 0;
}
