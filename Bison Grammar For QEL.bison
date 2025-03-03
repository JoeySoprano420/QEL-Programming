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

// Additional tokens for control structures
%token IF THEN ELSE WHILE FOR LPAREN RPAREN LBRACE RBRACE SEMICOLON

%%

statements:
    statement
    | statements statement
;

statement:
    if_statement
    | while_statement
    | for_statement
    | assignment_statement
;

if_statement:
    IF expression THEN statement ELSE statement {
        $$ = new IfExprNode($2, $4, $6);
    }
;

while_statement:
    WHILE expression statement {
        $$ = new WhileLoopNode($2, $3);
    }
;

for_statement:
    FOR LPAREN assignment_statement expression SEMICOLON expression RPAREN statement {
        $$ = new ForLoopNode($2, $3, $5, $7);
    }
;

