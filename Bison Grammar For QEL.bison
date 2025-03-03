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

statement:
    if_statement
    | while_statement
    | assignment_statement
    | error {
        std::cerr << "Syntax error encountered. Attempting recovery.\n";
        yyerrok; // Recover from the error
    }
;

%{
#include <stdio.h>
#include <stdlib.h>
%}

%token DEFINE WRAP ATTEMPT IF ELSE RETURN TIMES MINUS DIVIDED BY
%token NUMBER IDENTIFIER

%%

program:
    statements
;

statements:
    statements statement
    | statement
;

statement:
    DEFINE IDENTIFIER 'of' IDENTIFIER '→' IDENTIFIER '{' expr '}'
    | WRAP IDENTIFIER 'of' IDENTIFIER '{' expr '}'
    | RETURN expr
    | ATTEMPT expr IF expr
    | RETURN expr
;

expr:
    expr '+' expr
    | expr '-' expr
    | NUMBER
    | IDENTIFIER
    | IDENTIFIER '(' expr ')'
;
%%

%{
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

using namespace std;

enum class NodeType { 
    PROGRAM, 
    VAR_DECLARATION, 
    FUNC_DECLARATION, 
    ASSIGNMENT, 
    IF_STATEMENT, 
    WHILE_STATEMENT, 
    FOR_STATEMENT, 
    EXPRESSION, 
    CALL, 
    LITERAL, 
    BINARY_OP,
    FUNC_CALL,
    LAMBDA_EXP,
    CLASS_DECLARATION,
    OBJECT_CREATION
};

class Node {
public:
    virtual NodeType getNodeType() const = 0;
    virtual ~Node() {}
};

class Expression : public Node {
public:
    virtual void evaluate() = 0; 
};

class Program : public Node {
    vector<shared_ptr<Node>> statements;
public:
    void addStatement(shared_ptr<Node> stmt) {
        statements.push_back(stmt);
    }

    NodeType getNodeType() const override {
        return NodeType::PROGRAM;
    }

    void execute() {
        for (auto &stmt : statements) {
            stmt->execute();
        }
    }
};

%}

%union {
    int int_val;
    std::string* str_val;
    std::shared_ptr<Node> node;
}

%token <int_val> INT
%token <str_val> IDENTIFIER
%token IF WHILE FOR RETURN TRY CATCH
%token EQ NEQ LT GT LE GE
%token PLUS MINUS MUL DIV
%token LPAREN RPAREN LBRACE RBRACE SEMICOLON COMMA COLON
%token CLASS OBJECT LAMBDA

%type <node> program statement assignment if_statement while_statement for_statement expression function_declaration func_call return_statement try_catch_block block

%%

program:
    statements { 
        $$ = make_shared<Program>(); 
        for (auto &stmt : $1) {
            $$->addStatement(stmt);
        }
    }
;

statements:
    statement statements { $$ = $1; $$->addStatement($2); }
    | /* empty */ { $$ = make_shared<Program>(); }
;

statement:
    assignment SEMICOLON { $$ = $1; }
    | if_statement { $$ = $1; }
    | while_statement { $$ = $1; }
    | for_statement { $$ = $1; }
    | return_statement SEMICOLON { $$ = $1; }
    | try_catch_block { $$ = $1; }
    | func_call SEMICOLON { $$ = $1; }
    | block { $$ = $1; }
;

assignment:
    IDENTIFIER EQ expression { 
        $$ = make_shared<Assignment>($1, $3); 
    }
;

if_statement:
    IF LPAREN expression RPAREN block { 
        $$ = make_shared<IfStatement>($3, $5); 
    }
;

while_statement:
    WHILE LPAREN expression RPAREN block { 
        $$ = make_shared<WhileStatement>($3, $5); 
    }
;

for_statement:
    FOR LPAREN assignment expression SEMICOLON assignment RPAREN block { 
        $$ = make_shared<ForStatement>($3, $5, $7, $9); 
    }
;

return_statement:
    RETURN expression { 
        $$ = make_shared<ReturnStatement>($2); 
    }
;

try_catch_block:
    TRY block CATCH LPAREN IDENTIFIER RPAREN block {
        $$ = make_shared<TryCatchBlock>($2, $5, $7);
    }
;

block:
    LBRACE statements RBRACE { $$ = make_shared<Block>($2); }
;

function_declaration:
    IDENTIFIER LPAREN params RPAREN block { 
        $$ = make_shared<FuncDeclaration>($1, $3, $5); 
    }
;

params:
    IDENTIFIER { $$ = { $1 }; }
    | IDENTIFIER COMMA params { $$ = { $1 }; $$->insert($$.end(), $3.begin(), $3.end()); }
;

func_call:
    IDENTIFIER LPAREN arguments RPAREN { 
        $$ = make_shared<FuncCall>($1, $3); 
    }
;

arguments:
    expression { $$ = { $1 }; }
    | expression COMMA arguments { $$ = { $1 }; $$->insert($$.end(), $3.begin(), $3.end()); }
;

expression:
    IDENTIFIER { $$ = make_shared<VarReference>($1); }
    | INT { $$ = make_shared<Literal>($1); }
    | LPAREN expression RPAREN { $$ = $2; }
    | expression PLUS expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::ADD, $1, $3); 
    }
    | expression MINUS expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::SUBTRACT, $1, $3); 
    }
    | expression MUL expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::MULTIPLY, $1, $3); 
    }
    | expression DIV expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::DIVIDE, $1, $3); 
    }
    | IDENTIFIER LPAREN arguments RPAREN { 
        $$ = make_shared<FuncCall>($1, $3); 
    }
    | LAMBDA LPAREN params RPAREN expression { 
        $$ = make_shared<LambdaExpression>($3, $5);
    }
;

class_declaration:
    CLASS IDENTIFIER LBRACE class_body RBRACE { 
        $$ = make_shared<ClassDeclaration>($2, $4); 
    }
;

class_body:
    class_member class_body { $$ = $1; $$->addMember($2); }
    | /* empty */ { $$ = make_shared<ClassBody>(); }
;

class_member:
    IDENTIFIER IDENTIFIER SEMICOLON { 
        $$ = make_shared<ClassMember>($1, $2); 
    }
    | function_declaration { $$ = $1; }
;

object_creation:
    OBJECT IDENTIFIER LPAREN arguments RPAREN { 
        $$ = make_shared<ObjectCreation>($2, $4); 
    }
;

%%

int main() {
    yyparse();
    return 0;
}

%{
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>

using namespace std;

enum class NodeType { 
    PROGRAM, 
    VAR_DECLARATION, 
    FUNC_DECLARATION, 
    ASSIGNMENT, 
    IF_STATEMENT, 
    WHILE_STATEMENT, 
    FOR_STATEMENT, 
    EXPRESSION, 
    CALL, 
    LITERAL, 
    BINARY_OP,
    FUNC_CALL,
    LAMBDA_EXP,
    CLASS_DECLARATION,
    OBJECT_CREATION,
    ARRAY_DECLARATION,
    MAP_DECLARATION,
    ERROR_HANDLING,
    THREAD,
    ASYNC,
    AWAIT
};

class Node {
public:
    virtual NodeType getNodeType() const = 0;
    virtual ~Node() {}
};

class Expression : public Node {
public:
    virtual void evaluate() = 0;
};

class Program : public Node {
    vector<shared_ptr<Node>> statements;
public:
    void addStatement(shared_ptr<Node> stmt) {
        statements.push_back(stmt);
    }

    NodeType getNodeType() const override {
        return NodeType::PROGRAM;
    }

    void execute() {
        for (auto &stmt : statements) {
            stmt->execute();
        }
    }
};

%}

%union {
    int int_val;
    std::string* str_val;
    std::shared_ptr<Node> node;
    std::vector<std::shared_ptr<Node>>* node_list;
}

%token <int_val> INT
%token <str_val> IDENTIFIER
%token IF WHILE FOR RETURN TRY CATCH ASYNC AWAIT THREAD
%token EQ NEQ LT GT LE GE
%token PLUS MINUS MUL DIV
%token LPAREN RPAREN LBRACE RBRACE SEMICOLON COMMA COLON
%token CLASS OBJECT LAMBDA ARRAY MAP
%token T F

%type <node> program statement assignment if_statement while_statement for_statement return_statement try_catch_block block expression function_declaration func_call return_statement try_catch_block class_declaration object_creation array_declaration map_declaration thread_statement async_statement await_statement

%%

program:
    statements { 
        $$ = make_shared<Program>(); 
        for (auto &stmt : $1) {
            $$->addStatement(stmt);
        }
    }
;

statements:
    statement statements { $$ = $1; $$->push_back($2); }
    | /* empty */ { $$ = make_shared<vector<shared_ptr<Node>>>(); }
;

statement:
    assignment SEMICOLON { $$ = $1; }
    | if_statement { $$ = $1; }
    | while_statement { $$ = $1; }
    | for_statement { $$ = $1; }
    | return_statement SEMICOLON { $$ = $1; }
    | try_catch_block { $$ = $1; }
    | func_call SEMICOLON { $$ = $1; }
    | block { $$ = $1; }
    | thread_statement { $$ = $1; }
    | async_statement { $$ = $1; }
    | await_statement { $$ = $1; }
;

assignment:
    IDENTIFIER EQ expression { 
        $$ = make_shared<Assignment>($1, $3); 
    }
;

if_statement:
    IF LPAREN expression RPAREN block { 
        $$ = make_shared<IfStatement>($3, $5); 
    }
;

while_statement:
    WHILE LPAREN expression RPAREN block { 
        $$ = make_shared<WhileStatement>($3, $5); 
    }
;

for_statement:
    FOR LPAREN assignment expression SEMICOLON assignment RPAREN block { 
        $$ = make_shared<ForStatement>($3, $5, $7, $9); 
    }
;

return_statement:
    RETURN expression { 
        $$ = make_shared<ReturnStatement>($2); 
    }
;

try_catch_block:
    TRY block CATCH LPAREN IDENTIFIER RPAREN block {
        $$ = make_shared<TryCatchBlock>($2, $5, $7);
    }
;

block:
    LBRACE statements RBRACE { $$ = make_shared<Block>($2); }
;

function_declaration:
    IDENTIFIER LPAREN params RPAREN block { 
        $$ = make_shared<FuncDeclaration>($1, $3, $5); 
    }
;

params:
    IDENTIFIER { $$ = { $1 }; }
    | IDENTIFIER COMMA params { $$ = { $1 }; $$->insert($$.end(), $3.begin(), $3.end()); }
;

func_call:
    IDENTIFIER LPAREN arguments RPAREN { 
        $$ = make_shared<FuncCall>($1, $3); 
    }
;

arguments:
    expression { $$ = { $1 }; }
    | expression COMMA arguments { $$ = { $1 }; $$->insert($$.end(), $3.begin(), $3.end()); }
;

expression:
    IDENTIFIER { $$ = make_shared<VarReference>($1); }
    | INT { $$ = make_shared<Literal>($1); }
    | LPAREN expression RPAREN { $$ = $2; }
    | expression PLUS expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::ADD, $1, $3); 
    }
    | expression MINUS expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::SUBTRACT, $1, $3); 
    }
    | expression MUL expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::MULTIPLY, $1, $3); 
    }
    | expression DIV expression { 
        $$ = make_shared<BinaryOp>(BinaryOp::DIVIDE, $1, $3); 
    }
    | IDENTIFIER LPAREN arguments RPAREN { 
        $$ = make_shared<FuncCall>($1, $3); 
    }
    | LAMBDA LPAREN params RPAREN expression { 
        $$ = make_shared<LambdaExpression>($3, $5);
    }
;

class_declaration:
    CLASS IDENTIFIER LBRACE class_body RBRACE { 
        $$ = make_shared<ClassDeclaration>($2, $4); 
    }
;

class_body:
    class_member class_body { $$ = $1; $$->addMember($2); }
    | /* empty */ { $$ = make_shared<ClassBody>(); }
;

class_member:
    IDENTIFIER IDENTIFIER SEMICOLON { 
        $$ = make_shared<ClassMember>($1, $2); 
    }
    | function_declaration { $$ = $1; }
;

object_creation:
    OBJECT IDENTIFIER LPAREN arguments RPAREN { 
        $$ = make_shared<ObjectCreation>($2, $4); 
    }
;

array_declaration:
    ARRAY IDENTIFIER EQ LBRACE elements RBRACE { 
        $$ = make_shared<ArrayDeclaration>($2, $4); 
    }
;

elements:
    expression COMMA elements { $$ = { $1 }; $$->insert($$.end(), $3.begin(), $3.end()); }
    | expression { $$ = { $1 }; }
;

map_declaration:
    MAP IDENTIFIER EQ LBRACE key_value_pairs RBRACE { 
        $$ = make_shared<MapDeclaration>($2, $4); 
    }
;

key_value_pairs:
    expression COLON expression COMMA key_value_pairs { $$ = { $1, $3 }; $$->insert($$.end(), $5.begin(), $5.end()); }
    | expression COLON expression { $$ = { $1, $3 }; }
;

thread_statement:
    THREAD IDENTIFIER LPAREN arguments RPAREN block { 
        $$ = make_shared<ThreadStatement>($2, $4, $6);
    }
;

async_statement:
    ASYNC block { 
        $$ = make_shared<AsyncStatement>($2);
    }
;

await_statement:
    AWAIT expression { 
        $$ = make_shared<AwaitStatement>($2);
    }
;

%%

int main() {
    yyparse();
    return 0;
}
