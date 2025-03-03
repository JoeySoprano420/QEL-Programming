// ast.hpp
#ifndef AST_HPP
#define AST_HPP

#include <string>
#include <memory>

// Abstract base class for all AST nodes
class ExprNode {
public:
    virtual ~ExprNode() {}
    virtual void print() = 0;
};

// Concrete classes for different types of nodes
class IntegerNode : public ExprNode {
public:
    int value;
    IntegerNode(int val) : value(val) {}
    void print() override {
        std::cout << value;
    }
};

class VariableNode : public ExprNode {
public:
    std::string name;
    VariableNode(const std::string& n) : name(n) {}
    void print() override {
        std::cout << name;
    }
};

class AddNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    AddNode(std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r)
        : left(l), right(r) {}
    void print() override {
        left->print();
        std::cout << " + ";
        right->print();
    }
};

class SubtractNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    SubtractNode(std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r)
        : left(l), right(r) {}
    void print() override {
        left->print();
        std::cout << " - ";
        right->print();
    }
};

class IfExprNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> condition;
    std::shared_ptr<ExprNode> then_expr;
    std::shared_ptr<ExprNode> else_expr;
    IfExprNode(std::shared_ptr<ExprNode> cond, std::shared_ptr<ExprNode> then_e, std::shared_ptr<ExprNode> else_e)
        : condition(cond), then_expr(then_e), else_expr(else_e) {}
    void print() override {
        std::cout << "if ";
        condition->print();
        std::cout << " then ";
        then_expr->print();
        std::cout << " else ";
        else_expr->print();
    }
};

#endif

// AST for while and for loops
class WhileLoopNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> condition;
    std::shared_ptr<ExprNode> body;
    WhileLoopNode(std::shared_ptr<ExprNode> cond, std::shared_ptr<ExprNode> b)
        : condition(cond), body(b) {}
    void print() override {
        std::cout << "while ";
        condition->print();
        std::cout << " { ";
        body->print();
        std::cout << " }";
    }
};

class ForLoopNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> initialization;
    std::shared_ptr<ExprNode> condition;
    std::shared_ptr<ExprNode> increment;
    std::shared_ptr<ExprNode> body;
    ForLoopNode(std::shared_ptr<ExprNode> init, std::shared_ptr<ExprNode> cond, 
                std::shared_ptr<ExprNode> inc, std::shared_ptr<ExprNode> b)
        : initialization(init), condition(cond), increment(inc), body(b) {}
    void print() override {
        std::cout << "for (";
        initialization->print();
        std::cout << "; ";
        condition->print();
        std::cout << "; ";
        increment->print();
        std::cout << ") { ";
        body->print();
        std::cout << " }";
    }
};

class FunctionNode : public ExprNode {
public:
    std::string name;
    std::shared_ptr<ParameterNode> parameters;
    std::shared_ptr<ExprNode> body;

    FunctionNode(const std::string& n, std::shared_ptr<ParameterNode> p, std::shared_ptr<ExprNode> b)
        : name(n), parameters(p), body(b) {}

    void print() override {
        std::cout << "function " << name << "(";
        parameters->print();
        std::cout << ") { ";
        body->print();
        std::cout << " }";
    }
};

enum class Type {
    INT,
    BOOL,
    VOID
};

class TypeNode : public ExprNode {
public:
    Type type;

    TypeNode(Type t) : type(t) {}

    void print() override {
        switch (type) {
            case Type::INT:
                std::cout << "int";
                break;
            case Type::BOOL:
                std::cout << "bool";
                break;
            case Type::VOID:
                std::cout << "void";
                break;
        }
    }
};

