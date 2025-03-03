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
