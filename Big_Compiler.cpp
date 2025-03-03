#include <iostream>
#include <regex>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <functional>

// ---- [Token Types] ----
enum class TokenType {
    Keyword, Identifier, Number, Operator, Punctuation, EndOfFile, Invalid
};

enum class OperatorType {
    Plus, Minus, Multiply, Divide, Equal, LessThan, GreaterThan, NotEqual
};

// ---- [Token Structure] ----
struct Token {
    TokenType type;
    std::string value;
    size_t position;
};

// ---- [Lexer (Lexical Analyzer)] ----
class Lexer {
public:
    Lexer(const std::string& source) : source(source), position(0) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (position < source.size()) {
            char currentChar = source[position];

            // Skip whitespace
            if (isspace(currentChar)) {
                position++;
                continue;
            }

            // Check for keywords
            if (std::regex_match(std::string(1, currentChar), std::regex("[a-zA-Z_]"))) {
                std::string word = readWord();
                if (word == "if" || word == "else" || word == "while" || word == "return") {
                    tokens.push_back({TokenType::Keyword, word, position});
                } else {
                    tokens.push_back({TokenType::Identifier, word, position});
                }
                continue;
            }

            // Check for numbers
            if (isdigit(currentChar)) {
                std::string number = readNumber();
                tokens.push_back({TokenType::Number, number, position});
                continue;
            }

            // Check for operators
            if (std::regex_match(std::string(1, currentChar), std::regex("[+-*/=<>]"))) {
                tokens.push_back({TokenType::Operator, std::string(1, currentChar), position});
                position++;
                continue;
            }

            // Check for punctuation (e.g., semicolon, parentheses)
            if (std::regex_match(std::string(1, currentChar), std::regex("[;(){}]"))) {
                tokens.push_back({TokenType::Punctuation, std::string(1, currentChar), position});
                position++;
                continue;
            }

            // Invalid character
            tokens.push_back({TokenType::Invalid, std::string(1, currentChar), position});
            position++;
        }
        tokens.push_back({TokenType::EndOfFile, "", position});
        return tokens;
    }

private:
    std::string source;
    size_t position;

    std::string readWord() {
        std::string word;
        while (position < source.size() && std::isalnum(source[position])) {
            word += source[position++];
        }
        return word;
    }

    std::string readNumber() {
        std::string number;
        while (position < source.size() && std::isdigit(source[position])) {
            number += source[position++];
        }
        return number;
    }
};

// ---- [AST Nodes] ----
struct ASTNode {
    virtual ~ASTNode() = default;
};

struct NumberNode : public ASTNode {
    int value;
    NumberNode(int value) : value(value) {}
};

struct IdentifierNode : public ASTNode {
    std::string name;
    IdentifierNode(const std::string& name) : name(name) {}
};

struct BinaryOpNode : public ASTNode {
    OperatorType op;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;

    BinaryOpNode(OperatorType op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right)
        : op(op), left(std::move(left)), right(std::move(right)) {}
};

// ---- [Parser (Recursive Descent)] ----
class Parser {
public:
    Parser(std::vector<Token>& tokens) : tokens(tokens), position(0) {}

    std::unique_ptr<ASTNode> parse() {
        return parseExpression();
    }

private:
    std::vector<Token>& tokens;
    size_t position;

    Token currentToken() {
        return tokens[position];
    }

    void advance() {
        position++;
    }

    std::unique_ptr<ASTNode> parseExpression() {
        auto left = parseTerm();

        while (currentToken().type == TokenType::Operator && (currentToken().value == "+" || currentToken().value == "-")) {
            OperatorType op = currentToken().value == "+" ? OperatorType::Plus : OperatorType::Minus;
            advance();
            auto right = parseTerm();
            left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<ASTNode> parseTerm() {
        auto left = parseFactor();

        while (currentToken().type == TokenType::Operator && (currentToken().value == "*" || currentToken().value == "/")) {
            OperatorType op = currentToken().value == "*" ? OperatorType::Multiply : OperatorType::Divide;
            advance();
            auto right = parseFactor();
            left = std::make_unique<BinaryOpNode>(op, std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<ASTNode> parseFactor() {
        if (currentToken().type == TokenType::Number) {
            int value = std::stoi(currentToken().value);
            advance();
            return std::make_unique<NumberNode>(value);
        }

        if (currentToken().type == TokenType::Identifier) {
            std::string name = currentToken().value;
            advance();
            return std::make_unique<IdentifierNode>(name);
        }

        // Error Handling: Unmatched tokens, missing parentheses
        std::cerr << "Syntax error: Unexpected token " << currentToken().value << std::endl;
        exit(1);
    }
};

// ---- [Code Generation] ----
class CodeGenerator {
public:
    void generate(const ASTNode* root) {
        if (const NumberNode* numNode = dynamic_cast<const NumberNode*>(root)) {
            std::cout << "PUSH " << numNode->value << "\n";
        } else if (const IdentifierNode* idNode = dynamic_cast<const IdentifierNode*>(root)) {
            std::cout << "LOAD " << idNode->name << "\n";
        } else if (const BinaryOpNode* binOpNode = dynamic_cast<const BinaryOpNode*>(root)) {
            generate(binOpNode->left.get());
            generate(binOpNode->right.get());
            std::string opCode = opToStr(binOpNode->op);
            std::cout << opCode << "\n";
        }
    }

private:
    std::string opToStr(OperatorType op) {
        switch (op) {
            case OperatorType::Plus: return "ADD";
            case OperatorType::Minus: return "SUB";
            case OperatorType::Multiply: return "MUL";
            case OperatorType::Divide: return "DIV";
            default: return "UNKNOWN";
        }
    }
};

// ---- [Execution Loop] ----
int main() {
    std::string code = "3 + 4 * 5";
    
    // Step 1: Lexer
    Lexer lexer(code);
    std::vector<Token> tokens = lexer.tokenize();
    
    // Step 2: Parser
    Parser parser(tokens);
    auto ast = parser.parse();
    
    // Step 3: Code Generation
    CodeGenerator codeGen;
    codeGen.generate(ast.get());

    return 0;
}

#include <iostream>
#include <regex>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <functional>

// ---- [Token Types] ----
enum class TokenType {
    Keyword, Identifier, Number, Operator, Punctuation, Type, EndOfFile, Invalid
};

enum class OperatorType {
    Plus, Minus, Multiply, Divide, Equal, LessThan, GreaterThan, NotEqual
};

// ---- [AST Node Types] ----
enum class ASTNodeType {
    Number, Identifier, BinaryOp, Assignment, FunctionDecl, FunctionCall, VariableDecl
};

// ---- [Token Structure] ----
struct Token {
    TokenType type;
    std::string value;
    size_t position;
};

// ---- [AST Nodes] ----
struct ASTNode {
    virtual ~ASTNode() = default;
    virtual ASTNodeType nodeType() const = 0;
};

struct NumberNode : public ASTNode {
    int value;
    NumberNode(int value) : value(value) {}
    ASTNodeType nodeType() const override { return ASTNodeType::Number; }
};

struct IdentifierNode : public ASTNode {
    std::string name;
    IdentifierNode(const std::string& name) : name(name) {}
    ASTNodeType nodeType() const override { return ASTNodeType::Identifier; }
};

struct BinaryOpNode : public ASTNode {
    OperatorType op;
    std::unique_ptr<ASTNode> left;
    std::unique_ptr<ASTNode> right;

    BinaryOpNode(OperatorType op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right)
        : op(op), left(std::move(left)), right(std::move(right)) {}

    ASTNodeType nodeType() const override { return ASTNodeType::BinaryOp; }
};

struct AssignmentNode : public ASTNode {
    std::unique_ptr<IdentifierNode> variable;
    std::unique_ptr<ASTNode> expression;

    AssignmentNode(std::unique_ptr<IdentifierNode> variable, std::unique_ptr<ASTNode> expression)
        : variable(std::move(variable)), expression(std::move(expression)) {}

    ASTNodeType nodeType() const override { return ASTNodeType::Assignment; }
};

struct VariableDeclNode : public ASTNode {
    std::string type;
    std::unique_ptr<IdentifierNode> variable;

    VariableDeclNode(const std::string& type, std::unique_ptr<IdentifierNode> variable)
        : type(type), variable(std::move(variable)) {}

    ASTNodeType nodeType() const override { return ASTNodeType::VariableDecl; }
};

struct FunctionDeclNode : public ASTNode {
    std::string name;
    std::vector<std::string> parameters;
    std::unique_ptr<ASTNode> body;

    FunctionDeclNode(const std::string& name, const std::vector<std::string>& parameters, std::unique_ptr<ASTNode> body)
        : name(name), parameters(parameters), body(std::move(body)) {}

    ASTNodeType nodeType() const override { return ASTNodeType::FunctionDecl; }
};

struct FunctionCallNode : public ASTNode {
    std::string name;
    std::vector<std::unique_ptr<ASTNode>> arguments;

    FunctionCallNode(const std::string& name, std::vector<std::unique_ptr<ASTNode>> arguments)
        : name(name), arguments(std::move(arguments)) {}

    ASTNodeType nodeType() const override { return ASTNodeType::FunctionCall; }
};

// ---- [Lexer (Lexical Analysis)] ----
class Lexer {
public:
    Lexer(const std::string& source) : source(source), position(0) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (position < source.size()) {
            char currentChar = source[position];

            // Skip whitespace
            if (isspace(currentChar)) {
                position++;
                continue;
            }

            // Check for keywords (if, else, while, return, etc.)
            if (std::regex_match(std::string(1, currentChar), std::regex("[a-zA-Z_]"))) {
                std::string word = readWord();
                if (word == "if" || word == "else" || word == "while" || word == "return") {
                    tokens.push_back({TokenType::Keyword, word, position});
                } else if (word == "int" || word == "float" || word == "void") {
                    tokens.push_back({TokenType::Type, word, position});
                } else {
                    tokens.push_back({TokenType::Identifier, word, position});
                }
                continue;
            }

            // Check for numbers
            if (isdigit(currentChar)) {
                std::string number = readNumber();
                tokens.push_back({TokenType::Number, number, position});
                continue;
            }

            // Check for operators
            if (std::regex_match(std::string(1, currentChar), std::regex("[+-*/=<>]"))) {
                tokens.push_back({TokenType::Operator, std::string(1, currentChar), position});
                position++;
                continue;
            }

            // Check for punctuation (e.g., semicolon, parentheses)
            if (std::regex_match(std::string(1, currentChar), std::regex("[;(){}]"))) {
                tokens.push_back({TokenType::Punctuation, std::string(1, currentChar), position});
                position++;
                continue;
            }

            // Invalid character
            tokens.push_back({TokenType::Invalid, std::string(1, currentChar), position});
            position++;
        }
        tokens.push_back({TokenType::EndOfFile, "", position});
        return tokens;
    }

private:
    std::string source;
    size_t position;

    std::string readWord() {
        std::string word;
        while (position < source.size() && std::isalnum(source[position])) {
            word += source[position++];
        }
        return word;
    }

    std::string readNumber() {
        std::string number;
        while (position < source.size() && std::isdigit(source[position])) {
            number += source[position++];
        }
        return number;
    }
};

// ---- [Parser (Recursive Descent)] ----
class Parser {
public:
    Parser(std::vector<Token>& tokens) : tokens(tokens), position(0) {}

    std::unique_ptr<ASTNode> parse() {
        return parseProgram();
    }

private:
    std::vector<Token>& tokens;
    size_t position;

    Token currentToken() {
        return tokens[position];
    }

    void advance() {
        position++;
    }

    std::unique_ptr<ASTNode> parseProgram() {
        std::vector<std::unique_ptr<ASTNode>> statements;

        while (currentToken().type != TokenType::EndOfFile) {
            statements.push_back(parseStatement());
        }

        // Wrapping up statements into a block (could be expanded further)
        return std::make_unique<BinaryOpNode>(OperatorType::Plus, std::move(statements.front()), std::move(statements.back()));  // Temporary handling
    }

    std::unique_ptr<ASTNode> parseStatement() {
        if (currentToken().type == TokenType::Keyword && currentToken().value == "int") {
            return parseVariableDeclaration();
        } else if (currentToken().type == TokenType::Identifier) {
            return parseAssignmentOrFunctionCall();
        }
        
        // Handle error cases
        std::cerr << "Error: Unrecognized statement\n";
        exit(1);
    }

    std::unique_ptr<ASTNode> parseVariableDeclaration() {
        advance();  // Skip 'int' or other type
        if (currentToken().type != TokenType::Identifier) {
            std::cerr << "Error: Expected variable name after type declaration\n";
            exit(1);
        }
        
        std::string varName = currentToken().value;
        advance();
        return std::make_unique<VariableDeclNode>("int", std::make_unique<IdentifierNode>(varName));
    }

    std::unique_ptr<ASTNode> parseAssignmentOrFunctionCall() {
        std::string varName = currentToken().value;
        advance();
        
        if (currentToken().type == TokenType::Operator && currentToken().value == "=") {
            advance();
            auto expr = parseExpression();
            return std::make_unique<AssignmentNode>(std::make_unique<IdentifierNode>(varName), std::move(expr));
        } else {
            return parseFunctionCall(varName);
        }
    }

    std::unique_ptr<ASTNode> parseExpression() {
        // Placeholder for expression parsing, like adding, multiplying, etc.
        return std::make_unique<NumberNode>(5);  // Temporary logic
    }

    std::unique_ptr<ASTNode> parseFunctionCall(const std::string& funcName) {
        // Placeholder: This would handle function calls.
        return std::make_unique<FunctionCallNode>(funcName, {});
    }
};

// ---- [Code Generation] ----
class CodeGenerator {
public:
    void generate(const ASTNode* root) {
        if (root->nodeType() == ASTNodeType::Number) {
            const NumberNode* numNode = dynamic_cast<const NumberNode*>(root);
            std::cout << "PUSH " << numNode->value << "\n";
        } else if (root->nodeType() == ASTNodeType::Identifier) {
            const IdentifierNode* idNode = dynamic_cast<const IdentifierNode*>(root);
            std::cout << "LOAD " << idNode->name << "\n";
        } else if (root->nodeType() == ASTNodeType::Assignment) {
            const AssignmentNode* assignNode = dynamic_cast<const AssignmentNode*>(root);
            generate(assignNode->expression.get());
            std::cout << "STORE " << assignNode->variable->name << "\n";
        }
    }
};

// ---- [Main Execution] ----
int main() {
    std::string input = "int x = 5; x = 10;";

    Lexer lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator generator;
    generator.generate(ast.get());

    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <cassert>

// ---- [Tokenization] ----
enum class TokenType {
    INTEGER,
    PLUS,
    MINUS,
    MUL,
    DIV,
    LPAREN,
    RPAREN,
    IDENTIFIER,
    ASSIGN,
    END,
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string value;
};

// ---- [Lexer: Tokenizer] ----
class Lexer {
public:
    Lexer(const std::string& input) : input(input), pos(0) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (pos < input.length()) {
            char currentChar = input[pos];
            if (isdigit(currentChar)) {
                tokens.push_back({TokenType::INTEGER, parseNumber()});
            } else if (currentChar == '+') {
                tokens.push_back({TokenType::PLUS, "+"});
                pos++;
            } else if (currentChar == '-') {
                tokens.push_back({TokenType::MINUS, "-"});
                pos++;
            } else if (currentChar == '*') {
                tokens.push_back({TokenType::MUL, "*"});
                pos++;
            } else if (currentChar == '/') {
                tokens.push_back({TokenType::DIV, "/"});
                pos++;
            } else if (currentChar == '(') {
                tokens.push_back({TokenType::LPAREN, "("});
                pos++;
            } else if (currentChar == ')') {
                tokens.push_back({TokenType::RPAREN, ")"});
                pos++;
            } else if (isalpha(currentChar)) {
                tokens.push_back({TokenType::IDENTIFIER, parseIdentifier()});
            } else if (currentChar == '=') {
                tokens.push_back({TokenType::ASSIGN, "="});
                pos++;
            } else {
                tokens.push_back({TokenType::UNKNOWN, std::string(1, currentChar)});
                pos++;
            }
        }
        tokens.push_back({TokenType::END, ""});
        return tokens;
    }

private:
    std::string input;
    size_t pos;

    std::string parseNumber() {
        size_t start = pos;
        while (pos < input.length() && isdigit(input[pos])) pos++;
        return input.substr(start, pos - start);
    }

    std::string parseIdentifier() {
        size_t start = pos;
        while (pos < input.length() && isalnum(input[pos])) pos++;
        return input.substr(start, pos - start);
    }
};

// ---- [Math Functions for Quadratic Equations] ----
class QuadraticSolver {
public:
    // Solving ax^2 + bx + c = 0 using the quadratic formula
    static std::pair<double, double> solve(double a, double b, double c) {
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            throw std::invalid_argument("No real roots");
        }
        double sqrtDiscriminant = std::sqrt(discriminant);
        double root1 = (-b + sqrtDiscriminant) / (2 * a);
        double root2 = (-b - sqrtDiscriminant) / (2 * a);
        return {root1, root2};
    }
};

// ---- [Multi-State Conditional Assertions] ----
class ConditionalAssertion {
public:
    // Multi-state conditional check
    static bool assertMultiState(const std::function<bool()>& condition) {
        return condition();
    }
};

// ---- [Physics Laws] ----
class PhysicsLaws {
public:
    // Newton's Second Law: F = ma
    static double newtonsSecondLaw(double mass, double acceleration) {
        return mass * acceleration;
    }

    // Conservation of Energy: E = mc^2 (Simplified model)
    static double conservationOfEnergy(double mass) {
        const double speedOfLight = 299792458; // m/s
        return mass * speedOfLight * speedOfLight;
    }

    // Schrödinger Equation (simplified form): 
    // i*h_bar * d(psi)/dt = H*psi
    // Simplified to a constant energy for this example.
    static double schrodingerEquation(double waveFunction) {
        const double hBar = 1.0545718e-34; // Reduced Planck's constant
        return hBar * waveFunction;  // Simplified assumption for constant energy solution
    }
};

// ---- [Parser] ----
class Parser {
public:
    Parser(const std::vector<Token>& tokens) : tokens(tokens), pos(0) {}

    double parseExpression() {
        double result = parseTerm();
        while (tokens[pos].type == TokenType::PLUS || tokens[pos].type == TokenType::MINUS) {
            Token op = tokens[pos++];
            double term = parseTerm();
            if (op.type == TokenType::PLUS) {
                result += term;
            } else if (op.type == TokenType::MINUS) {
                result -= term;
            }
        }
        return result;
    }

private:
    std::vector<Token> tokens;
    size_t pos;

    double parseTerm() {
        double result = parseFactor();
        while (tokens[pos].type == TokenType::MUL || tokens[pos].type == TokenType::DIV) {
            Token op = tokens[pos++];
            double factor = parseFactor();
            if (op.type == TokenType::MUL) {
                result *= factor;
            } else if (op.type == TokenType::DIV) {
                result /= factor;
            }
        }
        return result;
    }

    double parseFactor() {
        if (tokens[pos].type == TokenType::INTEGER) {
            return std::stod(tokens[pos++].value);
        } else if (tokens[pos].type == TokenType::LPAREN) {
            pos++;  // Skip '('
            double result = parseExpression();
            if (tokens[pos].type == TokenType::RPAREN) pos++;  // Skip ')'
            return result;
        }
        throw std::runtime_error("Unexpected token: " + tokens[pos].value);
    }
};

// ---- [Code Generation] ----
class CodeGenerator {
public:
    void generate(double result) {
        std::cout << "Generated result: " << result << "\n";
    }
};

// ---- [Main Execution] ----
int main() {
    std::string input = "(3 * 5) + (x = 10)";

    Lexer lexer(input);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    double result = parser.parseExpression();
    std::cout << "Expression Result: " << result << "\n";

    // Example of solving a quadratic equation
    try {
        auto roots = QuadraticSolver::solve(1, -3, 2); // x^2 - 3x + 2 = 0
        std::cout << "Roots of the quadratic equation: " << roots.first << ", " << roots.second << "\n";
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << "\n";
    }

    // Example of a physical law: Newton's Second Law (F = ma)
    double force = PhysicsLaws::newtonsSecondLaw(5.0, 9.8); // mass = 5kg, acceleration = 9.8 m/s^2
    std::cout << "Force: " << force << " N\n";

    // Example of multi-state assertion
    bool assertion = ConditionalAssertion::assertMultiState([]() {
        return 3 > 2;  // Conditional check
    });
    std::cout << "Multi-State Assertion result: " << (assertion ? "True" : "False") << "\n";

    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <cassert>
#include <unordered_map>
#include <set>

// ---- [Tokenization] ----
enum class TokenType {
    INTEGER,
    PLUS,
    MINUS,
    MUL,
    DIV,
    LPAREN,
    RPAREN,
    IDENTIFIER,
    ASSIGN,
    END,
    UNKNOWN
};

struct Token {
    TokenType type;
    std::string value;
};

// ---- [Lexer: Tokenizer] ----
class Lexer {
public:
    Lexer(const std::string& input) : input(input), pos(0) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (pos < input.length()) {
            char currentChar = input[pos];
            if (isdigit(currentChar)) {
                tokens.push_back({TokenType::INTEGER, parseNumber()});
            } else if (currentChar == '+') {
                tokens.push_back({TokenType::PLUS, "+"});
                pos++;
            } else if (currentChar == '-') {
                tokens.push_back({TokenType::MINUS, "-"});
                pos++;
            } else if (currentChar == '*') {
                tokens.push_back({TokenType::MUL, "*"});
                pos++;
            } else if (currentChar == '/') {
                tokens.push_back({TokenType::DIV, "/"});
                pos++;
            } else if (currentChar == '(') {
                tokens.push_back({TokenType::LPAREN, "("});
                pos++;
            } else if (currentChar == ')') {
                tokens.push_back({TokenType::RPAREN, ")"});
                pos++;
            } else if (isalpha(currentChar)) {
                tokens.push_back({TokenType::IDENTIFIER, parseIdentifier()});
            } else if (currentChar == '=') {
                tokens.push_back({TokenType::ASSIGN, "="});
                pos++;
            } else {
                tokens.push_back({TokenType::UNKNOWN, std::string(1, currentChar)});
                pos++;
            }
        }
        tokens.push_back({TokenType::END, ""});
        return tokens;
    }

private:
    std::string input;
    size_t pos;

    std::string parseNumber() {
        size_t start = pos;
        while (pos < input.length() && isdigit(input[pos])) pos++;
        return input.substr(start, pos - start);
    }

    std::string parseIdentifier() {
        size_t start = pos;
        while (pos < input.length() && isalnum(input[pos])) pos++;
        return input.substr(start, pos - start);
    }
};

// ---- [Intermediate Representation (IR)] ----
class IRGenerator {
public:
    IRGenerator() {}

    void generateIR(const std::string& code) {
        // Here we transform the code into an intermediate representation (IR)
        // For the sake of simplicity, we will represent IR as simple assembly instructions
        irCode.push_back("MOV R0, " + code);  // Example of IR generation
    }

    void printIR() {
        for (const auto& line : irCode) {
            std::cout << line << std::endl;
        }
    }

private:
    std::vector<std::string> irCode;
};

// ---- [Optimizations] ----
class Optimizer {
public:
    Optimizer() {}

    // Example optimization: Constant folding (e.g., 2 + 2 => 4)
    void applyConstantFolding(IRGenerator& ir) {
        for (auto& line : irCode) {
            if (line.find("+") != std::string::npos) {
                // If we detect a simple addition operation, attempt constant folding
                size_t plusPos = line.find("+");
                size_t left = std::stoi(line.substr(0, plusPos));
                size_t right = std::stoi(line.substr(plusPos + 1));
                line = std::to_string(left + right);
            }
        }
    }

    // Dead code elimination or common subexpression elimination can be added here...

private:
    std::vector<std::string> irCode;
};

// ---- [Code Generation - Target Architecture] ----
class CodeGenerator {
public:
    CodeGenerator() {}

    void generateCode(const IRGenerator& ir) {
        std::cout << "Generating assembly for AMD Ryzen 3 Series 7000..." << std::endl;
        for (const auto& line : ir.getIR()) {
            std::cout << line << std::endl;
        }
    }
};

// ---- [Parser] ----
class Parser {
public:
    Parser(const std::vector<Token>& tokens) : tokens(tokens), pos(0) {}

    double parseExpression() {
        double result = parseTerm();
        while (tokens[pos].type == TokenType::PLUS || tokens[pos].type == TokenType::MINUS) {
            Token op = tokens[pos++];
            double term = parseTerm();
            if (op.type == TokenType::PLUS) {
                result += term;
            } else if (op.type == TokenType::MINUS) {
                result -= term;
            }
        }
        return result;
    }

private:
    std::vector<Token> tokens;
    size_t pos;

    double parseTerm() {
        double result = parseFactor();
        while (tokens[pos].type == TokenType::MUL || tokens[pos].type == TokenType::DIV) {
            Token op = tokens[pos++];
            double factor = parseFactor();
            if (op.type == TokenType::MUL) {
                result *= factor;
            } else if (op.type == TokenType::DIV) {
                result /= factor;
            }
        }
        return result;
    }

    double parseFactor() {
        if (tokens[pos].type == TokenType::INTEGER) {
            return std::stod(tokens[pos++].value);
        } else if (tokens[pos].type == TokenType::LPAREN) {
            pos++;  // Skip '('
            double result = parseExpression();
            if (tokens[pos].type == TokenType::RPAREN) pos++;  // Skip ')'
            return result;
        }
        throw std::runtime_error("Unexpected token: " + tokens[pos].value);
    }
};

// ---- [Main Compiler Logic] ----
int main() {
    // Example input code
    std::string input = "(3 * 5) + (2 + 4)";

    // Lexical analysis (tokenization)
    Lexer lexer(input);
    auto tokens = lexer.tokenize();

    // Syntax parsing
    Parser parser(tokens);
    double result = parser.parseExpression();
    std::cout << "Parsed result: " << result << std::endl;

    // Generate IR (Intermediate Representation)
    IRGenerator irGen;
    irGen.generateIR("MOV R0, 10");
    irGen.printIR();

    // Apply optimization techniques like constant folding
    Optimizer optimizer;
    optimizer.applyConstantFolding(irGen);

    // Generate optimized assembly code
    CodeGenerator codeGen;
    codeGen.generateCode(irGen);

    return 0;
}

#include <iostream>
#include <immintrin.h> // AVX2 for SIMD operations
#include <vector>

// Optimized loop: SIMD Vectorization, Loop Unrolling, Register Allocation

void optimized_addition(std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    // AVX2 SIMD optimization (8 floats per instruction)
    __m256 b_vec, c_vec, result_vec;

    // Loop unrolling by a factor of 8
    for (int i = 0; i < n; i += 8) {
        b_vec = _mm256_loadu_ps(&b[i]);  // Load 8 elements from b[i]
        c_vec = _mm256_loadu_ps(&c[i]);  // Load 8 elements from c[i]
        result_vec = _mm256_add_ps(b_vec, c_vec);  // SIMD addition of 8 elements

        _mm256_storeu_ps(&a[i], result_vec);  // Store results back into a[i]
    }
}

// Function for instruction scheduling - minimal dependency reordering
void process_dependencies(std::vector<int>& data) {
    // Simple instruction scheduling - swap elements after processing (for demonstration)
    for (size_t i = 0; i < data.size() - 1; i++) {
        std::swap(data[i], data[i + 1]);
    }
}

// Function for branch prediction optimization - Profile-guided decision
void branch_optimized_function(int x) {
    if (x > 0) {
        std::cout << "Positive value detected!" << std::endl;
    } else {
        std::cout << "Non-positive value detected!" << std::endl;
    }
}

// Register allocation simulation
void register_allocation_demo() {
    int a = 10;
    int b = 20;
    int c = a + b; // Imagine a, b, and c are placed in registers
    std::cout << "Sum: " << c << std::endl;
}

// Main loop that uses SIMD, instruction scheduling, and optimized register allocation
void optimized_loop_demo() {
    int n = 1000;
    std::vector<float> a(n, 0), b(n, 2.0), c(n, 3.0);

    // SIMD vectorization and loop unrolling for addition
    optimized_addition(a, b, c, n);

    // Simulate instruction scheduling for dependency reordering
    std::vector<int> data = {1, 2, 3, 4, 5};
    process_dependencies(data);

    // Branch prediction optimization example
    branch_optimized_function(1);

    // Simulate register allocation optimization
    register_allocation_demo();
}

// Linker and Assembler Simulation
void simulate_linker_and_assembler() {
    // Linking external libraries or resolving external references
    // Simulate assembly code generation (assembly would normally be generated by a real compiler backend)
    std::cout << "Linking external references..." << std::endl;
    std::cout << "Assembly code generation complete. Proceeding to final linking..." << std::endl;
    std::cout << "Final binary executable generated!" << std::endl;
}

// Main function that ties everything together
int main() {
    // Generate optimized code
    optimized_loop_demo();

    // Simulate the final steps of linking and assembling
    simulate_linker_and_assembler();

    return 0;
}

#include <iostream>
#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <queue>

// Graph coloring for register allocation: Basic example
class RegisterAllocator {
public:
    RegisterAllocator(int numRegisters) : numRegisters(numRegisters) {}

    void allocateRegisters(const std::vector<std::vector<int>>& interferenceGraph) {
        std::vector<int> registerAssignments(interferenceGraph.size(), -1);

        for (size_t node = 0; node < interferenceGraph.size(); ++node) {
            std::unordered_map<int, bool> usedRegisters;
            
            // Check which registers are used by adjacent nodes (interfering variables)
            for (int neighbor : interferenceGraph[node]) {
                if (neighbor < registerAssignments.size() && registerAssignments[neighbor] != -1) {
                    usedRegisters[registerAssignments[neighbor]] = true;
                }
            }

            // Assign the first available register
            for (int reg = 0; reg < numRegisters; ++reg) {
                if (usedRegisters.find(reg) == usedRegisters.end()) {
                    registerAssignments[node] = reg;
                    break;
                }
            }
        }

        printAssignments(registerAssignments);
    }

private:
    void printAssignments(const std::vector<int>& registerAssignments) {
        for (size_t i = 0; i < registerAssignments.size(); ++i) {
            std::cout << "Variable " << i << " -> Register " << registerAssignments[i] << std::endl;
        }
    }

    int numRegisters;
};

// Profile-guided optimization for branch prediction (simple version)
class ProfileGuidedOptimization {
public:
    ProfileGuidedOptimization(const std::vector<int>& branchProfiles) : branchProfiles(branchProfiles) {}

    void optimizeBranchOrder() {
        // Reorder branches based on frequency (larger frequency comes first)
        std::vector<int> indices(branchProfiles.size());
        for (int i = 0; i < branchProfiles.size(); ++i) {
            indices[i] = i;
        }
        
        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            return branchProfiles[a] > branchProfiles[b];
        });

        // Simulate reordering of branches
        std::cout << "Optimized Branch Order:" << std::endl;
        for (int idx : indices) {
            std::cout << "Branch " << idx << " with frequency " << branchProfiles[idx] << std::endl;
        }
    }

private:
    std::vector<int> branchProfiles;
};

// Auto-vectorization: Convert scalar operations to SIMD
void auto_vectorize_addition(std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    // AVX2 SIMD optimization (8 floats per instruction)
    __m256 b_vec, c_vec, result_vec;

    for (int i = 0; i < n; i += 8) {
        b_vec = _mm256_loadu_ps(&b[i]);  // Load 8 elements from b[i]
        c_vec = _mm256_loadu_ps(&c[i]);  // Load 8 elements from c[i]
        result_vec = _mm256_add_ps(b_vec, c_vec);  // SIMD addition of 8 elements

        _mm256_storeu_ps(&a[i], result_vec);  // Store results back into a[i]
    }
}

// Compiler optimization loop simulation: Combining all techniques
void compiler_optimization_simulation() {
    // 1. **Graph Coloring for Register Allocation**
    std::vector<std::vector<int>> interferenceGraph = {
        {1, 2},  // Node 0 interferes with 1, 2
        {0, 2},  // Node 1 interferes with 0, 2
        {0, 1},  // Node 2 interferes with 0, 1
    };
    RegisterAllocator registerAllocator(3);  // 3 registers for demo
    registerAllocator.allocateRegisters(interferenceGraph);

    // 2. **Profile-Guided Optimization (Branch Prediction)**
    std::vector<int> branchProfiles = {10, 5, 20, 15};  // Example branch frequencies
    ProfileGuidedOptimization pgo(branchProfiles);
    pgo.optimizeBranchOrder();

    // 3. **Auto-vectorization for SIMD**
    int n = 1000;
    std::vector<float> a(n, 0), b(n, 2.0), c(n, 3.0);
    auto_vectorize_addition(a, b, c, n);

    std::cout << "Optimized code execution complete!" << std::endl;
}

// Main function: simulates full compiler process
int main() {
    compiler_optimization_simulation();
    return 0;
}

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <immintrin.h>

// Data structure for the intermediate representation (IR) of the code
struct IRInstruction {
    std::string op;  // Operation (e.g., ADD, MUL, etc.)
    std::vector<std::string> operands;  // Operands for the operation
};

// Frontend: Simple lexer and parser (simplified for demonstration)
class Frontend {
public:
    std::vector<IRInstruction> parse(const std::string& sourceCode) {
        std::vector<IRInstruction> irInstructions;
        // Example source code parsing: simple ADD and MUL statements.
        // This is a very simplified parser for demonstration.
        irInstructions.push_back({"ADD", {"R1", "R2", "R3"}});
        irInstructions.push_back({"MUL", {"R4", "R5", "R6"}});
        return irInstructions;
    }
};

// Backend: Generate assembly or machine code for SIMD
class Backend {
public:
    void generateAssembly(const std::vector<IRInstruction>& irInstructions) {
        // Generate AVX2 assembly instructions for SIMD operations.
        for (const auto& ir : irInstructions) {
            if (ir.op == "ADD") {
                std::cout << "vaddps " << ir.operands[0] << ", " << ir.operands[1] << ", " << ir.operands[2] << "\n";
            } else if (ir.op == "MUL") {
                std::cout << "vmulps " << ir.operands[0] << ", " << ir.operands[1] << ", " << ir.operands[2] << "\n";
            }
        }
    }
};

// Optimization Passes: Apply loop peeling, induction variable analysis, and data dependency analysis

class Optimizer {
public:
    // Loop Peeling: Unroll loop for SIMD vectorization
    void applyLoopPeeling(std::vector<IRInstruction>& irInstructions) {
        for (auto& ir : irInstructions) {
            if (ir.op == "ADD" || ir.op == "MUL") {
                // Simulate loop unrolling (peeling) by creating multiple operations
                ir.operands.push_back("SIMD_PEEL");  // Example of peeled loop
                std::cout << "Loop peeled for optimization: " << ir.op << "\n";
            }
        }
    }

    // Induction Variable Analysis: Identifying loop indices and creating SIMD optimizations
    void applyInductionVariableAnalysis(std::vector<IRInstruction>& irInstructions) {
        // Simulate induction variable analysis
        for (auto& ir : irInstructions) {
            if (ir.op == "ADD") {
                // Introduce vectorized version of the operation
                ir.op = "VADD";
                std::cout << "Induction variable analysis applied: " << ir.op << "\n";
            }
        }
    }

    // Data Dependency Analysis: Identify data dependencies and parallelize loops
    void applyDataDependencyAnalysis(std::vector<IRInstruction>& irInstructions) {
        // Simulate data dependency analysis
        for (auto& ir : irInstructions) {
            if (ir.op == "MUL") {
                // This is an independent operation suitable for parallel execution
                ir.operands.push_back("PARALLEL");
                std::cout << "Data dependency analysis applied: " << ir.op << "\n";
            }
        }
    }
};

// Auto-vectorization for SIMD
void auto_vectorize_addition(std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    __m256 b_vec, c_vec, result_vec;
    for (int i = 0; i < n; i += 8) {
        b_vec = _mm256_loadu_ps(&b[i]);
        c_vec = _mm256_loadu_ps(&c[i]);
        result_vec = _mm256_add_ps(b_vec, c_vec);
        _mm256_storeu_ps(&a[i], result_vec);
    }
}

void auto_vectorize_multiplication(std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    __m256 b_vec, c_vec, result_vec;
    for (int i = 0; i < n; i += 8) {
        b_vec = _mm256_loadu_ps(&b[i]);
        c_vec = _mm256_loadu_ps(&c[i]);
        result_vec = _mm256_mul_ps(b_vec, c_vec);
        _mm256_storeu_ps(&a[i], result_vec);
    }
}

// Main function to simulate the compiler pipeline
int main() {
    // Step 1: Frontend - Parse the source code
    std::string sourceCode = "ADD R1, R2, R3; MUL R4, R5, R6;";  // Example source code
    Frontend frontend;
    std::vector<IRInstruction> irInstructions = frontend.parse(sourceCode);

    // Step 2: Optimization Passes - Apply optimizations
    Optimizer optimizer;
    optimizer.applyLoopPeeling(irInstructions);
    optimizer.applyInductionVariableAnalysis(irInstructions);
    optimizer.applyDataDependencyAnalysis(irInstructions);

    // Step 3: Backend - Generate assembly code (SIMD instructions)
    Backend backend;
    backend.generateAssembly(irInstructions);

    // Step 4: SIMD operations (auto-vectorization)
    int n = 1000;
    std::vector<float> a(n, 0), b(n, 2.0), c(n, 3.0);
    auto_vectorize_addition(a, b, c, n);
    auto_vectorize_multiplication(a, b, c, n);

    std::cout << "Optimization and assembly generation complete!" << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <immintrin.h>

// Data structure for the intermediate representation (IR) of the code
struct IRInstruction {
    std::string op;  // Operation (e.g., ADD, MUL, etc.)
    std::vector<std::string> operands;  // Operands for the operation
};

// Frontend: Simple lexer and parser (simplified for demonstration)
class Frontend {
public:
    std::vector<IRInstruction> parse(const std::string& sourceCode) {
        std::vector<IRInstruction> irInstructions;
        // Example source code parsing: simple ADD and MUL statements.
        // This is a very simplified parser for demonstration.
        irInstructions.push_back({"ADD", {"R1", "R2", "R3"}});
        irInstructions.push_back({"MUL", {"R4", "R5", "R6"}});
        return irInstructions;
    }
};

// Backend: Generate assembly or machine code for SIMD
class Backend {
public:
    void generateAssembly(const std::vector<IRInstruction>& irInstructions) {
        // Generate AVX2 assembly instructions for SIMD operations.
        for (const auto& ir : irInstructions) {
            if (ir.op == "ADD") {
                std::cout << "vaddps " << ir.operands[0] << ", " << ir.operands[1] << ", " << ir.operands[2] << "\n";
            } else if (ir.op == "MUL") {
                std::cout << "vmulps " << ir.operands[0] << ", " << ir.operands[1] << ", " << ir.operands[2] << "\n";
            }
        }
    }
};

// Optimization Passes: Apply loop peeling, induction variable analysis, and data dependency analysis

class Optimizer {
public:
    // Loop Peeling: Unroll loop for SIMD vectorization
    void applyLoopPeeling(std::vector<IRInstruction>& irInstructions) {
        for (auto& ir : irInstructions) {
            if (ir.op == "ADD" || ir.op == "MUL") {
                // Simulate loop unrolling (peeling) by creating multiple operations
                ir.operands.push_back("SIMD_PEEL");  // Example of peeled loop
                std::cout << "Loop peeled for optimization: " << ir.op << "\n";
            }
        }
    }

    // Induction Variable Analysis: Identifying loop indices and creating SIMD optimizations
    void applyInductionVariableAnalysis(std::vector<IRInstruction>& irInstructions) {
        // Simulate induction variable analysis
        for (auto& ir : irInstructions) {
            if (ir.op == "ADD") {
                // Introduce vectorized version of the operation
                ir.op = "VADD";
                std::cout << "Induction variable analysis applied: " << ir.op << "\n";
            }
        }
    }

    // Data Dependency Analysis: Identify data dependencies and parallelize loops
    void applyDataDependencyAnalysis(std::vector<IRInstruction>& irInstructions) {
        // Simulate data dependency analysis
        for (auto& ir : irInstructions) {
            if (ir.op == "MUL") {
                // This is an independent operation suitable for parallel execution
                ir.operands.push_back("PARALLEL");
                std::cout << "Data dependency analysis applied: " << ir.op << "\n";
            }
        }
    }
};

// Auto-vectorization for SIMD
void auto_vectorize_addition(std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    __m256 b_vec, c_vec, result_vec;
    for (int i = 0; i < n; i += 8) {
        b_vec = _mm256_loadu_ps(&b[i]);
        c_vec = _mm256_loadu_ps(&c[i]);
        result_vec = _mm256_add_ps(b_vec, c_vec);
        _mm256_storeu_ps(&a[i], result_vec);
    }
}

void auto_vectorize_multiplication(std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int n) {
    __m256 b_vec, c_vec, result_vec;
    for (int i = 0; i < n; i += 8) {
        b_vec = _mm256_loadu_ps(&b[i]);
        c_vec = _mm256_loadu_ps(&c[i]);
        result_vec = _mm256_mul_ps(b_vec, c_vec);
        _mm256_storeu_ps(&a[i], result_vec);
    }
}

// Main function to simulate the compiler pipeline
int main() {
    // Step 1: Frontend - Parse the source code
    std::string sourceCode = "ADD R1, R2, R3; MUL R4, R5, R6;";  // Example source code
    Frontend frontend;
    std::vector<IRInstruction> irInstructions = frontend.parse(sourceCode);

    // Step 2: Optimization Passes - Apply optimizations
    Optimizer optimizer;
    optimizer.applyLoopPeeling(irInstructions);
    optimizer.applyInductionVariableAnalysis(irInstructions);
    optimizer.applyDataDependencyAnalysis(irInstructions);

    // Step 3: Backend - Generate assembly code (SIMD instructions)
    Backend backend;
    backend.generateAssembly(irInstructions);

    // Step 4: SIMD operations (auto-vectorization)
    int n = 1000;
    std::vector<float> a(n, 0), b(n, 2.0), c(n, 3.0);
    auto_vectorize_addition(a, b, c, n);
    auto_vectorize_multiplication(a, b, c, n);

    std::cout << "Optimization and assembly generation complete!" << std::endl;

    return 0;
}

#include <iostream>
#include <stdexcept> // For handling exceptions
#include <thread>     // For simulating long-running tasks (optional)
#include <chrono>     // For using time functions (optional)

void simulateTask() {
    // Simulate a task that might throw an exception
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Simulate a delay
    // Uncomment the next line to simulate an error
    // throw std::runtime_error("An error occurred during task execution.");
}

int main() {
    bool running = true;

    while (running) {
        try {
            // Simulate some work or logic here
            std::cout << "Program is running... Press Enter to stop." << std::endl;
            simulateTask(); // This simulates work being done

            std::cout << "Press Enter to stop the program." << std::endl;
            
            // Wait for the user to press Enter
            if (std::cin.get()) {
                running = false; // Exit the loop when Enter is pressed
            }

        } catch (const std::exception& e) {
            // Handle any exceptions that occur during execution
            std::cout << "Error: " << e.what() << std::endl;
            std::cout << "The program will continue running." << std::endl;
        }
    }

    std::cout << "Program has been stopped. Press Enter to close the terminal." << std::endl;
    
    // Wait for the user to press Enter before fully closing
    std::cin.get(); // This keeps the program from closing immediately after finishing

    return 0;
}

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <memory>

enum class TokenType {
    IDENTIFIER,
    KEYWORD,
    NUMBER,
    OPERATOR,
    OPEN_PAREN,
    CLOSE_PAREN,
    SEMICOLON,
    EOF_TYPE
};

struct Token {
    TokenType type;
    std::string value;
};

class Lexer {
public:
    Lexer(const std::string& input) : input(input), pos(0) {}

    Token nextToken() {
        while (pos < input.length() && isspace(input[pos])) ++pos;

        if (pos == input.length()) return {TokenType::EOF_TYPE, ""};

        char currentChar = input[pos];

        if (isdigit(currentChar)) {
            return readNumber();
        }

        if (isalpha(currentChar)) {
            return readIdentifier();
        }

        if (currentChar == '(') {
            pos++;
            return {TokenType::OPEN_PAREN, "("};
        }

        if (currentChar == ')') {
            pos++;
            return {TokenType::CLOSE_PAREN, ")"};
        }

        if (currentChar == ';') {
            pos++;
            return {TokenType::SEMICOLON, ";"};
        }

        if (currentChar == '+') {
            pos++;
            return {TokenType::OPERATOR, "+"};
        }

        throw std::runtime_error("Invalid character");
    }

private:
    Token readNumber() {
        size_t start = pos;
        while (pos < input.length() && isdigit(input[pos])) ++pos;
        return {TokenType::NUMBER, input.substr(start, pos - start)};
    }

    Token readIdentifier() {
        size_t start = pos;
        while (pos < input.length() && isalnum(input[pos])) ++pos;
        return {TokenType::IDENTIFIER, input.substr(start, pos - start)};
    }

    std::string input;
    size_t pos;
};

class Parser {
public:
    Parser(Lexer& lexer) : lexer(lexer), currentToken(lexer.nextToken()) {}

    void parse() {
        while (currentToken.type != TokenType::EOF_TYPE) {
            statement();
        }
    }

private:
    void statement() {
        if (currentToken.type == TokenType::IDENTIFIER) {
            // Handle identifier statements (e.g., function definitions)
            std::cout << "Parsing identifier: " << currentToken.value << std::endl;
            currentToken = lexer.nextToken();
        } else if (currentToken.type == TokenType::KEYWORD) {
            // Handle keyword statements (e.g., if, while)
        } else {
            throw std::runtime_error("Unexpected token: " + currentToken.value);
        }
    }

    Lexer& lexer;
    Token currentToken;
};

// Main entry for code generation
int main() {
    std::string code = "define Fibonacci(n) { return n + 1; }";
    Lexer lexer(code);
    Parser parser(lexer);

    try {
        parser.parse();
    } catch (const std::exception& e) {
        std::cerr << "Error during parsing: " << e.what() << std::endl;
    }

    return 0;
}

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

class CodeGenerator {
public:
    void generate(ASTNode* root) {
        if (auto* node = dynamic_cast<NumberNode*>(root)) {
            std::cout << "PUSH " << node->value << std::endl;
        } else if (auto* node = dynamic_cast<BinaryOpNode*>(root)) {
            generate(node->left);
            generate(node->right);
            std::cout << "OP " << node->op << std::endl;
        }
    }
};

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Verifier.h>

// This is where you convert the AST into LLVM IR.
llvm::Value* generateCode(ExprNode* node, llvm::IRBuilder<>& builder) {
    if (auto intNode = dynamic_cast<IntegerNode*>(node)) {
        return llvm::ConstantInt::get(builder.getInt32Ty(), intNode->value);
    }
    if (auto varNode = dynamic_cast<VariableNode*>(node)) {
        // Fetch the variable from the symbol table
        return nullptr;  // Placeholder
    }
    if (auto addNode = dynamic_cast<AddNode*>(node)) {
        llvm::Value* leftVal = generateCode(addNode->left.get(), builder);
        llvm::Value* rightVal = generateCode(addNode->right.get(), builder);
        return builder.CreateAdd(leftVal, rightVal, "addtmp");
    }
    if (auto subNode = dynamic_cast<SubtractNode*>(node)) {
        llvm::Value* leftVal = generateCode(subNode->left.get(), builder);
        llvm::Value* rightVal = generateCode(subNode->right.get(), builder);
        return builder.CreateSub(leftVal, rightVal, "subtmp");
    }
    // Additional code generation for other types of AST nodes...
    return nullptr;
}

llvm::Value* generateCode(ExprNode* node, llvm::IRBuilder<>& builder) {
    if (auto intNode = dynamic_cast<IntegerNode*>(node)) {
        return llvm::ConstantInt::get(builder.getInt32Ty(), intNode->value);
    }
    if (auto varNode = dynamic_cast<VariableNode*>(node)) {
        // Check for variable type and handle accordingly
    }
    if (auto addNode = dynamic_cast<AddNode*>(node)) {
        llvm::Value* leftVal = generateCode(addNode->left.get(), builder);
        llvm::Value* rightVal = generateCode(addNode->right.get(), builder);
        // Ensure types match (e.g., both operands are integers)
        return builder.CreateAdd(leftVal, rightVal, "addtmp");
    }
    // Additional type checking for other operations
    return nullptr;
}

function_declaration:
    FUNCTION IDENTIFIER LPAREN parameters RPAREN block {
        $$ = new FunctionNode($2, $4, $6);
    }
;

parameters:
    parameter
    | parameters COMMA parameter
;

parameter:
    IDENTIFIER {
        $$ = new ParameterNode($1);
    }
;

struct ASTNode {
    virtual ~ASTNode() = default;
};

struct NumberNode : public ASTNode {
    int value;
    NumberNode(int value) : value(value) {}
};

struct BinaryOpNode : public ASTNode {
    char op;
    ASTNode* left;
    ASTNode* right;

    BinaryOpNode(char op, ASTNode* left, ASTNode* right) : op(op), left(left), right(right) {}
};

class Parser {
public:
    Parser(std::vector<Token>& tokens) : tokens(tokens), position(0) {}

    ASTNode* parse() {
        return parseExpression();
    }

private:
    std::vector<Token>& tokens;
    size_t position;

    ASTNode* parseExpression() {
        ASTNode* left = parseTerm();

        while (currentToken().type == TokenType::Operator && (currentToken().value == "+" || currentToken().value == "-")) {
            char op = currentToken().value[0];
            position++;
            ASTNode* right = parseTerm();
            left = new BinaryOpNode(op, left, right);
        }
        return left;
    }

    ASTNode* parseTerm() {
        ASTNode* left = parseFactor();

        while (currentToken().type == TokenType::Operator && (currentToken().value == "*" || currentToken().value == "/")) {
            char op = currentToken().value[0];
            position++;
            ASTNode* right = parseFactor();
            left = new BinaryOpNode(op, left, right);
        }
        return left;
    }

    ASTNode* parseFactor() {
        if (currentToken().type == TokenType::Number) {
            int value = std::stoi(currentToken().value);
            position++;
            return new NumberNode(value);
        }
        // Implement parsing of other constructs (e.g., parentheses, variables)
        return nullptr;
    }

    Token currentToken() {
        return tokens[position];
    }
};

int main() {
    llvm::LLVMContext context;
    llvm::Module module("QEL", context);
    llvm::IRBuilder<> builder(context);

    // Parse and generate AST from QEL code
    yyparse();  // Calls Bison's parser, which will use Flex lexer

    // Example AST node (e.g., a simple "1 + 2")
    std::shared_ptr<ExprNode> root = std::make_shared<AddNode>(std::make_shared<IntegerNode>(1), std::make_shared<IntegerNode>(2));

    // Generate LLVM IR from AST
    generateCode(root.get(), builder);

    // Print out the generated LLVM IR
    module.print(llvm::outs(), nullptr);

    return 0;
}

int main() {
    std::string source = "3 + 4 * 5";
    Lexer lexer(source);
    std::vector<Token> tokens = lexer.tokenize();

    Parser parser(tokens);
    ASTNode* root = parser.parse();

    CodeGenerator codeGen;
    codeGen.generate(root);

    // Clean up AST
    delete root;

    return 0;
}

class RuntimeError : public std::exception {
public:
    const char* message;
    RuntimeError(const char* msg) : message(msg) {}
    const char* what() const noexcept override {
        return message;
    }
};

void execute(ExprNode* node) {
    try {
        // Execute the node, and throw an exception if an error occurs
        if (dynamic_cast<InvalidOperationNode*>(node)) {
            throw RuntimeError("Invalid operation.");
        }
    } catch (const RuntimeError& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
    }
}

