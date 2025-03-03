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
