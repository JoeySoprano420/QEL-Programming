#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <unordered_map>

enum class TokenType {
    Keyword, Identifier, Number, Operator, Punctuation, EndOfFile, Invalid
};

struct Token {
    TokenType type;
    std::string value;
    size_t position;
};

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
                if (word == "if" || word == "else" || word == "while") {
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
