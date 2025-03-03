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
