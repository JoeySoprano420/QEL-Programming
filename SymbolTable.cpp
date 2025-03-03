class SymbolTable {
public:
    std::map<std::string, std::shared_ptr<ExprNode>> variables;
    std::map<std::string, std::shared_ptr<FunctionNode>> functions;

    void addVariable(const std::string& name, std::shared_ptr<ExprNode> value) {
        variables[name] = value;
    }

    std::shared_ptr<ExprNode> getVariable(const std::string& name) {
        if (variables.find(name) != variables.end())
            return variables[name];
        return nullptr; // Variable not found
    }

    void addFunction(const std::string& name, std::shared_ptr<FunctionNode> func) {
        functions[name] = func;
    }

    std::shared_ptr<FunctionNode> getFunction(const std::string& name) {
        if (functions.find(name) != functions.end())
            return functions[name];
        return nullptr; // Function not found
    }
};
