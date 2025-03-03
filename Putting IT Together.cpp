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
