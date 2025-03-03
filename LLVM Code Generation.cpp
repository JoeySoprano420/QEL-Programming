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
