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
