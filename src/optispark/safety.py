import ast
import re

class ReadOnlyValidator(ast.NodeVisitor):
    """
    AST Walker that blocks destructive DataFrame methods and SQL commands.
    Used to defensively parse LLM code before it hits the Python interpreter.
    """
    def __init__(self):
        self.destructive_methods = {'save', 'saveAsTable', 'insertInto', 'drop', 'delete', 'truncate'}
        self.destructive_attrs = {'write'}
        self.destructive_sql = re.compile(r'\b(DROP|DELETE|TRUNCATE|INSERT|UPDATE|CREATE)\b', re.IGNORECASE)

    def visit_Attribute(self, node):
        if node.attr in self.destructive_attrs:
            raise ValueError(f"Unsafe operation detected: blocked access to '.{node.attr}' attribute.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self.destructive_methods:
                raise ValueError(f"Unsafe operation detected: blocked destructive method '{node.func.attr}'.")
            
            if node.func.attr == 'sql':
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    if self.destructive_sql.search(node.args[0].value):
                        raise ValueError(f"Unsafe operation detected: blocked destructive SQL command.")
        
        self.generic_visit(node)


def secure_exec(generated_code, global_vars, local_vars):
    """Parses code for AST safety, then executes via exec()."""
    # 1. AST Safety Check
    tree = ast.parse(generated_code)
    ReadOnlyValidator().visit(tree)
    
    # 2. Execution
    exec(generated_code, global_vars, local_vars)


def validate_safety(code, target_df, max_safe_size_mb=50):
    """Validates AI-generated code against cluster safety constraints."""
    print("🛡️ [3/4] Running safety checks...")
    
    if "F.explode" in code or "salt_array" in code:
        if target_df is None:
            return False, "Dangerous operation detected, but no target_df provided for size validation."
            
        try:
            # Tap Catalyst Optimizer for metadata
            size_bytes = target_df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > max_safe_size_mb:
                return False, f"Risk of OOM. Target DF is {size_mb:.2f} MB (Threshold: {max_safe_size_mb} MB)."
            return True, f"Code passed safety thresholds. DF size: {size_mb:.4f} MB."
        except Exception as e:
            return False, f"Failed to calculate Catalyst stats: {str(e)}"
            
    return True, "No high-memory operations detected in code."