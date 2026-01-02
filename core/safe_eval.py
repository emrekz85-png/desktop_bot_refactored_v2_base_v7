#!/usr/bin/env python3
"""
Safe Expression Evaluator

Replaces dangerous eval() with a safe AST-based expression evaluator.
Only allows whitelisted operations and no code execution.

Security Features:
- No arbitrary code execution
- Whitelist-based approach
- AST validation before evaluation
- Safe operators only (comparisons, math, logic)
- No function calls or imports allowed
- Read-only context access
"""

import ast
import operator
from typing import Any, Dict


class SafeEvaluator:
    """Safe expression evaluator using AST parsing."""

    # Whitelisted operators
    ALLOWED_OPERATORS = {
        # Comparison operators
        ast.Gt: operator.gt,
        ast.Lt: operator.lt,
        ast.GtE: operator.ge,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,

        # Boolean operators
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,

        # Arithmetic operators
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,

        # Unary operators
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, max_expression_length: int = 500):
        """Initialize safe evaluator.

        Args:
            max_expression_length: Maximum allowed expression length
        """
        self.max_expression_length = max_expression_length

    def evaluate(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate expression with given context.

        Args:
            expression: Expression string to evaluate (e.g., "adx > 25")
            context: Dictionary of variables available to expression

        Returns:
            Result of expression evaluation

        Raises:
            ValueError: If expression is invalid or contains unsafe operations
            SyntaxError: If expression has syntax errors

        Example:
            >>> evaluator = SafeEvaluator()
            >>> context = {'adx': 30, 'rsi': 65}
            >>> evaluator.evaluate('adx > 25', context)
            True
            >>> evaluator.evaluate('adx > 25 and rsi < 70', context)
            True
        """
        # Validate expression length
        if len(expression) > self.max_expression_length:
            raise ValueError(
                f"Expression too long ({len(expression)} > {self.max_expression_length})"
            )

        # Parse expression into AST
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression syntax: {e}")

        # Validate AST nodes (ensure no dangerous operations)
        self._validate_ast(tree.body)

        # Evaluate expression safely
        try:
            return self._eval_node(tree.body, context)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

    def _validate_ast(self, node: ast.AST) -> None:
        """Validate that AST only contains safe operations.

        Args:
            node: AST node to validate

        Raises:
            ValueError: If node contains unsafe operations
        """
        # Allowed node types
        safe_nodes = (
            ast.Compare,      # Comparisons (>, <, ==, etc.)
            ast.BoolOp,       # Boolean operations (and, or)
            ast.UnaryOp,      # Unary operations (not, -, +)
            ast.BinOp,        # Binary operations (+, -, *, /)
            ast.Name,         # Variable names
            ast.Constant,     # Constants (numbers, strings, etc.)
            ast.Num,          # Numbers (Python 3.7 compatibility)
            ast.Str,          # Strings (Python 3.7 compatibility)
            ast.Load,         # Context for loading variables
        )

        # Allowed operator types (these are nested inside node.op)
        safe_operators = tuple(self.ALLOWED_OPERATORS.keys())

        # Recursively validate all nodes
        for child in ast.walk(node):
            # Check if node type is safe
            if not isinstance(child, safe_nodes + safe_operators):
                raise ValueError(
                    f"Unsafe operation detected: {child.__class__.__name__}. "
                    f"Only comparisons, math, and logic are allowed."
                )

    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Recursively evaluate AST node.

        Args:
            node: AST node to evaluate
            context: Variable context

        Returns:
            Evaluation result
        """
        # Constants (numbers, strings, True, False, None)
        if isinstance(node, ast.Constant):
            return node.value

        # Legacy constant nodes (Python 3.7 compatibility)
        if isinstance(node, (ast.Num, ast.Str)):
            return node.n if isinstance(node, ast.Num) else node.s

        # Variable lookup
        if isinstance(node, ast.Name):
            if node.id not in context:
                raise ValueError(f"Variable '{node.id}' not found in context")
            return context[node.id]

        # Binary operations (a + b, a * b, etc.)
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op_func = self.ALLOWED_OPERATORS[type(node.op)]
            return op_func(left, right)

        # Unary operations (-a, +a, not a)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op_func = self.ALLOWED_OPERATORS[type(node.op)]
            return op_func(operand)

        # Comparisons (a > b, a == b, etc.)
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)

            # Handle chained comparisons (a < b < c)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = self.ALLOWED_OPERATORS[type(op)]
                result = result and op_func(left, right)
                left = right  # For chained comparisons

                if not result:
                    break

            return result

        # Boolean operations (a and b, a or b)
        if isinstance(node, ast.BoolOp):
            # For 'and', stop at first False
            # For 'or', stop at first True
            is_or = isinstance(node.op, ast.Or)

            for value in node.values:
                result = self._eval_node(value, context)

                if is_or and result:
                    return True
                elif not is_or and not result:
                    return False

            # All values evaluated
            return result if is_or else True

        raise ValueError(f"Unsupported node type: {node.__class__.__name__}")


# Convenience function for simple usage
_default_evaluator = SafeEvaluator()

def safe_eval(expression: str, context: Dict[str, Any]) -> Any:
    """Safely evaluate expression (convenience function).

    Args:
        expression: Expression to evaluate
        context: Variable context

    Returns:
        Evaluation result

    Example:
        >>> safe_eval('x > 10', {'x': 15})
        True
        >>> safe_eval('a + b', {'a': 5, 'b': 3})
        8
    """
    return _default_evaluator.evaluate(expression, context)


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("SAFE EXPRESSION EVALUATOR - TEST")
    print("="*80)

    evaluator = SafeEvaluator()

    # Test cases
    test_cases = [
        # (expression, context, expected_result, should_pass)
        ("adx > 25", {"adx": 30}, True, True),
        ("adx > 25", {"adx": 20}, False, True),
        ("adx > 25 and rsi < 70", {"adx": 30, "rsi": 65}, True, True),
        ("adx > 25 or rsi < 30", {"adx": 20, "rsi": 25}, True, True),
        ("x + y", {"x": 5, "y": 3}, 8, True),
        ("x * y + z", {"x": 2, "y": 3, "z": 4}, 10, True),
        ("not (x > 10)", {"x": 5}, True, True),
        ("10 < x < 20", {"x": 15}, True, True),

        # These should fail (unsafe operations)
        ("import os", {}, None, False),
        ("eval('1+1')", {}, None, False),
        ("__import__('os')", {}, None, False),
        ("[x for x in range(10)]", {}, None, False),
    ]

    print("\n✅ SAFE EXPRESSIONS:")
    for expr, ctx, expected, should_pass in test_cases:
        if not should_pass:
            continue

        try:
            result = evaluator.evaluate(expr, ctx)
            status = "✅" if result == expected else "❌"
            print(f"{status} {expr:<30} → {result} (expected: {expected})")
        except Exception as e:
            print(f"❌ {expr:<30} → ERROR: {e}")

    print("\n🔒 UNSAFE EXPRESSIONS (should be rejected):")
    for expr, ctx, expected, should_pass in test_cases:
        if should_pass:
            continue

        try:
            result = evaluator.evaluate(expr, ctx)
            print(f"❌ {expr:<30} → SECURITY FAILURE! Result: {result}")
        except (ValueError, SyntaxError) as e:
            print(f"✅ {expr:<30} → BLOCKED: {str(e)[:50]}")

    print("\n" + "="*80)
    print("Safe evaluator is ready to use!")
    print("Replace eval() calls with safe_eval() or SafeEvaluator.evaluate()")
    print("="*80)
