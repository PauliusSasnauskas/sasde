from IPython.display import display_markdown
from sympy import latex

# Display sympy in latex style
def d(expr):
    display_markdown("$\displaystyle " + latex(expr) + "$", raw=True)

# Display number arrays unified style
def a(arr):
    return "[" + ", ".join(f"{item:.4f}" if item < 0 else f" {item:.4f}" for item in arr) + "]"