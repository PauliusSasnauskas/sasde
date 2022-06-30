import logging
from typing import Any, List
from IPython.display import display_markdown
from sympy import Expr, latex

# Display sympy in latex style

def d(expr: Expr) -> None:
    display_markdown("$\\displaystyle " + latex(expr) + "$", raw=True)

# Display number arrays unified style
def a(arr: List[Any]) -> str:
    return "[" + ", ".join(f"{item:.4f}" if item < 0 else f" {item:.4f}" for item in arr) + "]"

# Pad start with zeroes (2 symbols)
def pad(x, n=2) -> str:
    str_x = str(x)
    return '0' * (n - len(str_x)) + str_x

logger_stream = logging.StreamHandler()
logger_stream.setFormatter(logging.Formatter(
    "{asctime}.{msecs:0<3.0f} [{levelname}] {message}", datefmt='%H:%M:%S', style='{'
))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logger_stream]

def info(val, *args, **kwargs) -> None: # pylint: disable=missing-function-docstring
    logging.info(val, *args, **kwargs)
