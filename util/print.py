from IPython.display import display_markdown
from sympy import latex
import logging

# Display sympy in latex style
def d(expr):
    display_markdown("$\displaystyle " + latex(expr) + "$", raw=True)

# Display number arrays unified style
def a(arr):
    return "[" + ", ".join(f"{item:.4f}" if item < 0 else f" {item:.4f}" for item in arr) + "]"

# Pad start with zeroes (2 symbols)
def pad(x, n=2):
    str_x = str(x)
    return '0' * (n - len(str_x)) + str_x

loggerStream = logging.StreamHandler()
loggerStream.setFormatter(logging.Formatter("{asctime}.{msecs:0<3.0f} [{levelname}] {message}", datefmt='%H:%M:%S', style='{'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [loggerStream]

def info(input, *args, **kwargs):
    logging.info(input, *args, **kwargs)