import math

# ## Task 0.1
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply."""
    return x * y


def id(x: float) -> float:
    """Identity."""
    return x


def add(x: float, y: float) -> float:
    """Add."""
    return x + y


def neg(x: float) -> float:
    """Negate."""
    return -x


def lt(x: float, y: float) -> bool:
    """Less."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Equal."""
    return x == y


def max(x: float, y: float) -> float:
    """Maximum."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Close."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU."""
    return max(0, x)


def log(x: float) -> float:
    """Logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential."""
    return math.exp(x)


def inv(x: float) -> float:
    """Inverse."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Log backprop."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Inverse backprop."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """ReLU backprop."""
    return d if x > 0 else 0


def map(fn: Callable[[float], float], ls: Iterable[float]) -> list[float]:
    """Map."""
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> list[float]:
    """ZipWith."""
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    fn: Callable[[float, float], float], ls: Iterable[float], initial: float
) -> float:
    """Reduce."""
    result = initial
    for x in ls:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> list[float]:
    """Negate list."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
    """Add lists."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum."""
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Product."""
    return reduce(mul, ls, 1.0)
