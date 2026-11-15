import random
import string


def rand(a=0.0, b=1.0):
    """
    Generate a random float in [a, b).
    Shortcut for generating probabilities, e.g., rand() < 0.5.
    """
    return random.uniform(a, b)


def randstring(length=8):
    """
    Generate a random string of letters and digits.
    Useful for creating unique window names or temporary IDs.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


if __name__ == "__main__":
    print("rand() ->", [rand(1.0, 5.0) for _ in range(3)])
    print("randstring() ->", [randstring(8) for _ in range(3)])