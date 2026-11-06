import math
import time

def calculate_factorial(n):
    """Calculates the factorial of n on the CPU."""
    print(f"  [CPU Engine] Calculating factorial for n = {n}...", flush=True)
    start = time.time()
    
    # Compute the factorial
    result = math.factorial(n)
    
    elapsed = time.time() - start
    print(f"  [CPU Engine] Completed in {elapsed:.4f} seconds.", flush=True)
    return f"Factorial_Result_{result}"