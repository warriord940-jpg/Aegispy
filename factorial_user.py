def calculate_factorial(n):
    if n < 0:
        return "Factorial is not defined for negative numbers."
    elif n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    try:
        user_input = input("Enter a number to calculate its factorial: ")
        number = int(user_input)
        
        print(f"Calculating factorial for: {number}")
        output = calculate_factorial(number)
        print(f"Result: {output}")
        
    except ValueError:
        print("Please enter a valid integer.")