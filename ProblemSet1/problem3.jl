
function my_factorial(n)
    result = 1  # Initialize result to 1
    for i in 1:n  # Iterate from 1 to n
        result *= i  # Multiply result by i in each iteration
    end
    return result  # Return the final result
end