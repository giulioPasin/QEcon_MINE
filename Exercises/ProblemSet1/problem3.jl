# Define the function to calculate the factorial
function my_factorial(n::Int)
    # Initialize result as 1 (neutral element for multiplication)
    result = 1
    
    # Loop over each integer from 1 to n
    for i in 1:n
        result *= i # Multiply result by the current number
    end
    
    # Return the final result
    return result
end

# Test the function with some examples
println(my_factorial(5)) # Output: 120
println(my_factorial(7)) # Output: 5040
