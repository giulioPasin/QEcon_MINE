using Plot
function odd_or_even(n)
if n % 2 == 0  # Check if the remainder of n divided by 2 is 0 (even)
    println("Even")
    else  # If the remainder is not 0, it's odd
    println("Odd")
    end
end

odd_or_even(7)   # Output: Odd
odd_or_even(12)  # Output: Even
