
function compare_three(a, b, c)
 if a == 0 && b == 0 && c == 0  
       println("All numbers are zero")
    elseif a > 0 && b > 0 && c > 0  # Check if all numbers are positive
        println("All numbers are positive")
    else  # If any number is negative
        println("At least one number is negative")
    end
end

compare_three(1, 2, 3)    # Output: All numbers are positive
compare_three(-1, 5, 7)   # Output: At least one number is not positive
compare_three(0, -4, 3)   # Output: At least one number is not positive
compare_three(0, 0, 0)    # Output: All numbers are zero


