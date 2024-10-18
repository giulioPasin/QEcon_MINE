
function compare_three(a, b, c)
 if a == 0 && b == 0 && c == 0  
       println("All numbers are zero")
    elseif a > 0 && b > 0 && c > 0  # Check if all numbers are positive
        println("All numbers are positive")
    else  # If any number is negative
        println("At least one number is negative")
    end
end



