using Plots
function count_positives_broadcasting(arr)
    # Create a Boolean array for each true element if the corresponding element in arr is >0
    positive_bools = arr .> 0
    # Sum the Boolean array (true is 1, false is 0) to count the number of positive numbers
    return sum(positive_bools)
end

# Test cases
println(count_positives_broadcasting([1, -3, 4, 7, -2, 0]))  # Output: 3
println(count_positives_broadcasting([-5, -10, 0, 6]))       # Output: 1
