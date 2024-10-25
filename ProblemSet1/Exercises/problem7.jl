#PART1_calculate SD
# Function to calculate standard deviation
function standard_deviation(x::Vector{Float64})
    # Step 1: Calculate the mean
    mean_x = sum(x) / length(x)
    
    # Step 2: Calculate the squared differences (x - mean_x) ^ 2 using broadcasting
    squared_diffs = (x .- mean_x) .^ 2
    
    # Step 3: Sum the squared differences and divide by (n-1)
    variance = sum(squared_diffs) / (length(x) - 1)
    
    # Step 4: Return the square root of the variance (standard deviation)
    return sqrt(variance)
end

# Test the function with an example
x = [1.0, 2.0, 3.0, 4.0, 5.0]
println(standard_deviation(x))  # Output: 1.5811388300841898

PART2_calculateSquaredDifferences
# Define the function to calculate squared differences
function squared_differences(x::Vector{Float64})
    # Step 1: Calculate the mean
    mean_x = sum(x) / length(x)
    
    # Step 2: Calculate the element-wise differences using broadcasting
    d = x .- mean_x  # Broadcasting subtraction
    
    # Step 3: Square the differences using broadcasting
    squared_d = d .^ 2  # Broadcasting square operation
    
    # Return the squared differences
    return squared_d
end

# Test the function with an example array
x = [1.0, 2.0, 3.0, 4.0, 5.0]
println(squared_differences(x))  # Output: [4.0, 1.0, 0.0, 1.0, 4.0]

#PART3_calculate variance
# Define a function to calculate variance
function calculate_variance(x::Vector{Float64})
    # Step 1: Calculate the mean
    mean_x = sum(x) / length(x)
    
    # Step 2: Calculate the squared differences
    squared_d = (x .- mean_x) .^ 2
    
    # Step 3: Calculate the variance using degrees of freedom correction
    variance = sum(squared_d) / (length(x) - 1)
    
    return variance
end

# Test the function with an example
x = [1.0, 2.0, 3.0, 4.0, 5.0]
println(calculate_variance(x))  # Output: 2.5

#PART 4_calculate SD
# Function to calculate standard deviation
function calculate_standard_deviation(x::Vector{Float64})
    # Step 1: Calculate the mean
    mean_x = sum(x) / length(x)
    
    # Step 2: Calculate the squared differences
    squared_d = (x .- mean_x) .^ 2
    
    # Step 3: Calculate the variance using degrees of freedom correction
    variance = sum(squared_d) / (length(x) - 1)
    
    # Step 4: Calculate the standard deviation as the square root of variance
    standard_deviation = sqrt(variance)
    
    return standard_deviation
end

# Test the function with an example
x = [1.0, 2.0, 3.0, 4.0, 5.0]
println(calculate_standard_deviation(x))  # Output: 1.5811388300841898

#part 5_ return SD
# Function to calculate and return standard deviation
function calculate_standard_deviation(x::Vector{Float64})
    # Step 1: Calculate the mean
    mean_x = sum(x) / length(x)
    
    # Step 2: Calculate the squared differences
    squared_d = (x .- mean_x) .^ 2
    
    # Step 3: Calculate the variance using degrees of freedom correction
    variance = sum(squared_d) / (length(x) - 1)
    
    # Step 4: Calculate the standard deviation as the square root of variance
    standard_deviation = sqrt(variance)
    
    # Return the standard deviation value
    return standard_deviation
end

# Test the function with an example
 x = [1.0, 2.0, 3.0, 4.0, 5.0]
sd_value = calculate_standard_deviation(x)
println(sd_value)



