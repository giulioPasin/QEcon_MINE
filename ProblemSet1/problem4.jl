function count_positives(arr)
    counter = 0  # Initialize a counter to keep track of positive numbers
    for num in arr  # Loop through each number in the array
        if num > 0  # Check if the number is positive
            counter += 1  # Increment the counter if it's positive
        end
    end
    println(counter)  # Print the total count of positive numbers
end

