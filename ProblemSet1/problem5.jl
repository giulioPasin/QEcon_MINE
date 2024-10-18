# Import the Plots package
using Plots

# Define the function plot_powers to plot powers of x
function plot_powers(n)
    # Initialize an empty plot
    power_plot = plot(title="Powers of x", xlabel="x", ylabel="y")
    
    # Define the range for x from -10 to 10 with step size of 0.2
    x_vals = -10:0.2:10
    
    # Loop through each power from 1 to n
    for i in 1:n
        # Compute the corresponding power of x
        y_vals = x_vals .^ i
        
        # Plot the current power, updating the same plot
        plot!(x_vals, y_vals, label="x^$i", lw=3, linestyle=:dash)
    end
    
    # Return the final plot
    return power_plot
end

# Example usage: call the function with n=3 and display the plot
my_plot = plot_powers(3)
display(my_plot)
