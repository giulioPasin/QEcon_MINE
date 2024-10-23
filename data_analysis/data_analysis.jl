using Statistics, Plots, DelimitedFiles

#Let's get some practice working with vectors and matrices!
data_1 = readdlm("data_analysis//datasets//dataset_1.csv", ',',Float64)
data_2 = readdlm("data_analysis//datasets//dataset_2.csv", ',',Float64)

#Accessing the first (x) column of the data_1 matrix
data_1[:,1]

#Calculating mean of the first column of the data_1 matrix
mean(data_1[:,1])

#Calculating standard deviation of the first column of the data_1 matrix
std(data_1[:,1])

plot_1 = scatter(data_1[:,1], data_1[:,2]; legend=false, color=:blue, markersize = 5, opacity=0.7)
xaxis!(plot_1, "x")
yaxis!(plot_1, "y")
title!(plot_1, "Scatter plot of data_1")
display(plot_1)

plot_2 = scatter(data_2[:,1], data_2[:,2]; legend=false, color=:purple, markersize = 5)
xaxis!(plot_2, "x")
yaxis!(plot_2, "y")
title!(plot_2, "Scatter plot of data_2")
display(plot_2)

#Combining two plots into one
both_plots = plot(plot_1,plot_2,layout=(1,2),size=(600, 400))
savefig(both_plots, "data_analysis//both_plots.pdf")

#data_1 is a matrix, if we want to calculate the mean we need to specify the dimension!
mean(data_1, dims=1)
#dims=2 would produce mean over columns
mean(data_1, dims=2)

#in Julia there are many ways to compute mean of a vector: 
map(mean, eachcol(data_1))
[mean(col) for col in eachcol(data_1)]
 
#Standard deviation, again we need to specify the dimension!
std(data_1, dims=1)


#Pearson correlation coefficient  
cor(data_1)
#The above returns a matrix of correlations between all columns
cor(data_1)[1,2]
cor(data_1[:,1],data_1[:,2])

#Calculate correlations for both datasets!
cor_data_1 = cor(data_1)[1,2]
cor_data_2 = cor(data_2)[1,2]


#Note: This syntax with $(variable) is used to insert the value of a variable into a string
#It will be very useful for your homework!
plot_1 = scatter(data_1[:,1], data_1[:,2]; label="cor(x,y)=$(cor_data_1)", color=:blue, markersize = 5)


cor_data_1 = round(cor_data_1; digits=2)
cor_data_2 = round(cor_data_2; digits=2)

plot_1 = scatter(data_1[:,1], data_1[:,2]; label="cor(x,y)=$(cor_data_1)", color=:blue, markersize = 5)
xaxis!(plot_1, "x")
yaxis!(plot_1, "y")
title!(plot_1, "Scatter plot of data_1")
display(plot_1)



plot_2 = scatter(data_2[:,1], data_2[:,2]; label="cor(x,y)=$cor_data_2", color=:purple, markersize = 5)
xaxis!(plot_2, "x")
yaxis!(plot_2, "y")
title!(plot_2, "Scatter plot of data_2")
display(plot_2)
both_plots = plot(plot_1,plot_2,layout=(1,2))
savefig(both_plots, "code//data_analysis//both_plots.pdf")

#############################        QUICK TASK 1:         ############################# 
# a. Import dataset_3 as data_3.
# b. Calculate the correlation between x and y in the data.
# c. Plot the data as a red scatter plot, name it properly, and label it with the correlation coefficient.
# d. Combine plot_1, plot_2, and your plot_3 plot into one plot (use the option layout=(1,3)).
######################################################################################### 
###YOUR CODE:
# Import necessary libraries
using Statistics, Plots, DelimitedFiles

# a. Import dataset_3 as data_3
data_3 = readdlm("data_analysis//datasets//dataset_3.csv", ',', Float64)

# Load dataset_1 and dataset_2 as well
data_1 = readdlm("data_analysis//datasets//dataset_1.csv", ',', Float64)
data_2 = readdlm("data_analysis//datasets//dataset_2.csv", ',', Float64)

# b. Calculate the correlation between x and y in data_3
correlation_data_3 = cor(data_3[:,1], data_3[:,2])
correlation_data_3 = round(correlation_data_3; digits=2)

# c. Plot the data_3 as a red scatter plot
plot_3 = scatter(data_3[:,1], data_3[:,2]; label="cor(x,y)=$correlation_data_3", color=:red, markersize=5)
xaxis!(plot_3, "x")
yaxis!(plot_3, "y")
title!(plot_3, "Scatter plot of data_3")
display(plot_3)

# Let's reuse the code for data_1 and data_2 for their scatter plots
cor_data_1 = cor(data_1[:,1], data_1[:,2])
cor_data_1 = round(cor_data_1; digits=2)

cor_data_2 = cor(data_2[:,1], data_2[:,2])
cor_data_2 = round(cor_data_2; digits=2)

# Plotting data_1
plot_1 = scatter(data_1[:,1], data_1[:,2]; label="cor(x,y)=$cor_data_1", color=:blue, markersize=5)
xaxis!(plot_1, "x")
yaxis!(plot_1, "y")
title!(plot_1, "Scatter plot of data_1")
display(plot_1)

# Plotting data_2
plot_2 = scatter(data_2[:,1], data_2[:,2]; label="cor(x,y)=$cor_data_2", color=:purple, markersize=5)
xaxis!(plot_2, "x")
yaxis!(plot_2, "y")
title!(plot_2, "Scatter plot of data_2")
display(plot_2)

# d. Combine plot_1, plot_2, and plot_3 into a plot with layout=(1,3)
combined_plots = plot(plot_1, plot_2, plot_3, layout=(1,3), size=(900, 400))


# Display plotS
display(combined_plots)








#############################        QUICK TASK 2:         ############################# 
#Following the instruction on slides write your own function fit_regression(x,y)
#which accepts two vectors x,y and returns a vector of regression coefficients.

# HINTS: 
# 1. Do it in steps: define numerator, denominator, and then use those to get the coefficient β1.
# 2. Remember that you can use mean(), sum() and  broadcasting(you don't need any loops)!! to get the final result. 
# 3. Define:
x = data_1[:,1]
y = data_1[:,2]
# 4. Define a function fit_regression(x,y)



# 5. This call should return the coefficients!
β0,β1 = fit_regression(x,y)

#Check:
#See if your coefficient β1 is equal to:
cov(x,y)/var(x)

#MY CODE
function fit_regression(x, y)
    # Step 1: Calculate means
    mean_x = mean(x)
    mean_y = mean(y)
    
    # Step 2: Calculate numerator (covariance)
    numerator = sum((x .- mean_x) .* (y .- mean_y))
    
    # Step 3: Calculate denominator (variance)
    denominator = sum((x .- mean_x) .^ 2)
    
    # Step 4: Calculate coefficients
    β1 = numerator / denominator
    β0 = mean_y - β1 * mean_x

    return β0, β1
end

# data
x = data_1[:, 1]
y = data_1[:, 2]

# Fit regression and get coefficients
β0, β1 = fit_regression(x, y)

# Check if the coefficient β1 is equal to cov(x, y) / var(x)
expected_β1 = cov(x, y) / var(x)
println("Calculated β1: ", β1)
println("Expected β1: ", expected_β1)

# Check if equal
println("Are they equal? ", β1 ≈ expected_β1)

######################################################################################### 


scatter(x, y; label="Our data", color=:blue, markersize = 5)
#This will work only if you have defined β0 and β1 (thus fit_regression function!!)
plot!(x,β0.+β1.*x; label="Fitted line: y=$(round(β0,digits=2))+$(round(β1,digits=2))x",linewidth=4)
xaxis!(plot_1, "x")
yaxis!(plot_1, "y")
title!(plot_1, "Scatter plot of data_1 with fitted line")



