
using DelimitedFiles, Plots, Statistics

# Load the data 
data = readdlm("C:\\Users\\Algiu\\OneDrive\\Desktop\\Class Material QUANT.ECON\\QEcon_MINE\\ProblemSet1\\dataset.csv", ',', Float64)

# Separate the columns 
earnings = data[:, 1]         # First column: Earnings
education = data[:, 2]        # Second column: Education
hours_worked = data[:, 3]     # Third column: Hours Worked

# Plot Earnings vs Education
plot(education, earnings, seriestype = :scatter, color = "green", 
    xlabel = "Education Level", ylabel = "Earnings", 
    title = "Earnings vs Education", label = "Earnings vs Education")

display(plot!(legend = :topright))

# Plot Earnings vs Hours Worked
plot(hours_worked, earnings, seriestype = :scatter, color = "red", 
    xlabel = "Hours Worked per Week", ylabel = "Earnings", 
    title = "Earnings vs Hours Worked", label = "Earnings vs Hours Worked")

display(plot!(legend = :topright))

# Correlation between earnings and education
correlation_earnings_education = cor(earnings, education)

# Correlation between earnings and hours worked
correlation_earnings_hours = cor(earnings, hours_worked)

println("Correlation between Earnings and Education: ", correlation_earnings_education)
println("Correlation between Earnings and Hours Worked: ", correlation_earnings_hours)


#Analyze the results

#The correlation is in both cases positive so it should mean that there is a positive relation between the earning and education and earning and hours worked. Althought Corr>0 is not extremely positive which suggests that the relation between the two variables is not relevent or not as strong as we could think. I believe that the results could be correct since, if we are looking to a big or significatn number or cases, then obvisouly there can be cases in which earning and education or hours worked are not directly proportional as it could happen in a real life situation.