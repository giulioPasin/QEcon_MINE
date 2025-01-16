#PROBLEM 1
function solve_basil_problem(X, C, q, f, pmin, pmax, N)
    # Discretize prices
    prices = collect(pmin:0.1:pmax)
    price_prob = 1 / length(prices)  
    num_prices = length(prices)
    
    # Initialize value functions and policy functions
    V = zeros(N + 1)  
    σapproach = zeros(Bool, N + 1)  # Policy function for approaching
    σbuy = zeros(Bool, N + 1, num_prices)  # Policy function for buying

    # Backward induction
    for n in N:-1:1
        # Value of terminating 
        V_terminate = -n * C

        # Compute expected value
        for (i, p) in enumerate(prices)

            # Value of buying orchid at price p
            V_buy = X - p - n * C - f
            # Value of approaching a vendor
            V_search = q * V_buy + (1 - q) * V[n + 1] - f

            V[n] = max(V[n], max(V_terminate, V_search))
            
            if V_search > V_terminate
                σapproach[n] = true
            end
            if V_buy > V_search
                σbuy[n, i] = true
            end
        end
    end

    # Compute outputs
    prob_buy = q * sum(V[2:N + 1]) / N  
    expected_price = sum(prices .* vec(sum(σbuy, dims=1))) / sum(σbuy)
    expected_vendors = sum(σapproach)

    return V, σapproach, σbuy, prob_buy, expected_price, expected_vendors
end

# Set parameters
X = 50  # Maximum value of the orchid
C = 0.5  # Cost per vendor
q = 0.15  # Probability a vendor has the orchid
f = 0.2  # Mental cost of approaching a vendor
pmin = 10.0  # Minimum price
pmax = 100.0  # Maximum price
N = 50  # Maximum number of vendors


V, σapproach, σbuy, prob_buy, expected_price, expected_vendors = solve_basil_problem(X, C, q, f, pmin, pmax, N)

# results
println("Value function (V): ", V)
println("Policy function for approaching (σapproach): ", σapproach)
println("Policy function for buying (σbuy): ", σbuy)
println("Probability Basil will buy the orchid: ", prob_buy)
println("Expected price Basil will pay (conditional on buying): ", expected_price)
println("Expected number of vendors Basil will approach: ", expected_vendors)



#PROBLEM 2
using Plots

function solve_job_search(p_vals, β, c, wages, π)
    
    reservation_wages = []
    probabilities_q = []
    expected_durations = []

    for p in p_vals
        
        VU = zeros(length(wages))  # Value of being unemployed
        VE = zeros(length(wages))  # Value of being employed
        
       
        tol = 1e-6
        diff = Inf
        while diff > tol
            VU_new = zeros(length(wages))
            VE_new = zeros(length(wages))
            for (i, w) in enumerate(wages)
                
                VE_new[i] = w + β * ((1 - p) * VE[i] + p * sum(π .* VU))
                
                
                EU = sum(π .* VU)
                VU_new[i] = max(VE_new[i], c + β * EU)
            end
            diff = maximum(abs.(VU_new .- VU)) + maximum(abs.(VE_new .- VE))
            VU, VE = VU_new, VE_new
        end

        #reservation wage
        w_star_idx = findfirst(VU .> VE)
        w_star = w_star_idx !== nothing ? wages[w_star_idx] : minimum(wages)
        push!(reservation_wages, w_star)

        #probability of accepting a job (q)
        q = sum(π[wages .>= w_star])
        push!(probabilities_q, q)

        #expected duration of unemployment
        exp_duration = 1 / q
        push!(expected_durations, exp_duration)
    end

    return reservation_wages, probabilities_q, expected_durations
end


β = 0.95  # Discount factor
c = 2.0  # Utility of being unemployed
wages = collect(10:1:50)  # Wage grid
π = fill(1 / length(wages), length(wages))  # Uniform distribution of wages
p_vals = 0.0:0.05:0.5  # Range of separation probabilities


reservation_wages, probabilities_q, expected_durations = solve_job_search(p_vals, β, c, wages, π)

# results
plot1 = plot(p_vals, reservation_wages, label="Reservation Wage (w*)", xlabel="Separation Probability (p)", ylabel="Reservation Wage", lw=2)
plot2 = plot(p_vals, probabilities_q, label="Probability of Accepting Job (q)", xlabel="Separation Probability (p)", ylabel="Probability", lw=2)
plot3 = plot(p_vals, expected_durations, label="Expected Duration of Unemployment", xlabel="Separation Probability (p)", ylabel="Expected Duration", lw=2)


plot(plot1, plot2, plot3, layout=(3, 1), size=(800, 800))





#PROBLEM 3
