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
using Plots

# Function to compute the steady-state level of capital
function steady_state_capital(β, α, δ)
    return ((β * α) / (1 - β * (1 - δ)))^(1 / (1 - α))
end

# Function to simulate the Neoclassical Growth Model
function simulate_ngm(β, α, δ, γ_vals, k0, max_periods)
    k_star = steady_state_capital(β, α, δ)  # Steady-state capital
    results = []

    for γ in γ_vals
        k_vals = [k0]  # Initialize capital path
        for t in 1:max_periods
            k_prev = k_vals[end]
            c = (1 - β * (1 - δ)) * k_prev^α  # Consumption
            k_next = k_prev^α + (1 - δ) * k_prev - c  # Capital transition
            push!(k_vals, k_next)
            if k_star - k_next < 0.5 * (k_star - k0)  # Convergence criterion
                push!(results, (γ, t))
                break
            end
        end
    end

    return results
end

# Function to generate the figure with 4 panels
function plot_ngm_dynamics(β, α, δ, γ_vals, k0, max_periods)
    k_star = steady_state_capital(β, α, δ)
    time = 0:max_periods
    all_paths = Dict()

    for γ in γ_vals
        k_vals = [k0]
        c_vals = []
        i_vals = []
        y_vals = []

        for t in 1:max_periods
            k_prev = k_vals[end]
            y = k_prev^α
            c = (1 - β * (1 - δ)) * k_prev^α
            k_next = k_prev^α + (1 - δ) * k_prev - c
            i = k_next - (1 - δ) * k_prev

            push!(k_vals, k_next)
            push!(y_vals, y)
            push!(c_vals, c / y)
            push!(i_vals, i / y)
        end

        all_paths[γ] = (k_vals, y_vals, i_vals, c_vals)
    end

    # Plot the dynamics
    p1 = plot(title="Capital Over Time", xlabel="Time", ylabel="Capital (k)")
    p2 = plot(title="Output Over Time", xlabel="Time", ylabel="Output (y)")
    p3 = plot(title="Investment to Output Ratio", xlabel="Time", ylabel="I/Y")
    p4 = plot(title="Consumption to Output Ratio", xlabel="Time", ylabel="C/Y")

    for γ in γ_vals
        k_vals, y_vals, i_vals, c_vals = all_paths[γ]
        plot!(p1, time[1:length(k_vals)], k_vals, label="γ = $γ", lw=2)
        plot!(p2, time[1:length(y_vals)], y_vals, label="γ = $γ", lw=2)
        plot!(p3, time[1:length(i_vals)], i_vals, label="γ = $γ", lw=2)
        plot!(p4, time[1:length(c_vals)], c_vals, label="γ = $γ", lw=2)
    end

    plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 600))
end

# Parameters
β = 0.95  # Discount factor
α = 0.3  # Output elasticity of capital
δ = 0.05  # Depreciation rate
γ_vals = [0.5, 1.0, 2.0]  # Different values of γ
k0 = 0.5 * steady_state_capital(β, α, δ)  # Initial capital, half the steady state
max_periods = 500  # Maximum simulation periods

# Generate the table
convergence_results = simulate_ngm(β, α, δ, γ_vals, k0, max_periods)

# Display the table
println("γ\tPeriods to Halfway Steady-State")
for (γ, t) in convergence_results
    println("$γ\t$t")
end

# Generate and plot the dynamics
plot_ngm_dynamics(β, α, δ, γ_vals, k0, 100)




#PROBLEM 4
using LinearAlgebra

# Transition matrix P for Zt
P = [
    0.5 0.3 0.2;
    0.2 0.7 0.1;
    0.3 0.3 0.4
]

# States for Xt and Zt
Xt = 0:5
Zt = 1:3

# Policy function σ(Xt, Zt)
function σ(x, z)
    if z == 1
        return 0
    elseif z == 2
        return x
    elseif z == 3
        return x + 1 ≤ 5 ? x + 1 : 3
    end
end

# Generate the joint transition matrix for (Xt, Zt)
function joint_transition_matrix(Xt, Zt, P, σ)
    num_X = length(Xt)
    num_Z = length(Zt)
    num_states = num_X * num_Z
    T = zeros(num_states, num_states)

    for (i, x) in enumerate(Xt)
        for (j, z) in enumerate(Zt)
            current_state_idx = (i - 1) * num_Z + j
            for (k, z_next) in enumerate(Zt)
                next_x = σ(x, z)
                next_state_idx = (findfirst(==(next_x), Xt) - 1) * num_Z + k
                T[current_state_idx, next_state_idx] += P[z, k]
            end
        end
    end
    return T
end

# Calculate the stationary distribution
function stationary_distribution(T)
    eigs = eigen(T')
    # Find the eigenvector corresponding to eigenvalue closest to 1
    idx = argmax(abs.(eigs.values .- 1))
    stationary = eigs.vectors[:, idx]
    # Ensure the result is real-valued
    stationary = real(stationary)  # Take the real part
    stationary = stationary / sum(stationary)  # Normalize to sum to 1
    return stationary
end

# Marginal distribution of Xt
function marginal_distribution(stationary, Xt, Zt)
    num_X = length(Xt)
    num_Z = length(Zt)
    marginal_X = zeros(num_X)

    for (i, x) in enumerate(Xt)
        for (j, z) in enumerate(Zt)
            state_idx = (i - 1) * num_Z + j
            marginal_X[i] += stationary[state_idx]
        end
    end
    return marginal_X
end

# Expected value of Xt
function expected_value_Xt(marginal_X, Xt)
    return sum(marginal_X .* Xt)
end

# Main computation
T = joint_transition_matrix(Xt, Zt, P, σ)
stationary = stationary_distribution(T)
marginal_X = marginal_distribution(stationary, Xt, Zt)
expected_X = expected_value_Xt(marginal_X, Xt)

# Display results
println("Joint Transition Matrix for (Xt, Zt):")
println(T)

println("\nStationary Distribution for (Xt, Zt):")
println(stationary)

println("\nMarginal Distribution of Xt:")
println(marginal_X)

println("\nExpected Value of Xt:")
println(expected_X)
