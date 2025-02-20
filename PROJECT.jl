using Pkg
Pkg.add(["LinearAlgebra", "Statistics", "NLsolve", "Parameters", "Distributions", "Plots", "Roots"])
using LinearAlgebra, Statistics, Plots, Distributions, Roots

# Parameters
β = 0.96     # Discount factor (to be calibrated)
γ = 2.0      # Risk aversion
ρ = 0.9      # Persistence of productivity
σ = 0.4      # Variance of productivity shock
α = 0.36     # Capital share in production
δ = 0.08     # Depreciation rate (to be calibrated)
A = 1.0      # Productivity level (to be calibrated)
ϕ = 0.0      # Borrowing constraint
λ_values = [0.0, 0.15] # Tax progressivity levels

# Discretize productivity process using Tauchen method
function tauchen(N, μ, ρ, σ, m=3)
    z_std = sqrt(σ^2 / (1 - ρ^2))
    z_grid = collect(range(μ - m * z_std, μ + m * z_std, length=N)) # Ensure it's a vector
    p_mat = zeros(N, N)
    
    for j in 1:N
        for k in 1:N
            if k == 1
                p_mat[j, k] = cdf(Normal(), (z_grid[k] - ρ * z_grid[j] + z_grid[2] - z_grid[1]) / σ)
            elseif k == N
                p_mat[j, k] = 1 - cdf(Normal(), (z_grid[k] - ρ * z_grid[j] - (z_grid[2] - z_grid[1])) / σ)
            else
                p_mat[j, k] = cdf(Normal(), (z_grid[k] - ρ * z_grid[j] + (z_grid[2] - z_grid[1]) / 2) / σ) -
                              cdf(Normal(), (z_grid[k] - ρ * z_grid[j] - (z_grid[2] - z_grid[1]) / 2) / σ)
            end
        end
    end
    return z_grid, p_mat
end

# Solve for equilibrium wage and interest rate
function equilibrium_prices(α, A, K, L, δ)
    r = α * A * (K^(α-1)) * (L^(1-α)) - δ
    w = (1 - α) * A * (K^α) * (L^(-α))
    return r, w
end

# Compute Gini coefficient
function gini_coefficient(x)
    x_sorted = sort(x)
    n = length(x)
    return 1 - 2 * sum((n - i + 0.5) * x_sorted[i] for i in 1:n) / (n * sum(x))
end

# Generate required plots
function generate_plots(V, policy, grid_assets, grid_productivity)
    plot1 = plot(grid_assets, V, title="Value Functions for Both Economies")
    plot2 = plot(grid_assets, policy, title="Policy Functions for Both Economies")
    plot3 = histogram(grid_assets, title="Marginal Distribution of Assets")
    lorenz_x = cumsum(sort(grid_assets)) / sum(grid_assets)
    lorenz_y = collect(range(0, 1, length=length(lorenz_x)))
    plot4 = plot(lorenz_x, lorenz_y, title="Lorenz Curves for After-Tax Labor Income and Assets")
    display(plot1)
    display(plot2)
    display(plot3)
    display(plot4)
end

# Bellman equation iteration for value function and policy function
function solve_bellman(grid_assets, grid_productivity, transition_matrix, r, w, λ, β, γ)
    V = zeros(length(grid_assets), length(grid_productivity)) # Initial value function
    policy = zeros(length(grid_assets), length(grid_productivity)) # Policy function
    tol = 1e-6
    max_iter = 1000
    diff = tol + 1
    iter = 0
    
    while diff > tol && iter < max_iter
        V_new = copy(V)
        for i in 1:length(grid_productivity)
            z = grid_productivity[i]
            for j in 1:length(grid_assets)
                a = grid_assets[j]
                y = w * z
                best_val = -Inf
                best_a_prime = 0.0
                for k in 1:length(grid_assets)
                    a_prime = grid_assets[k]
                    c = max(y + (1 + r) * a - a_prime, 1e-10) # Enforce borrowing constraint
                    u = (c^(1-γ) - 1) / (1-γ) # Adjusted utility function
                    EV = sum(transition_matrix[i, :] .* V[k, :]) # Corrected expectation computation
                    val = u + β * EV
                    if val > best_val
                        best_val = val
                        best_a_prime = a_prime
                    end
                end
                V_new[j, i] = best_val
                policy[j, i] = best_a_prime
            end
        end
        diff = maximum(abs.(V_new - V))
        V = copy(V_new)
        iter += 1
    end
    return V, policy
end

# Main function to solve the model
function main()
    grid_assets = collect(range(-ϕ, 50, length=100))
    grid_productivity, transition_matrix = tauchen(5, 0, ρ, σ)
    
    results = []
    for λ in λ_values
        r, w = equilibrium_prices(α, A, 10, 1, δ)
        V, policy = solve_bellman(grid_assets, grid_productivity, transition_matrix, r, w, λ, β, γ)
        gini_assets = gini_coefficient(grid_assets)
        gini_income = gini_coefficient(grid_productivity .* w)
        push!(results, (λ, r, w, gini_assets, gini_income))
        println("λ = $λ: r = $r, w = $w, Gini(assets) = $gini_assets, Gini(income) = $gini_income")
        generate_plots(V, policy, grid_assets, grid_productivity)
    end
    return results
end

results = main()
