### ATTEMPTING TO MAKE A BAYESIAN APPROACH TAILORED SPECIFICALLY FOR LARGE PARAMTER SPACES ###
# The implementation would typically involve:
# 1. Defining the Hamiltonian dynamics of the model.
# 2. Implementing the leapfrog integrator to simulate the Hamiltonian dynamics.
# 3. Adapting the step size and trajectory length based on the acceptance rate.
# 4. Using a dual averaging scheme to adjust the step size dynamically.
# 5. Implementing the No-U-Turn condition to terminate the trajectory early if it starts to turn back on itself.
# 6. Returning samples from the posterior distribution of the model parameters.
# using LinearAlgebra
# using Statistics
# using Random
# using ForwardDiff

"""
Computes the log of the joint probability of the position θ and momentum r.
This is equivalent to the negative Hamiltonian.
"""
function log_joint(θ, log_posterior, r)
    return log_posterior(θ) - 0.5 * dot(r, r)
end

"""
Performs a single leapfrog step to update the position and momentum.
"""
function leapfrog(θ, r, ϵ, log_posterior_grad)
    # Half step for momentum
    r_half = r + 0.5 * ϵ * log_posterior_grad(θ)
    # Full step for position
    θ_new = θ + ϵ * r_half
    # Full step for momentum
    r_new = r_half + 0.5 * ϵ * log_posterior_grad(θ_new)
    return θ_new, r_new
end

function naivegradient(log_posterior, θ; s=0.1)
    # n = 123
    # p = 1_000
    # h² = 0.5
    # X = rand(Beta(2.0, 2.0), n, p) 
    # β = abs.(round.(rand(Laplace(0.0, 0.001), p), digits=2))
    # σ²ᵦ = var(X * β)
    # σ² = σ²ᵦ * (1.0 / h² - 1.0)
    # e = rand(Normal(0.0, σ²), n)
    # y = X*β + e
    # log_posterior = θ -> -sqrt(mean((X*θ - y).^2))
    # θ = β; s=0.1
    # # p = length(θ)
    # # grad = fill(0.0, p)
    # # θ_δ = fill(0.0, p)
    # # pb = ProgressMeter.Progress(p, desc="Calculating gradient")
    # # @inbounds for j in 1:p
    # #     # @show j
    # #     θ_δ[j] = δ
    # #     grad[j] = (log_posterior(θ .+ θ_δ) - log_posterior(θ .- θ_δ)) ./ (2 * δ)
    # #     θ_δ[j] = 0.0
    # #     ProgressMeter.next!(pb)
    # # end
    # # ProgressMeter.finish!(pb)
    # # return grad
    p = length(θ)
    # θ_δ = rand(Normal(0.0, s), p)
    θ_δ = fill(0.0, p)
    idx = sample(1:p, Int(ceil(p*0.5)), replace=false)
    θ_δ[idx] = rand(Normal(0.0, s), length(idx))
    c = (log_posterior(θ) - log_posterior(θ .+ θ_δ))
    θ .+ (θ_δ.*c)
end

"""
The No-U-Turn Sampler (Iterative Implementation).
"""
function NoUTurnSampler(initial_θ, log_posterior, n_samples; ϵ=0.1, max_depth=10)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    # n = 123
    # p = 100_000
    # h² = 0.5
    # X = rand(Beta(2.0, 2.0), n, p) 
    # β = abs.(round.(rand(Laplace(0.0, 0.001), p), digits=2))
    # σ²ᵦ = var(X * β)
    # σ² = σ²ᵦ * (1.0 / h² - 1.0)
    # e = rand(Normal(0.0, σ²), n)
    # y = X*β + e
    # UnicodePlots.histogram(y)
    # log_posterior = θ -> -sqrt(mean((X*θ - y).^2))
    # initial_θ = zeros(p)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

    # Automatically create the gradient function using ForwardDiff
    # log_posterior_grad = θ -> ForwardDiff.gradient(log_posterior, θ)
    # @time log_posterior_grad(initial_θ);
    # @time log_posterior_grad(initial_θ);
    log_posterior_grad = θ -> naivegradient(log_posterior, θ)
    # @time log_posterior_grad(initial_θ);

    D = length(initial_θ)
    samples = zeros(n_samples, D)
    samples[1, :] = initial_θ
    θ_current = initial_θ

    pb = ProgressMeter.Progress(n_samples, desc="Sampling progress")
    @inbounds for i in 2:n_samples
        # i = 2
        # if i % (n_samples ÷ 10) == 0
        #     println("Progress: $(round(i/n_samples*100))%")
        # end

        # Resample momentum from a standard normal distribution
        r0 = randn(D)

        # Determine the slice variable for the acceptance window
        # The log(rand()) is a trick to sample from an exponential distribution
        u = log(rand()) + log_joint(θ_current, log_posterior, r0)

        # Initialize the trajectory endpoints
        θ_minus, θ_plus = θ_current, θ_current
        r_minus, r_plus = r0, r0
        
        j = 0         # Tree depth
        θ_proposal = θ_current # The proposed next sample
        n = 1         # Number of valid points in the trajectory
        s = 1         # Trajectory validity flag (1 = valid, 0 = invalid/U-turn)

        # Iteratively build the trajectory tree until a U-turn is detected or max depth is reached
        while s == 1 && j < max_depth
            # Choose a random direction to expand the tree: -1 (backwards) or 1 (forwards)
            direction = rand([-1, 1])
            
            # --- Build a new subtree in the chosen direction ---
            θ_edge, r_edge = (direction == -1) ? (θ_minus, r_minus) : (θ_plus, r_plus)
            
            n_prime = 0          # Number of valid points in the new subtree
            s_prime = 1          # Validity of the new subtree
            θ_prime_subtree = θ_edge # The proposal from the new subtree

            # The number of steps to take depends on the tree depth (2^j)
            num_steps = 2^j
            @inbounds for _ in 1:num_steps
                # Take a leapfrog step
                θ_edge, r_edge = leapfrog(θ_edge, r_edge, direction * ϵ, log_posterior_grad)
                
                # Check for divergence
                if !isfinite(log_joint(θ_edge, log_posterior, r_edge))
                    s_prime = 0
                    break
                end

                # Check if the new point is within the slice
                logp_new = log_joint(θ_edge, log_posterior, r_edge)
                if u <= logp_new
                    n_prime += 1
                    # With probability 1/n_prime, accept this new point as the subtree's proposal
                    if rand() < 1.0 / n_prime
                        θ_prime_subtree = θ_edge
                    end
                end
            end
            
            # Update the main trajectory endpoints with the edge of the new subtree
            if direction == -1
                θ_minus, r_minus = θ_edge, r_edge
            else
                θ_plus, r_plus = θ_edge, r_edge
            end

            # --- Combine the new subtree with the existing trajectory ---
            if s_prime == 1 && n_prime > 0
                # Probabilistically choose between the proposal from the old trajectory and the new one
                if rand() < n_prime / (n + n_prime)
                    θ_proposal = θ_prime_subtree
                end
            end
            
            n += n_prime # Update the total count of valid points

            # Check for a U-turn across the full trajectory
            # This is the "No-U-Turn" condition. It checks if the trajectory has started to double back on itself.
            s = s_prime * (dot(θ_plus - θ_minus, r_minus) >= 0) * (dot(θ_plus - θ_minus, r_plus) >= 0)
            
            j += 1 # Increment tree depth
        end # end while

        samples[i, :] = θ_proposal
        θ_current = θ_proposal
        ProgressMeter.next!(pb)
    end # end for
    ProgressMeter.finish!(pb)
    println("Sampling complete.")
    return samples
end


X, y, β, log_posterior_mvn = let n = Int(round(rand()*1_000)), p = Int(round(rand()*500)), h² = 0.5
    X = rand(Beta(2.0, 2.0), n, p) 
    # β = abs.(round.(rand(Laplace(0.0, 0.001), p), digits=2))
    β = rand(Normal(0.0, 0.1), p)
    σ²ᵦ = var(X * β)
    σ² = σ²ᵦ * (1.0 / h² - 1.0)
    e = rand(Normal(0.0, σ²), n)
    y = X*β + e
    display(UnicodePlots.histogram(y))
    (X, y, β, θ -> -sqrt(mean((X*θ - y).^2)))
end
initial_θ = zeros(size(β))  # Initial position in the parameter space
n_samples = 10_000
step_size = 0.1
max_tree_depth = 3

@time samples = NoUTurnSampler(initial_θ, log_posterior_mvn, n_samples, ϵ=step_size, max_depth=max_tree_depth);
b_hat = mean(samples, dims=1)[1, :]
UnicodePlots.scatterplot(X*b_hat, y)
@show size(X)
@show cor(X*b_hat,  y)
