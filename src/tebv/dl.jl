"""
    makex(varex::Vector{String}; df::DataFrame)::Tuple{Matrix{Float64}, Vector{String}, Vector{String}}

Constructs a design matrix `X` based on categorical variables specified in `varex` from the given DataFrame `df`.

# Arguments
- `varex::Vector{String}`: A vector of column names in `df` representing categorical variables to be encoded.
- `df::DataFrame`: The input DataFrame containing the data.

# Returns
A tuple `(X, X_vars, X_labels)` where:
- `X::Matrix{Float64}`: The design matrix with one-hot encoded columns for each categorical variable.
- `X_vars::Vector{String}`: A vector indicating the variable name corresponding to each column in `X`.
- `X_labels::Vector{String}`: A vector of unique labels for each categorical variable.

# Example
```
```
"""
function makex(varex::Vector{String}; df::DataFrame)::Tuple{Matrix{Float64},Vector{String},Vector{String}}
    n = nrow(df)
    X = nothing
    X_vars = []
    X_labels = []
    for v in varex
        # v = varex[1]
        x = unique(df[!, v])
        A = zeros(n, length(x))
        for i = 1:n
            # i = 1
            A[i, findfirst(x .== df[i, v])] = 1.0
        end
        X = if isnothing(X)
            A
        else
            hcat(X, A)
        end
        X_vars = vcat(X_vars, repeat([v], length(x)))
        X_labels = vcat(X_labels, x)
    end
    (X, X_vars, X_labels)
end


# Under construction...
function analyseviaNN(
    trials::Trials,
    traits::Vector{String};
    grm::Union{GRM,Nothing} = nothing,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose::Bool = false,
)::Nothing
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); grm::Union{GRM, Nothing} = grmploidyaware(genomes; ploidy = 2, max_iter = 10, verbose = true); traits::Vector{String} = ["trait_1"]; other_covariates::Union{Vector{String}, Nothing} = ["trait_2"]; n_iter::Int64 = 1_000; n_burnin::Int64 = 100; seed::Int64 = 1234; verbose::Bool = true;
    # Check arguments
    if !checkdims(trials)
        error("The Trials struct is corrupted ☹.")
    end
    if length(traits) > 0
        for trait in traits
            if !(trait ∈ trials.traits)
                throw(ArgumentError("The `traits` ($traits) argument is not a trait in the Trials struct."))
            end
        end
    end
    # Omit GRM?!?!?!
    if !isnothing(grm)
        if !checkdims(grm)
            throw(ArgumentError("The GRM is corrupted ☹."))
        end
    end
    if !isnothing(other_covariates)
        for c in other_covariates
            if !(c ∈ trials.traits)
                throw(ArgumentError("The `other_covariates` ($c) argument is not a trait in the Trials struct."))
            end
        end
    end
    # Bayesian network is a directed acyclic graph (DAG) representing probabilistic relationships between variables.
    # We therefore want to create a Bayesian network structure that captures the relationships between the one or more traits traits, entries, and other covariates.
    # using Lux, Optimisers
    seed = 42
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    df = tabularise(trials)
    y, y_min, y_max, X, X_vars, X_labels = let trait_id = "trait_1", varex = ["years", "seasons", "sites", "entries"]
        # trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"];
        y::Vector{Float64} = df[!, trait_id]
        X, X_vars, X_labels = makex(varex; df = df)
        n, p = size(X)
        # Map into 0 to 1 range instead of standardising because we are not assuming a single distribution for the trait, i.e. it may be multi-modal
        y_min = minimum(y)
        y_max = maximum(y)
        # UnicodePlots.histogram(y)
        y = (y .- y_min) ./ (y_max - y_min)
        y = vcat(y, zeros(n))
        X = vcat(hcat(X, zeros(n, n)), hcat(zeros(n, p), diagm(ones(n))))
        X_vars = vcat(X_vars, repeat(["Σ"], n))
        X_labels = vcat(X_labels, [string("Σ_", i) for i = 1:n])
        (y, y_min, y_max, X, X_vars, X_labels)
    end

    # Instantiate output Fit
    activation = [sigmoid, sigmoid_fast, relu, tanh][3]
    use_cpu = false
    verbose = true
    n_layers = 3
    max_n_nodes = 256
    n_nodes_droprate = 0.50
    dropout_droprate = 0.25
    n_epochs = 1_000
    n, p = size(X)
    model = if n_layers == 1
        Chain(Dense(p, 1, activation))
    elseif n_layers == 2
        Chain(Dense(p, max_n_nodes, activation), Dense(max_n_nodes, 1, activation))
    else
        model = if dropout_droprate > 0.0
            Chain(Dense(p, max_n_nodes, activation), Dropout(dropout_droprate))
        else
            Chain(Dense(p, max_n_nodes, activation))
        end
        for i = 2:(n_layers-1)
            model = if dropout_droprate > 0.0
                in_dims = model.layers[end-1].out_dims
                out_dims = Int64(maximum([round(in_dims * n_nodes_droprate), 1]))
                dp = model.layers[end].p * dropout_droprate
                Chain(model, Dense(in_dims, out_dims, activation), Dropout(dp))
            else
                in_dims = model.layers[end].out_dims
                out_dims = Int64(maximum([round(in_dims * n_nodes_droprate), 1]))
                Chain(model, Dense(in_dims, out_dims, activation))
            end
        end
        in_dims = if dropout_droprate > 0.0
            model.layers[end-1].out_dims
        else
            in_dims = model.layers[end].out_dims
        end
        model = Chain(model, Dense(in_dims, 1, activation))
        model
    end
    # Get the device determined by Lux
    dev = if use_cpu
        cpu_device()
    else
        gpu_device()
    end


    # Move the data to the device (Note that we have to transpose X and y)
    X_transposed::Matrix{Float64} = X'
    x = dev(X_transposed)
    a = dev(reshape(y, 1, length(y)))
    # Parameter and State Variables
    ps, st = Lux.setup(rng, model) |> dev # ps => parameters => weights and biases; st => state variable
    ## First construct a TrainState
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(0.0001f0))

    # Defining a custom loss function
    function W(model, ps, st, (x, a))
        â, st = model(x, ps, st)
        n = Int(size(â, 2) / 2)
        y = view(a, 1:n)
        ŷ = view(â, 1:n)
        loss_y = mean((ŷ .- y) .^ 2)
        ŝ = view(â, (n+1):(2*n))
        Ŝ = Matrix{Float64}(ŝ * ŝ')
        # loss_S = logpdf(MvNormal(zeros(n), Matrix{Float64}(Ŝ) + diagm(fill(0.1, n))), ŷ)
        # loss_S = logpdf(MvNormal(zeros(n), Ŝ), ŷ)
        loss_S = if !LinearAlgebra.isposdef(Ŝ)
            2*loss_y
        else
            logpdf(mvnorm_lol(Ŝ), ŷ)
        end
        loss = loss_y + loss_S
        return loss, st, NamedTuple()
    end


    ### Train
    Lux.trainmode(st) # not really required as the default is training mode, this is more for debugging with metrics calculations below
    if verbose
        pb = ProgressMeter.Progress(n_epochs, desc = "Training progress")
    end
    for iter = 1:n_epochs
        # Compute the gradients
        gs, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), W, (x, a), train_state)
        # gs, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), MSELoss(), (x, a), train_state)
        ## Optimise
        train_state = Training.apply_gradients!(train_state, gs)
        # # Compute gradients and optimise as a single call
        # _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), MSELoss(), (x, a), train_state)
        if verbose
            # println("Iteration: $iter ($(round(100*iter/n_epochs))%)\tLoss: $loss")
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Metrics
    Lux.testmode(st)
    y_pred, st = Lux.apply(model, x, ps, st)
    ϕ_pred::Vector{Float64} = y_pred[1, :] * (y_max - y_min) .+ y_min
    ϕ_true::Vector{Float64} = a[1, :] * (y_max - y_min) .+ y_min
    display(UnicodePlots.scatterplot(ϕ_true, ϕ_pred))
    cor(ϕ_pred, ϕ_true) |> println
    (ϕ_pred - ϕ_true) .^ 2 |> mean |> sqrt |> println

    # ps[1].weight
    # ps[1].bias
    # DataFrame(ids = X_labels, weights = ps[1].weight[1, :])
    # # Extract effects per entry
    # bool_entries = (X_vars .== "entries") .|| (X_vars .== "sites")
    # A = deepcopy(X)
    # A[:, .!bool_entries] .= 0.0
    # As = [join(r) for r in eachrow(A)]
    # Asu = unique(As)
    # idx = [findfirst(As .== x) for x in Asu]
    # A = A[idx, :]

    # Set the entries
    idx = findall([!isnothing(match(Regex("entries"), x)) for x in X_vars])
    X_new = zeros(length(idx), p)
    for (i, j) in enumerate(idx)
        X_new[i, j] = 1.0
    end
    # # Set the environmental factors
    # env_levels = ["year_1", "season_1", "site_1"]
    # for env in env_levels
    #     # env = env_levels[2]
    #     # @show env
    #     idx = findall([!isnothing(match(Regex(env), x)) for x in X_labels])[1]
    #     X_new[:, idx] .= 1.0
    # end
    Lux.testmode(st)
    x_new = dev(X_new')
    y_preds, st = Lux.apply(model, x_new, ps, st)
    ŷ = Vector(y_preds[1, :])
    ŷ = (ŷ * (y_max - y_min)) .+ y_min
    k = 1
    for i = 1:length(simulated_effects)
        # i = 1
        if simulated_effects[i].id[1] == trait_id
            k = i
            break
        end
    end
    g =
        simulated_effects[k].additive_genetic +
        simulated_effects[k].dominance_genetic +
        simulated_effects[k].epistasis_genetic
    UnicodePlots.scatterplot(ŷ, g)
    DataFrame(g = g, ŷ = ŷ)
    @show cor(ŷ, g) >= 0.9



    # Covariance estimation possible???
    idx = findall([!isnothing(match(Regex("Σ"), x)) for x in X_vars])
    X_new = zeros(length(idx), p)
    for (i, j) in enumerate(idx)
        X_new[i, j] = 1.0
    end
    x_new = dev(X_new')
    σ, st = Lux.apply(model, x_new, ps, st)
    σ = Vector(σ[1, :])
    u = Σ * σ
    UnicodePlots.histogram(σ)
    UnicodePlots.histogram(u)
    UnicodePlots.histogram(y)
    UnicodePlots.histogram(ŷ)

    df.entries

end
