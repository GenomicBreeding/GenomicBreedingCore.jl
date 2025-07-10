function makex(; df::DataFrame, varex::Vector{String}, verbose::Bool=false)::Tuple{Matrix{Float64},Vector{String},Vector{String}}
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials);; varex = ["years", "seasons", "sites", "entries"]; verbose::Bool=false
    if sum([!(v ∈ names(df)) for v in varex]) > 0
        throw(ArgumentError("The explanatory variable/s: `$(join(varex[[!(v ∈ names(df)) for v in varex]], "`, `"))` do not exist in the DataFrame."))
    end
    X = nothing
    X_vars = []
    X_labels = []
    n = nrow(df)
    if verbose
        pb = ProgressMeter.Progress(length(varex), desc = "Preparing inputs")
    end
    @inbounds for v in varex
        # v = varex[end]
        A, x_vars, x_labels = try 
            x = Vector{Float64}(df[!, v])
            if sum(ismissing.(x) .|| isnan.(x) .|| isinf.(x)) > 0
                throw(ArgumentError("We expect the continuous numeric covariate ($v) to have no missing/NaN/Inf values relative to the response variable. Please remove these unsuitable values jointly across the response variable and covariates and/or remove the offending covariate ($v)."))
            end
            x_min = minimum(x)
            x_max = maximum(x)
            x = (x .- x_min) ./ (x_max - x_min)
            A = reshape(x, length(x), 1)
            (A, v, v)
        catch
            x = unique(df[!, v])
            A = zeros(n, length(x))
            for i = 1:n
                # i = 1
                A[i, findfirst(x .== df[i, v])] = 1.0
            end
            (A, repeat([v], length(x)), x)
        end
        X = if isnothing(X)
            A
        else
            hcat(X, A)
        end
        X_vars = vcat(X_vars, x_vars)
        X_labels = vcat(X_labels, x_labels)
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    (X, X_vars, X_labels)
end

function prepinputs(; df::DataFrame, varex::Vector{String}, trait_id::String, verbose::Bool=false)
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, sparsity=0.1, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries", "trait_3"]; verbose::Bool=false
    # Remove rows with missing, NaN or Inf values in the trait_id and varex
    if sum([!(v ∈ names(df)) for v in varex]) > 0
        throw(ArgumentError("The explanatory variable/s: `$(join(varex[[!(v ∈ names(df)) for v in varex]], "`, `"))` do not exist in the DataFrame."))
    end
    idx = []
    if verbose
        pb = ProgressMeter.Progress(nrow(df), desc = "Filtering rows with missing/NaN/Inf values")
    end
    @inbounds for i in 1:nrow(df)
        # i = 1
        bool = !ismissing(df[i, trait_id]) && !isnan(df[i, trait_id]) && !isinf(df[i, trait_id])
        if !bool
            continue
        end
        @inbounds for v in varex
            # v = varex[end]
            bool = try
                bool && (!ismissing(df[i, v]) && !isnan(df[i, v]) && !isinf(df[i, v]))
            catch
                bool && !ismissing(df[i, v])
            end
            if !bool
                break
            end
        end
        if !bool
            continue
        end
        push!(idx, i)
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    y::Vector{Float64} = df[idx, trait_id]
    y_min = minimum(y)
    y_max = maximum(y)
    y = (y .- y_min) ./ (y_max - y_min)
    X, X_vars, X_labels = makex(df = df[idx, :], varex=varex, verbose=verbose)
    n, p = size(X)
    y = vcat(
        y, 
        zeros(n), 
        zeros(n),
    )
    X = vcat(
        hcat(
            X, 
            zeros(n, n),
        ), 
        hcat(
            zeros(n, p), 
            diagm(ones(n)),
        ), 
        # zeros(n, n+p),
        hcat(
            zeros(n, p), 
            diagm(ones(n)),
        ), 
    )
    X_vars = vcat(X_vars, repeat(["Σ"], n))
    X_labels = vcat(X_labels, [string("Σ_", i) for i = 1:n])
    (y, y_min, y_max, X, X_vars, X_labels)
end

function prepmodel(;
    p, 
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    n_layers = 3,
    max_n_nodes = 256,
    n_nodes_droprate = 0.50,
    dropout_droprate = 0.25,
)
    if n_layers == 1
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
        Chain(model, Dense(in_dims, 1, activation))
    end
end

function lossϵΣ(model, ps, st, (x, a))
    # Forward pass through the model to get predictions
    â, st = model(x, ps, st)
    n = Int(size(â, 2) / 3)
    # Extract true values and predicted values for the trait
    y = view(a, 1:n)
    ŷ = view(â, 1:n)
    # Calculate MSE loss for the trait predictions
    loss_y = mean((ŷ .- y) .^ 2)
    # Calculate loss for covariance structure
    # using Mahalanobis distance: sqrt((y-μ)ᵀΣ⁻¹(y-μ))
    loss_S = begin
        s = view(â, (n+1):(2*n))
        S = (s * s') + CuArray(Array{Float32}(diagm(fill(0.01, n))))
        S_inv = inv(S)
        μ = view(â, (2*n+1):(3*n))
        diff = y - μ
        CUDA.allowscalar() do
            sqrt(diff' * S_inv * diff)
        end
    end
    # Combine both losses
    loss = loss_y + loss_S
    return loss, st, NamedTuple()
end

function goodnessoffit(;
    ϕ_true::Vector{Float64},
    ϕ_pred::Vector{Float64},
    y_max::Float64,
    y_min::Float64,
    μ::Vector{Float64},
    Σ::Matrix{Float64},
)
    ϕ_pred_remapped = ϕ_pred * (y_max - y_min) .+ y_min
    ϕ_true_remapped = ϕ_true * (y_max - y_min) .+ y_min
    corr_pearson = cor(ϕ_pred_remapped, ϕ_true_remapped)
    corr_spearman = corspearman(ϕ_pred_remapped, ϕ_true_remapped)
    mae = mean(abs.(ϕ_pred_remapped - ϕ_true_remapped))
    rmse = sqrt(mean((ϕ_pred_remapped - ϕ_true_remapped) .^ 2))
    corr_pearson_μ = cor(μ, ϕ_true_remapped)
    corr_spearman_μ = corspearman(μ, ϕ_true_remapped)
    mae_μ = mean(abs.(μ - ϕ_true_remapped))
    rmse_μ = sqrt(mean((μ - ϕ_true_remapped) .^ 2))
    loglik = logpdf(MvNormal(μ, Σ), ϕ_true)
    Dict(
        :ϕ_pred_remapped => ϕ_pred_remapped,
        :ϕ_true_remapped => ϕ_true_remapped,
        :corr_pearson => corr_pearson,
        :corr_spearman => corr_spearman,
        :mae => mae,
        :rmse => rmse,
        :corr_pearson_μ => corr_pearson_μ,
        :corr_spearman_μ => corr_spearman_μ,
        :mae_μ => mae_μ,
        :rmse_μ => rmse_μ,
        :loglik => loglik,
    )
end

function trainNN(
    df::DataFrame;
    trait_id::String,
    varex::Vector{String},
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    n_layers::Int64 = 3,
    max_n_nodes::Int64 = 256,
    n_nodes_droprate::Float64 = 0.50,
    dropout_droprate::Float64 = 0.25,
    n_epochs::Int64 = 1_000,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"];activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_layers = 3; max_n_nodes = 256; n_nodes_droprate = 0.50; dropout_droprate = 0.25; n_epochs = 1_000; use_cpu = false; seed=42;  verbose::Bool = true;
    y, y_min, y_max, X, X_vars, X_labels = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
    _n, p = size(X)
    model = prepmodel(
        p = p, 
        activation = activation, 
        n_layers = n_layers, 
        max_n_nodes = max_n_nodes, 
        n_nodes_droprate = n_nodes_droprate, 
        dropout_droprate = dropout_droprate,
    )
    dev = if use_cpu
        cpu_device()
    else
        gpu_device()
    end
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    n = Int(length(y)/3)
    idx_train = begin
        idx = sort(sample(rng, 1:n, Int(0.9*n), replace=false))
        vcat(idx, idx .+ n, idx .+ 2n)
    end
    idx_valid = sort(filter(x -> !(x ∈ idx_train), 1:3n))
    x = dev(X[idx_train, :]')
    a = dev(reshape(y[idx_train], 1, length(idx_train)))
    
    ps, st = Lux.setup(rng, model) |> dev # ps => parameters => weights and biases; st => state variable
    ## First construct a TrainState
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(0.0001f0))
    ### Train
    Lux.trainmode(st) # not really required as the default is training mode, this is more for debugging with metrics calculations below
    if verbose
        pb = ProgressMeter.Progress(n_epochs, desc = "Training progress")
    end
    t = []
    l = []
    for iter = 1:n_epochs
        # Compute the gradients
        gs, loss, stats, train_state = Lux.Training.compute_gradients(AutoZygote(), lossϵΣ, (x, a), train_state)
        ## Optimise
        train_state = Training.apply_gradients!(train_state, gs)
        # # Alternatively, compute gradients and optimise with a single call
        # _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), MSELoss(), (x, a), train_state)
        if verbose
            ProgressMeter.next!(pb)
        end
        push!(t, iter)
        push!(l, loss)
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # CUDA.reclaim()
    # CUDA.pool_status()
    Lux.testmode(st)
    y_pred, st = Lux.apply(model, x, ps, st);
    n = Int(length(y_pred) / 3);
    ϕ_pred::Vector{Float64} = y_pred[1, 1:n];
    ϕ_true::Vector{Float64} = a[1, 1:n];
    μ::Vector{Float64} = y_pred[1, (2*n+1):(3*n)];
    s::Vector{Float64} = y_pred[1, (n+1):(2*n)];
    Σ = Matrix{Float64}(s * s' + diagm(0.01 * ones(n))); # inflatediagonals!(Σ); det(Σ)
    stats = goodnessoffit(
        ϕ_true=ϕ_true,
        ϕ_pred=ϕ_pred,
        y_max=y_max,
        y_min=y_min,
        μ=μ,
        Σ=Σ,
    )
    if verbose
        # Plot the training loss
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("TRAINING LOSS:")
        display(UnicodePlots.scatterplot(t, l, xlabel = "Iteration", ylabel = "Loss", title = "Training Loss"))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED VALUES:")
        display(UnicodePlots.scatterplot(stats[:ϕ_true_remapped], stats[:ϕ_pred_remapped], xlabel = "Observed", ylabel = "Fitted", title = "Fitted vs Observed"))
        println("Pearson's product-moment correlation: ", round(100*stats[:corr_pearson], digits=2), "%")
        println("Spearman's rank correlation: ", round(100*stats[:corr_spearman], digits=2), "%")
        println("MAE: ", round(stats[:mae], digits=4))
        println("RMSE: ", round(stats[:rmse], digits=4))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED MULTIVARIATE NORMAL DISTRIBUTION:")
        # display(UnicodePlots.scatterplot(ϕ_true_remapped, μ, xlabel = "Observed", ylabel = "Fitted expectations of the multivariate normal distribution (μ)", title = "μ vs Observed"))
        # println("Pearson's product-moment correlation with μ: ", round(100*stats[:corr_pearson_μ], digits=2), "%")
        # println("Spearman's rank correlation with μ: ", round(100*stats[:corr_spearman_μ], digits=2), "%")
        # println("MAE with μ: ", round(stats[:mae_μ], digits=4))
        # println("RMSE with μ: ", round(stats[:rmse_μ], digits=4))
        display(UnicodePlots.heatmap(Σ, title="Fitted variance-covariance matrix (Σ)"))
        println("Goodness of fit in log-likelihood: ", round(stats[:loglik], digits=4))
    end
    # Marginal effects extraction, i.e. the effects of column in X keeping the other columns constant; all the while excluding the Σ variables
    marginals = Dict()
    for v in varex
        # varex[1]
        idx_marginals = findall([x == v for x in X_vars])
        idx_to_zeros = findall([!(x ∈ vcat(v, "Σ")) for x in X_vars])
        X_new = deepcopy(X)
        X_new[:, idx_to_zeros] .= 0.0
        x_new = dev(X_new')
        y_marginals, st = Lux.apply(model, x_new, ps, st)
        s = y_marginals[1, (n+1):(2*n)]
        Σ = Matrix{Float64}(s * s' + diagm(0.01 * ones(n)))
        ϕ_marginals::Vector{Float64} = y_marginals[1, idx_marginals]
        Σ_marginals::Matrix{Float64} = Σ[idx_marginals, idx_marginals]
        if verbose
            println("Marginal effects for variable: ", v)
            @show ϕ_marginals
            @show Σ_marginals
        end
        # Probably wrong...
        z = ϕ_marginals ./ diag(Σ_marginals)
        p_vals = 2 * (1 .- cdf(Normal(0.0, 1.0), abs.(z)))
        marginals[v] = Dict(
            "labels" => X_labels[idx_marginals],
            "ϕ_marginals" => ϕ_marginals,
            "Σ_marginals" => Σ_marginals,
            "z" => z,
            "p_vals" => p_vals,
        )
    end
    # # Model I/O
    # @save "temp_model.jld2" ps st
    # @load "temp_model.jld2" ps st
    return Dict(
        "model" => model,
        "parameters" => ps,
        "state" => st,
        "values" => ϕ_pred * (y_max - y_min) .+ y_min,
        "means" => μ,
        "covariances" => Σ,
        "marginals" => marginals,
        "stats" => stats,
    )
end

# Under construction...
function analyseviaNN(
    trials::Trials,
    traits::Vector{String};
    other_covariates::Union{Vector{String},Nothing} = nothing,
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    n_layers::Int64 = 3,
    max_n_nodes::Int64 = 256,
    n_nodes_droprate::Float64 = 0.50,
    dropout_droprate::Float64 = 0.25,
    n_epochs::Int64 = 1_000,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Nothing
    # genomes = simulategenomes(n=10, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, sparsity=0.1, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); traits = ["trait_1", "trait_2"]; other_covariates=["trait_3"]; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_layers = 3; max_n_nodes = 256; n_nodes_droprate = 0.50; dropout_droprate = 0.25; n_epochs = 1_000; use_cpu = false; seed=42;  verbose::Bool = true;
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
    if !isnothing(other_covariates)
        for c in other_covariates
            if !(c ∈ trials.traits)
                throw(ArgumentError("The `other_covariates` ($c) argument is not a trait in the Trials struct."))
            end
        end
    end
    # Tabularise the Trials struct
    df = tabularise(trials)
    
    


    # Define the explanatory variables
    # TODO: detect fixed columns and which spatial variable to use best
    varex_expected = if isnothing(other_covariates)
        ["years", "seasons", "sites", "harvests", "rows", "cols", "entries"]
    else
        vcat(["years", "seasons", "sites", "harvests", "rows", "cols", "entries"], other_covariates)
    end
    varex::Vector{String} = []
    for v in varex_expected
        if length(unique(df[!, v])) > 1
            push!(varex, v)
        end
    end
    trait_id = traits[1]





    fitted_nn = trainNN(
        df,
        trait_id=trait_id,
        varex=varex,
        activation=activation,
        n_layers=n_layers,
        max_n_nodes=max_n_nodes,
        n_nodes_droprate=n_nodes_droprate,
        dropout_droprate=dropout_droprate,
        n_epochs=n_epochs,
        use_cpu=use_cpu,
        seed=seed,
        verbose=verbose,
    )
    fitted_nn["marginals"]
    fitted_nn["marginals"]["seasons"]
    fitted_nn["marginals"]["sites"]
    fitted_nn["marginals"]["trait_3"]
    fitted_nn["marginals"]["rows"]
    fitted_nn["marginals"]["cols"]

    # y_valid = y[idx_valid][1:Int(length(idx_valid) / 3)]
    # n = length(y_valid)
    # x_valid = dev(X[idx_valid, :]')
    # ϕ_hat, st = Lux.apply(model, x_valid, ps, st);
    # y_hat::Vector{Float64} = ϕ_hat[1, 1:n]
    # display(UnicodePlots.scatterplot(y_valid, y_hat))
    # cor(y_hat, y_valid) |> println
    



end
