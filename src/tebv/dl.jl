function makex(; df::DataFrame, varex::Vector{String}, verbose::Bool=false)::Tuple{Matrix{Float16},Vector{String},Vector{String}}
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
            x = Vector{Float16}(df[!, v])
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
    y::Vector{Float16} = df[idx, trait_id]
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
        S = (s * s') + CuArray(Array{Float16}(diagm(fill(0.01, n))))
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
    ϕ_true::Vector{Float16},
    ϕ_pred::Vector{Float16},
    y_max::Float16,
    y_min::Float16,
    μ::Vector{Float16},
    Σ::Matrix{Float16},
)
    ϕ_pred_remapped = Float64.(ϕ_pred) * (Float64(y_max) - Float64(y_min)) .+ Float64(y_min)
    ϕ_true_remapped = Float64.(ϕ_true) * (Float64(y_max) - Float64(y_min)) .+ Float64(y_min)
    corr_pearson = cor(ϕ_pred_remapped, ϕ_true_remapped)
    corr_spearman = corspearman(ϕ_pred_remapped, ϕ_true_remapped)
    diff = ϕ_pred_remapped - ϕ_true_remapped
    mae = mean(abs.(diff))
    rmse = sqrt(mean(diff.^ 2))
    corr_pearson_μ = cor(Float64.(μ), Float64.(ϕ_true))
    corr_spearman_μ = corspearman(Float64.(μ), Float64.(ϕ_true))
    mae_μ = mean(abs.(Float64.(μ) - Float64.(ϕ_true)))
    rmse_μ = sqrt(mean((Float64.(μ) - Float64.(ϕ_true)) .^ 2))
    loglik = logpdf(MvNormal(Float64.(μ), Σ), Float64.(ϕ_true))
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
    n_nodes_droprate::Float16 = 0.50,
    dropout_droprate::Float16 = 0.25,
    n_epochs::Int64 = 1_000,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"];activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_layers = 3; max_n_nodes = 256; n_nodes_droprate = 0.50; dropout_droprate = 0.25; n_epochs = 1_000; use_cpu = false; seed=42;  verbose::Bool = true;
    # y_orig, y_min_origin, y_max_orig, X_orig, X_vars_orig, X_labels_orig = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
    # n_orig = Int(size(X_orig, 1) / 3)
    # b_orig = rand(Float16, size(X_orig, 2))
    # df[!, trait_id] = X_orig[1:n_orig, :] * b_orig

    y, y_min, y_max, X, X_vars, X_labels = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
    n, p = size(X)
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
    x = dev(X')
    a = dev(reshape(y, 1, n))
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
    # Memory clean-up
    CUDA.reclaim()
    if verbose
        CUDA.pool_status()
    end
    Lux.testmode(st)
    y_pred, st = Lux.apply(model, x, ps, st);
    n = Int(length(y_pred) / 3);
    ϕ_pred::Vector{Float16} = y_pred[1, 1:n];
    ϕ_true::Vector{Float16} = a[1, 1:n];
    μ::Vector{Float16} = y_pred[1, (2*n+1):(3*n)];
    s::Vector{Float16} = y_pred[1, (n+1):(2*n)];
    Σ = Matrix{Float16}(s * s' + diagm(0.01 * ones(n))); # inflatediagonals!(Σ); det(Σ)
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
    # TODO: Fix below as it is probably wrong...
    # Marginal effects extraction, i.e. the effects of column in X keeping the other columns constant; all the while excluding the Σ variables
    if verbose
        pb = ProgressMeter.Progress(length(varex) + 1, desc = "Extracting marginal effects")
    end
    marginals = Dict()
    # Per explanatory variable
    gxe_vars = ["years", "seasons", "harvests", "sites", "entries"]
    idx_varex = vcat([findall([x ∈ gxe_vars for x in varex])], [[x] for x in 1:length(varex)])
    for idx_1 in idx_varex
        # idx_1 = idx_varex[1]
        # How many rows in the new X matrix do we need?
        m = 1
        for v in varex[idx_1]
            # v = varex[idx_1][1]
            m *= sum(X_vars .== v)
        end
        X_new = Float16.(zeros(3*m, p))
        # Which column indexes correspond to the explanatory variables we wish to vary?
        combins = []
        for v in varex[idx_1]
            # v = varex[idx_1][1]
            idx_2 = findall(X_vars .== v)
            push!(combins, idx_2)
        end
        # Define the new X matrix using all possible combinations of the explanatory variables
        X_labels_new::Vector{String} = fill("", m)
        for (i, idx_3) in enumerate(collect(Iterators.product(combins...)))
            # @show idx_3
            for j in idx_3
                X_new[i, j] = 1
                X_labels_new[i] = if X_labels_new[i] == ""
                    X_labels[j]
                else
                    X_labels_new[i] * "|" * X_labels[j]
                end
            end
        end
        # Predict
        x_new = dev(X_new')
        y_marginals, st = Lux.apply(model, x_new, ps, st)
        # Extract
        ϕ_marginals::Vector{Float16} = y_marginals[1, 1:m]
        s = y_marginals[1, (m+1):(2*m)]
        Σ_marginals::Matrix{Float16} = Matrix{Float16}(s * s' + CuArray(diagm(0.01 * ones(m))))
        z = ϕ_marginals ./ diag(Σ_marginals)
        p_vals = 2 * (1 .- cdf(Normal(0.0, 1.0), abs.(z)))
        marginals[join(varex[idx_1], "|")] = Dict(
            "labels" => X_labels_new,
            "ϕ_marginals" => ϕ_marginals,
            "Σ_marginals" => Σ_marginals,
            "z" => z,
            "p_vals" => p_vals,
        )
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
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
    n_nodes_droprate::Float16 = 0.50,
    dropout_droprate::Float16 = 0.25,
    n_epochs::Int64 = 1_000,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Nothing
    # genomes = simulategenomes(n=10, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, sparsity=0.1, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); traits = ["trait_1", "trait_2"]; other_covariates=["trait_3"]; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_layers = 3; max_n_nodes = 256; n_nodes_droprate = Float16(0.50); dropout_droprate = Float16(0.25); n_epochs = 10_000; use_cpu = false; seed=42;  verbose::Bool = true;
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
    varex_expected = if isnothing(other_covariates)
        ["years", "seasons", "sites", "harvests", "rows", "cols", "blocks", "replications", "entries"]
    else
        vcat(["years", "seasons", "sites", "harvests", "rows", "cols", "blocks", "replications", "entries"], other_covariates)
    end
    varex::Vector{String} = []
    n_levels = Dict()
    for v in varex_expected
        n = length(unique(df[!, v]))
        if n > 1
            push!(varex, v)
        end
        n_levels[v] = n
    end
    n_rc = n_levels["rows"] * n_levels["cols"]
    n_b = n_levels["blocks"]
    n_r = n_levels["replications"]
    varex = if (n_rc > n_b) && (n_rc > n_r)
        filter(x -> x != "blocks" && x != "replications", varex)
    elseif !(n_rc > n_b) && (n_rc > n_r)
        filter(x -> x != "replications" && x != "rows" && x != "cols", varex)
    else
        filter(x -> x != "blocks" && x != "rows" && x != "cols", varex)
    end
    if verbose
        println("Explanatory variables: ", varex)
        println("Number of levels for each variable: ", n_levels)
        println("Number of rows in the DataFrame: ", nrow(df))
    end
    # for trait_id in traits
        trait_id = traits[1]
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("Training neural network for trait: ", trait_id)
        end
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
        # fitted_nn["marginals"]
        # fitted_nn["marginals"]["years"]
        # fitted_nn["marginals"]["seasons"]
        # fitted_nn["marginals"]["sites"]
        # fitted_nn["marginals"]["trait_3"]
        # fitted_nn["marginals"]["rows"]
        # fitted_nn["marginals"]["cols"]
        # fitted_nn["marginals"]["entries"]

        effects_years = Dict()
        for k in fitted_nn["marginals"]["years"]["labels"]
            effects_years[k] = 0.0
        end
        effects_seasons = Dict()
        for k in fitted_nn["marginals"]["seasons"]["labels"]
            effects_seasons[k] = 0.0
        end
        effects_sites = Dict()
        for k in fitted_nn["marginals"]["sites"]["labels"]
            effects_sites[k] = 0.0
        end
        for (K, D) in Dict(
            :year => effects_years, 
            :season => effects_seasons, 
            :site => effects_sites, 
        )
            # K = :additive_genetic; D = effects_entries
            for (k, v) in D
                # k = string.(keys(D))[1]
                for x in simulated_effects
                    # x = simulated_effects[1]
                    if k ∈ x.id
                        D[k] = getproperty(x, K)
                        break
                    end
                end
            end
        end
        effects_years
        fitted_nn["marginals"]["years"]["ϕ_marginals"]
        effects_seasons
        fitted_nn["marginals"]["seasons"]["ϕ_marginals"]
        effects_sites
        fitted_nn["marginals"]["sites"]["ϕ_marginals"]


        # y_valid = y[idx_valid][1:Int(length(idx_valid) / 3)]
        # n = length(y_valid)
        # x_valid = dev(X[idx_valid, :]')
        # ϕ_hat, st = Lux.apply(model, x_valid, ps, st);
        # y_hat::Vector{Float16} = ϕ_hat[1, 1:n]
        # display(UnicodePlots.scatterplot(y_valid, y_hat))
        # cor(y_hat, y_valid) |> println
    # end
    
    nothing


end
