function makex(; df::DataFrame, varex::Vector{String}, verbose::Bool=false)::Tuple{Matrix{Float16},Vector{String},Vector{String}}
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); varex = ["years", "seasons", "sites", "entries"]; verbose::Bool=false
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
    
    # TODO: consider SparseArrays.jl
    
    y = vcat(
        y, 
        zeros(n), 
        zeros(n),
    )
    # N = maximum([n, p])
    X = vcat(
        hcat(
            X, 
            zeros(n, n),
            # zeros(n, N-p),
        ), 
        # hcat(
        #     diagm(ones(n)),
        #     zeros(n, N-n)
        # ), 
        # hcat(
        #     diagm(ones(n)),
        #     zeros(n, N-n)
        # ), 
        hcat(
            zeros(n, p), 
            diagm(ones(n)),
        ), 
        hcat(
            zeros(n, p), 
            diagm(ones(n)),
        ), 
        # # zeros(n, n+p),
        # hcat(
        #     zeros(n, p), 
        #     diagm(ones(n)),
        # ), 
    )
    X_vars = vcat(X_vars, repeat(["Σ"], n))
    X_labels = vcat(X_labels, [string("Σ_", i) for i = 1:n])
    (y, y_min, y_max, X, X_vars, X_labels)
end

function prepmodel(;
    input_dims::Int64,
    hidden_dims::Int64,
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    output_dims::Int64 = 1,
    n_hidden_layers::Int64 = 3,
    dropout_rate::Float64 = 0.1,
)
    @compact(
        input=Dense(input_dims, hidden_dims),
        hidden=[Dense(hidden_dims, hidden_dims) for i in 1:n_hidden_layers],
        output=Dense(hidden_dims, output_dims),
        activation=activation,
        dropout=Dropout(dropout_rate),
    ) do x
        layers = activation.(input(x))
        for W in hidden
            W = activation.(W(layers))
            W = dropout(W)
        end
        out = output(layers)
        @return out
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
    loss_y = sum((ŷ .- y) .^ 2)
    # Calculate loss for covariance structure
    # using the square of the Mahalanobis distance: (y-μ)ᵀΣ⁻¹(y-μ)
    loss_S = begin
        s = view(â, (n+1):(2*n))
        S = (s * s') + CuArray(Array{Float16}(diagm(fill(maximum(s), n))))
        S_inv = inv(S)
        μ = view(â, (2*n+1):(3*n))
        diff = y - μ
        diff' * S_inv * diff
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
    μ_pred_remapped = Float64.(μ) * (Float64(y_max) - Float64(y_min)) .+ Float64(y_min)
    corr_pearson = cor(ϕ_true_remapped, ϕ_pred_remapped)
    corr_spearman = corspearman(ϕ_true_remapped, ϕ_pred_remapped)
    corr_lin = begin
        σ_true = std(ϕ_true_remapped)
        σ_pred = std(ϕ_pred_remapped)
        μ_true = mean(ϕ_true_remapped)
        μ_pred = mean(ϕ_pred_remapped)
        ρ = corr_pearson
        (2*ρ*σ_true*σ_pred) / (σ_true^2 + σ_pred^2 + (μ_true - μ_pred)^2)
    end
    diff = ϕ_pred_remapped - ϕ_true_remapped
    mae = mean(abs.(diff))
    rmse = sqrt(mean(diff.^ 2))
    R² = 1 - (sum((diff.^ 2)) / sum((ϕ_true_remapped .- mean(ϕ_true_remapped)).^2))
    loglik, mahalanobis_distance = begin
        counter = 0
        while !isposdef(Σ) && (counter < 100)
            Σ += Float16.(diagm(fill(0.1, size(Σ, 1))))
            counter += 1
        end
        if !isposdef(Σ)
            (NaN, NaN)
        else
            loglik = logpdf(MvNormal(Float64.(μ), Σ), Float64.(ϕ_true))
            mahalanobis_distance = sqrt(diff' * inv(Σ) * diff)
            (loglik, mahalanobis_distance)
        end
    end
    Dict(
        :ϕ_pred_remapped => ϕ_pred_remapped,
        :ϕ_true_remapped => ϕ_true_remapped,
        :μ_pred_remapped => μ_pred_remapped,
        :corr_pearson => corr_pearson,
        :corr_spearman => corr_spearman,
        :corr_lin => corr_lin,
        :mae => mae,
        :rmse => rmse,
        :R² => R²,
        :loglik => loglik,
        :mahalanobis_distance => mahalanobis_distance,
    )
end

function trainNN(
    df::DataFrame;
    trait_id::String,
    varex::Vector{String},
    idx_training::Union{Vector{Int64}, Nothing} = nothing,
    idx_validation::Union{Vector{Int64}, Nothing} = nothing,
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    optimiser = [
        Optimisers.Adam(0.0001f0),
        Optimisers.NAdam(),
        Optimisers.RAdam(),
        Optimisers.AdaMax(),
    ][4],
    n_hidden_layers::Int64 = 3,
    hidden_dims::Int64 = 256,
    dropout_rate::Float64 = 0.01,
    n_epochs::Int64 = 10_000,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"];activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = 0.00; n_epochs = 1_000; use_cpu = false; seed=42;  verbose::Bool = true; idx_training = sort(sample(1:nrow(df), Int(round(0.9*nrow(df))), replace=false)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); optimiser = [Optimisers.Adam(0.0001f0),Optimisers.NAdam(),Optimisers.RAdam(),Optimisers.AdaMax(),][4]
    # # y_orig, y_min_origin, y_max_orig, X_orig, X_vars_orig, X_labels_orig = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
    # # n_orig = Int(size(X_orig, 1) / 3)
    # # b_orig = rand(Float16, size(X_orig, 2))
    # # df[!, trait_id] = X_orig[1:n_orig, :] * b_orig
    # Checks
    errors::Vector{String} = []
    ϕ = df[!, trait_id]
    if sum(ismissing.(ϕ) .|| isnan.(ϕ) .|| isinf.(ϕ)) > 0
        push!(error, "Missing data in trait: $trait_id is not permitted. Please filter-out missing data first as these may potentially conflict with the supplied training and/or validation set indexes.")
    end
    if !("entries" ∈ names(df))
        push!(error, "The expected `entries` column is absent in the input data frame. We expect a tabularised Trials struct.")
    end
    # Checks
    if !isnothing(idx_training)
        if minimum(idx_training) < 1
            push!(errors, "Training set index starts below 1.")
        end
        if maximum(idx_training) > nrow(df)
            push!(errors, "Training set index is greater than the number of observations, i.e. above $(nrow(df)).")
        end
    end
    if !isnothing(idx_validation)
        if minimum(idx_validation) < 1
            push!(errors, "Validation set index starts below 1.")
        end
        if maximum(idx_validation) > nrow(df)
            push!(errors, "Validation set index is greater than the number of observations, i.e. above $(nrow(df)).")
        end
    end
    idx_training, idx_validation= if isnothing(idx_training) && isnothing(idx_validation)
        (collect(1:nrow(df)), [])
    elseif isnothing(idx_training) && !isnothing(idx_validation)
        (filter(x -> !(x ∈ idx_validation), 1:nrow(df)), sort(idx_validation))
    elseif !isnothing(idx_training) && isnothing(idx_validation)
        (sort(idx_training), filter(x -> !(x ∈ idx_training), 1:nrow(df)))
    else
        (sort(idx_training), sort(idx_validation))
    end
    if length(idx_training) < 2
        push!(errors, "There is less than 2 observations for training!")
    end
    if length(filter(x -> x ∈ idx_training, idx_validation)) > 0
        push!(errors, "There is data leakage!")
    end
    if length(errors) > 0
        throw(ArgumentError(string("\n\t‣ ", join(errors, "\n\t‣ "))))
    end
    y, X, y_validation, X_validation, y_min, y_max, X_vars, X_labels = begin
        y_ALL, y_min_ALL, y_max_ALL, X_ALL, X_vars_ALL, X_labels_ALL = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
        n = nrow(df)
        sort!(idx_training)
        sort!(idx_validation)
        idx_training_including_μ_and_Σ = vcat(idx_training, idx_training.+(n), idx_training.+(2*n))
        idx_validation_including_μ_and_Σ = vcat(idx_validation, idx_validation.+(n), idx_validation.+(2*n))
        (
            y_ALL[idx_training_including_μ_and_Σ],
            X_ALL[idx_training_including_μ_and_Σ, :],
            y_ALL[idx_validation_including_μ_and_Σ],
            X_ALL[idx_validation_including_μ_and_Σ, :],
            y_min_ALL, 
            y_max_ALL, 
            X_vars_ALL, 
            X_labels_ALL,
        )
    end
    n, p = size(X)
    model = prepmodel(
        input_dims=p,
        hidden_dims=hidden_dims,
        activation=activation,
        output_dims=1,
        n_hidden_layers=n_hidden_layers,
        dropout_rate=dropout_rate,
    )
    dev = if use_cpu
        cpu_device()
    else
        gpu_device()
    end
    rng = Random.seed!(seed)
    x = dev(X')
    x_validation = dev(X_validation')
    a = dev(reshape(y, 1, n))
    ps, st = Lux.setup(rng, model) |> dev # ps => parameters => weights and biases; st => state variable
    ## First construct a TrainState
    training_state = Lux.Training.TrainState(model, ps, st, optimiser)
    ### Train
    if verbose
        pb = ProgressMeter.Progress(n_epochs, desc = "Training progress")
    end
    t = []
    l = []
    v = []
    for iter = 1:n_epochs
        # Compute the gradients
        gradients, loss, stats, training_state = Lux.Training.compute_gradients(AutoZygote(), lossϵΣ, (x, a), training_state)
        ## Optimise
        training_state = Training.apply_gradients!(training_state, gradients)
        # # Alternatively, compute gradients and optimise with a single call
        # _, loss, _, training_state = Lux.Training.single_train_step!(AutoZygote(), MSELoss(), (x, a), training_state)
        # Check cross-validation performance
        validation_state = Lux.testmode(training_state.states)
        y_pred, _ = Lux.apply(model, x_validation, ps, validation_state);
        n = Int(length(y_pred) / 3);
        ϵ = Float64.(Vector(y_pred[1, 1:n])) - y_validation[1:n]

        if verbose
            ProgressMeter.next!(pb)
        end
        push!(t, iter)
        push!(l, loss)
        push!(v, sqrt(mean(ϵ.^2)))
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Memory clean-up
    CUDA.reclaim()
    if verbose
        CUDA.pool_status()
    end
    validation_state = Lux.testmode(training_state.states)
    y_pred, _ = Lux.apply(model, x, ps, validation_state);
    n = Int(length(y_pred) / 3);
    ϕ_pred::Vector{Float16} = y_pred[1, 1:n];
    ϕ_true::Vector{Float16} = a[1, 1:n];
    μ::Vector{Float16} = y_pred[1, (2*n+1):(3*n)];
    s::Vector{Float16} = y_pred[1, (n+1):(2*n)];
    Σ = Matrix{Float16}(s * s')
    stats = goodnessoffit(
        ϕ_true=ϕ_true,
        ϕ_pred=ϕ_pred,
        y_max=y_max,
        y_min=y_min,
        μ=μ,
        Σ=Σ,
    )
    # Cross-validation
    stats_validation = if length(idx_validation) > 0
        n = length(idx_validation)
        x_validation = dev(X_validation')
        a_validation = dev(reshape(y_validation, 1, 3*n))
        y_pred_validation, _ = Lux.apply(model, x_validation, ps, validation_state);
        ϕ_pred_validation::Vector{Float16} = y_pred_validation[1, 1:n];
        ϕ_true_validation::Vector{Float16} = a_validation[1, 1:n];
        μ_validation::Vector{Float16} = y_pred_validation[1, (2*n+1):(3*n)];
        s_validation::Vector{Float16} = y_pred_validation[1, (n+1):(2*n)];
        Σ_validation = Matrix{Float16}(s_validation * s_validation');
        goodnessoffit(
            ϕ_true=ϕ_true_validation,
            ϕ_pred=ϕ_pred_validation,
            y_max=y_max,
            y_min=y_min,
            μ=μ_validation,
            Σ=Σ_validation,
        )
    else
        nothing
    end
    if verbose
        # Plot the training loss
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("TRAINING LOSS:")
        display(UnicodePlots.scatterplot(t, l, xlabel = "Iteration", ylabel = "Loss", title = "Training Loss"))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("CROSS-VALIDATION RMSE:")
        display(UnicodePlots.scatterplot(t, v, xlabel = "Iteration", ylabel = "CV RMSE", title = "Cross-validation RMSE"))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED VALUES:")
        display(UnicodePlots.scatterplot(stats[:ϕ_true_remapped], stats[:ϕ_pred_remapped], xlabel = "Observed", ylabel = "Fitted", title = "Fitted vs Observed"))
        println("Pearson's product-moment correlation: ", round(100*stats[:corr_pearson], digits=2), "%")
        println("Spearman's rank correlation: ", round(100*stats[:corr_spearman], digits=2), "%")
        println("MAE: ", round(stats[:mae], digits=4))
        println("RMSE: ", round(stats[:rmse], digits=4))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED MULTIVARIATE NORMAL DISTRIBUTION:")
        # display(UnicodePlots.scatterplot(stats[:ϕ_true_remapped], stats[:μ_pred_remapped], xlabel = "Observed", ylabel = "Fitted expectations of the multivariate normal distribution (μ)", title = "μ vs Observed"))
        # println("Pearson's product-moment correlation with μ: ", round(100*stats[:corr_pearson_μ], digits=2), "%")
        # println("Spearman's rank correlation with μ: ", round(100*stats[:corr_spearman_μ], digits=2), "%")
        # println("MAE with μ: ", round(stats[:mae_μ], digits=4))
        # println("RMSE with μ: ", round(stats[:rmse_μ], digits=4))
        display(UnicodePlots.heatmap(Σ, title="Fitted variance-covariance matrix (Σ)"))
        println("Goodness of fit in log-likelihood: ", round(stats[:loglik], digits=4))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("CROSS-VALIDATION:")
        if isnothing(stats_validation)
            println("None")
        else
            display(UnicodePlots.scatterplot(stats_validation[:ϕ_true_remapped], stats_validation[:ϕ_pred_remapped]))
            println("Pearson's product-moment correlation: ", round(100*stats_validation[:corr_pearson], digits=2), "%")
            println("Spearman's rank correlation: ", round(100*stats_validation[:corr_spearman], digits=2), "%")
            println("MAE: ", round(stats_validation[:mae], digits=4))
            println("RMSE: ", round(stats_validation[:rmse], digits=4))
        end
    end
    # Marginal effects extraction, i.e. the effects of column in X keeping the other columns constant; all the while excluding the Σ variables
    if verbose
        pb = ProgressMeter.Progress(length(varex) + 1, desc = "Extracting marginal effects")
    end
    marginals = Dict()
    # Per explanatory variable
    gxe_vars = ["years", "seasons", "harvests", "sites", "entries"]
    idx_varex = vcat([findall([x ∈ gxe_vars for x in varex])], [[x] for x in 1:length(varex)])
    for idx_1 in idx_varex
        # idx_1 = idx_varex[end]
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
        y_marginals, _ = Lux.apply(model, x_new, ps, validation_state)
        # Extract
        ϕ_marginals::Vector{Float16} = y_marginals[1, 1:m]
        s = y_marginals[1, (m+1):(2*m)]
        Σ_marginals::Matrix{Float16} = (s * s')
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
    # @save "temp_model.jld2" training_state.parameters training_state.states
    # @load "temp_model.jld2" training_state.parameters training_state.states
    return Dict(
        "model" => model,
        "parameters" => training_state.parameters,
        "state" => training_state.states,
        "values" => ϕ_pred * (y_max - y_min) .+ y_min,
        "means" => μ,
        "covariances" => Σ,
        "marginals" => marginals,
        "stats" => stats,
        "stats_validation" => stats_validation,
    )
end


function optimNN(
    df::DataFrame;
    trait_id::String,
    varex::Vector{String},
    validation_rate::Float64 = 0.25,
    activation::Any = [sigmoid, sigmoid_fast, relu, tanh][3],
    optimiser = nothing,
    n_epochs::Union{Nothing, Int64} = nothing,
    n_hidden_layers::Union{Nothing, Int64} = nothing,
    hidden_dims::Union{Nothing, Int64} = nothing,
    dropout_rate::Union{Nothing, Float64} = nothing,
    use_cpu::Bool = false,
    n_random_searches::Int64 = 100,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Tuple{DataFrame, Int64}
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=2); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"]; validation_rate=0.25; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; optimiser = nothing; use_cpu = false; seed=42;  verbose::Bool = true; n_hidden_layers = nothing; hidden_dims = nothing; dropout_rate = nothing; n_epochs = nothing; seed = 42; n_random_searches=20
    choices_activation = if !isnothing(activation)
        [activation]
    else
        [sigmoid, sigmoid_fast, relu, tanh]
    end
    choices_optimiser = if !isnothing(optimiser)
        [optimiser]
    else
        [
            Optimisers.Adam(),
            Optimisers.NAdam(),
            Optimisers.RAdam(),
            Optimisers.AdaMax(),
        ]
    end
    choices_n_epochs = if !isnothing(n_epochs)
        n_epochs
    else
        [1_000, 5_000, 7_000, 10_000]
    end
    choices_n_hidden_layers = if !isnothing(n_hidden_layers)
        [n_hidden_layers]
    else
        [2, 3, 4, 5]
    end
    choices_hidden_dims = if !isnothing(hidden_dims)
        [hidden_dims]
    else
        [2^8, 2^9, 2^10, 2^11]
    end
    choices_dropout_rate = if !isnothing(dropout_rate)
        [dropout_rate]
    else
        [0.0, 0.001, 0.01, 0.1]
    end

    # Random search instead of grid search
    rng = Random.seed!(seed)
    params_all = collect(Iterators.product([
            choices_activation,
            choices_optimiser,
            choices_n_epochs,
            choices_n_hidden_layers,
            choices_hidden_dims,
            choices_dropout_rate,
        ]...
    ))
    params_drawn = sample(rng, params_all, n_random_searches, replace=false)
    df_stats = DataFrame(
        id=collect(1:n_random_searches),
        activation="",
        optimiser="",
        n_epochs=NaN,
        n_hidden_layers=NaN,
        hidden_dims=NaN,
        dropout_rate=NaN,
        corr_pearson=NaN,
        corr_spearman=NaN,
        corr_lin=NaN,
        R²=NaN,
        mae=NaN,
        rmse=NaN,
    )
    n = nrow(df)
    n_validation  = Int(round(validation_rate*n))
    if verbose
        pb = ProgressMeter.Progress(n_random_searches, desc="Fine-tuning hyperparameters")
    end
    for (i, params) in enumerate(params_drawn)
        # i = 12; params = params_drawn[i]
        @show i
        @show activation, optimiser, n_epochs, n_hidden_layers, hidden_dims, dropout_rate = params
        idx_validation = sort(sample(rng, 1:n, n_validation, replace=false))
        stats = trainNN(
            df,
            trait_id=trait_id,
            varex=varex,
            idx_training=nothing,
            idx_validation=idx_validation,
            activation=activation,
            optimiser=optimiser,
            n_hidden_layers=n_hidden_layers,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            n_epochs=n_epochs,
            use_cpu=use_cpu,
            seed=i,
            verbose=true,
        )
        df_stats.activation[i] = string(activation)
        df_stats.optimiser[i] = string(optimiser)
        df_stats.n_epochs[i] = n_epochs
        df_stats.n_hidden_layers[i] = Float64(n_hidden_layers)
        df_stats.hidden_dims[i] = Float64(hidden_dims)
        df_stats.dropout_rate[i] = dropout_rate
        df_stats.corr_pearson[i] = stats["stats_validation"][:corr_pearson]
        df_stats.corr_spearman[i] = stats["stats_validation"][:corr_spearman]
        df_stats.corr_lin[i] = stats["stats_validation"][:corr_lin]
        df_stats.R²[i] = stats["stats_validation"][:R²]
        df_stats.mae[i] = stats["stats_validation"][:mae]
        df_stats.rmse[i] = stats["stats_validation"][:rmse]
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Find optimum given observations alone
    z = [sum(vcat(x[1:(end-2)], -1.00 .* x[(end-1):end])) for x in eachrow(
        hcat(
            [(x .- mean(x)) ./ std(x) for x in eachcol(Matrix(df_stats[:, 8:end]))]
        )
    )]
    idx_opt = argmax(z)
    # df_stats[idx_opt, :]
    # Find the optimum via interpolation
    # using ScatteredInterpolation
    x_activation = [Float64(findall(choices_activation .== x)[1]) for x in df_stats.activation]
    x_optimiser = [Float64(findall(choices_optimiser .== x)[1]) for x in df_stats.optimiser]
    xs = hcat(
        x_activation,
        x_optimiser,
        df_stats.n_epochs,
        df_stats.n_hidden_layers,
        df_stats.hidden_dims,
        df_stats.dropout_rate,
    )'
    xs_all = hcat(
        [vcat(string(x[1]), string(x[2]), Vector(x[3:end])) for x in params_all]
    )

    interpolations = ScatteredInterpolation.interpolate(Multiquadratic(), xs, z)
    ẑ = ScatteredInterpolation.evaluate(interpolations, )

    # Output
    (df_stats, idx_opt)
end

# Testing trainNN
if false
    n_iter = 10
    n_replications = 1
    n_folds = 5
    iter = []
    rep = []
    fold = []
    dl_corr_pearson = []
    dl_corr_spearman = []
    dl_mae = []
    dl_rmse = []
    lmm_corr_pearson = []
    lmm_corr_spearman = []
    lmm_mae = []
    lmm_rmse = []
    for i in 1:n_iter
        # i = 1
        rng = Random.seed!(i)
        n_traits = 1
        genomes = simulategenomes(n=20, l=1_000, seed=i); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(rng, n_traits, 3), proportion_of_variance = rand(rng, 9, n_traits), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3, seed=i); 
        df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"];activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = 0.50; n_epochs = 1_000; use_cpu = false; seed=42;  verbose::Bool = true;
        # s_true = rand(rng, nrow(df))
        # Σ_true = (s_true * s_true') + diagm(ones(nrow(df)))
        # isposdef(Σ_true)
        # D_true = MvNormal(Float64.(df[!, trait_id]), Σ_true)
        # df[!, trait_id] = rand(D_true)
        df = filter(x -> !ismissing(x.trait_1), df)
        n = nrow(df)
        n_samples = Int(floor(n/n_folds))
        partitionings::Dict{String, Vector{Int64}} = Dict()
        for r in 1:n_replications
            # r = 1
            idx = sample(1:n, n, replace=false)
            for f in 1:n_folds
                # f = 1
                ini = (f-1)*n_samples+1
                fin = f < n_folds ? f*n_samples[1] : n
                partitionings[string("rep", r, "|fold", f)] = idx[ini:fin]
            end
        end
        for r in 1:n_replications
            for f in 1:n_folds
                # r=f=1
                # DL
                dl = begin
                    dl = trainNN(
                        df, 
                        trait_id=trait_id, 
                        idx_training=nothing,
                        idx_validation=partitionings[string("rep", r, "|fold", f)],
                        varex=varex,
                        n_epochs=10_000,
                        n_hidden_layers=5,
                        hidden_dims=256,
                        dropout_rate=0.00,
                        verbose=true,
                    )
                    display(UnicodePlots.scatterplot(dl["stats_validation"][:ϕ_true_remapped], dl["stats_validation"][:ϕ_pred_remapped]))
                    (
                        corr_pearson=dl["stats_validation"][:corr_pearson],
                        corr_spearman=dl["stats_validation"][:corr_spearman],
                        corr_lin=dl["stats_validation"][:corr_lin],
                        R²=dl["stats_validation"][:R²],
                        mae=dl["stats_validation"][:mae],
                        rmse=dl["stats_validation"][:rmse],
                    )
                end
                # LMM
                lmm = begin
                    df_tmp = deepcopy(df)
                    rename!(df_tmp, trait_id => "y")
                    idx_validation = sort(partitionings[string("rep", r, "|fold", f)])
                    idx_training = filter(x -> !(x ∈ idx_validation), 1:nrow(df_tmp))
                    # F = @formula y ~ years*seasons*sites*entries + (1|rows) + (1|cols)
                    # F = @formula y ~ seasons + sites + entries + (1|rows) + (1|cols)
                    # F = @formula y ~ seasons*sites*entries + (entries|rows) + (entries|cols)
                    F = @formula y ~ 1 + entries + (1|sites) + (1|years) + (1|seasons) + (1|sites&years) + (1|entries&sites) + (1|entries&years) + (1|sites&years&rows) + (1|sites&years&cols)
                    model = MixedModel(F, df_tmp[idx_training, :])
                    model.optsum.REML = true
                    model.optsum.maxtime = 360
                    fit!(model, progress = true)
                    y_true = Float64.(df_tmp.y[idx_validation])
                    y_pred = Float64.(predict(model, df_tmp[idx_validation, :]))
                    # Σ = model.λ * model.λ'
                    stats = goodnessoffit(;
                        ϕ_true=Float16.(y_true),
                        ϕ_pred=Float16.(y_pred),
                        y_max=Float16(1),
                        y_min=Float16(0),
                        μ=Float16.(y_pred),
                        Σ=Float16.(diagm(ones(length(y_pred)))),
                    )
                    display(UnicodePlots.scatterplot(y_true, y_pred))
                    (
                        corr_pearson=stats[:corr_pearson],
                        corr_spearman=stats[:corr_spearman],
                        corr_lin=stats[:corr_lin],
                        R²=stats[:R²],
                        mae=stats[:mae],
                        rmse=stats[:rmse],
                    )
                end
                @show dl
                @show lmm
                # Collect metrics
                push!(iter, i)
                push!(rep, r)
                push!(fold, f)
                push!(dl_corr_pearson, dl[1])
                push!(dl_corr_spearman, dl[2])
                push!(dl_mae, dl[3])
                push!(dl_rmse, dl[4])
                push!(lmm_corr_pearson, lmm[1])
                push!(lmm_corr_spearman, lmm[2])
                push!(lmm_mae, lmm[3])
                push!(lmm_rmse, lmm[4])
            end
        end
    end
    perf = DataFrame(
        iter=iter,
        rep=rep,
        fold=fold,
        dl_corr_pearson=dl_corr_pearson,
        dl_corr_spearman=dl_corr_spearman,
        dl_mae=dl_mae,
        dl_rmse=dl_rmse,
        lmm_corr_pearson=lmm_corr_pearson,
        lmm_corr_spearman=lmm_corr_spearman,
        lmm_mae=lmm_mae,
        lmm_rmse=lmm_rmse,
    )
end

# Under construction...
function analyseviaNN(
    trials::Trials,
    traits::Vector{String};
    other_covariates::Union{Vector{String},Nothing} = nothing,
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    n_hidden_layers::Int64 = 3,
    hidden_dims::Int64 = 256,
    dropout_rate::Float64 = 0.50,
    n_epochs::Int64 = 1_000,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Nothing
    # genomes = simulategenomes(n=10, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, sparsity=0.1, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); traits = ["trait_1", "trait_2"]; other_covariates=["trait_3"]; 
    # activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = Float16(0.50); n_epochs = 10_000; use_cpu = false; seed=42;  verbose::Bool = true;
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
            n_hidden_layers=n_hidden_layers,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
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
