function makex(; df::DataFrame, varex::Vector{String}, verbose::Bool=false)::Tuple{Matrix{Float64},Vector{String},Vector{String}}
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
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, sparsity=0.1, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries", "rows", "cols"]; verbose::Bool=false
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
    μ_y = mean(y)
    σ_y = std(y)
    if σ_y < 1e-12
        throw(ArgumentError("No variation in the input phenotype data."))
    end
    y = (y .- μ_y) ./ σ_y
    X, X_vars, X_labels = makex(df = df[idx, :], varex=varex, verbose=verbose)
    n, p = size(X)
    Y::Matrix{Float64} = hcat(
        y, 
        zeros(n, 1),
    )
    # Output
    (Y, μ_y, σ_y, X, X_vars, X_labels)
end

function prepmodel(;
    X::Matrix{Float64},
    Y::Matrix{Float64},
    hidden_dims::Int64,
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    n_hidden_layers::Int64 = 3,
    dropout_rate::Float64 = 0.0,
)
    input_dims = size(X, 2)
    output_dims = size(Y, 2)
    return @compact(
        input=Dense(input_dims, hidden_dims),
        hidden=[Dense(hidden_dims, hidden_dims) for i in 1:n_hidden_layers],
        output=Dense(hidden_dims, output_dims),
        activation=activation,
        dropout=Dropout(dropout_rate),
    ) do x
        layers = activation.(input(x))
        for W in hidden
            layers = activation.(W(layers))
            layers = dropout(layers)
        end
        out = output(layers)
        @return out
    end
end

function GenomicBreedingCore.lossϵΣ(model, ps, st, (x, y))
    m, n = size(y)
    p, _ = size(x)
    # Forward pass through the model to get predictions
    ŷ, st = model(x, ps, st)
    Ε = ŷ - y

    ϵ_y = view(Ε, 1, 1:n)
    ϵ_Σ = CuArray(reshape(view(Ε, 3:m, 1:n), (m-2)*n))

    # Calculate MSE loss for the trait predictions
    loss_y = (ϵ_y' * ϵ_y)/n
    loss_Σ = (ϵ_Σ' * ϵ_Σ)/((m-2)*n)
    # Calculate loss for covariance structure
    # using the Mahalanobis distance: (y-μ)ᵀΣ⁻¹(y-μ)
    loss_S = begin
        û = view(ŷ, 2, 1:p)
        Ŝ = view(ŷ, 3:m, 1:p)
        sqrt(abs(û' * inv(Ŝ) * û))/p
    end
    # Combine both losses
    loss = loss_y + loss_Σ + loss_S
    return loss, st, NamedTuple()
end

function extractpredictions(Ŷ)
    n = size(Ŷ, 2)
    ŷ = Vector(Ŷ[1, :])
    û = Vector(Ŷ[2, :])
    ū = mean(û)
    Ŝ = (1/(n-1)) * (û .- ū) * (û .- ū)'
    Ŝ = Matrix(Symmetric(Ŝ + diagm(fill(maximum(Ŝ), n))))
    (ŷ, Ŝ, û)
end

function goodnessoffit(;
    ϕ_true::Vector{Float64},
    ϕ_pred::Vector{Float64},
    σ_y::Float64,
    μ_y::Float64,
    Σ::Matrix{Float64},
)
    ϕ_pred_remapped = Float64.(ϕ_pred) * Float64(σ_y) .+ Float64(μ_y)
    ϕ_true_remapped = Float64.(ϕ_true) * Float64(σ_y) .+ Float64(μ_y)
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
            Σ += Float64.(diagm(fill(0.1, size(Σ, 1))))
            counter += 1
        end
        if !isposdef(Σ)
            (NaN, NaN)
        else
            loglik = logpdf(MvNormal(Float64.(ϕ_pred), Σ), Float64.(ϕ_true))
            mahalanobis_distance = sqrt(diff' * inv(Σ) * diff)
            (loglik, mahalanobis_distance)
        end
    end
    Dict(
        :ϕ_pred_remapped => ϕ_pred_remapped,
        :ϕ_true_remapped => ϕ_true_remapped,
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
        Optimisers.Adam(),
        Optimisers.NAdam(),
        Optimisers.OAdam(),
        Optimisers.AdaMax(),
    ][1],
    n_hidden_layers::Int64 = 3,
    hidden_dims::Int64 = 256,
    dropout_rate::Float64 = 0.01,
    n_epochs::Int64 = 10_000,
    n_patient_epochs::Int64 = 100,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    verbose::Bool = true,
)
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"];activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = 0.00; n_epochs = 10_000; use_cpu = false; seed=42;  verbose::Bool = true; idx_training = sort(sample(1:nrow(df), Int(round(0.9*nrow(df))), replace=false)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); optimiser = [Optimisers.Adam(),Optimisers.NAdam(),Optimisers.OAdam(),Optimisers.AdaMax(),][2]; n_patient_epochs=100;
    # # y_orig, μ_y_origin, σ_y_orig, X_orig, X_vars_orig, X_labels_orig = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
    # # n_orig = Int(size(X_orig, 1) / 3)
    # # b_orig = rand(Float64, size(X_orig, 2))
    # # df[!, trait_id] = X_orig[1:n_orig, :] * b_orig
    # df[!, trait_id] = rand(nrow(df))
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
    Y, μ_y, σ_y, X, X_vars, X_labels = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose=verbose)
    Y_training = Y[idx_training, :]
    Y_validation = Y[idx_validation, :]
    X_training = X[idx_training, :]
    X_validation = X[idx_validation, :]
    Σ_training = begin
        Σ = cov(X_training .* Y_training[:, 1])
        E = eigen(Σ)
        E.values[E.values .< 1e-12] .= 1e-12
        Matrix(Symmetric(E.vectors * diagm(E.values) * E.vectors'))
    end
    Φ_training = hcat(Y_training, vcat(Σ_training, zeros(size(Y_training, 1)-size(Σ_training, 1), size(Σ_training, 2))))
    model = prepmodel(
        X=X_training,
        Y=Φ_training,
        hidden_dims=hidden_dims,
        activation=activation,
        n_hidden_layers=n_hidden_layers,
        dropout_rate=dropout_rate,
    )
    if verbose
        display(model)
    end
    dev = if use_cpu
        cpu_device()
    else
        gpu_device()
    end
    x_training = dev(Matrix(X_training'))
    x_validation = dev(Matrix(X_validation'))
    y_training = dev(Matrix(Φ_training'))
    y_validation = dev(Matrix(Y_validation'))
    rng = Random.RandomDevice()
    Random.seed!(seed)
    ps, st = Lux.setup(rng, model) |> dev # ps => parameters => weights and biases; st => state variable
    ## First construct a TrainState
    training_state = Lux.Training.TrainState(model, ps, st, optimiser)
    ### Train
    if verbose
        pb = ProgressMeter.Progress(n_epochs, desc="Training for a maximum of $n_epochs epochs: ")
    end
    time = []
    training_loss = []
    training_rmse = []
    validation_rmse = []
    min_validation_rmse = sum(abs.(y_training[1, :]))
    for iter = 1:n_epochs
        # Compute the gradients
        # gradients, loss, stats, training_state = Lux.Training.compute_gradients(AutoZygote(), lossϵΣ, (x_training, y_training), training_state)
        # # Optimise
        # training_state = Training.apply_gradients!(training_state, gradients)
        # # Alternatively, compute gradients and optimise with a single call
        _, loss, _, training_state = Lux.Training.single_train_step!(AutoZygote(), lossϵΣ, (x_training, y_training), training_state)
        # Check cross-validation performance
        validation_state = Lux.testmode(training_state.states)
        y_training_pred, _ = Lux.apply(model, x_training, ps, validation_state);
        y_validation_pred, _ = Lux.apply(model, x_validation, ps, validation_state);
        ϵt = y_training_pred[1, :] - y_training[1, :]
        ϵv = y_validation_pred[1, :] - y_validation[1, :]
        if verbose
            ProgressMeter.next!(pb)
        end
        push!(time, iter)
        push!(training_loss, loss)
        push!(training_rmse, sqrt(mean(ϵt.^2)))
        push!(validation_rmse, sqrt(mean(ϵv.^2)))
        # Early stopping logic
        min_validation_rmse = if validation_rmse[end] < min_validation_rmse
            validation_rmse[end]
        else
            if length(validation_rmse) > n_patient_epochs
                println("Stopping early because validation RMSE is increasing!")
                break
            end
            if sum(isnan.(ϵt)) > 0
                println("Stopping early because of NaNs!")
                break
            end
            min_validation_rmse
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Memory clean-up
    CUDA.reclaim()
    if verbose
        CUDA.pool_status()
    end
    # Fit stats
    validation_state = Lux.testmode(training_state.states)
    Ŷ, _ = Lux.apply(model, x_training, ps, validation_state)
    ŷ, Ŝ, û = extractpredictions(Ŷ)
    stats = goodnessoffit(
        ϕ_true=Y_training[:, 1],
        ϕ_pred=Float64.(ŷ),
        σ_y=σ_y,
        μ_y=μ_y,
        Σ=Ŝ,
    )
    # Cross-validation
    stats_validation = if length(idx_validation) > 0
        Ŷ_validation, _ = Lux.apply(model, x_validation, ps, validation_state)
        ŷ_validation, Ŝ_validation = extractpredictions(Ŷ_validation)
        goodnessoffit(
            ϕ_true=Y_validation[:, 1],
            ϕ_pred=Float64.(ŷ_validation),
            σ_y=σ_y,
            μ_y=μ_y,
            Σ=Ŝ_validation,
        )
    else
        nothing
    end
    if verbose
        # Plot the training loss
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("TRAINING LOSS:")
        display(UnicodePlots.scatterplot(time, training_loss, xlabel = "Iteration", ylabel = "Training Loss", title = "Training Loss"))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("TRAINING AND VALIDATION RMSE:")
        display(UnicodePlots.lineplot(time, validation_rmse, xlabel="Iteration", ylabel="Training RMSE"))
        display(UnicodePlots.lineplot(time, training_rmse, xlabel="Iteration", ylabel="Validation RMSE"))
        all_rmse = filter(x -> !isnan(x) && !isinf(x), vcat(training_rmse, validation_rmse))
        ylim = (minimum(all_rmse), maximum(all_rmse))
        plt = UnicodePlots.lineplot(time, training_rmse, ylim=ylim, color=:red, name="Training RMSE", xlabel="Iteration", ylabel="RMSE")
        UnicodePlots.lineplot!(plt, time, validation_rmse, color=:green, name="Validation RMSE")
        display(plt)
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED VALUES:")
        display(UnicodePlots.scatterplot(stats[:ϕ_true_remapped], stats[:ϕ_pred_remapped], xlabel = "Observed", ylabel = "Fitted", title = "Fitted vs Observed\n(n=$(length(stats[:ϕ_pred_remapped])))"))
        println("Pearson's product-moment correlation: ", round(100*stats[:corr_pearson], digits=2), "%")
        println("Spearman's rank correlation: ", round(100*stats[:corr_spearman], digits=2), "%")
        println("MAE: ", round(stats[:mae], digits=4))
        println("RMSE: ", round(stats[:rmse], digits=4))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED MULTIVARIATE NORMAL DISTRIBUTION:")
        display(UnicodePlots.heatmap(Ŝ, title="Fitted variance-covariance matrix (Σ)"))
        println("Goodness of fit in log-likelihood: ", round(stats[:loglik], digits=4))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("CROSS-VALIDATION:")
        if isnothing(stats_validation)
            println("None")
        else
            display(UnicodePlots.scatterplot(stats_validation[:ϕ_true_remapped], stats_validation[:ϕ_pred_remapped], xlabel = "Observed", ylabel = "Predicted", title = "Cross-validation\n(n=$(length(stats_validation[:ϕ_pred_remapped])))"))
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
    p = size(X, 2)
    for idx_1 in idx_varex
        # idx_1 = idx_varex[1]
        # How many rows in the new X matrix do we need?
        m = 1
        for v in varex[idx_1]
            # v = varex[idx_1][1]
            m *= sum(X_vars .== v)
        end
        X_new = Float64.(zeros(m, p))
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
        Ŷ_marginals, _ = Lux.apply(model, x_new, ps, validation_state)
        ŷ_marginals, Ŝ_marginals = extractpredictions(Ŷ_marginals)
        z = ŷ_marginals ./ sqrt.(diag(Ŝ_marginals))
        p_vals = 2 * (1 .- cdf(Normal(0.0, 1.0), abs.(z)))
        marginals[join(varex[idx_1], "|")] = Dict(
            "labels" => X_labels_new,
            "ϕ_marginals" => ŷ_marginals,
            "Σ_marginals" => Ŝ_marginals,
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
        "training_progress" => DataFrame(epoch=time, training_loss=training_loss, training_rmse=training_rmse, validation_rmse=validation_rmse),
        "values" => ŷ * σ_y .+ μ_y,
        "covariances" => Ŝ,
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
    dropout_rate::Union{Nothing, Float64} = 0.0, # using drop-out rate > 0.0 results to inefficiencies
    optimiser = nothing,
    n_epochs::Union{Nothing, Int64} = nothing,
    n_hidden_layers::Union{Nothing, Int64} = nothing,
    hidden_dims::Union{Nothing, Int64} = nothing,
    use_cpu::Bool = false,
    n_random_searches::Int64 = 100,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Tuple{DataFrame, Int64}
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=2); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"]; validation_rate=0.25; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; dropout_rate = 0.0; optimiser = nothing; use_cpu = false; seed=42;  verbose::Bool = true; n_hidden_layers = nothing; hidden_dims = nothing; n_epochs = nothing; seed = 42; n_random_searches=10
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
            Optimisers.OAdam(),
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
        [1, 2, 3, 4, 5]
    end
    choices_hidden_dims = if !isnothing(hidden_dims)
        [hidden_dims]
    else
        n = nrow(df)
        [1*n, 2*n, 3*n, 4*n]
    end
    choices_dropout_rate = if !isnothing(dropout_rate)
        [dropout_rate]
    else
        [0.0, 0.001, 0.01, 0.1, 0.5]
    end

    # Random search instead of grid search
    rng = Random.RandomDevice()
    Random.seed!(seed)
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
    if verbose
        pb = ProgressMeter.Progress(n_random_searches, desc="Fine-tuning hyperparameters using $(length(params_drawn))/$(length(params_all)) randomly chosen paramater combinations: ")
    end
    n = nrow(df)
    n_validation  = Int(round(validation_rate*n))
    @time for (i, params) in enumerate(params_drawn)
        # i = 19; params = params_drawn[i]
        @show i
        @show activation, optimiser, n_epochs, n_hidden_layers, hidden_dims, dropout_rate = params
        idx_validation = sort(sample(Random.seed!(i), 1:n, n_validation, replace=false))
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
    df_stats = filter(x -> !isnan(x.R²), df_stats)
    df_stats.z = [sum(vcat(x[1:(end-2)], -1.00 .* x[(end-1):end])) for x in eachrow(
        hcat(
            [(x .- mean(x)) ./ std(x) for x in eachcol(Matrix(df_stats[:, 8:end]))]
        )
    )]
    idx_opt = argmax(df_stats.z)
    # df_stats[idx_opt, :]
    # Find the optimum via interpolation
    xs_all = nothing
    for params in params_all
        # params = params_all[1]
        x = vcat(
            [
                Float64(findall(string.(choices_activation) .== string(params[1]))[1]),
                Float64(findall(string.(choices_optimiser) .== string(params[2]))[1]),
            ],
            vcat(params[3:end]...)
        )
        xs_all = if isnothing(xs_all)
            reshape(x, length(x), 1)
        else
            hcat(xs_all, x)
        end
    end
    x_activation = [Float64(findall(string.(choices_activation) .== x)[1]) for x in df_stats.activation]
    x_optimiser = [Float64(findall(string.(choices_optimiser) .== x)[1]) for x in df_stats.optimiser]
    xs = hcat(
        x_activation,
        x_optimiser,
        df_stats.n_epochs,
        df_stats.n_hidden_layers,
        df_stats.hidden_dims,
        df_stats.dropout_rate,
    )'
    z = df_stats.z
    interpolations = ScatteredInterpolation.interpolate(Multiquadratic(), xs, z)
    ẑ = ScatteredInterpolation.evaluate(interpolations, xs_all)
    p_tmp = xs[:, argmax(z)]
    stats_tmp = trainNN(
        df,
        trait_id=trait_id,
        varex=varex,
        activation=choices_activation[Int(p_tmp[1])],
        optimiser=choices_optimiser[Int(p_tmp[2])],
        n_epochs=Int64(p_tmp[3]),
        n_hidden_layers=Int64(p_tmp[4]),
        hidden_dims=Int64(p_tmp[5]),
        dropout_rate=p_tmp[6],
        use_cpu=use_cpu,
        seed=1,
        verbose=true,
    )

    p_opt = xs_all[:, argmax(ẑ)]
    stats_opt = trainNN(
        df,
        trait_id=trait_id,
        varex=varex,
        activation=choices_activation[Int(p_opt[1])],
        optimiser=choices_optimiser[Int(p_opt[2])],
        n_epochs=Int64(p_opt[3]),
        n_hidden_layers=Int64(p_opt[4]),
        hidden_dims=Int64(p_opt[5]),
        dropout_rate=p_opt[6],
        use_cpu=use_cpu,
        seed=1,
        verbose=true,
    )

    stats_tmp["stats_validation"][:R²]
    stats_opt["stats_validation"][:R²]


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
                        optimiser=Optimisers.AdaMax(),
                        n_epochs=10_000,
                        n_hidden_layers=2,
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
                mlm = begin
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
                        ϕ_true=Float64.(y_true),
                        ϕ_pred=Float64.(y_pred),
                        σ_y=Float64(1),
                        μ_y=Float64(0),
                        Σ=Float64.(diagm(ones(length(y_pred)))),
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
                @show mlm
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
    # activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = Float64(0.50); n_epochs = 10_000; use_cpu = false; seed=42;  verbose::Bool = true;
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
        # y_hat::Vector{Float64} = ϕ_hat[1, 1:n]
        # display(UnicodePlots.scatterplot(y_valid, y_hat))
        # cor(y_hat, y_valid) |> println
    # end
    
    nothing


end
