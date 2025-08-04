function makex(; df::DataFrame, varex::Vector{String}, verbose::Bool = false)::Tuple{Matrix{Float64},Vector{String},Vector{String}}
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); varex = ["years", "seasons", "sites", "entries"]; verbose::Bool=false
    if sum([!(v ∈ names(df)) for v in varex]) > 0
        throw(
            ArgumentError(
                "The explanatory variable/s: `$(join(varex[[!(v ∈ names(df)) for v in varex]], "`, `"))` do not exist in the DataFrame.",
            ),
        )
    end
    X = nothing
    feature_groups = []
    feature_names = []
    n = nrow(df)
    if verbose
        pb = ProgressMeter.Progress(length(varex), desc = "Preparing inputs")
    end
    @inbounds for v in varex
        # v = varex[end]
        A, x_vars, x_labels = try
            x = Vector{Float64}(df[!, v])
            if sum(ismissing.(x) .|| isnan.(x) .|| isinf.(x)) > 0
                throw(
                    ArgumentError(
                        "We expect the continuous numeric covariate ($v) to have no missing/NaN/Inf values relative to the response variable. Please remove these unsuitable values jointly across the response variable and covariates and/or remove the offending covariate ($v).",
                    ),
                )
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
        feature_groups = vcat(feature_groups, x_vars)
        feature_names = vcat(feature_names, x_labels)
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    (X, feature_groups, feature_names)
end

function prepinputs(; df::DataFrame, varex::Vector{String}, trait_id::String, verbose::Bool = false)
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, sparsity=0.1, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries", "rows", "cols"]; verbose::Bool=false
    # Remove rows with missing, NaN or Inf values in the trait_id and varex
    if sum([!(v ∈ names(df)) for v in varex]) > 0
        throw(
            ArgumentError(
                "The explanatory variable/s: `$(join(varex[[!(v ∈ names(df)) for v in varex]], "`, `"))` do not exist in the DataFrame.",
            ),
        )
    end
    idx = []
    if verbose
        pb = ProgressMeter.Progress(nrow(df), desc = "Filtering rows with missing/NaN/Inf values")
    end
    @inbounds for i = 1:nrow(df)
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
    row_names = [join(x, "|") for x in eachrow(df[idx, [x ∈ varex for x in names(df)]])]
    y::Vector{Float64} = df[idx, trait_id]
    μ_y = mean(y)
    σ_y = std(y)
    if σ_y < 1e-12
        throw(ArgumentError("No variation in the input phenotype data."))
    end
    y = (y .- μ_y) ./ σ_y
    X, feature_groups, feature_names = makex(df = df[idx, :], varex = varex, verbose = verbose)
    Y = hcat(y)
    # Output
    (Y, μ_y, σ_y, X, feature_groups, feature_names, row_names)
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
        input = Dense(input_dims, hidden_dims),
        hidden = [Dense(hidden_dims, hidden_dims) for i = 1:n_hidden_layers],
        output = Dense(hidden_dims, output_dims),
        activation = activation,
        dropout = Dropout(dropout_rate),
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

function goodnessoffit(; ϕ_true::Vector{Float64}, ϕ_pred::Vector{Float64}, σ_y::Float64, μ_y::Float64)
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
        (2 * ρ * σ_true * σ_pred) / (σ_true^2 + σ_pred^2 + (μ_true - μ_pred)^2)
    end
    diff = ϕ_pred_remapped - ϕ_true_remapped
    mae = mean(abs.(diff))
    rmse = sqrt(mean(diff .^ 2))
    R² = 1 - (sum((diff .^ 2)) / sum((ϕ_true_remapped .- mean(ϕ_true_remapped)) .^ 2))
    Dict(
        :ϕ_pred_remapped => ϕ_pred_remapped,
        :ϕ_true_remapped => ϕ_true_remapped,
        :corr_pearson => corr_pearson,
        :corr_spearman => corr_spearman,
        :corr_lin => corr_lin,
        :mae => mae,
        :rmse => rmse,
        :R² => R²,
    )
end

function checkinputs(;
    df::DataFrame,
    trait_id::String,
    varex::Vector{String},
    idx_training::Vector{Int64},
    idx_validation::Vector{Int64},
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    optimiser = [Optimisers.Adam(), Optimisers.NAdam(), Optimisers.OAdam(), Optimisers.AdaMax()][1],
    n_hidden_layers::Int64 = 3,
    hidden_dims::Int64 = 256,
    dropout_rate::Float64 = 0.01,
    n_epochs::Int64 = 10_000,
    n_patient_epochs::Int64 = 100,
)
    # Checks
    errors::Vector{String} = []
    ϕ = df[!, trait_id]
    if sum(ismissing.(ϕ) .|| isnan.(ϕ) .|| isinf.(ϕ)) > 0
        push!(
            error,
            "Missing data in trait: $trait_id is not permitted. Please filter-out missing data first as these may potentially conflict with the supplied training and/or validation set indexes.",
        )
    end
    if !("entries" ∈ names(df))
        push!(
            error,
            "The expected `entries` column is absent in the input data frame. We expect a tabularised Trials struct.",
        )
    end
    for v in varex
        if !(v ∈ names(df))
            push!(errors, "The explantory variable, $v is absent in the data frame `df`.")
        end
    end
    if !isnothing(idx_training)
        if minimum(idx_training) < 1
            push!(errors, "Training set index starts below 1.")
        end
        if maximum(idx_training) > nrow(df)
            push!(errors, "Training set index is greater than the number of observations, i.e. above $(nrow(df)).")
        end
    end
    if length(idx_validation) > 0
        if minimum(idx_validation) < 1
            push!(errors, "Validation set index starts below 1.")
        end
        if maximum(idx_validation) > nrow(df)
            push!(errors, "Validation set index is greater than the number of observations, i.e. above $(nrow(df)).")
        end
    end
    if length(idx_training) < 2
        push!(errors, "There is less than 2 observations for training!")
    end
    if length(filter(x -> x ∈ idx_training, idx_validation)) > 0
        push!(errors, "There is data leakage!")
    end
    if !(string(activation) ∈ string.([sigmoid, sigmoid_fast, relu, tanh]))
        push!(errors, "Activation function: `$(string(activation))` is invalid.")
    end
    if !(string(optimiser) ∈ string.([Optimisers.Adam(), Optimisers.NAdam(), Optimisers.OAdam(), Optimisers.AdaMax()]))
        push!(errors, "Optimiser: `$(string(optimiser))` is invalid.")
    end
    if n_hidden_layers < 1
        push!(errors, "The number of hidden layers (`n_hidden_layers`) should be at least 1.")
    end
    if prod(hidden_dims) < 1
        push!(errors, "The dimensions of the hidden layers (`hidden_dims`) should be non-zero.")
    end
    if (dropout_rate < 0.0) || (dropout_rate > 1.0)
        push!(errors, "The drop-out rate (`dropout_rate`) should range from 0.0 to 1.0.")
    end
    if n_epochs < 1
        push!(errors, "The number of training epochs (`n_epochs`) should be non-zero.")
    end
    if n_patient_epochs < 1
        push!(errors, "The number of training epochs prior to assessing over-fitting (`n_patient_epochs`) should be non-zero.")
    end
    if length(errors) > 0
        throw(ArgumentError(string("\n\t‣ ", join(errors, "\n\t‣ "))))
    else
        nothing
    end
end

function trainNN(
    df::DataFrame;
    trait_id::String,
    varex::Vector{String},
    idx_training::Vector{Int64},
    idx_validation::Vector{Int64},
    activation = [sigmoid, sigmoid_fast, relu, tanh][3],
    optimiser = [Optimisers.Adam(), Optimisers.NAdam(), Optimisers.OAdam(), Optimisers.AdaMax()][1],
    n_hidden_layers::Int64 = 3,
    hidden_dims::Int64 = 256,
    dropout_rate::Float64 = 0.01,
    n_epochs::Int64 = 10_000,
    n_patient_epochs::Int64 = 100,
    use_cpu::Bool = false,
    seed::Int64 = 42,
    save_model::Bool = false,
    verbose::Bool = true,
)::DLModel
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"]; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = 0.00; n_epochs = 10_000; use_cpu = false; seed=42;  verbose::Bool = true; idx_training = sort(sample(1:nrow(df), Int(round(0.9*nrow(df))), replace=false)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); optimiser = [Optimisers.Adam(),Optimisers.NAdam(),Optimisers.OAdam(),Optimisers.AdaMax(),][2]; n_patient_epochs=1_000; save_model::Bool = false;
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"]; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; n_hidden_layers = 3; hidden_dims = 256; dropout_rate = 0.00; n_epochs = 10_000; use_cpu = false; seed=42;  verbose::Bool = true; idx_training = collect(1:nrow(df)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); optimiser = [Optimisers.Adam(),Optimisers.NAdam(),Optimisers.OAdam(),Optimisers.AdaMax(),][2]; n_patient_epochs=1_000; save_model::Bool = false;
    # Checks
    checkinputs(
        df=df,
        trait_id=trait_id,
        varex=varex,
        idx_training=idx_training,
        idx_validation=idx_validation,
        activation=activation,
        optimiser=optimiser,
        n_hidden_layers=n_hidden_layers,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        n_epochs=n_epochs,
        n_patient_epochs=n_patient_epochs,
    )
    # Prepare model inputs
    Y, μ_y, σ_y, X, feature_groups, feature_names, row_names = prepinputs(df = df, varex = varex, trait_id = trait_id, verbose = verbose)
    Y_training = Y[idx_training, :]
    Y_validation = Y[idx_validation, :]
    X_training = X[idx_training, :]
    X_validation = X[idx_validation, :]
    # Instantiate the model
    model = prepmodel(
        X = X_training,
        Y = Y_training,
        hidden_dims = hidden_dims,
        activation = activation,
        n_hidden_layers = n_hidden_layers,
        dropout_rate = dropout_rate,
    )
    if verbose
        display(model)
    end
    dev = if use_cpu
        cpu_device()
    else
        gpu_device()
    end
    x_training = dev(X_training')
    x_validation = dev(X_validation')
    y_training = dev(Y_training')
    y_validation = dev(Y_validation')
    rng = Random.RandomDevice()
    Random.seed!(seed)
    ps, st = Lux.setup(rng, model) |> dev # ps => parameters => weights and biases; st => state variable
    ## First construct a TrainState
    training_state = Lux.Training.TrainState(model, ps, st, optimiser)
    ### Train
    if verbose
        pb = ProgressMeter.Progress(n_epochs, desc = "Training for a maximum of $n_epochs epochs: ")
    end
    time = []
    training_loss = []
    training_rmse = []
    validation_rmse = []
    min_training_rmse = Inf64
    min_validation_rmse = Inf64
    for iter = 1:n_epochs
        # Compute the gradients
        # gradients, loss, stats, training_state = Lux.Training.compute_gradients(AutoZygote(), lossϵΣ, (x_training, y_training), training_state)
        # # Optimise
        # training_state = Training.apply_gradients!(training_state, gradients)
        # # Alternatively, compute gradients and optimise with a single call
        _, loss, _, training_state = Lux.Training.single_train_step!(
            AutoZygote(), 
            MSELoss(), 
            (x_training, y_training), 
            training_state, 
        )
        # Check cross-validation performance
        validation_state = Lux.testmode(training_state.states)
        y_training_pred, _ = Lux.apply(model, x_training, ps, validation_state)
        ϵt = y_training_pred[1, :] - y_training[1, :]
        ϵv = if length(idx_validation) > 0
            y_validation_pred, _ = Lux.apply(model, x_validation, ps, validation_state)
            y_validation_pred[1, :] - y_validation[1, :]
        else
            [NaN]
        end
        push!(time, iter)
        push!(training_loss, loss)
        push!(training_rmse, sqrt(mean(ϵt .^ 2)))
        push!(validation_rmse, sqrt(mean(ϵv .^ 2)))
        if verbose
            ProgressMeter.next!(pb)
        end
        # Early stopping logic
        if length(validation_rmse) > n_patient_epochs
            if training_rmse[end] > min_training_rmse
                verbose ? println("Stopping early because training RMSE is increasing!") : nothing
                break
            end
            if (length(idx_validation) > 0) && (validation_rmse[end] > min_validation_rmse)
                verbose ? println("Stopping early because validation RMSE is increasing!") : nothing
                break
            end
            if sum(isnan.(ϵt)) > 0
                verbose ? println("Stopping early because of NaNs!") : nothing
                break
            end
        end
        min_training_rmse = training_rmse[end] < min_training_rmse ? training_rmse[end] : min_training_rmse
        min_validation_rmse = validation_rmse[end] < min_validation_rmse ? validation_rmse[end] : min_validation_rmse
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Memory clean-up
    CUDA.reclaim()
    if verbose
        CUDA.pool_status()
    end
    # Training progress
    training_progress = DataFrame(
        epoch=time, 
        training_loss=training_loss, 
        training_rmse=training_rmse, 
        validation_rmse=validation_rmse
    )
    # Fit stats
    validation_state = Lux.testmode(training_state.states)
    Ŷ, _ = Lux.apply(model, x_training, ps, validation_state)
    stats = goodnessoffit(ϕ_true = Y_training[:, 1], ϕ_pred = Vector{Float64}(Ŷ[1, :]), σ_y = σ_y, μ_y = μ_y)
    # Cross-validation
    Ŷ_validation, stats_validation = if length(idx_validation) > 0
        Ŷ_validation, _ = Lux.apply(model, x_validation, ps, validation_state)
        (
            Ŷ_validation,
            goodnessoffit(ϕ_true = Y_validation[:, 1], ϕ_pred = Vector{Float64}(Ŷ_validation[1, :]), σ_y = σ_y, μ_y = μ_y)
        )
    else
        (nothing, Dict(:corr_pearson => nothing))
    end
    # Messages/info
    if verbose
        # Plot the training loss
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("TRAINING LOSS:")
        display(
            UnicodePlots.scatterplot(
                training_progress.epoch,
                training_progress.training_loss,
                xlabel = "Iteration",
                ylabel = "Training Loss",
                title = "Training Loss",
            ),
        )
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("TRAINING RMSE:")
        display(UnicodePlots.lineplot(training_progress.epoch, training_progress.training_rmse, xlabel = "Iteration", ylabel = "Validation RMSE"))
        if length(idx_validation) > 0
            println("VALIDATION RMSE:")
            display(UnicodePlots.lineplot(training_progress.epoch, training_progress.validation_rmse, xlabel = "Iteration", ylabel = "Training RMSE"))
            all_rmse = filter(x -> !isnan(x) && !isinf(x), vcat(training_rmse, validation_rmse))
            ylim = (minimum(all_rmse), maximum(all_rmse))
            plt = UnicodePlots.lineplot(
                training_progress.epoch,
                training_progress.training_rmse,
                ylim = ylim,
                color = :red,
                name = "Training RMSE",
                xlabel = "Iteration",
                ylabel = "RMSE",
            )
            UnicodePlots.lineplot!(plt, training_progress.epoch, training_progress.validation_rmse, color = :green, name = "Validation RMSE")
            display(plt)
        end
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("FITTED VALUES:")
        display(
            UnicodePlots.scatterplot(
                stats[:ϕ_true_remapped],
                stats[:ϕ_pred_remapped],
                xlabel = "Observed",
                ylabel = "Fitted",
                title = "Fitted vs Observed\n(n=$(length(stats[:ϕ_pred_remapped])))",
            ),
        )
        println("Pearson's product-moment correlation: ", round(100 * stats[:corr_pearson], digits = 2), "%")
        println("Spearman's rank correlation: ", round(100 * stats[:corr_spearman], digits = 2), "%")
        println("MAE: ", round(stats[:mae], digits = 4))
        println("RMSE: ", round(stats[:rmse], digits = 4))
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("CROSS-VALIDATION:")
        if isnothing(stats_validation[:corr_pearson])
            println("None")
        else
            display(
                UnicodePlots.scatterplot(
                    stats_validation[:ϕ_true_remapped],
                    stats_validation[:ϕ_pred_remapped],
                    xlabel = "Observed",
                    ylabel = "Predicted",
                    title = "Cross-validation\n(n=$(length(stats_validation[:ϕ_pred_remapped])))",
                ),
            )
            println(
                "Pearson's product-moment correlation: ",
                round(100 * stats_validation[:corr_pearson], digits = 2),
                "%",
            )
            println("Spearman's rank correlation: ", round(100 * stats_validation[:corr_spearman], digits = 2), "%")
            println("MAE: ", round(stats_validation[:mae], digits = 4))
            println("RMSE: ", round(stats_validation[:rmse], digits = 4))
        end
    end
    # Model I/O
    if save_model
        fname_model = string("model-", trait_id, "-", hash(rand()), ".jld2")
        @save fname_model model=model training_state=training_state
        # @load fname_model model training_state
        if verbose
            println("Please find the model: '$(joinpath(pwd(), fname_model))'")
        end
    end
    # Output
    training_observed, training_predicted, training_labels = begin
        (
            Y_training[:, 1] .* σ_y .+ μ_y, 
            Vector{Float64}(Ŷ[1, :]) .* σ_y .+ μ_y, 
            row_names[idx_training]
        )
    end
    validation_observed, validation_predicted, validation_labels = if length(idx_validation) == 0
        ([], [], [])
    else
        (
            Y_validation[:, 1] .* σ_y .+ μ_y,
            Vector{Float64}(Ŷ_validation[1, :]) .* σ_y .+ μ_y,
            row_names[idx_validation],
        )
    end
    DLModel(
        model,
        dev,
        μ_y,
        σ_y,
        row_names,
        feature_groups,
        feature_names,
        training_state,
        training_progress,
        training_observed,
        training_predicted,
        training_labels,
        validation_observed,
        validation_predicted,
        validation_labels,
        stats,
        stats_validation,
    )
end

function makexnew(model::DLModel, gxe_vars::Vector{String})
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; idx_training = sort(sample(1:nrow(df), Int(round(0.9*nrow(df))), replace=false)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); varex = ["years", "seasons", "sites", "entries"]; model = trainNN(df, trait_id=trait_id, varex=varex, idx_training=idx_training, idx_validation=idx_validation)
    # gxe_vars = filter(x -> x ∈ unique(model.feature_groups), ["years", "seasons", "entries"])
    indexes = Dict()
    for v in gxe_vars
        idx = findall(model.feature_groups .== v)
        indexes[v] = idx
    end
    _, p = size(model.training_state.parameters.input.weight)
    M = prod([length(x) for (_, x) in indexes])
    m = M
    X_new = zeros(m, p)
    for (v, idx) in indexes
        # v = string.(keys(indexes))[2]; idx = indexes[v]
        rep_inner = Int(m/length(idx))
        rep_outer = Int(M/(rep_inner*length(idx)))
        m = rep_inner
        # println("rep_inner=$rep_inner; rep_outer=$rep_outer")
        idx_cols = repeat(repeat(idx, inner=rep_inner), outer=rep_outer)
        for (i, j) in enumerate(idx_cols)
            X_new[i, j] = 1.0
        end
    end
    X_new
end

function extracteffects(model::DLModel)
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; idx_training = sort(sample(1:nrow(df), Int(round(0.9*nrow(df))), replace=false)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); varex = ["years", "seasons", "sites", "entries"]; model = trainNN(df, trait_id=trait_id, varex=varex, idx_training=idx_training, idx_validation=idx_validation)
    gxe_vars = begin
        avail_vars = [[x] for x in filter(x -> x ∈ unique(model.feature_groups), ["years", "seasons", "harvests", "sites", "entries"])]
        gxe_vars = []
        for i in 1:length(avail_vars)
            # i = 2
            v = if i == 1
                avail_vars
            else
                filter(x -> "entries" ∈ x, [vcat(x...) for x in combinations(avail_vars, i)])
            end
            append!(gxe_vars, v)
        end
        gxe_vars
    end
    validation_state = Lux.testmode(model.training_state.states)
    Φ = Dict()
    for v in gxe_vars
        # v = gxe_vars[5]
        X_new = makexnew(model, v)
        labels = begin
            labels = []
            for x in eachrow(X_new)
                # x = X_new[1, :]
                push!(labels, join(model.feature_names[x .== 1.0], "|"))
            end
            labels
        end
        x_new = model.dev(X_new')
        Ŷ_new, _ = Lux.apply(model.model, x_new, model.training_state.parameters, validation_state)
        ẑ = Vector{Float64}(Ŷ_new[1, :])
        ŷ = (ẑ .* model.σ_y) .+ model.μ_y
        Φ[join(v, "|")] = (ŷ=ŷ, ẑ=ẑ, labels=labels)
    end
    Φ
end

function extractcovariances(model::DLModel)
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=3); df = tabularise(trials); trait_id = "trait_1"; idx_training = sort(sample(1:nrow(df), Int(round(0.9*nrow(df))), replace=false)); idx_validation = filter(x -> !(x ∈ idx_training), 1:nrow(df)); varex = ["years", "seasons", "sites", "entries"]; model = trainNN(df, trait_id=trait_id, varex=varex, idx_training=idx_training, idx_validation=idx_validation)
    gxe_vars = filter(x -> x ∈ unique(model.feature_groups), ["years", "seasons", "harvests", "sites", "entries"])
    indexes = Dict()
    for v in gxe_vars
        idx = findall(model.feature_groups .== v)
        indexes[v] = idx
    end
    _, p = size(model.training_state.parameters.input.weight)
    M = prod([length(x) for (_, x) in indexes])
    m = M
    X_new = makexnew(model, gxe_vars)
    # X_new = zeros(m, p)
    # for (v, idx) in indexes
    #     # v = string.(keys(indexes))[2]; idx = indexes[v]
    #     rep_inner = Int(m/length(idx))
    #     rep_outer = Int(M/(rep_inner*length(idx)))
    #     m = rep_inner
    #     # println("rep_inner=$rep_inner; rep_outer=$rep_outer")
    #     idx_cols = repeat(repeat(idx, inner=rep_inner), outer=rep_outer)
    #     for (i, j) in enumerate(idx_cols)
    #         X_new[i, j] = 1.0
    #     end
    # end
    validation_state = Lux.testmode(model.training_state.states)
    x_new = model.dev(X_new')
    Ŷ_new, _ = Lux.apply(model.model, x_new, model.training_state.parameters, validation_state)
    ẑ = Vector{Float64}(Ŷ_new[1, :])
    ŷ = (ẑ .* model.σ_y) .+ model.μ_y
    Â = X_new .* ŷ
    Σ = Dict()
    for v in gxe_vars
        # v = gxe_vars[2]
        idx = findall(model.feature_groups .== v)
        B̂ = hcat([filter(x -> abs(x) > 1e-12, x) for x in eachcol(Â[:, idx])])
        Σ[v] = (Σ=cov(B̂), labels=model.feature_names[idx])
    end
    Σ
end

function optimNN(
    df::DataFrame;
    trait_id::String,
    varex::Vector{String},
    validation_rate::Float64 = 0.25,
    activation::Any = [sigmoid, sigmoid_fast, relu, tanh][3],
    dropout_rate::Union{Nothing,Float64} = 0.0, # using drop-out rate > 0.0 results to inefficiencies
    n_epochs::Union{Nothing,Int64} = 10_000, # using a fixed n_epochs = 10_000 because we are employing early stopping with separate training and validation sets
    n_patient_epochs::Union{Nothing,Int64} = nothing,
    optimiser = nothing,
    n_hidden_layers::Union{Nothing,Int64} = nothing,
    hidden_dims::Union{Nothing,Int64} = nothing,
    use_cpu::Bool = false,
    n_random_searches::Int64 = 100,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Tuple{DataFrame,Int64}
    # genomes = simulategenomes(n=20, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, f_add_dom_epi = rand(10,3), n_years=3, n_seasons=4, n_harvests=1, n_sites=3, n_replications=2); df = tabularise(trials); trait_id = "trait_1"; varex = ["years", "seasons", "sites", "entries"]; validation_rate=0.25; activation = [sigmoid, sigmoid_fast, relu, tanh][3]; dropout_rate = 0.0; optimiser = nothing; use_cpu = false; seed=42;  verbose::Bool = true; n_hidden_layers = nothing; hidden_dims = nothing; n_patient_epochs = nothing; n_epochs = 10_000; seed = 42; n_random_searches=10
    choices_activation = if !isnothing(activation)
        [activation]
    else
        [sigmoid, sigmoid_fast, relu, tanh]
    end
    choices_optimiser = if !isnothing(optimiser)
        [optimiser]
    else
        [Optimisers.Adam(), Optimisers.NAdam(), Optimisers.OAdam(), Optimisers.AdaMax()]
    end
    choices_n_epochs = if !isnothing(n_epochs)
        n_epochs
    else
        [1_000, 5_000, 7_000, 10_000]
    end
    choices_n_patient_epochs = if !isnothing(n_patient_epochs)
        n_patient_epochs
    else
        Int64.([0.10, 0.25, 0.50, 0.75, 0.90] .* minimum(choices_n_epochs))
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
        [1 * n, 2 * n, 3 * n, 4 * n]
    end
    choices_dropout_rate = if !isnothing(dropout_rate)
        [dropout_rate]
    else
        [0.0, 0.001, 0.01, 0.1, 0.5]
    end

    # Random search instead of grid search
    rng = Random.seed!(seed)
    params_all = collect(
        Iterators.product(
            [
                choices_activation,
                choices_optimiser,
                choices_n_epochs,
                choices_n_patient_epochs,
                choices_n_hidden_layers,
                choices_hidden_dims,
                choices_dropout_rate,
            ]...,
        ),
    )
    params_drawn = sample(rng, params_all, n_random_searches, replace = false)
    stat_names = ["corr_pearson", "corr_spearman", "corr_lin", "R²", "mae", "rmse"]
    df_stats = DataFrame(
        id = collect(1:n_random_searches),
        activation = "",
        optimiser = "",
        n_epochs = NaN,
        n_patient_epochs = NaN,
        n_hidden_layers = NaN,
        hidden_dims = NaN,
        dropout_rate = NaN,
    )
    for s in stat_names
        df_stats[:, s] .= NaN
    end
    if verbose
        pb = ProgressMeter.Progress(
            n_random_searches,
            desc = "Fine-tuning hyperparameters using $(length(params_drawn))/$(length(params_all)) randomly chosen paramater combinations: ",
        )
    end
    n = nrow(df)
    n_validation = Int(round(validation_rate * n))
    idx_validation = sort(sample(rng, 1:n, n_validation, replace = false))
    idx_training = sort(filter(x -> !(x ∈ idx_validation), 1:nrow(df)))
    @time for (i, params) in enumerate(params_drawn)
        # i = 1; params = params_drawn[i]
        # @show i
        # @show activation, optimiser, n_epochs, n_patient_epochs, n_hidden_layers, hidden_dims, dropout_rate = params
        activation, optimiser, n_epochs, n_patient_epochs, n_hidden_layers, hidden_dims, dropout_rate = params
        stats = trainNN(
            df,
            trait_id = trait_id,
            varex = varex,
            idx_training = idx_training,
            idx_validation = idx_validation,
            activation = activation,
            optimiser = optimiser,
            n_epochs = n_epochs,
            n_patient_epochs = n_patient_epochs,
            n_hidden_layers = n_hidden_layers,
            hidden_dims = hidden_dims,
            dropout_rate = dropout_rate,
            use_cpu = use_cpu,
            seed = i,
            verbose = false,
        )
        df_stats.activation[i] = string(activation)
        df_stats.optimiser[i] = string(optimiser)
        df_stats.n_epochs[i] = n_epochs
        df_stats.n_patient_epochs[i] = n_patient_epochs
        df_stats.n_hidden_layers[i] = Float64(n_hidden_layers)
        df_stats.hidden_dims[i] = Float64(hidden_dims)
        df_stats.dropout_rate[i] = dropout_rate
        for s in stat_names
            df_stats[i, s] = stats.stats_validation[Symbol(s)]
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Find optimum given observations alone
    df_stats = filter(x -> !isnan(x.corr_pearson), df_stats)
    for s in stat_names
        x = Vector{Float64}(df_stats[!, s])
        z = (x .- mean(x)) ./ std(x)
        df_stats[!, s] = (s == "mae") || (s == "rmse") ? -1.00 .* z : z
    end
    df_stats.z = mean(Matrix(df_stats[:, stat_names]), dims=2)[:, 1]
    if verbose
        println("Empirical stats:")
        display(df_stats)
        println("Best paramters given empirical stats (i.e. prior to interpolation):")
        display(df_stats[argmax(df_stats.z), :])
    end
    # Find the optimum via interpolation
    xs_all = nothing
    for params in params_all
        # params = params_all[1]
        x = vcat(
            [
                Float64(findall(string.(choices_activation) .== string(params[1]))[1]),
                Float64(findall(string.(choices_optimiser) .== string(params[2]))[1]),
            ],
            vcat(params[3:end]...),
        )
        xs_all = if isnothing(xs_all)
            reshape(x, length(x), 1)
        else
            hcat(xs_all, x)
        end
    end
    x_activation = [Float64(findall(string.(choices_activation) .== x)[1]) for x in df_stats.activation]
    x_optimiser = [Float64(findall(string.(choices_optimiser) .== x)[1]) for x in df_stats.optimiser]
    xs =
        hcat(
            x_activation,
            x_optimiser,
            df_stats.n_epochs,
            df_stats.n_patient_epochs,
            df_stats.n_hidden_layers,
            df_stats.hidden_dims,
            df_stats.dropout_rate,
        )'
    z = df_stats.z
    interpolations = ScatteredInterpolation.interpolate(Multiquadratic(), xs, z)
    ẑ = ScatteredInterpolation.evaluate(interpolations, xs_all)
    idx_opt = argmax(ẑ)
    p_opt = xs_all[:, idx_opt]
    # Final training
    idx_training = collect(1:nrow(df))
    idx_validation = sort(filter(x -> !(x ∈ idx_training), 1:nrow(df)))
    model = trainNN(
        df,
        trait_id = trait_id,
        varex = varex,
        idx_training=idx_training,
        idx_validation=idx_validation,
        activation = choices_activation[Int(p_opt[1])],
        optimiser = choices_optimiser[Int(p_opt[2])],
        n_epochs = Int64(p_opt[3]),
        n_patient_epochs = Int64(p_opt[4]),
        n_hidden_layers = Int64(p_opt[5]),
        hidden_dims = Int64(p_opt[6]),
        dropout_rate = p_opt[7],
        use_cpu = use_cpu,
        seed = 1,
        verbose = true,
    )
    # Extract effects and variance-covariance matrices
    Φ = extracteffects(model)
    Σ = extractcovariances(model)
    


    # Output
    # TODO: Include the interpolations into df_stats 
    (df_stats, model, Φ, Σ)
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
        vcat(
            ["years", "seasons", "sites", "harvests", "rows", "cols", "blocks", "replications", "entries"],
            other_covariates,
        )
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
        trait_id = trait_id,
        varex = varex,
        activation = activation,
        n_hidden_layers = n_hidden_layers,
        hidden_dims = hidden_dims,
        dropout_rate = dropout_rate,
        n_epochs = n_epochs,
        use_cpu = use_cpu,
        seed = seed,
        verbose = verbose,
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
    for (K, D) in Dict(:year => effects_years, :season => effects_seasons, :site => effects_sites)
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
