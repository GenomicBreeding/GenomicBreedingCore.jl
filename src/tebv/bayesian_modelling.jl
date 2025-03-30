function analyse(
    trials::Trials,
    traits::Vector{String};
    GRM::Union{Matrix{Float64},UniformScaling} = I,
    other_covariates::Union{String,Nothing} = nothing,
    verbose::Bool = false,
)::TEBV
    # genomes = simulategenomes(n=500, l=1_000)
    # trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3)
    # GRM::Union{Matrix{Float64}, UniformScaling} = I; traits = ["trait_1"]; other_covariates::Union{String, Nothing} = nothing; verbose::Bool = true;
    # Check arguments
    if !checkdims(trials)
        error("The Trials struct is corrupted.")
    end
    if length(traits) > 0
        for trait in traits
            if !(trait ∈ trials.traits)
                throw(ArgumentError("The `traits` ($traits) argument is not a trait in the Trials struct."))
            end
        end
    end
    if !isnothing(other_covariates)
        if !(other_covariates ∈ trials.traits)
            throw(
                ArgumentError(
                    "The `other_covariates` ($other_covariates) argument is not a trait in the Trials struct.",
                ),
            )
        end
    end
    # Extract the entries which we want the estimated breeding values for
    entries = sort(unique(trials.entries))
    n = length(entries)
    # Define the traits
    traits = if length(traits) == 0
        trials.traits
    else
        traits
    end
    # Instantiate output
    tebv = TEBV(
        traits = traits,
        formulae = string.(traits, " ~ 1"),
        models = ([""], rand(1), rand(1, 1)),
        df_BLUEs = fill(DataFrame(), length(traits)),
        df_BLUPs = fill(DataFrame(), length(traits)),
        phenomes = [
            begin
                p = Phenomes(n = n, t = 1)
                p.entries = entries
                p.traits = [t]
                p
            end for t in traits
        ],
    )
    # Tabularise the trials data
    df = tabularise(trials)
    # Identify non-fixed factors
    factors_all::Vector{String} = ["years", "seasons", "harvests", "sites", "blocks", "rows", "cols", "entries"]
    factors::Vector{String} = []
    for f in factors_all
        # f = factors_all[1]
        if length(unique(df[!, f])) > 1
            push!(factors, f)
        end
    end
    # If both blocks and rows are present, then we remove blocks as they are expected to be redundant
    if ("blocks" ∈ factors) && ("rows" ∈ factors)
        factors = filter(x -> x != "blocks", factors)
    end
    # Check for potential out-of-memory error
    D = dimensions(trials)
    total_parameters = 1
    for f in factors
        # f = factors[1]
        total_parameters *= (D[string("n_", f)] - 1)
    end
    total_X_size_in_Gb = nrow(df) * (total_parameters + 1) * sizeof(Float64) / (1024^3)
    @warn "The size of the design matrix is ~$(round(total_X_size_in_Gb)) GB. This may cause out-of-memory errors."


    # To prevent OOM errors, we will perform spatial analyses per harvest per site, i.e. remove spatial effects per replication
    # and perform the potentially GxE analysis on the residulals
    if ("blocks" ∈ factors) || ("rows" ∈ factors) || ("cols" ∈ factors)
        # Define spatial factors
        spatial_factors = factors[.!isnothing.(match.(Regex("blocks|rows|cols"), factors))]
        # Define the contrast for one-hot encoding, i.e. including the intercept, i.e. p instead of the p-1
        contrasts::Dict{Symbol, StatsModels.FullDummyCoding} = Dict()
        for f in spatial_factors
            # f = spatial_factors[1]
            contrasts[Symbol(f)] = StatsModels.FullDummyCoding()
        end
        # Make sure that each harvest is year- and site-specific
        df.harvests = string.(df.years, "|", df.sites, "|", df.harvests)
        for harvest in unique(df.harvests)
            # harvest = unique(df.harvests)[1]
            df_sub = filter(x -> x.harvests == harvest, df)
            for trait in traits
                # trait = traits[1]
                formula_struct = term(trait) ~ term(1) + foldl(*, term.(spatial_factors))
                # Extract the names of the coefficients and the design matrix (n x F(p-1); excludes the base level/s) to used for the regression
                _, col_labels = coefnames(apply_schema(formula_struct, schema(formula_struct, df_sub)))
                X = modelmatrix(formula_struct, df_sub)
                # Extract the names of all the coefficients including the base level/s and the full design matrix (n x F(p)).
                # This is not used for regression, rather it is used for the extraction of coefficiencts of all factor levels including the base level/s.
                mf = ModelFrame(formula_struct, df_sub, contrasts = contrasts)
                _, col_labels_ALL = coefnames(apply_schema(formula_struct, mf.schema))
                X_ALL = modelmatrix(mf)
                # Check
                size(X) < size(X_ALL)
                # Separate each factor from one another
                coefficient_names = string.(foldl(*, term.(spatial_factors)))
                vector_of_Xs = []
                vector_of_Xs_ALL = []
                for f in coefficient_names
                    # f = coefficient_names[1]
                    f_split = split(f, " ")
                    # Main design matrices
                    bool = fill(true, length(col_labels))
                    for x in f_split
                        bool = bool .&& .!isnothing.(match.(Regex(x), col_labels))
                    end
                    if length(f_split) == 1
                        bool = bool .&& isnothing.(match.(Regex("&"), col_labels))
                    end
                    push!(vector_of_Xs, X[:, bool])
                    # Full design matrices
                    bool = fill(true, length(col_labels_ALL))
                    for x in f_split
                        bool = bool .&& .!isnothing.(match.(Regex(x), col_labels_ALL))
                    end
                    if length(f_split) == 1
                        bool = bool .&& isnothing.(match.(Regex("&"), col_labels_ALL))
                    end
                    push!(vector_of_Xs_ALL, X_ALL[:, bool])
                end
                # Define the linear model
                y::Vector{Float64} = df_sub[!, trait]
                Turing.@model function blr_spatial(vector_of_Xs, y)
                    # Set intercept prior.
                    intercept ~ Normal(0.0, 10.0)
                    # Set variance predictors
                    P = length(vector_of_Xs)
                    σ²s = fill(0.0, P)
                    βs = [fill(0.0, size(x,2)) for x in vector_of_Xs]
                    μ = fill(0.0, size(vector_of_Xs[1],1)) .+ intercept
                    for i in 1:P
                        σ²s[i] ~ Exponential(1.0)
                        βs[i] ~ MvNormal(zeros(length(βs[i])), σ²s[i]*I)
                        μ += vector_of_Xs[i] * βs[i]
                    end
                    # Residual variance
                    σ² ~ Exponential(10.0)
                    # Return the distrbution of the response variable, from which the likelihood will be derived
                    return y ~ Distributions.MvNormal(μ, σ² * I)
                end
                function mcmc()
                    seed = 1234
                    n_iter = 1000
                    n_burnin = 150
                    verbose = true
                    rng::TaskLocalRNG = Random.seed!(seed)
                    model = blr_spatial(vector_of_Xs, y)
                    sampling_function =
                        NUTS(n_burnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.2; adtype = AutoReverseDiff(compile = true))
                    chain = Turing.sample(rng, model, sampling_function, n_iter - n_burnin, progress = verbose)
                    # Use the mean paramter values after 150 burn-in iterations
                    params = Turing.get_params(chain[(n_burnin+1):end, :, :])
                end
        

            end
        end
    end
        


    # Iterate per trait
    for trait in traits
        # trait = traits[1]
        # Define the formula for the model
        formula_string = string(trait, " ~ 1 + ", join(factors, " * "))
        formula_struct = @eval(@string2formula($formula_string))
        X = modelmatrix(formula_struct, df)



        f0 = @eval(@string2formula($(replace(formula_string, "y" => "0"))))
        _, col_labels = coefnames(apply_schema(f0, schema(f0, df)))
        size(df)
        length(col_labels)
    end
end


