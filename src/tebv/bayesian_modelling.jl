"""
    extractvariablesandnames(; trait::String, factors::Vector{String}, df::DataFrame, 
                      other_covariates::Union{Vector{String}, Nothing}=nothing, 
                      verbose::Bool=false)::Dict

Extract design matrices and response variable for Bayesian modeling of factorial experiments.

# Arguments
- `trait::String`: Name of the response variable (dependent variable) in the DataFrame
- `factors::Vector{String}`: Vector of factor names (independent variables) to be included in the model
- `df::DataFrame`: DataFrame containing the data
- `other_covariates::Union{Vector{String}, Nothing}=nothing`: Additional numeric covariates to include in the model
- `verbose::Bool=false`: If true, prints additional information during execution

# Returns
A dictionary with the following keys:
- `"trait"`: String, name of the response variable
- `"col_labels"`: Vector{String}, column labels for the design matrix (excluding base levels)
- `"X"`: Matrix{Float64}, design matrix (n × F(p-1)) excluding base levels
- `"col_labels_ALL"`: Vector{String}, column labels for the full design matrix (including base levels)
- `"X_ALL"`: Matrix{Float64}, full design matrix (n × F(p)) including base levels
- `"coefficient_names"`: Vector{String}, names of the coefficients in the model
- `"vector_of_Xs"`: Vector{Matrix{Bool}}, vector of Boolean design matrices for each factor combination (excluding base levels)
- `"vector_of_Xs_ALL"`: Vector{Matrix{Bool}}, vector of Boolean design matrices for each factor combination (including base levels)
- `"y"`: Vector{Float64}, response variable vector

# Details
Creates design matrices for factorial experiments using both standard and one-hot encoding approaches.
The function processes main effects and interaction terms, handling categorical factors and
continuous covariates differently. It utilizes StatsModels.jl for formula processing and 
contrast encoding.

# Implementation Notes
- Converts all factors to strings for categorical treatment
- Uses FullDummyCoding for contrast encoding
- Validates numeric nature of trait and covariates
- Processes interaction terms between factors automatically
- Handles both categorical factors (as Boolean matrices) and continuous covariates (as Float64 matrices)
- Creates separate design matrices with and without base levels

# Example
```
# TODO
```
"""
function extractvariablesandnames(;
    trait::String,
    factors::Vector{String},
    df::DataFrame,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    verbose::Bool = false,
)::Dict{String,Union{String,Vector{String},Vector{Float64},Matrix{Float64},Vector{Matrix{Union{Bool,Float64}}}}}
    # genomes = simulategenomes(n=500, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3);
    # trait = "trait_1"; factors = ["rows", "cols"]; df = tabularise(trials); other_covariates::Union{Vector{String}, Nothing} = nothing; verbose = true;
    # Check arguments
    if !(trait ∈ names(df))
        throw(ArgumentError("The input data frame does not include the trait: `$trait`."))
    end
    try
        [Float64(x) for x in df[!, trait]]
    catch
        throw(ArgumentError("The other trait: `$trait` is non-numeric and/or possibly have missing data."))
    end
    for f in factors
        if !(f ∈ names(df))
            throw(ArgumentError("The input data frame does not include the factor: `$f`."))
        end
        # Make sure the factors are strings, i.e. strictly non-numeric
        df[!, f] = string.(df[!, f])
    end
    if !isnothing(other_covariates)
        for c in other_covariates
            if !(c ∈ names(df))
                throw(ArgumentError("The input data frame does not include the other covariate: `$c`."))
            end
            try
                [Float64(x) for x in df[!, c]]
            catch
                throw(ArgumentError("The other covariate: `$c` is non-numeric and/or possibly have missing data."))
            end
            # Make sure the other covariates are strictly numeric
            co::Vector{Float64} = df[!, c]
            df[!, c] = co
        end
    end
    # Define the formula
    formula_struct, coefficients = if isnothing(other_covariates)
        coefficients = foldl(*, term.(factors))
        (term(trait) ~ term(1) + coefficients), coefficients
    else
        # f = term(trait) ~ term(1) + foldl(*, term.(factors)) + foldl(+, term.(other_covariates))
        coefficients =
            foldl(*, term.(factors)) +
            foldl(+, [concrete_term(term(c), df[!, c], ContinuousTerm) for c in other_covariates])
        (term(trait) ~ term(1) + coefficients), coefficients
    end
    # Extract the names of the coefficients and the design matrix (n x F(p-1); excludes the base level/s) to used for the regression
    if verbose
        println("Extracting the design matrix of the model: `$formula_struct`.")
    end
    _, col_labels = coefnames(apply_schema(formula_struct, schema(formula_struct, df)))
    X = modelmatrix(formula_struct, df)
    # Extract the names of all the coefficients including the base level/s and the full design matrix (n x F(p)).
    # This is not used for regression, rather it is used for the extraction of coefficients of all factor levels including the base level/s.
    # But first, define the contrast for one-hot encoding, i.e. including the intercept, i.e. p instead of the p-1
    if verbose
        println(
            "Extracting the design matrix of the one-hot encoding model, i.e. including all the base levels.\nThis will not be used in model fitting; but used in coefficient extraction.",
        )
    end
    contrasts::Dict{Symbol,StatsModels.FullDummyCoding} = Dict()
    coefficient_names::Vector{String} = [string(x) for x in coefficients]
    for f in coefficient_names
        # f = factors[1]
        contrasts[Symbol(f)] = StatsModels.FullDummyCoding()
    end
    mf = ModelFrame(formula_struct, df, contrasts = contrasts)
    _, col_labels_ALL = coefnames(apply_schema(formula_struct, mf.schema))
    X_ALL = modelmatrix(mf)
    # Make sure that we have extracted all base levels in X_ALL
    @assert size(X) < size(X_ALL)
    # Separate each factor from one another
    vector_of_Xs::Vector{Matrix{Union{Bool,Float64}}} = []
    vector_of_Xs_ALL::Vector{Matrix{Union{Bool,Float64}}} = []
    for f in coefficient_names
        # f = coefficient_names[end]
        f_split = split(f, " ")
        # Main design matrices
        bool = fill(true, length(col_labels))
        for x in f_split
            bool = bool .&& .!isnothing.(match.(Regex(x), col_labels))
        end
        if length(f_split) == 1
            bool = bool .&& isnothing.(match.(Regex("&"), col_labels))
        end
        if !(f ∈ other_covariates)
            # For categorical factors --> boolean matrix for memory-efficiency
            push!(vector_of_Xs, Bool.(X[:, bool]))
        else
            # For the continuous numeric other covariates --> Float64 matrix
            push!(vector_of_Xs, X[:, bool])
        end
        # Full design matrices
        bool = fill(true, length(col_labels_ALL))
        for x in f_split
            bool = bool .&& .!isnothing.(match.(Regex(x), col_labels_ALL))
        end
        if length(f_split) == 1
            bool = bool .&& isnothing.(match.(Regex("&"), col_labels_ALL))
        end
        if !(f ∈ other_covariates)
            # For categorical factors --> boolean matrix for memory-efficiency
            push!(vector_of_Xs_ALL, Bool.(X_ALL[:, bool]))
        else
            # For the continuous numeric other covariates --> Float64 matrix
            push!(vector_of_Xs_ALL, X_ALL[:, bool])
        end
    end
    # Define the linear model
    y::Vector{Float64} = df[!, trait]
    # Output
    Dict(
        "trait" => trait,
        "col_labels" => col_labels,
        "X" => X,
        "col_labels_ALL" => col_labels_ALL,
        "X_ALL" => X_ALL,
        "coefficient_names" => coefficient_names,
        "vector_of_Xs" => vector_of_Xs,
        "vector_of_Xs_ALL" => vector_of_Xs_ALL,
        "y" => y,
    )
end

"""
    turingblr(vector_of_Xs::Vector{Matrix{Bool}}, y::Vector{Float64})

Bayesian Linear Regression model implemented using Turing.jl.

# Arguments
- `vector_of_Xs::Vector{Matrix{Bool}}`: Vector of predictor matrices, where each matrix contains binary (Bool) predictors
- `y::Vector{Float64}`: Vector of response variables

# Model Details
- Includes an intercept term with Normal(0.0, 10.0) prior
- For each predictor matrix in `vector_of_Xs`:
  - Variance parameter (σ²) with Exponential(1.0) prior
  - Regression coefficients (β) with multivariate normal prior: MvNormal(0, σ² * I)
- Residual variance (σ²) with Exponential(10.0) prior
- Response variable modeled as multivariate normal: MvNormal(μ, σ² * I)

# Returns
- A Turing model object that can be used for MCMC sampling
"""
Turing.@model function turingblr(
    vector_of_Xs::Vector{Matrix{Bool}},
    vector_of_Σs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}},
    y::Vector{Float64},
)
    # Set intercept prior.
    intercept ~ Normal(0.0, 10.0)
    # Set variance predictors
    P = length(vector_of_Xs)
    σ²s = fill(0.0, P)
    βs = [fill(0.0, size(x, 2)) for x in vector_of_Xs]
    μ = fill(0.0, size(vector_of_Xs[1], 1)) .+ intercept
    for i = 1:P
        σ²s[i] ~ Exponential(1.0)
        βs[i] ~ MvNormal(zeros(length(βs[i])), σ²s[i] * vector_of_Σs[i])
        μ += Float64.(vector_of_Xs[i]) * βs[i]
    end
    # Residual variance
    σ² ~ Exponential(10.0)
    # Return the distribution of the response variable, from which the likelihood will be derived
    return y ~ Distributions.MvNormal(μ, σ² * I)
end


"""
    turingblrmcmc(;
        vector_of_Xs::Vector{Matrix{Bool}}, 
        y::Vector{Float64},
        n_iter::Int64 = 10_000,
        n_burnin::Int64 = 1_000,
        seed::Int64 = 1234,
        verbose = true
    )::BLR

Perform Markov Chain Monte Carlo (MCMC) sampling for Bayesian Linear Regression using the No-U-Turn Sampler (NUTS).

# Arguments
- `vector_of_Xs::Vector{Matrix{Bool}}`: A vector of design matrices for different model components
- `y::Vector{Float64}`: Vector of response variables
- `n_iter::Int64`: Number of MCMC iterations (default: 10,000)
- `n_burnin::Int64`: Number of burn-in iterations to discard (default: 1,000)
- `seed::Int64`: Random seed for reproducibility (default: 1234)
- `verbose::Bool`: Whether to show progress during sampling (default: true)

# Returns
- `BLR`: A Bayesian Linear Regression model object containing:
    - Estimated coefficients
    - Fitted values
    - Residuals
    - Variance components
    - Design matrices
    - Other model information

# Notes
- The function performs convergence diagnostics using Gelman-Rubin statistics (R̂)
- Warns if any parameters haven't converged (R̂ > 1.1 or R̂ < 0.9)
- Uses AutoReverseDiff for automatic differentiation
- Computes posterior means for all parameters after burn-in
- Includes base levels for categorical variables in the final coefficient vector

# Example
```
# TODO
```
"""
function turingblrmcmc(;
    D::Dict{String,Union{String,Vector{String},Vector{Float64},Matrix{Float64},Vector{Matrix{Union{Bool,Float64}}}}},
    vector_of_Σs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}},
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose = true,
)::BLR
    # trait = D["trait"]
    col_labels = D["col_labels"]
    # X = D["X"]
    col_labels_ALL = D["col_labels_ALL"]
    X_ALL = D["X_ALL"]
    coefficient_names = D["coefficient_names"]
    vector_of_Xs = D["vector_of_Xs"]
    vector_of_Xs_ALL = D["vector_of_Xs_ALL"]
    y = D["y"]
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turing_blr(vector_of_Xs, vector_of_Σs, y)
    sampling_function =
        NUTS(n_burnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.2; adtype = AutoReverseDiff(compile = true))
    chain = Turing.sample(rng, model, sampling_function, n_iter, discard_initial = n_burnin, progress = verbose)
    # Diagnostics
    R̂ = DataFrame(MCMCDiagnosticTools.rhat(chain))
    n_params_which_may_not_have_converged = sum((R̂.rhat .> 1.1) .|| (R̂.rhat .< 0.9))
    if n_params_which_may_not_have_converged > 0
        @warn "There are $n_params_which_may_not_have_converged parameters (out of $(nrow(R̂)) total parameters) which may not have converged."
    end
    # Use the mean parameter values which excludes the first n_burnin iterations
    params = Turing.get_params(chain[:, :, :])
    β0 = mean(params.intercept)
    βs = [mean(params.βs[i]) for i in eachindex(params.βs)]
    σ² = mean(params.σ²)
    σ²s = [mean(params.σ²s[i]) for i in eachindex(params.σ²s)]
    # Extract all the coefficients including the base level/s of each factor
    βs_ALL = fill(0.0, length(col_labels_ALL))
    βs_ALL[1] = β0
    for i in eachindex(βs_ALL)
        # i = 200
        if i == 1
            # intercept
            continue
        end
        # Less 1 because the X matrix does not include the intercept
        idx = findall(col_labels .== col_labels_ALL[i]) .- 1
        if length(idx) == 0
            # base level
            continue
        end
        βs_ALL[i] = βs[idx[1]]
    end
    ŷ = X_ALL * βs_ALL
    ϵ = y - ŷ
    design_matrices::Dict{String,Matrix{Bool}} = Dict("intercept" => Bool.(ones(length(y), 1)))
    Σs::Dict{String,Union{Matrix{Float64},UniformScaling{Float64}}} = Dict("σ²" => σ² * I)
    var_comp::Dict{String,Int64} = Dict("σ²" => 1)
    for i = 1:length(σ²s)
        # i = 1
        design_matrices[coefficient_names[i]] = vector_of_Xs_ALL[i]
        Σs["σ²s "*coefficient_names[i]] = σ²s[i] * I
        var_comp["σ²s "*coefficient_names[i]] = size(vector_of_Xs_ALL[i], 2)
    end
    n, p = size(X_ALL)
    blr_model = begin
        blr_model = BLR(n = n, p = p, var_comp = var_comp)
        @assert length(blr_model.entries) == length(df_sub.entries)
        blr_model.entries = df_sub.entries
        @assert length(blr_model.coefficient_names) == length(col_labels_ALL)
        blr_model.coefficient_names = col_labels_ALL
        @assert length(blr_model.y) == length(y)
        blr_model.y = y
        @assert length(blr_model.design_matrices) == length(design_matrices)
        blr_model.design_matrices = design_matrices
        @assert !isnothing(blr_model.other_covariates) ?
                length(blr_model.other_covariates) == length(other_covariates) : true
        blr_model.other_covariates = other_covariates
        @assert length(blr_model.coefficients) == length(βs_ALL)
        blr_model.coefficients = βs_ALL
        @assert length(blr_model.ŷ) == length(ŷ)
        blr_model.ŷ = ŷ
        @assert length(blr_model.ϵ) == length(ϵ)
        blr_model.ϵ = ϵ
        @assert length(blr_model.Σs) == length(Σs)
        blr_model.Σs = Σs
        blr_model
    end
    blr_model
end

function analyse(
    trials::Trials,
    traits::Vector{String};
    GRM::Union{Matrix{Float64},UniformScaling} = I,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    verbose::Bool = false,
)::TEBV
    # genomes = simulategenomes(n=500, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3); GRM::Union{Matrix{Float64}, UniformScaling} = I; traits = ["trait_1"]; other_covariates::Union{Vector{String}, Nothing} = ["trait_2"]; verbose::Bool = true;
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
        for c in other_covariates
            if !(c ∈ trials.traits)
                throw(ArgumentError("The `other_covariates` ($c) argument is not a trait in the Trials struct."))
            end
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
        if ("blocks" ∈ factors) || ("rows" ∈ factors) || ("cols" ∈ factors)
            continue
        end
        total_parameters *= (D[string("n_", f)] - 1)
    end
    total_X_size_in_Gb = nrow(df) * (total_parameters + 1) * sizeof(Float64) / (1024^3)
    @warn "The size of the design matrix is ~$(round(total_X_size_in_Gb)) GB. This may cause out-of-memory errors."

    # To prevent OOM errors, we will perform spatial analyses per harvest per site, i.e. remove spatial effects per replication
    # and perform the potentially GxE analysis on the residulals
    if ("blocks" ∈ factors) || ("rows" ∈ factors) || ("cols" ∈ factors)
        # Define spatial factors
        spatial_factors = factors[.!isnothing.(match.(Regex("blocks|rows|cols"), factors))]
        # Make sure that each harvest is year- and site-specific
        df.harvests = string.(df.years, "|", df.sites, "|", df.harvests)
        for harvest in unique(df.harvests)
            # harvest = unique(df.harvests)[1]
            df_sub = filter(x -> x.harvests == harvest, df)
            for i in eachindex(traits)
                # i = 1
                trait = traits[i]
                # Extract the Xs, ys, labels, and coefficient names
                D = extractvariablesandnames(
                    trait = trait,
                    factors = spatial_factors,
                    other_covariates = other_covariates,
                    df = df_sub,
                    verbose = verbose,
                )
                # Define the variance-covariance matrices of each design matrix and other_covariates matrix
                vector_of_Σs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}} = []
                for i in eachindex(D["vector_of_Xs_ALL"])
                    push!(vector_of_Σs, 1.0 * I)
                end
                # Bayesian linear regression
                blr_model = turingblrmcmc(
                    D = D,
                    vector_of_Σs = vector_of_Σs,
                    n_iter = 1_000,
                    n_burnin = 500,
                    seed = 1234,
                    verbose = verbose,
                )

            end
        end
    end

    # Instantiate output
    models::Vector{BLR} = fill(BLR(n = 1, p = 1), length(traits))
    tebv = TEBV(
        traits = traits,
        formulae = string.(traits, " ~ 1"),
        models = models,
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
