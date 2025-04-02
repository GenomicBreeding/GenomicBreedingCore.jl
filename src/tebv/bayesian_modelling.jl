"""
    instantiateblr(; trait::String, factors::Vector{String}, df::DataFrame, 
                      other_covariates::Union{Vector{String}, Nothing}=nothing, 
                      verbose::Bool=false)::BLR

Extract design matrices and response variable for Bayesian modelling of factorial experiments.

# Arguments
- `trait::String`: Name of the response variable (dependent variable) in the DataFrame
- `factors::Vector{String}`: Vector of factor names (independent variables) to be included in the model
- `df::DataFrame`: DataFrame containing the data
- `other_covariates::Union{Vector{String}, Nothing}=nothing`: Additional numeric covariates to include in the model
- `verbose::Bool=false`: If true, prints additional information during execution

# Returns
A dictionary with the following keys:
- `"trait"`: String, name of the response variable
- `"coefficient_names"`: Vector{String}, column labels for the design matrix (excluding base levels)
- `"X"`: Matrix{Float64}, design matrix (n × F(p-1)) excluding base levels but includes the intercept
- `"b_labels"`: Vector{String}, column labels for the full design matrix (including base levels)
- `"X_ALL"`: Matrix{Float64}, full design matrix (n × F(p)) including base levels and intercept
- `"explain_var_names"`: Vector{String}, names of the coefficients in the model
- `"vector_of_Xs_noint"`: Vector{Matrix{Bool}}, vector of Boolean design matrices for each factor combination (excluding base levels and intercept)
- `"vector_of_Xs_noint_ALL"`: Vector{Matrix{Bool}}, vector of Boolean design matrices for each factor combination (including base levels but excludes the intercept)
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
julia> genomes = simulategenomes(n=500, l=1_000, verbose=false); 

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr, vector_of_Xs_noint = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = nothing, verbose = false);

julia> blr2, vector_of_Xs_noint2 = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = [trials.traits[2]], verbose = false);

julia> blr.coefficient_names != blr2.coefficient_names
true

julia> sum([size(x,2) for x in vector_of_Xs_noint]) < sum([size(x,2) for x in vector_of_Xs_noint2])
true
```
"""
function instantiateblr(;
    trait::String,
    factors::Vector{String},
    df::DataFrame,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    verbose::Bool = false,
)::BLR
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
    _, coefficient_names = coefnames(apply_schema(formula_struct, schema(formula_struct, df)))
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
    explain_var_names::Vector{String} = [string(x) for x in coefficients]
    for f in explain_var_names
        # f = factors[1]
        contrasts[Symbol(f)] = StatsModels.FullDummyCoding()
    end
    mf = ModelFrame(formula_struct, df, contrasts = contrasts)
    _, b_labels = coefnames(apply_schema(formula_struct, mf.schema))
    X_ALL = modelmatrix(mf)
    # Make sure that we have extracted all base levels in X_ALL
    @assert size(X) < size(X_ALL)
    # Initialise the BLR struct
    blr = begin
        n, p = size(X)
        blr = BLR(n=n, p=p)
        blr.entries = df.entries
        delete!(blr.Xs, "dummy")
        delete!(blr.Σs, "dummy")
        delete!(blr.coefficients, "dummy")
        delete!(blr.coefficient_names, "dummy")
        blr.y = zeros(n)
        blr.ŷ = zeros(n)
        blr.ϵ = zeros(n)
        blr
    end
    blr_ALL = begin
        n, p = size(X_ALL)
        blr = BLR(n=n, p=p)
        blr.entries = df.entries
        delete!(blr.Xs, "dummy")
        delete!(blr.Σs, "dummy")
        delete!(blr.coefficients, "dummy")
        delete!(blr.coefficient_names, "dummy")
        blr.y = zeros(n)
        blr.ŷ = zeros(n)
        blr.ϵ = zeros(n)
        blr
    end
    # Separate each explanatory variable from one another
    for v in explain_var_names
        # v = explain_var_names[end]
        v_split = filter(x -> x != "&", split(v, " "))
        # Main design matrices (bool) full design matrices (bool_ALL)
        bool = fill(true, length(coefficient_names))
        bool_ALL = fill(true, length(b_labels))
        for x in v_split
            bool = bool .&& .!isnothing.(match.(Regex(x), coefficient_names))
            bool_ALL = bool_ALL .&& .!isnothing.(match.(Regex(x), b_labels))
        end
        # Make sure we are extract the correct coefficients
        if length(v_split) == 1
            bool = bool .&& isnothing.(match.(Regex("&"), coefficient_names))
            bool_ALL = bool_ALL .&& isnothing.(match.(Regex("&"), b_labels))
        else
            m = length(v_split) - 1
            bool = bool .&& [!isnothing(x) ? sum(.!isnothing.(match.(Regex("&"), split(coefficient_names[i], " ")))) == m : false for (i, x) in enumerate(match.(Regex("&"), coefficient_names))]
            bool_ALL = bool_ALL .&& [!isnothing(x) ? sum(.!isnothing.(match.(Regex("&"), split(b_labels[i], " ")))) == m : false for (i, x) in enumerate(match.(Regex("&"), b_labels))]
        end
        if !isnothing(other_covariates) && !(v ∈ other_covariates)
            # For categorical variables --> boolean matrix for memory-efficiency
            blr.Xs[v] = Bool.(X[:, bool])
            blr_ALL.Xs[v] = Bool.(X_ALL[:, bool_ALL])
        else
            # For the continuous numeric other covariates --> Float64 matrix
            blr.Xs[v] = X[:, bool]
            blr_ALL.Xs[v] = X_ALL[:, bool_ALL]
        end
        blr.Σs[v] = 1.0*I
        blr_ALL.Σs[v] = 1.0*I
        blr.coefficients[v] = zeros(sum(bool_ALL))
        blr_ALL.coefficients[v] = zeros(sum(bool_ALL))
        blr.coefficient_names[v] = b_labels[bool_ALL]
        blr_ALL.coefficient_names[v] = b_labels[bool_ALL]
    end
    # Output
    if !checkdims(blr)
        throw(ErrorException("The resulting BLR struct for the model to be fit is corrupted."))
    end
    if !checkdims(blr_ALL)
        throw(ErrorException("The resulting BLR struct for the full model is corrupted."))
    end
    (blr, blr_ALL)
end

"""
    turingblr(vector_of_Xs_noint::Vector{Matrix{Bool}}, y::Vector{Float64})

Bayesian Linear Regression model implemented using Turing.jl.

# Arguments
- `vector_of_Xs_noint::Vector{Matrix{Bool}}`: Vector of predictor matrices, where each matrix contains binary (Bool) predictors
- `y::Vector{Float64}`: Vector of response variables

# Model Details
- Includes an intercept term with Normal(0.0, 10.0) prior
- For each predictor matrix in `vector_of_Xs_noint`:
  - Variance parameter (σ²) with Exponential(1.0) prior
  - Regression coefficients (β) with multivariate normal prior: MvNormal(0, σ² * I)
- Residual variance (σ²) with Exponential(10.0) prior
- Response variable modeled as multivariate normal: MvNormal(μ, σ² * I)

# Returns
- A Turing model object that can be used for MCMC sampling
"""
Turing.@model function turingblr(
    vector_of_Xs_noint::Vector{Matrix{Union{Bool, Float64}}},
    vector_of_Σs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}},
    y::Vector{Float64},
)
    # Set intercept prior.
    intercept ~ Normal(0.0, 10.0)
    # Set variance predictors
    P = length(vector_of_Xs_noint)
    σ²s = fill(0.0, P)
    βs = [fill(0.0, size(x, 2)) for x in vector_of_Xs_noint]
    μ = fill(0.0, size(vector_of_Xs_noint[1], 1)) .+ intercept
    for i = 1:P
        σ²s[i] ~ Exponential(1.0)
        βs[i] ~ MvNormal(zeros(length(βs[i])), σ²s[i] * vector_of_Σs[i])
        μ += Float64.(vector_of_Xs_noint[i]) * βs[i]
    end
    # Residual variance
    σ² ~ Exponential(10.0)
    # Return the distribution of the response variable, from which the likelihood will be derived
    return y ~ Distributions.MvNormal(μ, σ² * I)
end


"""
    turingblrmcmc(;
        vector_of_Xs_noint::Vector{Matrix{Bool}}, 
        y::Vector{Float64},
        n_iter::Int64 = 10_000,
        n_burnin::Int64 = 1_000,
        seed::Int64 = 1234,
        verbose = true
    )::BLR

Perform Markov Chain Monte Carlo (MCMC) sampling for Bayesian Linear Regression using the No-U-Turn Sampler (NUTS).

# Arguments
- `vector_of_Xs_noint::Vector{Matrix{Bool}}`: A vector of design matrices for different model components
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
julia> genomes = simulategenomes(n=500, l=1_000, verbose=false); 

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr, vector_of_Xs_noint = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);

julia> turingblrmcmc(blr, vector_of_Xs_noint=vector_of_Xs_noint, n_iter=1_000, n_burnin=200, seed=123, verbose=false);

julia> 
```
"""
function turingblrmcmc!(
    blr;
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose = true,
)::BLR
    
    # n_iter = 1_000
    # n_burnin = 500
    # seed = 1234
    # verbose = true

    # Check arguments
    entries = blr.entries
    coefficient_names = blr.coefficient_names
    # b_labels = blr.b_labels
    # X_ALL = blr.X_ALL
    # explain_var_names = blr.explain_var_names
    y = blr.y
    vector_of_Σs = blr.Σs
    X, _b, b_labels = extractXb(blr)

    n, p = size(X)
    if n != length(y)
        throw(ArgumentError("The number of entries in y (`n=$(length(y))`) is not equal to the number of rows in X_ALL (`n=$n`)."))
    end
    if p != length(b_labels)
        throw(ArgumentError("The number of parameters in b_labels (`n=$(length(b_labels))`) is not equal to the number of columns in X_ALL (`n=$p`)."))
    end
    if length(blr.coefficient_names) != (1 + length(vector_of_Xs_noint))
        throw(ArgumentError("The BLR struct is incompatible with `vector_of_Xs_noint`."))
    end
    for i in eachindex(vector_of_Xs_noint)
        if n != size(vector_of_Xs_noint[i], 1)
            throw(ArgumentError("The number of entries (`n=$n`) is not equal to the number of rows in the $(i)th X (`size(vector_of_Xs_noint[$i], 1)=$(size(vector_of_Xs_noint[i], 1))`)."))
        end
    end
    for (i, x) in blr.Xs
        if n != size(x, 1)
            throw(ArgumentError("The number of entries (`n=$n`) is not equal to the number of rows in the $(i)th X_ALL (`size(vector_of_Xs_noint_ALL[$i], 1)=$(size(vector_of_Xs_noint_ALL[i], 1))`)."))
        end
    end
    # Instantiate the RNG, model, and sampling function
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turingblr(vector_of_Xs_noint, vector_of_Σs, y)
    sampling_function =
        NUTS(n_burnin, 0.65, max_depth = 5, Δ_max = 1000.0, init_ϵ = 0.2; adtype = AutoReverseDiff(compile = true))
    # MCMC
    chain = Turing.sample(rng, model, sampling_function, n_iter, discard_initial = n_burnin, progress = verbose)
    # Diagnostics
    # R̂ = DataFrame(MCMCDiagnosticTools.rhat(chain, kind=:basic, split_chains=5)) # classic R̂
    R̂ = DataFrame(MCMCDiagnosticTools.rhat(chain, kind=:rank, split_chains=5)) # new R̂: maximum R̂ of :bulk and :tail
    n_params_which_may_not_have_converged = sum(abs.(1.0 .- R̂.rhat) .> 0.1)
    if verbose && (n_params_which_may_not_have_converged > 0)
        @warn "There are $n_params_which_may_not_have_converged parameters (out of $(nrow(R̂)) total parameters) which may not have converged."
    end
    # Use the mean parameter values which excludes the first n_burnin iterations
    params = Turing.get_params(chain[:, :, :])
    β0 = mean(params.intercept)
    βs = [mean(params.βs[i]) for i in eachindex(params.βs)]
    σ² = mean(params.σ²)
    σ²s = [mean(params.σ²s[i]) for i in eachindex(params.σ²s)]
    # Extract all the coefficients including the base level/s of each factor
    βs_ALL = fill(0.0, length(b_labels))
    βs_ALL[1] = β0
    for i in eachindex(βs_ALL)
        # i = 200
        if i == 1
            # intercept
            continue
        end
        # Less 1 because the X matrix does not include the intercept
        idx = findall(coefficient_names .== b_labels[i]) .- 1
        if length(idx) == 0
            # base level
            continue
        end
        βs_ALL[i] = βs[idx[1]]
    end
    ŷ = X_ALL * βs_ALL
    ϵ = y - ŷ
    Xs::Dict{String,Matrix{Union{Bool, Float64}}} = Dict("intercept" => Bool.(ones(length(y), 1)))
    Σs::Dict{String,Union{Matrix{Float64},UniformScaling{Float64}}} = Dict("σ²" => σ² * I)
    dict_coefficients::Dict{String, Vector{Float64}} = Dict("intercept" => [β0])
    dict_coefficient_names::Dict{String, Vector{String}} = Dict("intercept" => ["intercept"])
    var_comp::Dict{String,Int64} = Dict("σ²" => 1)
    for i = 1:length(σ²s)
        # i = 3
        v = explain_var_names[i]
        k = size(vector_of_Xs_noint_ALL[i], 2)
        p_ini = sum(values(var_comp)) + 1
        p_fin = (p_ini-1) + k
        Xs[v] = vector_of_Xs_noint_ALL[i]
        Σs[v] = σ²s[i] * I
        dict_coefficients[v] = βs_ALL[p_ini:p_fin]
        dict_coefficient_names[v] = b_labels[p_ini:p_fin]
        var_comp[v] = k
    end
    blr_model = begin
        n, p = size(X_ALL)
        blr_model = BLR(n = n, p = p, var_comp = var_comp)
        blr_model.entries = entries
        blr_model.Xs = Xs
        blr_model.Σs = Σs
        blr_model.coefficients = dict_coefficients
        blr_model.coefficient_names = dict_coefficient_names
        blr_model.y = y
        blr_model.ŷ = ŷ
        blr_model.ϵ = ϵ
        blr_model
    end
    # Output
    if !checkdims(blr_model)
        throw(ErrorException("Error generating the BLR struct."))
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
                # Instantiate the BLR struct
                blr, vector_of_Xs_noint = instantiateblr(
                    trait = trait,
                    factors = spatial_factors,
                    other_covariates = other_covariates,
                    df = df_sub,
                    verbose = verbose,
                )
                blr.y = df[:, trait]
                # Bayesian linear regression
                turingblrmcmc!(
                    blr,
                    vector_of_Xs_noint = vector_of_Xs_noint,
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
        _, coefficient_names = coefnames(apply_schema(f0, schema(f0, df)))
        size(df)
        length(coefficient_names)
    end
end
