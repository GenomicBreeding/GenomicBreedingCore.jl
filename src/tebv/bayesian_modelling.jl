"""
    instantiateblr(; trait::String, factors::Vector{String}, df::DataFrame, 
                      other_covariates::Union{Vector{String}, Nothing}=nothing, 
                      verbose::Bool=false)::Tuple{BLR,BLR}

Extract design matrices and response variable for Bayesian modelling of factorial experiments.

# Arguments
- `trait::String`: Name of the response variable (dependent variable) in the DataFrame
- `factors::Vector{String}`: Vector of factor names (independent variables) to be included in the model 
- `df::DataFrame`: DataFrame containing the data
- `other_covariates::Union{Vector{String}, Nothing}=nothing`: Additional numeric covariates to include in the model
- `verbose::Bool=false`: If true, prints additional information during execution

# Returns
A tuple of two BLR structs:
1. BLR struct for model fitting (excluding base levels)
2. BLR struct for full model (including base levels)

Each BLR struct contains:
- `entries`: Vector of entry identifiers
- `y`: Response variable vector
- `ŷ`: Predicted values 
- `ϵ`: Residuals
- `Xs`: Dict mapping factors to design matrices
- `Σs`: Dict mapping factors to covariance matrices
- `coefficients`: Dict mapping factors to coefficient vectors
- `coefficient_names`: Dict mapping factors to coefficient names

# Details
Creates design matrices for factorial experiments using both standard and one-hot encoding approaches.
The function processes main effects and interaction terms, handling categorical factors and
continuous covariates differently. Other implementation notes:
- Converts all factors to strings for categorical treatment
- Uses FullDummyCoding for contrast encoding
- Validates numeric nature of trait and covariates
- Processes interaction terms between factors automatically
- Handles both categorical factors (as Boolean matrices) and continuous covariates (as Float64 matrices)
- Creates separate design matrices with and without base levels

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=500, l=1_000, verbose=false); 

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr, blr_ALL = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = nothing, verbose = false);

julia> blr2, blr_ALL2 = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = [trials.traits[2]], verbose = false);

julia> sum([length(x) for (_, x) in blr.coefficients]) < sum([length(x) for (_, x) in blr_ALL.coefficients])
true

julia> sum([length(x) for (_, x) in blr2.coefficients]) < sum([length(x) for (_, x) in blr_ALL2.coefficients])
true

julia> sum([length(x) for (_, x) in blr.coefficients]) ==  sum([length(x) for (_, x) in blr2.coefficients]) - 1
true

julia> sum([length(x) for (_, x) in blr_ALL.coefficients]) == sum([length(x) for (_, x) in blr_ALL2.coefficients]) - 1
true
```
"""
function instantiateblr(;
    trait::String,
    factors::Vector{String},
    df::DataFrame,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    verbose::Bool = false,
)::Tuple{BLR,BLR}
    # genomes = simulategenomes(n=500, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3);
    # trait = "trait_1"; factors = ["rows", "cols"]; df = tabularise(trials); other_covariates::Union{Vector{String}, Nothing} = ["trait_2", "trait_3"]; verbose = true;
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
        (concrete_term(term(trait), df[!, trait], ContinuousTerm) ~ term(1) + coefficients), coefficients
    else
        # f = term(trait) ~ term(1) + foldl(*, term.(factors)) + foldl(+, term.(other_covariates))
        coefficients =
            foldl(*, term.(factors)) +
            foldl(+, [concrete_term(term(c), df[!, c], ContinuousTerm) for c in other_covariates])
        (concrete_term(term(trait), df[!, trait], ContinuousTerm) ~ term(1) + coefficients), coefficients
    end
    # Extract the names of the coefficients and the design matrix (n x F(p-1); excludes the base level/s) to used for the regression
    if verbose
        println("Extracting the design matrix of the model: `$formula_struct`.")
    end
    _, coefficient_names = coefnames(apply_schema(formula_struct, schema(formula_struct, df)))
    y, X = modelcols(apply_schema(formula_struct, schema(formula_struct, df)), df)
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
    _, coefficient_names_ALL = coefnames(apply_schema(formula_struct, mf.schema))
    y_ALL, X_ALL = modelcols(apply_schema(formula_struct, mf.schema), df)
    # Make sure the extract ys and Xs are as expected
    if size(y) != size(y_ALL)
        throw(
            ErrorException(
                "The extracted response variable is not the same for the model to be fit and the full model.",
            ),
        )
    end
    if !(size(X) < size(X_ALL))
        throw(
            ErrorException(
                "The one-hot encoding explanatory matrix does not have more columns than the model matrix to be fit.",
            ),
        )
    end
    # Initialise the BLR struct
    blr, blr_ALL = begin
        n, p = size(X)
        blr = BLR(n = n, p = p)
        blr.entries = df.entries
        delete!(blr.Xs, "dummy")
        delete!(blr.Σs, "dummy")
        delete!(blr.coefficients, "dummy")
        delete!(blr.coefficient_names, "dummy")
        blr.y = y
        blr.ŷ = zeros(n)
        blr.ϵ = zeros(n)
        blr, clone(blr)
    end
    # Separate each explanatory variable from one another
    for v in explain_var_names
        # v = explain_var_names[end]
        v_split = filter(x -> x != "&", split(v, " "))
        # Main design matrices (bool) full design matrices (bool_ALL)
        bool = fill(true, length(coefficient_names))
        bool_ALL = fill(true, length(coefficient_names_ALL))
        for x in v_split
            bool = bool .&& .!isnothing.(match.(Regex(x), coefficient_names))
            bool_ALL = bool_ALL .&& .!isnothing.(match.(Regex(x), coefficient_names_ALL))
        end
        # Make sure we are extract the correct coefficients
        if length(v_split) == 1
            bool = bool .&& isnothing.(match.(Regex("&"), coefficient_names))
            bool_ALL = bool_ALL .&& isnothing.(match.(Regex("&"), coefficient_names_ALL))
        else
            m = length(v_split) - 1
            bool =
                bool .&& [
                    !isnothing(x) ? sum(.!isnothing.(match.(Regex("&"), split(coefficient_names[i], " ")))) == m :
                    false for (i, x) in enumerate(match.(Regex("&"), coefficient_names))
                ]
            bool_ALL =
                bool_ALL .&& [
                    !isnothing(x) ? sum(.!isnothing.(match.(Regex("&"), split(coefficient_names_ALL[i], " ")))) == m :
                    false for (i, x) in enumerate(match.(Regex("&"), coefficient_names_ALL))
                ]
        end
        v = if !isnothing(other_covariates) && (v ∈ other_covariates)
            # For the continuous numeric other covariates --> Float64 matrix
            # Rename the variance component to "other_covariates"
            v = "other_covariates"
            blr.Xs[v] = try
                hcat(blr.Xs[v], X[:, bool])
            catch
                X[:, bool]
            end
            blr_ALL.Xs[v] = try
                hcat(blr_ALL.Xs[v], X_ALL[:, bool_ALL])
            catch
                X_ALL[:, bool_ALL]
            end
            v
        else
            # For categorical variables --> boolean matrix for memory-efficiency
            blr.Xs[v] = Bool.(X[:, bool])
            blr_ALL.Xs[v] = Bool.(X_ALL[:, bool_ALL])
            v
        end
        blr.Σs[v] = 1.0 * I
        blr_ALL.Σs[v] = 1.0 * I
        blr.coefficients[v] = try
            vcat(blr.coefficients[v], zeros(sum(bool)))
        catch
            zeros(sum(bool))
        end
        blr_ALL.coefficients[v] = try
            vcat(blr_ALL.coefficients[v], zeros(sum(bool_ALL)))
        catch
            zeros(sum(bool_ALL))
        end
        blr.coefficient_names[v] = try
            vcat(blr.coefficient_names[v], coefficient_names[bool])
        catch
            coefficient_names[bool]
        end
        blr_ALL.coefficient_names[v] = try
            vcat(blr_ALL.coefficient_names[v], coefficient_names_ALL[bool_ALL])
        catch
            coefficient_names_ALL[bool_ALL]
        end
    end
    # Output
    if !checkdims(blr)
        throw(ErrorException("The resulting BLR struct for the model to be fit is corrupted ☹."))
    end
    if !checkdims(blr_ALL)
        throw(ErrorException("The resulting BLR struct for the full model is corrupted ☹."))
    end
    if blr == blr_ALL
        throw(
            ErrorException(
                "The BLR structs for the model to be fit and full model should not be the same but they are.",
            ),
        )
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
    vector_of_Xs_noint::Vector{Matrix{Union{Bool,Float64}}},
    vector_of_Δs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}},
    length_of_σs::Vector{Int64},
    y::Vector{Float64},
)
    # Set intercept prior.
    intercept ~ Normal(0.0, 10.0)
    # Set variance predictors
    p = size(vector_of_Xs_noint[1], 1)
    k = length(vector_of_Xs_noint)
    σ²s = [fill(0.0, length_of_σs[i]) for i = 1:k]
    βs = [fill(0.0, size(vector_of_Xs_noint[i], 2)) for i = 1:k]
    μ = fill(0.0, p) .+ intercept
    for i = 1:k
        σ²s[i] ~ filldist(Exponential(1.0), length(σ²s[i]))
        σ² = repeat(σ²s[i], 1 + length(βs[i]) - length(σ²s[i]))
        Σ = Diagonal(σ²) * vector_of_Δs[i] * Diagonal(σ²)
        βs[i] ~ MvNormal(zeros(length(βs[i])), Σ)
        μ += Float64.(vector_of_Xs_noint[i]) * βs[i]
    end
    # Residual variance
    σ² ~ Exponential(10.0)
    # Return the distribution of the response variable, from which the likelihood will be derived
    return y ~ Distributions.MvNormal(μ, σ² * I)
end

"""
    extractmodelinputs(blr_and_blr_ALL::Tuple{BLR,BLR}; multiple_σs::Union{Nothing, Dict{String, Bool}}=nothing)::Dict

Extract model inputs from BLR objects for Bayesian modeling.

# Arguments
- `blr_and_blr_ALL::Tuple{BLR,BLR}`: A tuple containing two BLR (Bayesian Linear Regression) objects
- `multiple_σs::Union{Nothing, Dict{String, Bool}}=nothing`: Optional dictionary specifying whether each variance component should use single (false) or multiple (true) variance scalers. If nothing, defaults to single scaler for all components.

# Returns
A dictionary containing:
- `"y"`: Response variable vector
- `"vector_of_Xs_noint"`: Vector of design matrices for each variance component
- `"vector_of_Δs"`: Vector of variance-covariance matrices or uniform scaling factors
- `"length_of_σs"`: Vector specifying number of variance components for each term
- `"variance_components"`: Vector of variance component names
- `"vector_coefficient_names"`: Vector of coefficient names for all variance components

# Notes
- Excludes intercept from variance components 
- For each variance component:
  - If multiple_σs[component] = false: Uses single variance scaler
  - If multiple_σs[component] = true: Uses separate variance scaler per coefficient
- Processes design matrices, variance-covariance structures, and coefficient names from the BLR objects

# Throws
- `ErrorException`: If a variance component is not specified in the multiple_σs dictionary

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase)
julia> genomes = simulategenomes(n=500, l=1_000, verbose=false); 

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr_and_blr_ALL = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);

julia> model_inputs_1 = extractmodelinputs(blr_and_blr_ALL, multiple_σs = nothing);

julia> model_inputs_2 = extractmodelinputs(blr_and_blr_ALL, multiple_σs = Dict("rows" => true, "cols" => true, "rows & cols" => false, "other_covariates" => true));

julia> (model_inputs_1["y"] == model_inputs_2["y"]) && (model_inputs_1["vector_of_Xs_noint"] == model_inputs_2["vector_of_Xs_noint"]) && (model_inputs_1["vector_of_Δs"] == model_inputs_2["vector_of_Δs"]) && (model_inputs_1["vector_coefficient_names"] == model_inputs_2["vector_coefficient_names"])
true

julia> sum(model_inputs_1["length_of_σs"]) < sum(model_inputs_2["length_of_σs"])
true
```
"""
function extractmodelinputs(
    blr_and_blr_ALL::Tuple{BLR,BLR};
    multiple_σs::Union{Nothing,Dict{String,Bool}} = nothing,
)::Dict{
    String,
    Union{
        Vector{Float64},
        Vector{Matrix{Union{Bool,Float64}}},
        Vector{Union{Matrix{Float64},UniformScaling{Float64}}},
        Vector{Int64},
        Vector{String},
    },
}
    # Extract the response variable, y, and the vectors of Xs, and Σs for model fitting, as well as the coefficient names for each variance component
    y::Vector{Float64} = blr_and_blr_ALL[1].y
    vector_of_Xs_noint::Vector{Matrix{Union{Bool,Float64}}} = []
    vector_of_Δs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}} = []
    length_of_σs::Vector{Int64} = []
    vector_coefficient_names::Vector{String} = []
    variance_components::Vector{String} = filter(x -> x != "intercept", string.(keys(blr_and_blr_ALL[1].Xs)))
    multiple_σs = if isnothing(multiple_σs)
        multiple_σs = Dict()
        for v in variance_components
            multiple_σs[v] = false
        end
        multiple_σs
    else
        multiple_σs
    end
    for v in variance_components
        if !(v ∈ string.(keys(multiple_σs)))
            throw(
                ErrorException(
                    "The variance component: $v is not in the dictionary specifying if each variance component should be estimated with a single or multiple variance scaler, `multiple_σs`.",
                ),
            )
        end
    end
    for v in variance_components
        # v = variance_components[1]
        push!(vector_of_Xs_noint, blr_and_blr_ALL[1].Xs[v])
        push!(vector_of_Δs, blr_and_blr_ALL[1].Σs[v])
        # if isa(blr_and_blr_ALL[1].Σs[v], UniformScaling{Float64})
        if !multiple_σs[v]
            # If only 1 variance component is requested, i.e. common/spherical variance
            push!(length_of_σs, 1)
        else
            # Otherwise, define separate variance scaler per coefficient
            push!(length_of_σs, size(blr_and_blr_ALL[1].Xs[v], 2))
        end
        push!(length_of_σs)
        vector_coefficient_names = vcat(vector_coefficient_names, blr_and_blr_ALL[1].coefficient_names[v])
    end
    Dict(
        "y" => y,
        "vector_of_Xs_noint" => vector_of_Xs_noint,
        "vector_of_Δs" => vector_of_Δs,
        "length_of_σs" => length_of_σs,
        "vector_coefficient_names" => vector_coefficient_names,
        "variance_components" => variance_components,
    )
end

"""
    turingblrmcmc!(
        blr_and_blr_ALL::Tuple{BLR,BLR};
        multiple_σs::Union{Nothing, Dict{String, Bool}} = nothing,
        turing_model::Function = turingblr,
        n_iter::Int64 = 10_000,
        n_burnin::Int64 = 1_000,
        δ::Float64 = 0.65,
        max_depth::Int64 = 5,
        Δ_max::Float64 = 1000.0,
        init_ϵ::Float64 = 0.2,
        adtype::AutoReverseDiff = AutoReverseDiff(compile = true),
        seed::Int64 = 1234,
        verbose::Bool = true
    )::Nothing

Perform MCMC sampling for Bayesian Linear Regression on a tuple of BLR models using NUTS sampler.

# Arguments
- `blr_and_blr_ALL::Tuple{BLR,BLR}`: Tuple containing two BLR models - first for fitting, second for inferring
- `multiple_σs::Union{Nothing, Dict{String, Bool}}`: Optional dictionary specifying multiple (true) or single (false) variance scalers per component
- `turing_model::Function`: The Turing model function to use (default: turingblr)
- `n_iter::Int64`: Number of MCMC iterations (default: 10,000)
- `n_burnin::Int64`: Number of burn-in iterations to discard (default: 1,000) 
- `δ::Float64`: Target acceptance rate for dual averaging (default: 0.65)
- `max_depth::Int64`: Maximum doubling tree depth (default: 5)
- `Δ_max::Float64`: Maximum divergence during doubling tree (default: 1000.0)
- `init_ϵ::Float64`: Initial step size; 0 means auto-search using heuristics (default: 0.2)
- `adtype::AutoReverseDiff`: Automatic differentiation type (default: AutoReverseDiff(compile = true))
- `seed::Int64`: Random seed for reproducibility (default: 1234)
- `verbose::Bool`: Whether to show progress during sampling (default: true)

# Returns
- `Nothing`: Updates the input BLR models in-place

# Notes
- Performs model validation checks before fitting
- Uses NUTS (No-U-Turn Sampler) with specified AD type
- Computes convergence diagnostics using Gelman-Rubin statistics (R̂)
- Warns if parameters haven't converged (|1.0 - R̂| > 0.1)
- Updates both models with estimated coefficients, variance components, predicted values, and residuals
- Ensures consistency between fitted and full models

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase)
julia> genomes = simulategenomes(n=500, l=1_000, verbose=false); 

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr_and_blr_ALL = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);

julia> turingblrmcmc!(blr_and_blr_ALL, n_iter=1_000, n_burnin=200, seed=123, verbose=false);

julia> mean([mean(x) for (_, x) in blr_and_blr_ALL[1].coefficients]) != 0.0
true

julia> mean([mean(x) for (_, x) in blr_and_blr_ALL[2].coefficients]) != 0.0
true

julia> cor(blr_and_blr_ALL[1].y, blr_and_blr_ALL[1].ŷ) > 0.0
true

julia> cor(blr_and_blr_ALL[1].ŷ, blr_and_blr_ALL[2].ŷ) > 0.99
true
```
"""
function turingblrmcmc!(
    blr_and_blr_ALL::Tuple{BLR,BLR};
    multiple_σs::Union{Nothing,Dict{String,Bool}} = nothing,
    turing_model::Function = turingblr,
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    δ::Float64 = 0.65, #  Target acceptance rate for dual averaging
    max_depth::Int64 = 5, # Maximum doubling tree depth
    Δ_max::Float64 = 1000.0, # Maximum divergence during doubling tree
    init_ϵ::Float64 = 0.2, # Initial step size; 0 means automatically searching using a heuristic procedure
    adtype::AutoReverseDiff = AutoReverseDiff(compile = true),
    seed::Int64 = 1234,
    verbose::Bool = true,
)::Nothing
    # genomes = simulategenomes(n=500, l=1_000, verbose=false); 
    # trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);
    # df = tabularise(trials);
    # blr_and_blr_ALL = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);
    # # multiple_σs::Union{Nothing, Dict{String, Bool}} = nothing
    # multiple_σs::Union{Nothing, Dict{String, Bool}} = Dict("rows" => true, "cols" => true, "rows & cols" => false, "other_covariates" => true)
    # turing_model = turingblr 
    # n_iter = 1_000
    # n_burnin = 500
    # δ = 0.65
    # max_depth = 5
    # Δ_max = 1000.0
    # init_ϵ = 0.2
    # adtype = AutoReverseDiff(compile = true)
    # seed = 1234
    # verbose = true
    # Check arguments
    if !checkdims(blr_and_blr_ALL[1])
        throw(ArgumentError("The BLR struct to be fit is corrupted ☹."))
    end
    if !checkdims(blr_and_blr_ALL[2])
        throw(ArgumentError("The BLR struct of the full model is corrupted ☹."))
    end
    if blr_and_blr_ALL[1].entries != blr_and_blr_ALL[2].entries
        throw(ArgumentError("The BLR structs have incompatible entries."))
    end
    if keys(blr_and_blr_ALL[1].coefficient_names) != keys(blr_and_blr_ALL[2].coefficient_names)
        throw(ArgumentError("The BLR structs have incompatible variance components."))
    end
    if n_iter < 100
        throw(
            ArgumentError(
                "The number of MCMC iterations (`n_iter=$n_iter`) is less than 100. Please be reasonable and add more iterations or else your model will likely not converge.",
            ),
        )
    end
    if n_burnin > n_iter
        throw(
            ArgumentError(
                "The number of burn-in stepes (`n_burnin=$n_burnin`) is greater than or equal to the number of MCMC iterations (`n_iter=$n_iter`).",
            ),
        )
    end
    # # Extract the response variable, y, and the vectors of Xs, and Σs for model fitting, as well as the coefficient names for each variance component
    model_inputs = extractmodelinputs(blr_and_blr_ALL, multiple_σs = multiple_σs)
    # Instantiate the RNG, model, and sampling function
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turing_model(
        model_inputs["vector_of_Xs_noint"],
        model_inputs["vector_of_Δs"],
        model_inputs["length_of_σs"],
        model_inputs["y"],
    )
    sampling_function = NUTS(n_burnin, δ, max_depth = max_depth, Δ_max = Δ_max, init_ϵ = init_ϵ; adtype = adtype)
    # MCMC
    if verbose
        println(
            "Single chain MCMC sampling for $n_iter total iterations where the first $n_burnin iteractions are omitted.",
        )
    end
    chain = Turing.sample(rng, model, sampling_function, n_iter, discard_initial = n_burnin, progress = verbose)
    # Diagnostics
    if verbose
        println("Diagnosing the MCMC chain for convergence by dividing the chain into 5 and finding the maximum R̂.")
    end
    R̂ = DataFrame(MCMCDiagnosticTools.rhat(chain, kind = :rank, split_chains = 5)) # new R̂: maximum R̂ of :bulk and :tail
    n_params_which_may_not_have_converged = sum(abs.(1.0 .- R̂.rhat) .> 0.1)
    if verbose
        if n_params_which_may_not_have_converged > 0
            @warn "There are $n_params_which_may_not_have_converged parameters (out of $(nrow(R̂)) total parameters) which may not have converged. Please consider increasing the number of iterations which is currently at $n_iter."
        else
            println("All parameters have converged.")
        end
    end
    # Use the mean parameter values which excludes the first n_burnin iterations
    params = Turing.get_params(chain[:, :, :])
    β0 = mean(params.intercept)
    βs = [mean(params.βs[i]) for i in eachindex(params.βs)]
    σ² = mean(params.σ²)
    # TODO: parse the σ²s properly when there are vectors
    σ²s = [mean(params.σ²s[i]) for i in eachindex(params.σ²s)]
    # Extract coefficients
    if verbose
        println("Extracting MCMC-derived coefficients")
    end
    blr_and_blr_ALL[1].coefficients["intercept"] = blr_and_blr_ALL[2].coefficients["intercept"] = [β0]
    blr_and_blr_ALL[1].Σs["σ²"] = blr_and_blr_ALL[2].Σs["σ²"] = σ² * blr_and_blr_ALL[1].Σs["σ²"]
    ini = 0
    ini_σ = 0
    fin = 0
    fin_σ = 0
    for (i, v) in enumerate(model_inputs["variance_components"])
        # i = 1; v = model_inputs["variance_components"][i];
        ini = fin + 1
        fin = (ini - 1) + size(model_inputs["vector_of_Xs_noint"][i], 2)
        ini_σ = fin_σ + 1
        fin_σ = (ini_σ - 1) + model_inputs["length_of_σs"][i]
        # Model fit
        if blr_and_blr_ALL[1].coefficient_names[v] != model_inputs["vector_coefficient_names"][ini:fin]
            throw(ErrorException("The expected coefficient names do not match for the variance component: $v."))
        end
        blr_and_blr_ALL[1].coefficients[v] = βs[ini:fin]
        blr_and_blr_ALL[1].Σs[v] =
            if isa(model_inputs["vector_of_Δs"][i], UniformScaling{Float64}) && (fin_σ - ini_σ == 0)
                σ²s[ini_σ] * model_inputs["vector_of_Δs"][i]
            else
                σ² = repeat(σ²s[ini_σ:fin_σ], 1 + length(blr_and_blr_ALL[1].coefficients[v]) - length(σ²s[ini_σ:fin_σ]))
                Matrix(Diagonal(σ²) * model_inputs["vector_of_Δs"][i] * Diagonal(σ²))
            end
        # Full model
        for (j, c) in enumerate(blr_and_blr_ALL[1].coefficient_names[v])
            # j = 29; c = blr_and_blr_ALL[1].coefficient_names[v][j]
            idx = findall(blr_and_blr_ALL[2].coefficient_names[v] .== c)[1]
            blr_and_blr_ALL[2].coefficients[v][idx] = blr_and_blr_ALL[1].coefficients[v][j]
        end
        blr_and_blr_ALL[2].Σs[v] = blr_and_blr_ALL[1].Σs[v]
    end
    # Define the predicted ys and residuals
    X, b, _b_labels = extractXb(blr_and_blr_ALL[1])
    blr_and_blr_ALL[1].ŷ = X * b
    blr_and_blr_ALL[1].ϵ = blr_and_blr_ALL[1].y .- blr_and_blr_ALL[1].ŷ
    X, b, _b_labels = extractXb(blr_and_blr_ALL[2])
    blr_and_blr_ALL[2].ŷ = X * b
    blr_and_blr_ALL[2].ϵ = blr_and_blr_ALL[2].y .- blr_and_blr_ALL[2].ŷ
    # Output checks
    if !checkdims(blr_and_blr_ALL[1]) || !checkdims(blr_and_blr_ALL[2])
        throw(ErrorException("Error updating the BLR structs after MCMC."))
    end
    if (mean(abs.(blr_and_blr_ALL[1].ŷ - blr_and_blr_ALL[2].ŷ)) > 1e-7) ||
       (mean(abs.(blr_and_blr_ALL[1].ϵ - blr_and_blr_ALL[2].ϵ)) > 1e-7)
        throw(ErrorException("The BLR struts do not yield the same ŷ and ϵ. After fitting and updating the fields."))
    end
    nothing
end

"""
    removespatialeffects(; df::DataFrame, factors::Vector{String}, traits::Vector{String}, 
                        other_covariates::Union{Vector{String},Nothing} = nothing,
                        autoregressive_Σ::Bool = true,
                        ρs::Dict{String, Float64} = Dict("rows" => 0.5, "cols" => 0.5),
                        n_iter::Int64 = 10_000, n_burnin::Int64 = 1_000, 
                        seed::Int64 = 1234, verbose::Bool = false)::Tuple{DataFrame, Vector{String}}

Remove spatial effects from trait measurements in field trials using Bayesian linear regression.

# Arguments
- `df::DataFrame`: DataFrame containing trial data with columns for traits, spatial factors, and harvests
- `factors::Vector{String}`: Vector of factor names to be considered in the model 
- `traits::Vector{String}`: Vector of trait names to be spatially adjusted
- `other_covariates::Union{Vector{String},Nothing}`: Optional vector of additional covariates
- `autoregressive_Σ::Bool`: Whether to use autoregressive covariance structure for spatial factors (default: true)
- `ρs::Dict{String, Float64}`: Correlation parameters for autoregressive structure, keys are factor names (default: rows=0.5, cols=0.5)
- `n_iter::Int64`: Number of MCMC iterations (default: 10_000)
- `n_burnin::Int64`: Number of burn-in iterations (default: 1_000) 
- `seed::Int64`: Random seed for reproducibility (default: 1234)
- `verbose::Bool`: If true, prints detailed information (default: false)

# Returns
A tuple containing:
1. Modified DataFrame with new spatially adjusted traits (prefix "SPATADJ-")  
2. Updated factors vector with spatial factors removed

# Details
This function performs spatial adjustment for traits measured in field trials by:

1. Identifying spatial factors (blocks, rows, columns) and creating design matrices
2. Fitting Bayesian linear model per harvest to account for:
   - Spatial effects (blocks, rows, columns and interactions) 
   - Autoregressive covariance structure for spatial dependence if enabled
   - Additional numeric covariates if specified
3. Creating spatially adjusted traits by adding intercept and residuals
4. Removing redundant factors and retaining only unique design matrices

The spatial adjustment is only performed if blocks, rows or columns are present.
Each harvest is treated separately to allow for year and site-specific spatial effects.

Variance component scalers are specified as follows:
- Rows, columns and other covariates use unique variance scalers (multiple_σs = true)
- Row-by-column interactions use a single spherical variance-covariance matrix (multiple_σs = false)
- This improves model tractability while allowing for flexible variance structures where needed

# Notes
- Requires "entries" and "harvests" columns in input DataFrame 
- Uses Bayesian linear regression via MCMC for spatial modeling
- Creates new columns with "SPATADJ-" prefix for adjusted traits
- Returns original DataFrame if no spatial factors present
- Automatically detects spatial factors via regex pattern "blocks|rows|cols"
- Supports autoregressive correlation structure for ordered spatial factors

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase)
julia> genomes = simulategenomes(n=500, l=1_000, verbose=false); 

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> df, factors = removespatialeffects(df = df, factors = ["rows", "cols"], traits = ["trait_1", "trait_2"], other_covariates=["trait_3"], n_iter=1_000, n_burnin=200, verbose = false);

julia> ("SPATADJ-trait_1" ∈ names(df)) && ("SPATADJ-trait_2" ∈ names(df))
true
```
"""
function removespatialeffects(;
    df::DataFrame,
    factors::Vector{String},
    traits::Vector{String},
    other_covariates::Union{Vector{String},Nothing} = nothing,
    autoregressive_Σ::Bool = true,
    ρs::Dict{String, Float64} = Dict("rows" => 0.5, "cols" => 0.5),
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose::Bool = false,
)::Tuple{DataFrame,Vector{String}}
    # genomes = simulategenomes(n=500, l=1_000, verbose=false); 
    # trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);
    # df = tabularise(trials);
    # factors = ["rows", "cols"]; traits = ["trait_1", "trait_2"]; other_covariates=["trait_3"]; autoregressive_Σ::Bool = true; ρs::Dict{String, Float64} = Dict("rows" => 0.5, "cols" => 0.5); n_iter=1_000; n_burnin=200; seed=123;  verbose = false;
    # Check arguments
    if !(("blocks" ∈ factors) || ("rows" ∈ factors) || ("cols" ∈ factors))
        # No spatial adjustment possible/required
        @warn "Spatial adjustment is not required as there are no blocks, rows or columns in the input data frame."
        return (df, factors)
    end
    if !("entries" ∈ names(df))
        throw(ArgumentError("There is no `entries` in the input data frame (`df`)."))
    end
    if (size(df, 1) == 0) || (size(df, 2) == 0)
        throw(ArgumentError("The input data frame is empty."))
    end
    for f in factors
        if !(f ∈ names(df))
            throw(ArgumentError("The input data frame does not include the factor: `$f`."))
        end
    end
    for t in traits
        if !(t ∈ names(df))
            throw(ArgumentError("The input data frame does not include the trait: `$t`."))
        end
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
    if autoregressive_Σ
        for (k, _) in ρs
            if !(k ∈ names(df))
                throw(ArgumentError("The input data frame does not include the factor: `$k`."))
            end
        end
    end
    # Make sure the "__new_spatially_adjusted_trait__" for adding new spatially adjusted traits does not exist
    if sum(names(df) .== "__new_spatially_adjusted_trait__") > 0
        throw(
            ErrorException(
                "Please rename the trait `__new_spatially_adjusted_trait__` in the Trials table to allow us to define spatially adjusted traits.",
            ),
        )
    end
    # Define spatial factors
    spatial_factors = filter(x -> !isnothing(match(Regex("blocks|rows|cols"), x)), factors)
    # Make sure that each harvest is year- and site-specific
    for harvest in unique(df.harvests)
        # harvest = unique(df.harvests)[1]
        idx_rows = findall(df.harvests .== harvest)
        if length(idx_rows) == 0
            continue
        end
        df_sub = df[idx_rows, :]
        for i in eachindex(traits)
            # i = 1
            trait = traits[i]
            if verbose
                println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                println("Spatial modelling for harvest: $harvest; and trait: $trait")
                println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            end
            # Add spatially adjusted trait to df
            new_spat_adj_trait_name = string("SPATADJ-", trait)
            if sum(names(df) .== new_spat_adj_trait_name) == 0
                df[:, "__new_spatially_adjusted_trait__"] .= df[:, trait]
                rename!(df, "__new_spatially_adjusted_trait__" => new_spat_adj_trait_name)
            end
            # Instantiate the BLR struct for spatial analysis to remove the spatial effects as well as the effects of the other covariates
            blr_and_blr_ALL = instantiateblr(
                trait = trait,
                factors = spatial_factors,
                other_covariates = other_covariates,
                df = df_sub,
                verbose = verbose,
            )
            # Define autoregressive variance-covariance matrix for the spatial factors
            # We assume that the coefficient names of the factor which will have an autoregressive variance-covariance matrix
            # can be sorted sensibly, e.g. names like "rows_01", "rows_02", "rows_03" to "row_99" can be sorted to get the order.
            if autoregressive_Σ
                for varcomp in string.(keys(ρs))
                    # varcomp = string.(keys(ρs))[1]
                    idx = sortperm(blr_and_blr_ALL[1].coefficient_names[varcomp])
                    p = length(idx)
                    AR1 = Matrix(Diagonal(ones(p)))
                    for i in idx
                        for j in idx
                            # i = idx[1]; j = idx[2];
                            if i == j
                                continue
                            end
                            # Farther factor levels less correlated
                            AR1[i, j] = ρs[varcomp]^abs(i-j)
                        end
                    end
                    inflatediagonals!(AR1)
                    det(AR1)
                    blr_and_blr_ALL[1].Σs[varcomp] = AR1
                end
            end
            # Set-up variance component multipliers/scalers such that row, columns and other covariates have unique multipliers;
            # while the row-by-col interaction only has 1, i.e. spherical variance-covariance matrix for model tractability
            multiple_σs::Union{Nothing,Dict{String,Bool}} = Dict()
            for varcomp in string.(keys(blr_and_blr_ALL[1].Σs))
                # varcomp = string.(keys(blr_and_blr_ALL[1].Σs))[1]
                if varcomp == "σ²"
                    continue
                end
                if (varcomp ∈ spatial_factors) || (varcomp == "other_covariates")
                    multiple_σs[varcomp] = true
                else
                    multiple_σs[varcomp] = false
                end
            end
            # Spatial analysis via Bayesian linear regression
            turingblrmcmc!(
                blr_and_blr_ALL,
                multiple_σs = multiple_σs,
                n_iter = n_iter,
                n_burnin = n_burnin,
                seed = seed,
                verbose = verbose,
            )
            # Update the spatially adjusted trait with the intercept + residuals of the spatial model above
            # cor(df[idx_rows, new_spat_adj_trait_name], blr_and_blr_ALL[1].coefficients["intercept"] .+ blr_and_blr_ALL[1].ϵ)
            df[idx_rows, new_spat_adj_trait_name] = blr_and_blr_ALL[1].coefficients["intercept"] .+ blr_and_blr_ALL[1].ϵ
        end
    end
    # Is the entries the only factor remaining?
    if length(factors) == 1
        return (df, factors)
    end
    # Factors with more than 1 level after correcting for spatial effects other than entries
    factors_out = ["entries"]
    Xs = [Bool.(modelmatrix(term("entries"), df))]
    for i in findall([x != "entries" for x in factors])
        # i = findall([x != "entries" for x in factors])[2]
        if (factors[i] ∈ spatial_factors) || (length(unique(df[:, factors[i]])) == 1)
            continue
        end
        xi = Bool.(modelmatrix(term(factors[i]), df))
        matched::Bool = false
        for j in eachindex(Xs)
            # j = 1
            if Xs[j] == xi
                matched = true
                break
            end
        end
        if !matched
            push!(Xs, xi)
            push!(factors_out, factors[i])
        end
    end
    # Output
    (df, factors_out)
end

# TODO: Add flexibility to specify whether to use one or multiple variance scalers in the GxE model
function analyse(
    trials::Trials,
    traits::Vector{String};
    grm::Union{GRM,Nothing} = nothing,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose::Bool = false,
)::TEBV
    # genomes = simulategenomes(n=500, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=2, n_replications=3); grm::Union{GRM, Nothing} = grmploidyaware(genomes; ploidy = 2, max_iter = 10, verbose = true); traits = ["trait_1"]; other_covariates::Union{Vector{String}, Nothing} = ["trait_2"]; n_iter::Int64 = 1_000; n_burnin::Int64 = 100; seed::Int64 = 1234; verbose::Bool = true;
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
    # Make sure the harvests are year- and site-specific
    df.harvests = string.(df.years, "|", df.sites, "|", df.harvests)
    # Identify non-fixed factors
    factors_all::Vector{String} = ["years", "seasons", "sites", "harvests", "blocks", "rows", "cols", "entries"]
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
    total_system_RAM_in_GB = Sys.free_memory() / (1024^3)
    if verbose && (total_X_size_in_Gb > 0.9 * total_system_RAM_in_GB)
        @warn "The size of the design matrix is ~$(round(total_X_size_in_Gb)) GB. This may cause out-of-memory errors."
    end
    # Spatial analyses per harvest-site-year
    # This is to prevent OOM errors, we will perform spatial analyses per harvest per site per year, i.e. remove spatial effects per harvest-site-year
    # as well as remove the effects of continuous numeric covariate/s.
    df, factors = removespatialeffects(
        df = df,
        factors = factors,
        traits = traits,
        other_covariates = other_covariates,
        n_iter = n_iter,
        n_burnin = n_burnin,
        seed = seed,
        verbose = verbose,
    )
    # GxE modelling excluding the effects of spatial factors and continuous covariates
    vector_of_BLRs::Vector{BLR} = []
    non_traits = vcat("id", "replications", "populations", factors_all)
    traits = filter(x -> !(x ∈ non_traits), names(df)) # includes spatially adjusted and non-spatially adjusted traits
    for (i, trait) in enumerate(traits)
        # i = length(traits); trait = traits[i];
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("GxE modelling for trait: $trait")
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        end
        # Instantiate the BLR struct for GxE analysis
        # Note that the covariate is now excluded as we should have controlled for them in the per harvest-site-year spatial analyses
        blr_and_blr_ALL =
            instantiateblr(trait = trait, factors = factors, other_covariates = nothing, df = df, verbose = verbose)


        # Proper variance partitioning
        # Prepare the variance-covariance matrix for the entries effects, i.e. using I or a GRM
        if !isnothing(grm)
            for idx in 1:2
                # idx = 1
                # Replace the variance-covariance matrix for the factors involving entries with the GRM
                factors_with_entries = begin
                    x = string.(keys(blr_and_blr_ALL[idx].coefficient_names))
                    x[.!isnothing.(match.(Regex("entries"), x))]
                end
                for f in factors_with_entries
                    # f = factors_with_entries[1]
                    n = length(blr_and_blr_ALL[idx].coefficient_names[f])
                    blr_and_blr_ALL[idx].Σs[f] = zeros(n, n)
                    for (i, entry_1) in enumerate(blr_and_blr_ALL[idx].coefficient_names[f])
                        for (j, entry_2) in enumerate(blr_and_blr_ALL[idx].coefficient_names[f])
                            # i = 1; entry_1 = blr_and_blr_ALL[idx].coefficient_names[f][i]
                            # j = 3; entry_2 = blr_and_blr_ALL[idx].coefficient_names[f][j]
                            k = findall(grm.entries .== join(split(entry_1, ": ")[2:end], ": "))[idx]
                            l = findall(grm.entries .== join(split(entry_2, ": ")[2:end], ": "))[idx]
                            blr_and_blr_ALL[idx].Σs[f][i, j] = grm.genomic_relationship_matrix[k, l]
                        end
                    end
                end
            end
        else
            # Keep the identity matrix for the entries variance-covariance matrix
            nothing
        end



        # TODO: Set-up variance component multipliers/scalers, for now it's just based on a maximumnumber of coefficients thresholds
        multiple_σs::Union{Nothing,Dict{String,Bool}} = Dict()
        for v in factors
            multiple_σs[v] = if size(blr_and_blr_ALL[1].Xs[v], 2) > 100
                false
            else
                true
            end
        end




        # GxE analysis via Bayesian linear regression
        turingblrmcmc!(blr_and_blr_ALL, n_iter = n_iter, n_burnin = n_burnin, seed = seed, verbose = verbose)
        # blr_and_blr_ALL[1].Xs
        # blr_and_blr_ALL[1].Σs
        # blr_and_blr_ALL[1].coefficients
        # blr_and_blr_ALL[1].coefficient_names["other_covariates"]
        # blr_and_blr_ALL[1].y
        # blr_and_blr_ALL[1].ŷ
        # blr_and_blr_ALL[1].ϵ
        # Collect the full BLR model, i.e. one-hot encoding which includes all factor levels (and not the df-1 model or the model that was fitted)
        push!(vector_of_BLRs, blr_and_blr_ALL[2])
    end

end
