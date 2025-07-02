"""
    checkandfocalterms(trait::String, factors::Vector{String}, df::DataFrame, other_covariates::Union{Vector{String},Nothing} = nothing)::Vector{String}

Validate input data and generate model terms for Bayesian analysis of field trials.

# Arguments
- `trait::String`: Name of the response variable (trait) column in the DataFrame
- `factors::Vector{String}`: Vector of factor names (categorical variables) to include in the model
- `df::DataFrame`: DataFrame containing the trial data
- `other_covariates::Union{Vector{String},Nothing}`: Optional vector of numeric covariate column names

# Returns
- `Vector{String}`: A vector of model terms including main effects, GxE interaction effects, and spatial effects

# Details
The function performs several tasks:
1. Validates that all specified columns exist in the DataFrame
2. Ensures trait and covariates are numeric
3. Converts factors to strings
4. Generates appropriate model terms based on available factors:
   - Main effects (entries, sites, seasons, years)
   - GxE interaction effects (various combinations of entries × environmental factors)
   - Spatial effects (blocks, rows, columns and their interactions)

# Throws
- `ArgumentError`: If trait or factors are not found in DataFrame
- `ArgumentError`: If trait or covariates are non-numeric or contain missing values

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase, DataFrames)
julia> genomes = simulategenomes(n=5, l=1_000, verbose=false);

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> focal_terms_1 = checkandfocalterms(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end])
2-element Vector{String}:
 "rows"
 "cols"

julia> focal_terms_2 = checkandfocalterms(trait = trials.traits[1], factors = ["years", "seasons", "sites"], df = df)
5-element Vector{String}:
 "years"
 "seasons"
 "sites"
 "years:sites"
 "seasons:sites"
```
"""
function checkandfocalterms(;
    trait::String,
    factors::Vector{String},
    df::DataFrame,
    other_covariates::Union{Vector{String},Nothing} = nothing,
)::Vector{String}
    # genomes = simulategenomes(n=500, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3);
    # trait = "trait_1"; factors = ["rows", "cols"]; df = tabularise(trials); other_covariates::Union{Vector{String}, Nothing} = ["trait_2", "trait_3"]; saturated_model = false; verbose = true;
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
    # Main effects and interaction terms we are most interested in fitting
    main_effects = filter(x -> x ∈ ["entries", "sites", "seasons", "years"], factors)
    # Interaction effects between environmental components and the entries
    exe_gxe_effects = if ("years" ∈ factors) && ("seasons" ∈ factors) && ("sites" ∈ factors)
        # Excludes other interaction effects with years for parsimony
        ["years:sites", "seasons:sites", "entries:seasons:sites"]
    elseif ("years" ∈ factors) && ("seasons" ∈ factors) && !("sites" ∈ factors)
        ["years:seasons", "entries:seasons"]
    elseif ("years" ∈ factors) && !("seasons" ∈ factors) && ("sites" ∈ factors)
        ["years:sites", "entries:sites"]
    elseif !("years" ∈ factors) && ("seasons" ∈ factors) && ("sites" ∈ factors)
        ["seasons:sites", "entries:seasons:sites"]
    elseif ("years" ∈ factors) && !("seasons" ∈ factors) && !("sites" ∈ factors)
        # Only years-by-entries interaction effects
        ["entries:years"]
    elseif !("years" ∈ factors) && ("seasons" ∈ factors) && !("sites" ∈ factors)
        # Only seasons-by-entries interaction effects
        ["entries:seasons"]
    elseif !("years" ∈ factors) && !("seasons" ∈ factors) && ("sites" ∈ factors)
        # Only sites-by-entries interaction effects
        ["entries:sites"]
    else
        nothing
    end
    # Site-specific spatial effects per harvest (i.e. stage-1 effects per year-site-season-harvest combination)
    spatial_effects = begin
        spatial_terms = if ("rows" ∈ factors) && ("cols" ∈ factors)
            # Regardless of whether or not the blocks are present, use only the rows and columns for parsimony
            ["rows", "cols"]
        elseif ("blocks" ∈ factors) && ("rows" ∈ factors) && !("cols" ∈ factors)
            # Check if the blocks and rows are unique, use the non-fixed one if not, else use both plus their interaction
            br = unique(string(df.blocks, "\t", df.rows))
            b = unique(string(df.blocks))
            r = unique(string(df.rows))
            if length(br) == length(b)
                ["blocks"]
            elseif length(br) == length(r)
                ["rows"]
            else
                ["blocks", "rows"]
            end
        elseif ("blocks" ∈ factors) && !("rows" ∈ factors) && ("cols" ∈ factors)
            # Check if the blocks and cols are unique, use the non-fixed one if not, else use both plus their interaction
            bc = unique(string(df.blocks, "\t", df.cols))
            b = unique(string(df.blocks))
            c = unique(string(df.cols))
            if length(bc) == length(b)
                ["blocks"]
            elseif length(bc) == length(c)
                ["cols"]
            else
                ["blocks", "cols"]
            end
        elseif ("blocks" ∈ factors) && !("rows" ∈ factors) && !("cols" ∈ factors)
            ["blocks"]
        elseif !("blocks" ∈ factors) && ("rows" ∈ factors) && !("cols" ∈ factors)
            ["rows"]
        elseif !("blocks" ∈ factors) && !("rows" ∈ factors) && ("cols" ∈ factors)
            ["cols"]
        else
            nothing
        end
        # Nest the spatial terms within years and/or seasons and/or sites
        nesters = if length(main_effects) > 0
            join(filter(x -> isnothing(match(Regex("entries"), x)), main_effects), ":")
        else
            []
        end
        spatial_effects = if (length(nesters) > 0) && !isnothing(spatial_terms)
            nesters .* ":" .* spatial_terms
        else
            spatial_terms
        end
        spatial_effects
    end
    # Filter the focal terms
    focal_terms = []
    for t in vcat(main_effects, exe_gxe_effects, spatial_effects)
        if isnothing(t)
            continue
        end
        ts = split(t, ":")
        if length(intersect(ts, factors)) == length(ts)
            push!(focal_terms, t)
        end
    end
    # Output
    focal_terms
end


"""
    instantiateblr(; trait::String, factors::Vector{String}, df::DataFrame, 
                  other_covariates::Union{Vector{String}, Nothing}=nothing,
                  saturated_model::Bool=false,
                  verbose::Bool=false)::BLR

Extract design matrices and response variable for Bayesian modelling of factorial experiments.

# Arguments
- `trait::String`: Name of the response variable (dependent variable) in the DataFrame
- `factors::Vector{String}`: Vector of factor names (independent variables) to be included in the model 
- `df::DataFrame`: DataFrame containing the data
- `other_covariates::Union{Vector{String}, Nothing}=nothing`: Additional numeric covariates to include in the model
- `saturated_model::Bool=false`: If true, includes all possible interactions between factors
- `verbose::Bool=false`: If true, prints additional information during execution

# Returns
A BLR struct containing:
- `entries`: Vector of entry identifiers 
- `y`: Response variable vector
- `ŷ`: Predicted values vector
- `ϵ`: Residuals vector
- `Xs`: Dict mapping factors to design matrices 
- `Σs`: Dict mapping factors to covariance matrices
- `coefficients`: Dict mapping factors to coefficient vectors
- `coefficient_names`: Dict mapping factors to coefficient names
- `diagnostics`: DataFrame with MCMC diagnostics (added after sampling)

# Model Details
Creates design matrices for hierarchical factorial experiments with:

1. Main effects:
- Genetic effects: `entries`
- Environmental effects: `sites`, `seasons`, `years`
- Spatial effects: `blocks`, `rows`, `cols`

2. Interaction effects:
- GxE interactions (entries × environment)
- Environmental interactions (between environment factors) 
- Spatial interactions (between spatial factors)

# Implementation Notes
- Uses FullDummyCoding for categorical factors
- Converts factors to strings
- Validates numeric traits/covariates
- Creates design matrices with full levels
- Handles continuous covariates as Float64
- Uses memory efficient boolean matrices for factors
- Assigns identity matrices as initial covariance structures

# Throws
- `ArgumentError`: For missing/invalid inputs
- `ErrorException`: For matrix extraction failures

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=5, l=1_000, verbose=false);

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr_1 = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = nothing, verbose = false);

julia> blr_2 = instantiateblr(trait = trials.traits[1], factors = ["years", "seasons", "sites", "entries"], df = df, other_covariates = [trials.traits[2]], verbose = false);

julia> length(blr_1.Xs) == 3
true

julia> size(blr_1.Xs["rows"]) == (15, 3)
true

julia> length(blr_2.Xs) == 9
true

julia> size(blr_2.Xs["entries & seasons & sites"]) == (15, 5)
true
```
"""
function instantiateblr(;
    trait::String,
    factors::Vector{String},
    df::DataFrame,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    saturated_model::Bool = false,
    verbose::Bool = false,
)::BLR
    # genomes = simulategenomes(n=500, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3);
    # trait = "trait_1"; factors = ["rows", "cols"]; df = tabularise(trials); other_covariates::Union{Vector{String}, Nothing} = ["trait_2", "trait_3"]; saturated_model = false; verbose = true;
    # Check arguments and extract focal terms
    focal_terms = checkandfocalterms(trait = trait, factors = factors, df = df, other_covariates = other_covariates)
    # Define the coefficients excluding possible additional continuous covariates
    coefficients_base = if !saturated_model
        vector_of_terms = []
        for t in focal_terms
            # t = focal_terms[5]
            ts = split(t, ":")
            if length(intersect(ts, factors)) == length(ts)
                if length(ts) > 1
                    # t = "rows:cols"
                    push!(vector_of_terms, foldl(*, term.(ts))[end]) ### insert only the last most complex term
                else
                    push!(vector_of_terms, term(t))
                end
            end
        end
        foldl(+, vector_of_terms)
    else
        foldl(*, term.(factors))
    end
    # Define the formula
    formula_struct, coefficients = if isnothing(other_covariates)
        (concrete_term(term(trait), df[!, trait], ContinuousTerm) ~ term(1) + coefficients_base), coefficients_base
    else
        # f = term(trait) ~ term(1) + foldl(*, term.(factors)) + foldl(+, term.(other_covariates))
        coefficients =
            coefficients_base + foldl(+, [concrete_term(term(c), df[!, c], ContinuousTerm) for c in other_covariates])
        (concrete_term(term(trait), df[!, trait], ContinuousTerm) ~ term(1) + coefficients), coefficients
    end
    # Extract the names of all the coefficients including the base level/s and the full design matrix (n x F(p)).
    if verbose
        println("Extracting the design matrices for the model to be fit...")
    end
    # Define the contrast for one-hot encoding, i.e. including the intercept, i.e. p instead of the p-1
    contrasts::Dict{Symbol,StatsModels.FullDummyCoding} = Dict()
    explain_var_names::Vector{String} = [string(x) for x in coefficients]
    for f in explain_var_names
        # f = factors[1]
        contrasts[Symbol(f)] = StatsModels.FullDummyCoding()
    end
    mf = ModelFrame(formula_struct, df, contrasts = contrasts)
    _, coefficient_names = coefnames(apply_schema(formula_struct, mf.schema))
    y, X = modelcols(apply_schema(formula_struct, mf.schema), df)
    # Initialise the BLR struct
    blr = begin
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
        blr
    end
    # Separate each explanatory variable from one another
    for v in explain_var_names
        # v = explain_var_names[3]
        v_split = filter(x -> x != "&", split(v, " "))
        # Main design matrices (bool) full design matrices (bool_ALL)
        bool = fill(true, length(coefficient_names))
        for x in v_split
            bool .*= .!isnothing.(match.(Regex(x), coefficient_names))
        end
        # B = sum(hcat([.!isnothing.(match.(Regex(x), coefficient_names)) for x in v_split]...), dims=2)[:, 1] .== 2
        # bool == B
        # Make sure we are extracting the correct coefficients, i.e. exclude unintended interaction terms
        if length(v_split) == 1
            bool .*= isnothing.(match.(Regex("&"), coefficient_names))
        else
            m = length(v_split) - 1
            bool .*= [
                !isnothing(x) ? sum(.!isnothing.(match.(Regex("&"), split(coefficient_names[i], " ")))) == m : false for (i, x) in enumerate(match.(Regex("&"), coefficient_names))
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
            v
        else
            # For categorical variables --> boolean matrix for memory-efficiency
            blr.Xs[v] = Bool.(X[:, bool])
            v
        end
        blr.Σs[v] = 1.0 * I
        blr.coefficients[v] = try
            vcat(blr.coefficients[v], zeros(sum(bool)))
        catch
            zeros(sum(bool))
        end
        blr.coefficient_names[v] = try
            vcat(blr.coefficient_names[v], coefficient_names[bool])
        catch
            coefficient_names[bool]
        end
    end
    # Output
    if !checkdims(blr)
        throw(ErrorException("The resulting BLR struct for the model to be fit is corrupted ☹."))
    end
    blr
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
        Σ = Symmetric(Diagonal(σ²) * vector_of_Δs[i] * Diagonal(σ²))
        βs[i] ~ MvNormal(zeros(length(βs[i])), Σ)
        μ += Float64.(vector_of_Xs_noint[i]) * βs[i]
    end
    # Residual variance
    σ² ~ Exponential(10.0)
    # Return the distribution of the response variable, from which the likelihood will be derived
    return y ~ Distributions.MvNormal(μ, σ² * I)
end

"""
    extractmodelinputs(blr::BLR; multiple_σs::Union{Nothing, Dict{String, Bool}}=nothing)::Dict

Extract model inputs from BLR object for Bayesian modelling.

# Arguments
- `blr::BLR`: A Bayesian Linear Regression model object 
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
- Processes design matrices, variance-covariance structures, and coefficient names from the BLR object

# Throws
- `ErrorException`: If a variance component is not specified in the multiple_σs dictionary

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase)
julia> genomes = simulategenomes(n=5, l=1_000, verbose=false);

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);

julia> model_inputs_1 = extractmodelinputs(blr, multiple_σs = nothing);

julia> model_inputs_2 = extractmodelinputs(blr, multiple_σs = Dict("rows" => true, "cols" => true, "rows & cols" => false, "other_covariates" => true));

julia> (model_inputs_1["y"] == model_inputs_2["y"]) && (model_inputs_1["vector_of_Xs_noint"] == model_inputs_2["vector_of_Xs_noint"]) && (model_inputs_1["vector_of_Δs"] == model_inputs_2["vector_of_Δs"]) && (model_inputs_1["vector_coefficient_names"] == model_inputs_2["vector_coefficient_names"])
true

julia> sum(model_inputs_1["length_of_σs"]) < sum(model_inputs_2["length_of_σs"])
true
```
"""
function extractmodelinputs(
    blr::BLR;
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
    y::Vector{Float64} = blr.y
    vector_of_Xs_noint::Vector{Matrix{Union{Bool,Float64}}} = []
    vector_of_Δs::Vector{Union{Matrix{Float64},UniformScaling{Float64}}} = []
    length_of_σs::Vector{Int64} = []
    vector_coefficient_names::Vector{String} = []
    variance_components::Vector{String} = filter(x -> x != "intercept", string.(keys(blr.Xs)))
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
        push!(vector_of_Xs_noint, blr.Xs[v])
        push!(vector_of_Δs, blr.Σs[v])
        # if isa(blr.Σs[v], UniformScaling{Float64})
        if !multiple_σs[v]
            # If only 1 variance component is requested, i.e. common/spherical variance
            push!(length_of_σs, 1)
        else
            # Otherwise, define separate variance scaler per coefficient
            push!(length_of_σs, size(blr.Xs[v], 2))
        end
        push!(length_of_σs)
        vector_coefficient_names = vcat(vector_coefficient_names, blr.coefficient_names[v])
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
        blr::BLR;
        multiple_σs::Union{Nothing, Dict{String, Bool}} = nothing,
        turing_model::Function = turingblr,
        n_iter::Int64 = 10_000,
        n_burnin::Int64 = 1_000,
        δ::Float64 = 0.65,
        max_depth::Int64 = 5,
        Δ_max::Float64 = 1000.0,
        init_ϵ::Float64 = 0.2,
        adtype::AutoReverseDiff = AutoReverseDiff(compile = true),
        diagnostics_threshold_std_lt::Float64 = 0.05,
        diagnostics_threshold_ess_ge::Int64 = 100, 
        diagnostics_threshold_rhat_lt::Float64 = 1.01,
        seed::Int64 = 1234,
        verbose::Bool = true
    )::Nothing

Perform MCMC sampling for Bayesian Linear Regression on a BLR model using NUTS sampler.

# Arguments
- `blr::BLR`: The BLR model to fit
- `multiple_σs::Union{Nothing, Dict{String, Bool}}`: Optional dictionary specifying multiple (true) or single (false) variance scalers per component
- `turing_model::Function`: The Turing model function to use (default: turingblr)
- `n_iter::Int64`: Number of MCMC iterations (default: 10,000)
- `n_burnin::Int64`: Number of burn-in iterations to discard (default: 1,000) 
- `δ::Float64`: Target acceptance rate for dual averaging (default: 0.65)
- `max_depth::Int64`: Maximum doubling tree depth (default: 5)
- `Δ_max::Float64`: Maximum divergence during doubling tree (default: 1000.0)
- `init_ϵ::Float64`: Initial step size; 0 means auto-search using heuristics (default: 0.2)
- `adtype::AutoReverseDiff`: Automatic differentiation type (default: AutoReverseDiff(compile = true))
- `diagnostics_threshold_std_lt::Float64`: Threshold for MCSE/std ratio convergence (default: 0.05)  
- `diagnostics_threshold_ess_ge::Int64`: Minimum effective sample size for convergence (default: 100)
- `diagnostics_threshold_rhat_lt::Float64`: Maximum R-hat for convergence (default: 1.01)
- `seed::Int64`: Random seed for reproducibility (default: 1234)
- `verbose::Bool`: Whether to show progress during sampling (default: true)

# Returns
- `Nothing`: The function mutates the input BLR struct in-place, updating its:
    + coefficients, 
    + variance components, 
    + predicted values, 
    + residuals, and 
    + MCMC diagnostics

# Notes
- Performs model validation checks before fitting
- Uses NUTS (No-U-Turn Sampler) with specified AD type
- Computes convergence diagnostics by splitting the chain into 5 sub-chains to calculate:
    + improved Gelman-Rubin statistics (R̂; https://doi.org/10.1214/20-BA1221),
    + effective sample size (ESS), and
    + Monte Carlo standard error (MCSE)
- Warns if parameters haven't converged based on:
    + R̂ >= diagnostics_threshold_rhat_lt
    + ESS < diagnostics_threshold_ess_ge  
    + MCSE/std ratio >= diagnostics_threshold_std_lt
- Updates the model with estimated coefficients, variance components, predicted values, and residuals
- Mutates the input BLR model in-place

```jldoctest; setup = :(using GenomicBreedingCore, StatsBase, DataFrames)
julia> genomes = simulategenomes(n=5, l=1_000, verbose=false);

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> blr = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);

julia> turingblrmcmc!(blr, n_iter=1_000, n_burnin=200, seed=123, verbose=false);

julia> mean([mean(x) for (_, x) in blr.coefficients]) != 0.0
true

julia> cor(blr.y, blr.ŷ) > 0.0
true

julia> (sum(blr.diagnostics.rhat .< 1.01) < nrow(blr.diagnostics))
true

julia> (sum(blr.diagnostics.ess .>= 100) < nrow(blr.diagnostics))
true
```
"""
function turingblrmcmc!(
    blr::BLR;
    multiple_σs::Union{Nothing,Dict{String,Bool}} = nothing,
    turing_model::Function = turingblr,
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    δ::Float64 = 0.65,
    max_depth::Int64 = 5,
    Δ_max::Float64 = 1000.0,
    init_ϵ::Float64 = 0.2,
    adtype::AutoReverseDiff = AutoReverseDiff(compile = true),
    diagnostics_threshold_std_lt = 0.05,
    diagnostics_threshold_ess_ge = 100,
    diagnostics_threshold_rhat_lt = 1.01,
    seed::Int64 = 1234,
    verbose::Bool = true,
)::Nothing
    # genomes = simulategenomes(n=500, l=1_000, verbose=false); 
    # trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);
    # df = tabularise(trials);
    # blr = instantiateblr(trait = trials.traits[1], factors = ["rows", "cols"], df = df, other_covariates = trials.traits[2:end], verbose = false);
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
    if !checkdims(blr)
        throw(ArgumentError("The BLR struct to be fit is corrupted ☹."))
    end
    if n_iter < 100
        throw(
            ArgumentError(
                "The number of MCMC iterations (`n_iter=$n_iter`) is less than 100. Please be reasonable and add more iterations or else your model will likely not converge.",
            ),
        )
    end
    # Extract the response variable, y, and the vectors of Xs, and Σs for model fitting, as well as the coefficient names for each variance component
    model_inputs = extractmodelinputs(blr, multiple_σs = multiple_σs)
    # Instantiate the RNG, model, and sampling function
    rng::TaskLocalRNG = Random.seed!(seed)
    model = turing_model(
        model_inputs["vector_of_Xs_noint"],
        model_inputs["vector_of_Δs"],
        model_inputs["length_of_σs"],
        model_inputs["y"],
    )
    # sampling_function = NUTS(n_burnin, δ, max_depth = max_depth, Δ_max = Δ_max, init_ϵ = init_ϵ; adtype = adtype)
    sampling_function = NUTS(δ, max_depth = max_depth, Δ_max = Δ_max, init_ϵ = init_ϵ; adtype = adtype)
    # MCMC
    if verbose
        println(
            "Single chain MCMC sampling for $n_iter iterations (excluding the first $n_burnin adaptation or warm-up iterations).",
        )
    end
    chain = Turing.sample(rng, model, sampling_function, n_iter, discard_initial = n_burnin, progress = verbose)
    # Diagnostics
    if verbose
        println(
            "Diagnosing the MCMC chain for convergence by dividing the chain into 5 and finding the maximum R̂ and estimating the effective sample size.",
        )
    end
    diagnostics = disallowmissing(
        leftjoin(
            leftjoin(
                DataFrame(Turing.summarystats(chain))[:, 1:3], # parameter names, mean and std only
                DataFrame(MCMCDiagnosticTools.ess_rhat(chain, split_chains = 5)), # new R̂: maximum R̂ of :bulk and :tail
                on = :parameters,
            ),
            DataFrame(MCMCDiagnosticTools.mcse(chain, split_chains = 5)),
            on = :parameters,
        ),
    )
    p = size(diagnostics, 1)
    n_mcse_converged = sum(diagnostics.mcse ./ diagnostics.std .< diagnostics_threshold_std_lt)
    n_ess_converged = sum(diagnostics.ess .>= diagnostics_threshold_ess_ge)
    n_rhat_converged = sum(diagnostics.rhat .< diagnostics_threshold_rhat_lt)
    if verbose
        display(UnicodePlots.histogram(diagnostics.mcse, title = "MCSE", xlabel = "", ylabel = ""))
        display(UnicodePlots.histogram(diagnostics.ess, title = "ESS", xlabel = "", ylabel = ""))
        display(UnicodePlots.histogram(diagnostics.rhat, title = "R̂", xlabel = "", ylabel = ""))
        if ((p - n_rhat_converged) > 0) || ((p - n_ess_converged) > 0) || ((p - n_mcse_converged) > 0)
            @warn "Convergence rates:\n" *
                  "\t‣ $(round(n_mcse_converged*100 / p))% ($(p - n_mcse_converged) out of $p parameter/s did not converge based on Monte Carlo standard error-to-posterior distribution standard deviatin ration (< 5%))\n" *
                  "\t‣ $(round(n_ess_converged*100 / p))% ($(p - n_ess_converged) out of $p parameter/s did not converge based on effective sample size (< 100))\n" *
                  "\t‣ $(round(n_rhat_converged*100 / p))% ($(p - n_rhat_converged) out of $p parameter/s did not converge based on R̂ (>= 1.01))\n" *
                  "Please consider increasing the number of iterations which is currently at $n_iter."
        else
            println("All parameters have converged.")
        end
    end
    # Check the chain size
    # Chain size is: 
    #   - n_iter iterations
    #   - p parameters + 13 internal variables, i.e. [:lp, :n_steps, :is_accept, :acceptance_rate, :log_density, :hamiltonian_energy, :hamiltonian_energy_error, :max_hamiltonian_energy_error, :tree_depth, :numerical_error, :step_size, :nom_step_size]
    #   - 1 chain
    p = sum([length(v) for (k, v) in blr.coefficient_names]) + sum(model_inputs["length_of_σs"])
    if size(chain) != (n_iter, p + 13, 1)
        throw(ErrorException("The MCMC chain is not of the expected size."))
    end
    # Extract the posterior distributions of the parameters
    # Use the mean parameter values which excludes the first n_burnin iterations
    params = Turing.get_params(chain)
    β0 = mean(params.intercept)
    βs = [mean(params.βs[i]) for i in eachindex(params.βs)]
    σ² = mean(params.σ²)
    σ²s = [mean(params.σ²s[i]) for i in eachindex(params.σ²s)]
    # Extract coefficients
    if verbose
        println("Extracting MCMC-derived coefficients")
    end
    blr.coefficients["intercept"] = [β0]
    blr.Σs["σ²"] = σ² * blr.Σs["σ²"]
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
        if blr.coefficient_names[v] != model_inputs["vector_coefficient_names"][ini:fin]
            throw(ErrorException("The expected coefficient names do not match for the variance component: $v."))
        end
        blr.coefficients[v] = βs[ini:fin]
        blr.Σs[v] = if (isa(model_inputs["vector_of_Δs"][i], UniformScaling{Float64}) && (fin_σ - ini_σ == 0))
            # Single variance component multiplier
            σ²s[ini_σ] * model_inputs["vector_of_Δs"][i]
        else
            # Multiple variance component multipliers
            # σ² = repeat(σ²s[ini_σ:fin_σ], 1 + length(blr.coefficients[v]) - length(σ²s[ini_σ:fin_σ]))
            σ² = σ²s[ini_σ:fin_σ]
            Matrix(Diagonal(σ²) * model_inputs["vector_of_Δs"][i] * Diagonal(σ²))
        end
    end
    # Define the predicted ys and residuals
    X, b, _b_labels = extractXb(blr)
    blr.ŷ = X * b
    blr.ϵ = blr.y .- blr.ŷ
    # Insert the diagnostics into the BLR struct
    blr.diagnostics = diagnostics
    # Output
    if !checkdims(blr)
        throw(ErrorException("Error updating the BLR struct after MCMC."))
    end
    return nothing
end


# ### ATTEMPTING TO MAKE A BAYESIAN APPROACH TAILORED SPECIFICALLY FOR LARGE PARAMTER SPACES ###
# # The implementation would typically involve:
# # 1. Defining the Hamiltonian dynamics of the model.
# # 2. Implementing the leapfrog integrator to simulate the Hamiltonian dynamics.
# # 3. Adapting the step size and trajectory length based on the acceptance rate.
# # 4. Using a dual averaging scheme to adjust the step size dynamically.
# # 5. Implementing the No-U-Turn condition to terminate the trajectory early if it starts to turn back on itself.
# # 6. Returning samples from the posterior distribution of the model parameters.
# # using LinearAlgebra
# # using Statistics
# # using Random
# # using ForwardDiff

# """
# Computes the log of the joint probability of the position θ and momentum r.
# This is equivalent to the negative Hamiltonian.
# """
# function log_joint(θ, log_posterior, r)
#     return log_posterior(θ) - 0.5 * dot(r, r)
# end

# """
# Performs a single leapfrog step to update the position and momentum.
# """
# function leapfrog(θ, r, ϵ, log_posterior_grad)
#     # Half step for momentum
#     r_half = r + 0.5 * ϵ * log_posterior_grad(θ)
#     # Full step for position
#     θ_new = θ + ϵ * r_half
#     # Full step for momentum
#     r_new = r_half + 0.5 * ϵ * log_posterior_grad(θ_new)
#     return θ_new, r_new
# end


# """
# The No-U-Turn Sampler (Iterative Implementation).
# """
# function NoUTurnSampler(initial_θ, log_posterior, n_samples; ϵ=0.1, max_depth=10)
#     # Automatically create the gradient function using ForwardDiff
#     log_posterior_grad = θ -> ForwardDiff.gradient(log_posterior, θ)

#     D = length(initial_θ)
#     samples = zeros(n_samples, D)
#     samples[1, :] = initial_θ
#     θ_current = initial_θ

#     println("Starting sampling...")
#     @inbounds for i in 2:n_samples
#         if i % (n_samples ÷ 10) == 0
#             println("Progress: $(round(i/n_samples*100))%")
#         end

#         # Resample momentum from a standard normal distribution
#         r0 = randn(D)

#         # Determine the slice variable for the acceptance window
#         # The log(rand()) is a trick to sample from an exponential distribution
#         u = log(rand()) + log_joint(θ_current, log_posterior, r0)

#         # Initialize the trajectory endpoints
#         θ_minus, θ_plus = θ_current, θ_current
#         r_minus, r_plus = r0, r0
        
#         j = 0         # Tree depth
#         θ_proposal = θ_current # The proposed next sample
#         n = 1         # Number of valid points in the trajectory
#         s = 1         # Trajectory validity flag (1 = valid, 0 = invalid/U-turn)

#         # Iteratively build the trajectory tree until a U-turn is detected or max depth is reached
#         while s == 1 && j < max_depth
#             # Choose a random direction to expand the tree: -1 (backwards) or 1 (forwards)
#             direction = rand([-1, 1])
            
#             # --- Build a new subtree in the chosen direction ---
#             θ_edge, r_edge = (direction == -1) ? (θ_minus, r_minus) : (θ_plus, r_plus)
            
#             n_prime = 0          # Number of valid points in the new subtree
#             s_prime = 1          # Validity of the new subtree
#             θ_prime_subtree = θ_edge # The proposal from the new subtree

#             # The number of steps to take depends on the tree depth (2^j)
#             num_steps = 2^j
#             @inbounds for _ in 1:num_steps
#                 # Take a leapfrog step
#                 θ_edge, r_edge = leapfrog(θ_edge, r_edge, direction * ϵ, log_posterior_grad)
                
#                 # Check for divergence
#                 if !isfinite(log_joint(θ_edge, log_posterior, r_edge))
#                     s_prime = 0
#                     break
#                 end

#                 # Check if the new point is within the slice
#                 logp_new = log_joint(θ_edge, log_posterior, r_edge)
#                 if u <= logp_new
#                     n_prime += 1
#                     # With probability 1/n_prime, accept this new point as the subtree's proposal
#                     if rand() < 1.0 / n_prime
#                         θ_prime_subtree = θ_edge
#                     end
#                 end
#             end
            
#             # Update the main trajectory endpoints with the edge of the new subtree
#             if direction == -1
#                 θ_minus, r_minus = θ_edge, r_edge
#             else
#                 θ_plus, r_plus = θ_edge, r_edge
#             end

#             # --- Combine the new subtree with the existing trajectory ---
#             if s_prime == 1 && n_prime > 0
#                 # Probabilistically choose between the proposal from the old trajectory and the new one
#                 if rand() < n_prime / (n + n_prime)
#                     θ_proposal = θ_prime_subtree
#                 end
#             end
            
#             n += n_prime # Update the total count of valid points

#             # Check for a U-turn across the full trajectory
#             # This is the "No-U-Turn" condition. It checks if the trajectory has started to double back on itself.
#             s = s_prime * (dot(θ_plus - θ_minus, r_minus) >= 0) * (dot(θ_plus - θ_minus, r_plus) >= 0)
            
#             j += 1 # Increment tree depth
#         end # end while

#         samples[i, :] = θ_proposal
#         θ_current = θ_proposal
#     end # end for

#     println("Sampling complete.")
#     return samples
# end

# # --- Example Usage ---
# # 🎯 Define a target log-posterior (e.g., a multivariate normal distribution)
# const target_mean = [0.5, -0.5]
# const target_cov = [1.0 0.8; 0.8 1.0]
# const target_cov_inv = inv(target_cov)

# function log_posterior_mvn(θ)
#     -0.5 * (θ - target_mean)' * target_cov_inv * (θ - target_mean)
# end
# # Set initial parameters
# initial_position = [0.0, 0.0]
# num_samples = 10_000
# step_size = 0.1
# max_tree_depth = 10
# @time samples = NoUTurnSampler(initial_position, log_posterior_mvn, num_samples, ϵ=step_size, max_depth=max_tree_depth);


# n = 123
# p = 100
# h² = 0.5
# X = rand(Beta(2.0, 2.0), n, p) 
# β = abs.(round.(rand(Laplace(0.0, 0.001), p), digits=2))
# σ²ᵦ = var(X * β)
# σ² = σ²ᵦ * (1.0 / h² - 1.0)
# e = rand(Normal(0.0, σ²), n)
# y = X*β + e
# UnicodePlots.histogram(y)
# function log_posterior_mvn(θ)
#     -sqrt(mean((X*θ - y).^2))
# end
# initial_position = zeros(p)  # Initial position in the parameter space

# # Run the sampler
# # The first run might be slower due to Just-In-Time (JIT) compilation
# samples = NoUTurnSampler(initial_position, log_posterior_mvn, num_samples, ϵ=step_size, max_depth=max_tree_depth)

# # Analyze the results 📊
# println("\nTarget Mean:")
# println(target_mean)
# println("\nSample Mean:")
# println(mean(samples, dims=1))

# println("\nTarget Covariance:")
# println(target_cov)
# println("\nSample Covariance:")
# println(cov(samples))




# Assessing how computationally efficient or how reasonably fast/slow the fitting of the full model via Bayesian linear regression is...
function fitfullmodel()
    genomes = simulategenomes(n = 1_000, l = 1_000, n_populations = 3, verbose = true)
    trials, simulated_effects = simulatetrials(
        genomes = genomes,
        f_add_dom_epi = rand(13, 3),
        n_years = 5,
        n_seasons = 4,
        n_harvests = 1,
        n_sites = 10,
        n_replications = 4,
        sparsity = 0.05,
        verbose = true,
    )
    trials = removemissnaninf(trials)
    df = tabularise(trials)
    blr = instantiateblr(
        trait = trials.traits[1],
        factors = ["years", "seasons", "sites", "rows", "cols"],
        df = df,
        other_covariates = trials.traits[2:end],
        verbose = true,
    )

    # # Testing Turing.jl for this HUGE model...
    # # turingblrmcmc!(blr, n_iter=1_000, n_burnin=200, seed=123, verbose=true);

    # # Turing.jl is very slow for estimating many parameters so we will test BAT.jl for now...
    n = 123
    p = 1_000
    h² = 0.5
    X = rand(Beta(2.0, 2.0), n, p) 
    β = abs.(round.(rand(Laplace(0.0, 0.001), p), digits=2))
    σ²ᵦ = var(X * β)
    σ² = σ²ᵦ * (1.0 / h² - 1.0)
    e = rand(Normal(0.0, σ²), n)
    y = X*β + e
    # UnicodePlots.histogram(y)
    # struct SomeDist <: ContinuousUnivariateDistribution
    #     a::Real
    #     b::Real
    # end
    # function Distributions.rand(rng::AbstractRNG, d::SomeDist)::Real
    #     # d = SomeDist(1.0, 1.0)
    #     # rand()
    #     rand(Normal(d.a, d.b))
    # end
    # function Distributions.logpdf(d::SomeDist, x::Real)::Real
    #     # d = SomeDist(1.0, 1.0)
    #     # -log(abs(x*(d.a + d.b)) + 1e-21)
    #     logpdf(Normal(d.a, d.b), x)
    # end
    # Distributions.minimum(d::SomeDist) = -Inf
    # Distributions.maximum(d::SomeDist) = Inf

    Turing.@model toy(X, y) = begin
        _n, p = size(X)
        # σ²ᵦ ~ filldist(Normal(0.0, 0.01), p)
        # β ~ MvNormal(fill(0.0, p), abs.(σ²ᵦ))
        # β ~ filldist(SomeDist(0.0, 1.0), p)
        # β ~ filldist(Normal(0.0, 1.0), p)
        β ~ MvNormal(fill(0.0, p), 1.0)
        σ² ~ InverseGamma(1,1)
        y ~ MvNormal(X*β, σ²)
    end
    model = toy(X, y)
    n_iter::Int64 = 10_000
    n_burnin::Int64 = 1_000
    n_adapts::Int64 = 500
    δ::Float64 = 0.65
    max_depth::Int64 = 5
    Δ_max::Float64 = 1000.0
    init_ϵ::Float64 = 0.2
    adtype::AutoReverseDiff = AutoReverseDiff(compile = true)
    sampling_function = NUTS(n_adapts, δ, max_depth = max_depth, Δ_max = Δ_max, init_ϵ = init_ϵ; adtype = adtype)
    diagnostics_threshold_std_lt = 0.05
    diagnostics_threshold_ess_ge = 100
    diagnostics_threshold_rhat_lt = 1.01
    @time chain = Turing.sample(model, sampling_function, n_iter, discard_initial = n_burnin, progress = true);

    q_init = q_fullrank_gaussian(model)  # initial variational approximation
    vi(model, q_init, 1000, show_progress=true) # perform VI with the default algorithm on `m` for 1000 iterations


    θs = Turing.get_params(chain);
    β_hat = [mean(θs.β[i]) for i in eachindex(θs.β)];
    UnicodePlots.scatterplot(β, β_hat)
    UnicodePlots.scatterplot(y, X*β_hat)
    cor(y, X*β_hat)


    

    # diagnostics = disallowmissing(
    #     leftjoin(
    #         leftjoin(
    #             DataFrame(Turing.summarystats(chain))[:, 1:3], # parameter names, mean and std only
    #             DataFrame(MCMCDiagnosticTools.ess_rhat(chain, split_chains = 5)), # new R̂: maximum R̂ of :bulk and :tail
    #             on = :parameters,
    #         ),
    #         DataFrame(MCMCDiagnosticTools.mcse(chain, split_chains = 5)),
    #         on = :parameters,
    #     ),
    # )
    # p = size(diagnostics, 1)
    # n_mcse_converged = sum(diagnostics.mcse ./ diagnostics.std .< diagnostics_threshold_std_lt)
    # n_ess_converged = sum(diagnostics.ess .>= diagnostics_threshold_ess_ge)
    # n_rhat_converged = sum(diagnostics.rhat .< diagnostics_threshold_rhat_lt)
end




















"""
    removespatialeffects!(df::DataFrame; factors::Vector{String}, traits::Vector{String}, 
                         other_covariates::Union{Vector{String},Nothing} = nothing,
                         Σ_type::Symbol = :empirical,
                         ρs::Dict{String, Float64} = Dict("rows" => 0.5, "cols" => 0.5),
                         n_iter::Int64 = 10_000, n_burnin::Int64 = 1_000, 
                         seed::Int64 = 1234, verbose::Bool = false)::Tuple{Vector{String}, Dict{String, DataFrame}}

Remove spatial effects from trait measurements in field trials using Bayesian linear regression.

# Arguments
- `df::DataFrame`: DataFrame containing trial data with columns for traits, spatial factors, and harvests
- `factors::Vector{String}`: Vector of factor names to be considered in the model 
- `traits::Vector{String}`: Vector of trait names to be spatially adjusted
- `other_covariates::Union{Vector{String},Nothing}`: Optional vector of additional covariates
- `Σ_type::Symbol`: Type of covariance structure to use (:empirical, :autoregressive, or :spherical) (default: :empirical)
- `ρs::Dict{String, Float64}`: Correlation parameters for autoregressive structure, keys are factor names (default: rows=0.5, cols=0.5)
- `n_iter::Int64`: Number of MCMC iterations (default: 10_000)
- `n_burnin::Int64`: Number of burn-in iterations (default: 1_000) 
- `seed::Int64`: Random seed for reproducibility (default: 1234)
- `verbose::Bool`: If true, prints detailed information (default: false)

# Returns
In addition to mutating `df` by adding the spatially adjusted traits as new columns with the prefix "SPATADJ-",
it returns a tuple containing:
1. Vector of remaining significant factors after spatial adjustment
2. Dictionary mapping harvest-trait combinations to diagnostic DataFrame results

# Details
This function performs spatial adjustment for traits measured in field trials by:

1. Identifying spatial factors (blocks, rows, columns) and creating design matrices
2. Fitting Bayesian linear model per harvest to account for:
   - Spatial effects (blocks, rows, columns and interactions) 
   - Choice of covariance structure:
     + Empirical: Estimated using linear shrinkage with unequal variances
     + Autoregressive: AR(1) structure with specified correlation parameters
     + Spherical: Identity matrix scaled by variance component
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
- Supports empirical, autoregressive or spherical covariance structures
- Modifies the input DataFrame in-place

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase, DataFrames)
julia> genomes = simulategenomes(n=5, l=1_000, verbose=false);

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);

julia> df = tabularise(trials);

julia> factors, diagnostics = removespatialeffects!(df, factors = ["rows", "cols", "blocks"], traits = ["trait_1", "trait_2"], other_covariates=["trait_3"], n_iter=1_000, n_burnin=200, verbose = false);

julia> ("SPATADJ-trait_1" ∈ names(df)) && ("SPATADJ-trait_2" ∈ names(df))
true

julia> [(sum(d.rhat .< 1.01) < nrow(d)) && (sum(d.ess .>= 100) < nrow(d)) for (_, d) in diagnostics] == [true, true]
true
```
"""
function removespatialeffects!(
    df::DataFrame;
    factors::Vector{String},
    traits::Vector{String},
    other_covariates::Union{Vector{String},Nothing} = nothing,
    Σ_type::Symbol = [:empirical, :autoregressive, :spherical][1],
    ρs::Dict{String,Float64} = Dict("rows" => 0.5, "cols" => 0.5),
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose::Bool = false,
)::Tuple{Vector{String},Dict{String,DataFrame}}
    # genomes = simulategenomes(n=500, l=1_000, verbose=false); 
    # trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=3, verbose=false);
    # df = tabularise(trials);
    # factors = ["rows", "cols"]; traits = ["trait_1", "trait_2"]; other_covariates=["trait_3"]; Σ_type::Symbol = [:empirical, :autoregressive][1]; ρs::Dict{String, Float64} = Dict("rows" => 0.5, "cols" => 0.5); n_iter=1_000; n_burnin=200; seed=123;  verbose = true;
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
    if !(Σ_type ∈ [:empirical, :autoregressive, :spherical])
        throw(
            ArgumentError(
                "The covariance structure type must be one of [:empirical, :autoregressive, :spherical]. You provided: :$Σ_type",
            ),
        )
    end
    if Σ_type != :spherical
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
    harvests = unique(df.harvests)
    # Instantiate the diagnostics dictionary for each harvest-by-trait combination
    spatial_diagnostics::Dict{String,DataFrame} = Dict()
    # Loop through each harvest
    for harvest in harvests
        # harvest = harvests[1]
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
                println(
                    " ($(1 + length(spatial_diagnostics))/$(length(harvests) * length(traits))) Spatial modelling for harvest: $harvest; and trait: $trait",
                )
                println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            end
            # Add spatially adjusted trait to df
            new_spat_adj_trait_name = string("SPATADJ-", trait)
            if sum(names(df) .== new_spat_adj_trait_name) == 0
                df[:, "__new_spatially_adjusted_trait__"] .= df[:, trait]
                rename!(df, "__new_spatially_adjusted_trait__" => new_spat_adj_trait_name)
            end
            # Instantiate the BLR struct for spatial analysis to remove the spatial effects as well as the effects of the other covariates
            blr = instantiateblr(
                trait = trait,
                factors = spatial_factors,
                other_covariates = other_covariates,
                df = df_sub,
                verbose = verbose,
            )
            # Define variance-covariance matrix for the spatial factors
            if Σ_type == :spherical
                # Spherical variance-covariance matrix
                # Default in instantiateblr() is to use the identity matrix scaled by the variance component
            else
                if Σ_type == :autoregressive
                    # Autoregressive (degree=1) variance-covariance matrix
                    # We assume that the coefficient names of the factor which will have an autoregressive variance-covariance matrix
                    # can be sorted sensibly, e.g. names like "rows_01", "rows_02", "rows_03" to "row_99" can be sorted to get the order.
                    for varcomp in string.(keys(ρs))
                        # varcomp = string.(keys(ρs))[1]
                        idx = sortperm(blr.coefficient_names[varcomp])
                        p = length(idx)
                        AR1 = Matrix(Diagonal(ones(p)))
                        for i in idx
                            for j in idx
                                # i = idx[1]; j = idx[2];
                                if i == j
                                    continue
                                end
                                # Farther factor levels less correlated
                                AR1[i, j] = ρs[varcomp]^abs(i - j)
                            end
                        end
                        inflatediagonals!(AR1)
                        blr.Σs[varcomp] = AR1
                    end
                else
                    # Empirical variance-covariance matrix
                    for varcomp in string.(keys(ρs))
                        # varcomp = string.(keys(ρs))[1]
                        X = blr.y .* blr.Xs[varcomp]
                        K = Matrix(cov(LinearShrinkage(DiagonalUnequalVariance(), :ss), X))
                        blr.Σs[varcomp] = K
                    end
                end
            end
            # Set-up variance component multipliers/scalers such that row, columns and other covariates have unique multipliers;
            # while the row-by-col interaction only has 1, i.e. spherical variance-covariance matrix for model tractability
            multiple_σs::Union{Nothing,Dict{String,Bool}} = Dict()
            for varcomp in string.(keys(blr.Σs))
                # varcomp = string.(keys(blr.Σs))[1]
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
                blr,
                multiple_σs = multiple_σs,
                n_iter = n_iter,
                n_burnin = n_burnin,
                seed = seed,
                verbose = verbose,
            )
            # Update the spatially adjusted trait with the intercept + residuals of the spatial model above
            # cor(df[idx_rows, new_spat_adj_trait_name], blr.coefficients["intercept"] .+ blr.ϵ)
            df[idx_rows, new_spat_adj_trait_name] = blr.coefficients["intercept"] .+ blr.ϵ
            # Update the spatial_diagnostics
            spatial_diagnostics[string(harvest, "|", new_spat_adj_trait_name)] = blr.diagnostics
            # Clean-up
            blr = nothing
            Base.GC.gc()
        end
    end
    # Is the entries the only factor remaining?
    if length(factors) == 1
        return (factors, spatial_diagnostics)
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
    (factors_out, spatial_diagnostics)
end


"""
    analyseviaBLR(
        trials::Trials,
        traits::Vector{String};
        grm::Union{GRM,Nothing} = nothing,
        other_covariates::Union{Vector{String},Nothing} = nothing,
        empirical_Σs::Bool = true,
        multiple_σs_threshold::Int64 = 500,
        n_iter::Int64 = 10_000,
        n_burnin::Int64 = 1_000,
        seed::Int64 = 1234,
        verbose::Bool = false,
    )::Tuple{TEBV,Dict{String,DataFrame}}

Perform Bayesian linear mixed model analysis on trial data for genetic evaluation.

# Arguments
- `trials::Trials`: A Trials struct containing the experimental data
- `traits::Vector{String}`: Vector of trait names to analyze. If empty, all traits in trials will be analyzed
- `grm::Union{GRM,Nothing}=nothing`: Optional genomic relationship matrix
- `other_covariates::Union{Vector{String},Nothing}=nothing`: Additional covariates to include in the model
- `multiple_σs_threshold::Int64=500`: Threshold for determining multiple variance components
- `n_iter::Int64=10_000`: Number of MCMC iterations
- `n_burnin::Int64=1_000`: Number of burn-in iterations
- `seed::Int64=1234`: Random seed for reproducibility
- `verbose::Bool=false`: Whether to print progress information

# Returns
A tuple containing:
- `TEBV`: Total estimated breeding values struct with model results. Note that:
    + only the variance-covariance components represent the p-1 factor levels; while,
    + the rest have the full number of levels, i.e. using the one-hot encoding vectors and matrices).
    + This means that the `Σs` have less rows and columns than the number of elements in `coefficient_names`.
- `Dict{String,DataFrame}`: Spatial diagnostics information

# Details
Performs a two-stage analysis:

1. Stage-1: Spatial analysis per harvest-site-year combination
- Creates temporary JLD2 file with spatially adjusted data to manage memory
- Corrects for spatial effects:
    + With rows and columns regardless of whether blocks are present:
        - `rows` + `cols` + `rows:cols`
    + With blocks and one spatial factor:
        - `blocks` + `rows` + `blocks:rows`
        - `blocks` + `cols` + `blocks:cols` 
    + With single spatial factor:
        - `blocks`
        - `rows`
        - `cols`
- Removes effects of continuous covariates
- Returns spatially adjusted traits with "SPATADJ-" prefix

2. Stage-2: GxE modeling excluding spatial effects and continuous covariates
    2.a. Genotypic effects:
        - `entries` (required)
    2.b. Environmental main effects:
        - `sites` if present
        - `seasons` if present
        - `years` if present 
    2.c Environmental interactions:
        - With all 3 environment factors:
            + `years:sites`
            + `seasons:sites`  
            + `entries:seasons:sites`
        - With 2 environment factors:
            + `years:seasons` + `entries:seasons` (no sites)
            + `years:sites` + `entries:sites` (no seasons)
            + `seasons:sites` + `entries:seasons:sites` (no years)
        - With 1 environment factor:
            + `entries:years`
            + `entries:seasons`
            + `entries:sites`

The analysis includes:
- Genomic relationship matrix (GRM) integration if provided:
  + GRM replaces identity matrices for entry-related effects
  + Diagonals are inflated if resulting matrices not positive definite
  + Inflation repeated up to 10 times to ensure stability
- Variance component estimation:
  + Single vs multiple variance scalers determined by threshold
  + Separate parameters for complex interaction terms
- MCMC-based Bayesian inference with:
  + Burn-in period for chain convergence
  + Diagnostic checks for convergence and mixing

# Notes
- Automatically handles memory management for large design matrices
- Creates temporary JLD2 file "TEMP-df_spat_adj-[hash].jld2" to store spatially adjusted data
- Automatically removes temporary JLD2 file after analysis is complete
- Supports both identity and genomic relationship matrices for genetic effects
- Performs automatic model diagnostics and variance component scaling
- Excludes continuous covariates from Stage-2 as they are handled in Stage-1

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase, DataFrames)
julia> genomes = simulategenomes(n=5, l=1_000, verbose=false);

julia> grm = grmploidyaware(genomes, ploidy=2, max_iter=10);

julia> trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=3, n_replications=3, verbose=false);

julia> tebv_1, spatial_diagnostics_1 = analyseviaBLR(trials, ["trait_1"], n_iter = 1_000, n_burnin = 100);

julia> tebv_2, spatial_diagnostics_2 = analyseviaBLR(trials, ["trait_1", "trait_2"], other_covariates = ["trait_3"], n_iter = 1_000, n_burnin = 100);

julia> tebv_3, spatial_diagnostics_3 = analyseviaBLR(trials, ["trait_3"], grm = grm, n_iter = 1_000, n_burnin = 100);

julia> (length(tebv_1.phenomes[1].entries) == 5) && (length(tebv_1.phenomes[2].entries) == 30) && (length(spatial_diagnostics_1) == 6)
true

julia> (length(tebv_2.phenomes[1].entries) == 5) && (length(tebv_2.phenomes[2].entries) == 30) && (length(spatial_diagnostics_2) == 12)
true

julia> (length(tebv_3.phenomes[1].entries) == 5) && (length(tebv_3.phenomes[2].entries) == 30) && (length(spatial_diagnostics_3) == 6)
true
```
"""
function analyseviaBLR(
    trials::Trials,
    traits::Vector{String};
    grm::Union{GRM,Nothing} = nothing,
    other_covariates::Union{Vector{String},Nothing} = nothing,
    empirical_Σs::Bool = true,
    multiple_σs_threshold::Int64 = 500,
    n_iter::Int64 = 10_000,
    n_burnin::Int64 = 1_000,
    seed::Int64 = 1234,
    verbose::Bool = false,
)::Tuple{TEBV,Dict{String,DataFrame}}
    # genomes = simulategenomes(n=5, l=1_000); trials, simulated_effects = simulatetrials(genomes = genomes, n_years=1, n_seasons=2, n_harvests=1, n_sites=3, n_replications=3); grm::Union{GRM, Nothing} = grmploidyaware(genomes; ploidy = 2, max_iter = 10, verbose = true); traits::Vector{String} = ["trait_1"]; other_covariates::Union{Vector{String}, Nothing} = ["trait_2"]; empirical_Σs::Bool = true; multiple_σs_threshold = 500; n_iter::Int64 = 1_000; n_burnin::Int64 = 100; seed::Int64 = 1234; verbose::Bool = true;
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
    df.harvests = string.(df.years, "|", df.seasons, "|", df.sites, "|", df.harvests)
    # Identify non-fixed factors
    factors_all::Vector{String} = ["years", "seasons", "sites", "harvests", "blocks", "rows", "cols", "entries"]
    factors::Vector{String} = []
    for f in factors_all
        # f = factors_all[4]
        if length(unique(df[!, f])) > 1
            push!(factors, f)
        end
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
    factors, spatial_diagnostics = removespatialeffects!(
        df,
        factors = factors,
        traits = traits,
        other_covariates = other_covariates,
        n_iter = n_iter,
        n_burnin = n_burnin,
        seed = seed,
        verbose = verbose,
    )
    # Save the spatially adjusted data into a temporary JLD2 file
    fname_df_spat_adj_tmp_jl2 = joinpath(pwd(), string("TEMP-df_spat_adj-", hash(df), ".jld2"))
    JLD2.save(fname_df_spat_adj_tmp_jl2, Dict("df" => df))
    # GxE modelling excluding the effects of spatial factors and continuous covariates
    BLRs::Dict{String,BLR} = Dict()
    traits = filter(x -> !isnothing(match(Regex("^SPATADJ-"), x)), names(df)) # includes only the spatially adjusted traits
    for (i, trait) in enumerate(traits)
        # i = length(traits); trait = traits[i];
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("GxE modelling for trait: $trait")
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        end
        # Instantiate the BLR struct for GxE analysis
        # Note that the covariate is now excluded as we should have controlled for them in the per harvest-site-year spatial analyses
        # - Stage-1 effects (spatial effects per year-season-site-harvest combination):
        #     + rows
        #     + cols
        #     + rows:cols
        # - Stage-2 effects (GxE effects after spatial corrections per year-season-site-harvest combination):
        #     + entries
        #     + sites
        #     + seasons
        #     + years
        #     + seasons:sites
        #     + years:sites
        #     + entries:seasons:sites
        blr = instantiateblr(trait = trait, factors = factors, other_covariates = nothing, df = df, verbose = verbose)
        # Prepare the variance-covariance matrix for the entries effects, i.e. using I or a GRM
        if !isnothing(grm)
            # Replace the variance-covariance matrix for the factors involving entries with the GRM
            factors_with_entries = begin
                x = string.(keys(blr.coefficient_names))
                x[.!isnothing.(match.(Regex("entries"), x))]
            end
            for fentries in factors_with_entries
                # fentries = factors_with_entries[end]
                n = length(blr.coefficient_names[fentries])
                blr.Σs[fentries] = zeros(n, n)
                for (i, name_1) in enumerate(blr.coefficient_names[fentries])
                    for (j, name_2) in enumerate(blr.coefficient_names[fentries])
                        # i = 1; name_1 = blr.coefficient_names[fentries][i]
                        # j = 1; name_2 = blr.coefficient_names[fentries][j]
                        split_1 = split(name_1, " ")
                        split_2 = split(name_2, " ")
                        if sum(split_1 .== split_2) < (length(split_1) - 1)
                            continue
                        end
                        entry_1 = split_1[.!isnothing.(match.(Regex("entry_"), split_1))][end]
                        entry_2 = split_2[.!isnothing.(match.(Regex("entry_"), split_2))][end]
                        k = findall(grm.entries .== entry_1)[end]
                        l = findall(grm.entries .== entry_2)[end]
                        blr.Σs[fentries][i, j] = grm.genomic_relationship_matrix[k, l]
                    end
                end
                # Inflate the diagonals if not positive definite
                counter = 0
                while !isposdef(blr.Σs[fentries])
                    if counter > 10
                        break
                    end
                    inflatediagonals!(blr.Σs[fentries], verbose = verbose)
                    counter += 1
                end
                if !isposdef(blr.Σs[fentries])
                    throw(
                        ErrorException(
                            "The variance-covariance matrix for the entries: $fentries remains non-positive definite after 10 inflation attempts.",
                        ),
                    )
                end
            end
        end
        # Set-up the variance-covariance matrix for the other factors if empirical_Σs = true
        if empirical_Σs
            for varcomp in string.(keys(blr.Σs))
                # varcomp = string.(keys(blr.Σs))[1]
                if varcomp == "σ²"
                    continue
                end
                X = blr.y .* blr.Xs[varcomp]
                K = Matrix(cov(LinearShrinkage(DiagonalUnequalVariance(), :ss), X))
                blr.Σs[varcomp] = K
            end
        end
        # Set-up variance component multipliers/scalers, for now it's just based on a maximum number of coefficients thresholds
        multiple_σs::Union{Nothing,Dict{String,Bool}} = Dict()
        for v in string.(keys(blr.coefficient_names))
            # v = string.(keys(blr.coefficient_names))[2]
            if v == "intercept"
                continue
            end
            multiple_σs[v] = if size(blr.Xs[v], 2) > multiple_σs_threshold
                false
            else
                true
            end
        end
        # GxE analysis via Bayesian linear regression
        turingblrmcmc!(
            blr,
            multiple_σs = multiple_σs,
            n_iter = n_iter,
            n_burnin = n_burnin,
            seed = seed,
            verbose = verbose,
        )
        # Add the fitted BLR struct to the dictionary of BLRs
        BLRs[trait] = blr
    end
    # Instantiate and populate the TEBV struct
    traits = []
    formulae::Vector{String} = []
    models::Vector{BLR} = []
    phenomes::Vector{Phenomes} = []
    for (k, v) in BLRs
        # k = string.(keys(BLRs))[1]; v = BLRs[k];
        factors_with_entries = begin
            x = string.(keys(v.coefficient_names))
            x[.!isnothing.(match.(Regex("entries"), x))]
        end
        for fentries in factors_with_entries
            # fentries = factors_with_entries[1]
            # fentries = factors_with_entries[2]
            trait = string(k, "|", fentries)
            push!(traits, trait)
            push!(formulae, string(trait, "~", join(string.(keys(v.coefficients)), "+")))
            push!(models, v)
            n = length(v.coefficient_names[fentries])
            ϕ = begin
                ϕ = Phenomes(n = n, t = 1)
                ϕ.entries = v.coefficient_names[fentries]
                ϕ.populations .= "unspecified"
                ϕ.traits = [trait]
                ϕ.phenotypes = reshape(v.coefficients[fentries], (n, 1))
                ϕ.mask .= true
                ϕ
            end
            push!(phenomes, ϕ)
        end
    end
    # tabularise(phenomes[2])
    # Output
    tebv = TEBV(
        traits = convert.(String, traits),
        formulae = formulae,
        models = models,
        df_BLUEs = repeat([DataFrame()], length(traits)),
        df_BLUPs = repeat([DataFrame()], length(traits)),
        phenomes = phenomes,
    )
    if !checkdims(tebv)
        throw(ErrorException("Error corrupted TEBV struct after BLR."))
    end
    # Clean-up and emit output
    rm(fname_df_spat_adj_tmp_jl2)
    (tebv, spatial_diagnostics)
end
