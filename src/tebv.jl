"""
    clone(x::TEBV)::TEBV

Create a deep copy of a TEBV (Trial-Estimated Breeding Value) object.

Returns a new TEBV instance with all fields deeply copied from the input object,
ensuring complete independence between the original and cloned objects.

# Arguments
- `x::TEBV`: The source TEBV object to be cloned

# Returns
- `TEBV`: A new TEBV object containing deep copies of all fields from the input

# Examples
```jldoctest; setup = :(using GBCore, MixedModels, DataFrames)
julia> tebv = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> copy_tebv = clone(tebv);

julia> copy_tebv.traits == tebv.traits
true

julia> copy_tebv.phenomes == tebv.phenomes
true
```
"""
function clone(x::TEBV)::TEBV
    Φ::TEBV = TEBV(
        traits = deepcopy(x.traits),
        formulae = deepcopy(x.formulae),
        models = deepcopy(x.models),
        df_BLUEs = deepcopy(x.df_BLUEs),
        df_BLUPs = deepcopy(x.df_BLUPs),
        phenomes = deepcopy(x.phenomes),
    )
    Φ
end


"""
    Base.hash(x::TEBV, h::UInt)::UInt

Calculate a hash value for a TEBV (Trial-Estimated Breeding Value) struct.

This method implements hashing for TEBV objects by combining the hash values of selected fields:
- traits: Vector of trait names
- formulae: Vector of formula strings
- phenomes: Vector of Phenomes objects

Note: For performance reasons, the following fields are deliberately excluded from the hash calculation:
- models
- df_BLUEs
- df_BLUPs

# Arguments
- `x::TEBV`: The TEBV struct to be hashed
- `h::UInt`: The hash value to be mixed with the object's hash

# Returns
- `UInt`: A unique hash value for the TEBV object

# Examples
```jldoctest; setup = :(using GBCore, MixedModels, DataFrames)
julia> tebv = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> typeof(hash(tebv))
UInt64
```
"""
function Base.hash(x::TEBV, h::UInt)::UInt
    # hash(TEBV, hash(x.traits, hash(x.formulae, hash(x.models, hash(x.df_BLUEs, hash(x.df_BLUPs, hash(x.phenomes, h)))))))
    hash(TEBV, hash(x.traits, hash(x.formulae, hash(x.phenomes, h))))
end


"""
    ==(x::TEBV, y::TEBV)::Bool

Compare two TEBV (Trial-Estimated Breeding Values) objects for equality.

This method implements equality comparison for TEBV structs by comparing their hash values.
Two TEBV objects are considered equal if they have identical values for all their fields:
traits, formulae, models, df_BLUEs, df_BLUPs, and phenomes.

# Arguments
- `x::TEBV`: First TEBV object to compare
- `y::TEBV`: Second TEBV object to compare

# Returns
- `Bool`: `true` if the TEBV objects are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GBCore, MixedModels, DataFrames)
julia> tebv_1 = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> tebv_2 = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> tebv_3 = TEBV(traits=["SOMETHING_ELSE"], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> tebv_1 == tebv_2
true

julia> tebv_1 == tebv_3
false
```
"""
function Base.:(==)(x::TEBV, y::TEBV)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(y::TEBV)::Bool

Check if all fields in the TEBV struct have compatible dimensions. The function verifies that
the length of all arrays in the TEBV struct match the number of traits.

# Arguments
- `tebv::TEBV`: A TEBV (Trial-estimated Breeding Values) struct containing traits,
  formulae, models, BLUEs, BLUPs, and phenomes.

# Returns
- `Bool`: Returns `true` if all fields have matching dimensions (equal to the number of traits),
  `false` otherwise.

# Details
The function checks if the following fields have the same length as `traits`:
- formulae
- unique models
- unique BLUEs DataFrames
- unique BLUPs DataFrames
- unique phenomes

# Examples
julia> tebv = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> checkdims(tebv)
true
```
"""
function checkdims(tebv::TEBV)::Bool
    t = length(tebv.traits)
    if (t != length(tebv.formulae)) ||
       (t != length(unique(tebv.models))) ||
       (t != length(unique(tebv.df_BLUEs))) ||
       (t != length(unique(tebv.df_BLUPs))) ||
       (t != length(unique(tebv.phenomes)))
        return false
    end
    true
end

"""
    dimensions(tebv::TEBV)::Dict{String, Int64}

Calculate various dimensional metrics for a TEBV (Trial-Estimated Breeding Values) struct.

# Arguments
- `tebv::TEBV`: A TEBV struct containing traits, formulae, models, BLUEs, BLUPs, and phenomes data

# Returns
A dictionary containing the following counts:
- `"n_entries"`: Number of unique entries across all phenomes
- `"n_populations"`: Number of unique populations across all phenomes
- `"n_traits"`: Number of traits in the TEBV struct
- `"n_total"`: Total number of observations across all traits
- `"n_zeroes"`: Total number of zero values across all traits
- `"n_missing"`: Total number of missing values across all traits
- `"n_nan"`: Total number of NaN values across all traits
- `"n_inf"`: Total number of Infinite values across all traits

# Throws
- `ArgumentError`: If the TEBV struct dimensions are inconsistent or corrupted

# Examples
```jldoctest; setup = :(using GBCore, MixedModels, DataFrames)
julia> tebv = TEBV(traits=["trait_1"], formulae=["trait_1 ~ 1 + 1|entries"], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> dimensions(tebv)
Dict{String, Int64} with 8 entries:
  "n_total"       => 1
  "n_zeroes"      => 0
  "n_nan"         => 0
  "n_entries"     => 1
  "n_traits"      => 1
  "n_inf"         => 0
  "n_populations" => 1
  "n_missing"     => 1
```
"""
function dimensions(tebv::TEBV)::Dict{String,Int64}
    if !checkdims(tebv)
        throw(ArgumentError("TEBV struct is corrupted."))
    end
    entries = tebv.phenomes[1].entries
    populations = tebv.phenomes[1].populations
    n_traits = length(tebv.traits)
    n_total = 0
    n_zeroes = 0
    n_missing = 0
    n_nan = 0
    n_inf = 0
    for i = 1:n_traits
        # i, trait = 1, tebv.traits[1]
        phenomes = tebv.phenomes[i]
        entries = unique(vcat(entries, tebv.phenomes[1].entries))
        populations = unique(vcat(populations, tebv.phenomes[1].populations))
        dims = dimensions(phenomes)
        n_total += dims["n_total"]
        n_zeroes += dims["n_zeroes"]
        n_missing += dims["n_missing"]
        n_nan += dims["n_nan"]
        n_inf += dims["n_inf"]
    end
    Dict(
        "n_entries" => length(entries),
        "n_populations" => length(populations),
        "n_traits" => n_traits,
        "n_total" => n_total,
        "n_zeroes" => n_zeroes,
        "n_missing" => n_missing,
        "n_nan" => n_nan,
        "n_inf" => n_inf,
    )
end

"""
    countlevels(df::DataFrame; column_names::Vector{String})::Int64

Count the total number of unique values (factor levels) across specified columns in a DataFrame.

# Arguments
- `df::DataFrame`: Input DataFrame to analyze
- `column_names::Vector{String}`: Vector of column names to count unique values from

# Returns
- `Int64`: Sum of unique values across all specified columns

# Throws
- `ArgumentError`: If any of the specified column names are not found in the DataFrame
"""
function countlevels(df::DataFrame; column_names::Vector{String})::Int64
    # trials, simulated_effects = simulatetrials(genomes = simulategenomes()); df::DataFrame = tabularise(trials); column_names::Vector{String} = ["years", "entries"]
    if length(names(df) ∩ column_names) != length(column_names)
        throw(ArgumentError("The supplied column names are not found in the dataframe."))
    end
    m::Int64 = 0
    for column_name in column_names
        m += length(unique(df[!, column_name]))
    end
    return m
end


"""
    @string2formula(x::String)

Convert a string representation of a formula into a `Formula` object.

This macro parses a string containing a formula expression and evaluates it into
a proper `Formula` object that can be used in statistical modeling.

# Arguments
- `x::String`: A string containing the formula expression (e.g., "y ~ x + z")

# Returns
- `Formula`: A Formula object representing the parsed expression
"""
macro string2formula(x)
    @eval(@formula($(Meta.parse(x))))
end


"""
    trialsmodelsfomulae!(df::DataFrame; trait::String, max_levels::Int64 = 100)::Tuple{Vector{String},Vector{Int64}}

Generate mixed model formulae for analyzing multi-environment trial data.

# Arguments
- `df::DataFrame`: Input DataFrame containing trial data, will be modified in-place
- `trait::String`: Name of the response variable column
- `max_levels::Int64=100`: Maximum number of factor levels allowed in interaction terms

# Returns
A tuple containing:
- `Vector{String}`: Collection of mixed model formulae with increasing complexity
- `Vector{Int64}`: Corresponding number of factor levels for each formula

# Details
The function:
1. Identifies available trial design variables (nesters, spatial components, targets)
2. Creates interaction terms between variables and adds them to the DataFrame
3. Generates model formulae considering:
   - Single and multi-environment models
   - Fixed and random entry effects
   - Spatial error components
   - Nested random effects
4. Filters redundant models and sorts by complexity

# Notes
- Warns if trials are unreplicated
- Throws error if only one entry is present
- Automatically removes block effects when both row and column effects are present
- Removes redundant nesting structures

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _simulated_effects = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> df = tabularise(trials);

julia> size(df)
(12800, 14)

julia> formulae, n_levels = trialsmodelsfomulae!(df, trait="trait_1");

julia> size(df)
(12800, 134)

julia> length(formulae)
76

julia> sum(n_levels .== sort(n_levels))
76
```
"""
function trialsmodelsfomulae!(
    df::DataFrame;
    trait::String,
    max_levels::Int64 = 100,
)::Tuple{Vector{String},Vector{Int64}}
    # trials, simulated_effects = simulatetrials(genomes = simulategenomes()); df::DataFrame = tabularise(trials); trait::String=trials.traits[1]; max_levels::Int64=100;
    # Define the totality of all expected variables
    nester_variables = ["years", "seasons", "harvests", "sites"]
    spatial_variables = ["blocks", "rows", "cols"]
    target_variables = ["entries", "populations"]
    residual_variable = ["replications"]
    # Extract the available (non-fixed) subset of all the variables listed above
    explanatory_variables = filter(
        x -> length(unique(df[!, x])) > 1,
        vcat(nester_variables, spatial_variables, target_variables, residual_variable),
    )
    # Warn if unreplicated
    if sum(explanatory_variables .== "replications") == 0
        @warn "Unreplicated trial data!"
    end
    # We expect to extract BLUEs/BLUPs of the entries
    if sum(explanatory_variables .== "entries") == 0
        throw(ErrorException("Only one entry in the entire trial!"))
    end
    # Exclude replications (residuals estimator), entries (what we're most interested in), and population (entry-specific and can be used instead of entries) in the formula
    explanatory_variables = filter(x -> x != "replications", explanatory_variables)
    explanatory_variables = filter(x -> x != "entries", explanatory_variables)
    explanatory_variables = filter(x -> x != "populations", explanatory_variables)
    # Permute
    vars::Array{Vector{String},1} = [[x] for x in explanatory_variables]
    for nv in explanatory_variables
        tmp_v = []
        for v in vars
            v = copy(v)
            push!(v, nv)
            sort!(v)
            unique!(v)
            push!(tmp_v, v)
        end
        sort!(tmp_v)
        unique!(tmp_v)
        append!(vars, tmp_v)
    end
    sort!(vars)
    unique!(vars)
    # Define models where we nest entries, blocks, rows and cols
    formulae::Vector{String} = []
    n_levels::Vector{Int64} = []
    for var in vars
        # var = vars[7]
        # Sort by decreasing nester
        spatials = var[[findall(var .== v)[1] for v in spatial_variables ∩ var]]
        nesters = var[[findall(var .== v)[1] for v in nester_variables ∩ var]]
        # Define the spatial errors nested within the nester variables
        spatial_error_column::String = ""
        if length(spatials) > 0
            spatial_error_column = join(vcat(nesters, spatials), "_x_")
            try
                df[!, spatial_error_column]
            catch
                nesters_spatials = vcat(nesters[2:end], spatials)
                df[!, spatial_error_column] = df[!, nesters_spatials[1]]
                for v in nesters_spatials[2:end]
                    df[!, spatial_error_column] = string.(df[!, spatial_error_column], "_x_", df[!, v])
                end
            end
        end

        # Single environment model
        if length(nesters) == 0
            if spatial_error_column != ""
                if countlevels(df; column_names = [spatial_error_column]) <= max_levels
                    # Fixed entries
                    push!(formulae, string(trait, " ~ 1 + entries + (0 + ", spatial_error_column, "|entries)"))
                    push!(n_levels, countlevels(df; column_names = ["entries", spatial_error_column]))
                    # Random entries
                    push!(formulae, string(trait, " ~ 1 + (1|entries) + (0 + ", spatial_error_column, "|entries)"))
                    push!(n_levels, countlevels(df; column_names = ["entries", spatial_error_column]))
                end
            else
                # Fixed entry effects
                # ... BLUEs would simply be means.
                # Random entries
                push!(formulae, string(trait, " ~ 1 + (0 + 1|entries)"))
                push!(n_levels, countlevels(df; column_names = ["entries"]))
            end
        end

        # Multi-environemnt models
        for i in eachindex(nesters)
            # i = 2
            # Divide the variables into fixed effects and random interaction effects
            fixed::Vector{String} = nesters[1:(end-i)]
            random_interaction::Vector{String} = nesters[(length(nesters)-(i-1)):end]
            if length(random_interaction) > 1
                interaction_column::String = join(random_interaction, "_x_")
                try
                    df[!, interaction_column]
                catch
                    df[!, interaction_column] = df[!, random_interaction[1]]
                    for v in random_interaction[2:end]
                        df[!, interaction_column] = string.(df[!, interaction_column], "_x_", df[!, v])
                    end
                end
                if length(unique(df[!, interaction_column])) > max_levels
                    continue
                end
            else
                interaction_column = random_interaction[1]
            end
            # Append the model
            if length(fixed) > 0
                if countlevels(df; column_names = [interaction_column]) <= max_levels
                    push!(
                        formulae,
                        string(trait, " ~ 1 + ", join(fixed, " + "), " + (0 + ", interaction_column, "|entries)"),
                    )
                    push!(n_levels, countlevels(df; column_names = vcat(["entries"], fixed, interaction_column)))
                end
                if (spatial_error_column != "") &&
                   (countlevels(df; column_names = [interaction_column, spatial_error_column]) <= max_levels)
                    push!(
                        formulae,
                        string(
                            trait,
                            " ~ 1 + ",
                            join(fixed, " + "),
                            " + (0 + ",
                            interaction_column,
                            "|entries)",
                            " + (0 + ",
                            spatial_error_column,
                            "|entries)",
                        ),
                    )
                    push!(
                        n_levels,
                        countlevels(
                            df;
                            column_names = vcat(["entries"], fixed, interaction_column, spatial_error_column),
                        ),
                    )
                end
            else
                if countlevels(df; column_names = [interaction_column]) <= max_levels
                    push!(formulae, string(trait, " ~ 1 + (0 + ", interaction_column, "|entries)"))
                    push!(n_levels, countlevels(df; column_names = ["entries", interaction_column]))
                end
                if (spatial_error_column != "") &&
                   (countlevels(df; column_names = [interaction_column, spatial_error_column]) <= max_levels)
                    push!(
                        formulae,
                        string(
                            trait,
                            " ~ 1 + (0 + ",
                            interaction_column,
                            "|entries)",
                            " + (0 + ",
                            spatial_error_column,
                            "|entries)",
                        ),
                    )
                    push!(
                        n_levels,
                        countlevels(df; column_names = ["entries", interaction_column, spatial_error_column]),
                    )
                end
            end
        end
    end
    # Keep only unique formulae
    idx = unique(i -> formulae[i], eachindex(formulae))
    formulae = formulae[idx]
    n_levels = n_levels[idx]
    # Sort formulae by complexity-ish
    idx = sortperm(n_levels)
    formulae = formulae[idx]
    n_levels = n_levels[idx]
    # Remove models with blocks if rows and columns exist
    if length(unique(vcat(vars...)) ∩ ["rows", "cols"]) == 2
        idx = findall(match.(r"blocks", formulae) .== nothing)
        formulae = formulae[idx]
        n_levels = n_levels[idx]
    end
    # Remove models with redundant nesters in the main and residual terms
    idx = findall([
        (sum(match.(r"years_x_seasons_x_harvests_x_sites", x) .!= nothing) < 2) &&
        (sum(match.(r"years_x_seasons_x_sites", x) .!= nothing) < 2) for x in split.(formulae, " + (")
    ])
    formulae = formulae[idx]
    n_levels = n_levels[idx]
    # Output in addition to the mutated `df`
    (formulae, n_levels)
end


"""
    analyse(
        df::DataFrame; 
        formulae::Vector{String},
        idx_parallel_models::Vector{Int64},
        idx_iterative_models::Vector{Int64},
        max_time_per_model::Int64 = 60,
        verbose::Bool=false
    )::Tuple{String, Any, DataFrame, DataFrame, Phenomes}

Fit univariate linear mixed models to extract entry effects from the best-fitting model.

# Arguments
- `df::DataFrame`: Input data frame containing trial data with columns for entries, traits, and other experimental factors
- `formulae::Vector{String}`: Vector of model formulae strings to be tested
- `idx_parallel_models::Vector{Int64}`: Indices of simpler models to be fitted in parallel
- `idx_iterative_models::Vector{Int64}`: Indices of complex models to be fitted iteratively
- `max_time_per_model::Int64`: Maximum time in seconds allowed for fitting each model (default: 60)
- `verbose::Bool`: Whether to display progress information (default: false)

# Returns
A tuple containing:
1. String: Formula of the best-fitting model
2. Any: The fitted model object
3. DataFrame: BLUEs (Best Linear Unbiased Estimates) results
4. DataFrame: BLUPs (Best Linear Unbiased Predictions) results
5. Phenomes: Struct containing consolidated phenotypic predictions

# Details
The function implements a mixed model fitting strategy with the following principles:
- Avoids over-parameterization
- Uses unstructured variance-covariance matrix for random effects
- Prefers REML over ML estimation
- Compares BLUEs vs BLUPs of entries
- Handles both parallel and iterative model fitting based on model complexity

# Notes
- All formulae must model the same trait
- Models are fitted using REML
- Simple models are fitted in parallel while complex models are fitted iteratively to avoid memory issues
- Returns empty results if no models can be successfully fitted

# Examples
```jldoctest; setup = :(using GBCore, StatsBase, DataFrames, MixedModels)
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> df = tabularise(trials);

julia> formulae, n_levels = trialsmodelsfomulae!(df; trait = "trait_1", max_levels = 10);

julia> idx_parallel_models::Vector{Int64} = findall(n_levels .<= (15));

julia> idx_iterative_models::Vector{Int64} = findall((n_levels .<= (15)) .!= true);

julia> formula_string, model, df_BLUEs, df_BLUPs, phenomes = analyse(df, formulae=formulae, idx_parallel_models=idx_parallel_models, idx_iterative_models=idx_iterative_models);

julia> length(phenomes.entries) == length(unique(df.entries))
true

julia> df_2 = df[(df.years .== df.years[1]) .&& (df.harvests .== df.harvests[1]) .&& (df.seasons .== df.seasons[1]) .&& (df.sites .== df.sites[1]) .&& (df.replications .== df.replications[1]), :];

julia> formula_string_2, model_2, df_BLUEs_2, df_BLUPs_2, phenomes_2 = analyse(df_2, formulae=["trait_1 ~ 1 + 1|entries"]);

julia> cor(phenomes_2.phenotypes[sortperm(phenomes_2.entries),1], df_2.trait_1[sortperm(df_2.entries)]) > 0.99
true
```
"""
function analyse(
    df::DataFrame;
    formulae::Vector{String},
    idx_parallel_models::Vector{Int64} = [1],
    idx_iterative_models::Vector{Int64} = [1],
    max_time_per_model::Int64 = 60,
    verbose::Bool = false,
)::Tuple{String,Any,DataFrame,DataFrame,Phenomes}
    # trials, _ = simulatetrials(genomes = simulategenomes()); max_levels::Int64=100; max_time_per_model::Int64=60; verbose::Bool = true;
    # df::DataFrame = tabularise(trials)
    # formulae, n_levels = trialsmodelsfomulae!(df; trait = "trait_1", max_levels = 10)
    # # formulae = ["trait_1 ~ 1 + years + (1|entries)"]
    # idx_parallel_models::Vector{Int64} = findall(n_levels .<= (1.5 * max_levels))
    # idx_iterative_models::Vector{Int64} = findall((n_levels .<= (1.5 * max_levels)) .!= true)
    # max_time_per_model = 60; verbose = true
    # Extract entries
    entries::Vector{String} = sort(unique(df[!, "entries"]))
    # Extract the trait
    trait = strip(split(formulae[1], "~")[1])
    for i in eachindex(formulae)
        if strip(split(formulae[i], "~")[1]) != trait
            throw(
                ArgumentError(
                    "All the formulae need to model the same trait, i.e. " *
                    trait *
                    "; but the `formulae[" *
                    string(i) *
                    "] models the trait: " *
                    strip(split(formulae[i], "~")[1]) *
                    ".",
                ),
            )
        end
    end
    if length(idx_parallel_models) > 0
        if (minimum(idx_parallel_models) < 1) || maximum(idx_parallel_models) > length(formulae)
            throw(
                ArgumentError(
                    "Incorrect `idx_parallel_models`: indexes for the formula ranges from 1 to " *
                    string(length(formulae)) *
                    ".",
                ),
            )
        end
    end
    if length(idx_iterative_models) > 0
        if (minimum(idx_iterative_models) < 1) || maximum(idx_iterative_models) > length(formulae)
            throw(
                ArgumentError(
                    "Incorrect `idx_iterative_models`.: indexes for the formula ranges from 1 to " *
                    string(length(formulae)) *
                    ".",
                ),
            )
        end
    end
    # Remove intersection between models to processed in parallel and iteratively
    idx_parallel_models = setdiff(idx_parallel_models, idx_iterative_models)
    if verbose
        println(string("\t ‣ models to fit = ", length(formulae)))
        println(string("\t ‣ parallel fitting for the ", length(idx_parallel_models), " simplest model/s"))
        println(string("\t ‣ iterative fitting for the ", length(idx_iterative_models), " most complex model/s"))
        println(string("\t ‣ simplest model: ", formulae[1]))
        println(string("\t ‣ most complex model: ", formulae[end]))
    end
    # Fit the models
    models::Vector{LinearMixedModel{Float64}} = Vector{LinearMixedModel{Float64}}(undef, length(formulae))
    deviances::Vector{Float64} = fill(Inf64, length(formulae))
    # Parallel fitting of models with total number of levels at or below `(1.5 * max_levels)`
    if verbose
        pb = Progress(length(idx_parallel_models); desc = "Trials analyses | parallel model fitting: ")
    end
    thread_lock::ReentrantLock = ReentrantLock()
    Threads.@threads for i in idx_parallel_models
        # for i in idx_parallel_models
        f = @eval(@string2formula $(formulae[i]))
        model = try
            MixedModel(f, df)
        catch
            continue
        end
        @lock thread_lock model.optsum.REML = true
        @lock thread_lock model.optsum.maxtime = max_time_per_model
        @lock thread_lock try
            fit!(model, progress = false)
        catch
            try
                fit!(model, progress = false)
            catch
                continue
            end
        end
        @lock thread_lock models[i] = model
        @lock thread_lock deviances[i] = deviance(model)
        if verbose
            next!(pb)
        end
    end
    if verbose
        finish!(pb)
    end
    # Iterative fitting of models with total number of levels above `(1.5 * max_levels)` to minimise OOM errors from occuring
    if verbose
        pb = Progress(length(idx_parallel_models); desc = "Trials analyses | iterative model fitting: ")
    end
    for i in idx_iterative_models
        # println(i)
        f = @eval(@string2formula $(formulae[i]))
        model = try
            MixedModel(f, df)
        catch
            continue
        end
        model.optsum.REML = true
        model.optsum.maxtime = max_time_per_model
        try
            fit!(model, progress = false)
        catch
            try
                fit!(model, progress = false)
            catch
                continue
            end
        end
        models[i] = model
        deviances[i] = deviance(model)
        if verbose
            next!(pb)
        end
    end
    if verbose
        finish!(pb)
    end
    # Return an empty handed if we failed to fit any model
    if sum(.!isinf.(deviances)) == 0
        # No mixed model fitting possible
        return ("", missing, DataFrame(), DataFrame(), Phenomes(n = 0, t = 0))
    end
    # Find the best-fitting model
    model = models[argmin(deviances)]
    # Extract the dataframes of the fixed and random effects
    df_BLUEs = DataFrame(coeftable(model))
    df_BLUEs.Name = replace.(df_BLUEs.Name, "entries: " => "")
    df_BLUPs = DataFrame(raneftables(model)[1])
    for i = 1:length(raneftables(model))
        df_tmp = DataFrame(raneftables(model)[i])
        if names(df_tmp)[1] == "entries"
            df_BLUPs = df_tmp
        end
    end
    # Accummulate fixed effects first
    ϕ_blues::Vector{Float64} = fill(0.0, length(entries))
    intercept = 0.0
    idx_intercept = findall(df_BLUEs.Name .== "(Intercept)")
    if length(idx_intercept) == 1
        intercept = df_BLUEs[idx_intercept[1], "Coef."]
        idx_entries = findall([sum(entries .== x) > 0 for x in df_BLUEs.Name])
        if length(idx_entries) == (length(entries) - 1)
            df_BLUEs.Name[idx_intercept] = setdiff(entries, df_BLUEs.Name[idx_entries])
            df_BLUEs[idx_intercept[1], "Coef."] = 0.0
        end
    end
    for (i, entry) in enumerate(entries)
        # i = 2; entry = entries[i]
        idx = findall(df_BLUEs.Name .== entry)
        if length(idx) == 1
            ϕ_blues[i] = df_BLUEs[idx[1], "Coef."]
        end
    end
    if length(idx_intercept) == 1
        ϕ_blues .+= intercept
    end
    # Include the random effects
    idx_entries = findall(names(df_BLUPs) .== "entries")
    idx_intercept = findall(names(df_BLUPs) .== "(Intercept)")
    idx_col::Vector{Int64} = findall(match.(r"^entries$|^\(Intercept\)$|blocks|rows|cols", names(df_BLUPs)) .== nothing)
    ψ_blups, blup_names =
        if ((length(idx_entries) == 1) .&& (length(idx_intercept) == 1)) ||
           ((length(idx_entries) == 1) .&& (length(idx_col) > 0))
            # Instantiate the output matrix and blup names
            ψ_blups::Matrix{Float64} = fill(0.0, length(entries), length(idx_intercept) + length(idx_col))
            blup_names::Vector{String} = fill("", length(idx_intercept) + length(idx_col))
            if length(idx_intercept) == 1
                # Find non-spatial interaction effects with the entries (Assumes the random effects have 1 intercept and only 1 intercept)
                ψ_blups[:, 1] = df_BLUPs[:, idx_intercept[1]]
                blup_names[1] = "intercept"
                # Add the intercept effects to the nester variable levels
                if length(idx_col) > 0
                    ψ_blups[:, 2:end] .= df_BLUPs[:, idx_col] .+ df_BLUPs[:, idx_intercept[1]]
                    # Find which nester variable level is the intercept
                    variables_present = unique([x[1] for x in split.(names(df_BLUPs)[idx_col], ": ")])
                    levels_present = [x[2] for x in split.(names(df_BLUPs)[idx_col], ": ")]
                    levels_all = []
                    for var in variables_present
                        append!(levels_all, unique(df[!, var]))
                    end
                    blup_names[1] = string("intercept_", join(setdiff(levels_all, levels_present), "_"))
                    blup_names[2:end] = colnames_df_BLUPs[idx_col]
                end
            elseif length(idx_col) > 0
                ψ_blups = Matrix(df_BLUPs[:, idx_col])
                blup_names = levels_present = [x[2] for x in split.(names(df_BLUPs)[idx_col], ": ")]
            end
            (ψ_blups, blup_names)
        else
            (fill(0.0, 0, 0), [])
        end
    # Find the final set of phenotypes and trait names
    Y, trait_names = if size(ψ_blups, 1) == length(entries)
        (ϕ_blues .+ ψ_blups, string.(trait, "|", blup_names))
    else
        (hcat(ϕ_blues), [trait])
    end
    if size(Y) != (length(entries), length(trait_names))
        throw(ErrorException("Error extracting and consolidating BLUEs and BLUPs."))
    end
    # Output
    phenomes = Phenomes(n = length(entries), t = length(trait_names))
    phenomes.entries = entries
    for (i, entry) in enumerate(phenomes.entries)
        idx = findall(df.entries .== entry)[1]
        phenomes.populations[i] = df.populations[idx]
    end
    phenomes.traits = trait_names
    phenomes.phenotypes = Y
    phenomes.mask .= true
    if !checkdims(phenomes)
        throw(ErrorException("Error building the phenomes struct from the BLUEs and BLUPs"))
    end
    # Output
    (formulae[argmin(deviances)], model, df_BLUEs, df_BLUPs, phenomes)
end


"""
    analyse(
        trials::Trials,
        formula_string::String = "";
        traits::Union{Nothing,Vector{String}} = nothing,
        max_levels::Int64 = 100,
        max_time_per_model::Int64 = 60,
        covariates_continuous::Union{Nothing,Vector{String}} = nothing,
        verbose::Bool = true
    )::TEBV

Analyze trial data using linear mixed models to estimate Best Linear Unbiased Estimates (BLUEs) 
and Best Linear Unbiased Predictions (BLUPs).

# Arguments
- `trials`: A Trials struct containing the experimental data
- `formula_string`: Optional model formula string. If empty, automatic model selection is performed
- `traits`: Optional vector of trait names to analyze. If nothing, all traits are analyzed
- `max_levels`: Maximum number of levels for non-entry random effects (default: 100)
- `max_time_per_model`: Maximum fitting time in seconds per model (default: 60)
- `covariates_continuous`: Optional vector of continuous covariates to include in models
- `verbose`: Whether to display analysis progress (default: true)

# Returns
A `TEBV` struct containing:
- `traits`: Vector of analyzed trait names
- `formulae`: Vector of best-fitting model formulae
- `models`: Vector of fitted LinearMixedModel objects
- `df_BLUEs`: Vector of DataFrames containing BLUEs
- `df_BLUPs`: Vector of DataFrames containing BLUPs
- `phenomes`: Vector of Phenomes objects with predicted values

# Details
The function implements a mixed model fitting strategy with the following principles:
- Avoids over-parameterization
- Uses unstructured variance-covariance matrix for random effects
- Prefers REML over ML estimation
- Compares BLUEs vs BLUPs of entries
- Handles both parallel and iterative model fitting based on model complexity

# Notes
- Models are fitted using REML
- Simple models are fitted in parallel while complex models are fitted iteratively to avoid memory issues
- Returns empty results if no models can be successfully fitted

# Examples
```jldoctest; setup = :(using GBCore, StatsBase, Suppressor)
julia> trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);

julia> tebv_1 = analyse(trials, "trait_1 ~ 1 + (1|entries)", max_levels=50, verbose=false);

julia> tebv_1.traits
3-element Vector{String}:
 "trait_1"
 "trait_2"
 "trait_3"

julia> tebv_2 = analyse(trials, max_levels=50, verbose=false);

julia> mean(tebv_2.phenomes[1].phenotypes) < mean(tebv_2.phenomes[2].phenotypes)
true

julia> trials = addcompositetrait(trials, composite_trait_name = "covariate", formula_string = "(trait_1 + trait_2) / (trait_3 + 0.0001)");

julia> tebv_3 = Suppressor.@suppress analyse(trials, "y ~ 1 + covariate + entries + (1|blocks)", max_levels=50, verbose=false);

julia> mean(tebv_3.phenomes[1].phenotypes) < mean(tebv_3.phenomes[2].phenotypes)
true

julia> tebv_4 = Suppressor.@suppress analyse(trials, max_levels=50, covariates_continuous=["covariate"], verbose=false);

julia> mean(tebv_4.phenomes[1].phenotypes) < mean(tebv_4.phenomes[2].phenotypes)
true
```
"""
function analyse(
    trials::Trials,
    formula_string::String = "";
    traits::Union{Nothing,Vector{String}} = nothing,
    max_levels::Int64 = 100,
    max_time_per_model::Int64 = 60,
    covariates_continuous::Union{Nothing,Vector{String}} = nothing,
    verbose::Bool = true,
)::TEBV
    # trials, simulated_effects = simulatetrials(genomes = simulategenomes()); formula_string="y ~ 1 + (1|entries)"; max_levels::Int64=100; max_time_per_model::Int64=60; verbose::Bool = true;
    # trials, simulated_effects = simulatetrials(genomes = simulategenomes(n=5), n_years=2, n_seasons=2, n_harvests=1, n_sites=2, n_replications=10); formula_string=""; max_levels::Int64=100; max_time_per_model::Int64=60; verbose::Bool = true;
    # trials, simulated_effects = simulatetrials(genomes = simulategenomes(n=5), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10); formula_string=""; max_levels::Int64=100; max_time_per_model::Int64=60; verbose::Bool = true;
    # fname = "/mnt/c/Users/jp3h/Downloads/Lucerne-2024-10-leaf_to_stem_ratio.txt"; using GBIO; trials = GBIO.readdelimited(Trials, fname=fname, sep="\t"); formula_string=""; max_levels::Int64=10; max_time_per_model::Int64=2; verbose::Bool = true;
    # Check Arguments
    if !checkdims(trials)
        throw(ArgumentError("Trials struct is corrupted."))
    end
    traits = if isnothing(traits)
        unique(trials.traits)
    else
        idx_missing_traits = findall([sum(trials.traits .== trait) == 0 for trait in traits])
        if length(idx_missing_traits) > 0
            throw(
                ArgumentError(
                    "The following requested traits were not found in the Trials struct:\n\t‣ " *
                    join(traits[idx_missing_traits], "\n\t‣ "),
                ),
            )
        end
        traits
    end
    # Tabularise
    df::DataFrame = tabularise(trials)
    # Rename for operation symbol_strings into underscores in the trait names
    symbol_strings::Vector{String} = ["+", "-", "*", "/", "%"]
    for i in eachindex(symbol_strings)
        trials.traits = replace.(trials.traits, symbol_strings[i] => "_")
        rename!(df, replace.(names(df), symbol_strings[i] => "_"))
    end
    # Number of entries whose BLUEs or BLUPs we wish to extract
    entries::Vector{String} = sort(unique(df[!, "entries"]))
    n::Int64 = length(entries)
    if verbose
        println("Analysing trial data with:")
        println(string("\t ‣ total observations = ", nrow(df)))
        println(string("\t ‣ entries = ", n))
        println(string("\t ‣ populations = ", length(unique(trials.populations))))
        println(string("\t ‣ traits = ", length(trials.traits)))
        println(string("\t ‣ years = ", length(unique(trials.years))))
        println(string("\t ‣ seasons = ", length(unique(trials.seasons))))
        println(string("\t ‣ harvests = ", length(unique(trials.harvests))))
        println(string("\t ‣ sites = ", length(unique(trials.sites))))
        println(string("\t ‣ replications = ", length(unique(trials.replications))))
        println(string("\t ‣ blocks = ", length(unique(trials.blocks))))
        println(string("\t ‣ rows = ", length(unique(trials.rows))))
        println(string("\t ‣ cols = ", length(unique(trials.cols))))
    end
    # Iterate across each trait
    out_traits::Vector{String} = []
    out_formulae::Vector{String} = []
    out_models::Vector{LinearMixedModel{Float64}} = []
    out_df_BLUEs::Vector{DataFrame} = []
    out_df_BLUPs::Vector{DataFrame} = []
    out_phenomes::Vector{Phenomes} = []
    for trait in traits
        # trait = traits[1]
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println(trait)
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        end
        # Define the formulae
        formulae::Vector{String} = []
        idx_parallel_models::Vector{Int64} = []
        idx_iterative_models::Vector{Int64} = []
        if formula_string == ""
            formulae, n_levels = trialsmodelsfomulae!(df; trait = trait, max_levels = max_levels)
            idx_parallel_models = findall(n_levels .<= (1.5 * max_levels))
            idx_iterative_models = findall((n_levels .<= (1.5 * max_levels)) .!= true)
            # Reset formula_string string for the next trait
            formula_string = ""
        else
            formulae = [replace(formula_string, split(formula_string, "~")[1] => trait)]
            idx_parallel_models = []
            idx_iterative_models = [1]
        end
        # Append the covariates singly or in pairs right after the intercept of each formula,
        # then add them to the list of models to be iteratively fit 9so that they may be easily debugged)
        if !isnothing(covariates_continuous)
            c = length(covariates_continuous)
            covariates_singly_and_pairs = deepcopy(covariates_continuous)
            if c > 1
                for i = 1:(c-1)
                    for j = (i+1):c
                        push!(
                            covariates_singly_and_pairs,
                            string(covariates_continuous[i], " + ", covariates_continuous[j]),
                        )
                    end
                end
            end
            f = length(formulae)
            for i = 1:f
                for v in covariates_singly_and_pairs
                    # v = covariates_singly_and_pairs[4]
                    push!(formulae, replace(formulae[i], "~ 1" => string("~ 1 + ", v)))
                    push!(idx_iterative_models, length(formulae))
                end
            end
        end
        formula_optim, model, df_BLUEs, df_BLUPs, phenomes = analyse(
            df;
            formulae = formulae,
            idx_parallel_models = idx_parallel_models,
            idx_iterative_models = idx_iterative_models,
            max_time_per_model = max_time_per_model,
            verbose = verbose,
        )
        if ismissing(model)
            continue
        end
        # Update output vectors
        push!(out_traits, trait)
        push!(out_formulae, formula_optim)
        push!(out_models, model)
        push!(out_df_BLUEs, df_BLUEs)
        push!(out_df_BLUPs, df_BLUPs)
        push!(out_phenomes, phenomes)
    end
    # Output
    tebv = TEBV(
        traits = out_traits,
        formulae = out_formulae,
        models = out_models,
        df_BLUEs = out_df_BLUEs,
        df_BLUPs = out_df_BLUPs,
        phenomes = out_phenomes,
    )
    if !checkdims(tebv)
        throw(ErrorException("Error analysing the trials."))
    end
    tebv
end

"""
    extractphenomes(tebv::TEBV)::Phenomes

Extract phenotypic values from a Trial-Estimated Breeding Value (TEBV) object.

This function processes phenotypic data from a TEBV object, handling intercept effects
and merging multiple phenomes if present. It performs the following operations:

1. Validates input TEBV dimensions
2. Processes intercept effects if present by:
   - Identifying intercept terms
   - Combining intercept values with trait effects
   - Adjusting trait names and phenotypic values accordingly
3. Merges multiple phenomes if present
4. Renames traits to match input TEBV traits if dimensions align
5. Validates output Phenomes dimensions

# Arguments
- `tebv::TEBV`: A Trial Estimated Breeding Value object containing phenotypic data

# Returns
- `Phenomes`: A Phenomes object containing processed phenotypic values

# Throws
- `ArgumentError`: If input TEBV or output Phenomes dimensions are invalid

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);

julia> tebv = analyse(trials, max_levels=50, verbose=false);

julia> phenomes = extractphenomes(tebv);

julia> phenomes.traits == ["trait_1", "trait_2", "trait_3"]
true
```
"""
function extractphenomes(tebv::TEBV)::Phenomes
    # trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);
    # tebv = analyse(trials, max_levels=50, verbose=false);
    if !checkdims(tebv)
        throw(ArgumentError("The TEBV struct is corrupted."))
    end
    phenomes = Phenomes(n = length(tebv.phenomes[1].entries), t = 1)
    for i in eachindex(tebv.phenomes)
        # i = 2
        phenomes_i = clone(tebv.phenomes[i])
        # Add intercept effects if present
        bool_intercept = .!isnothing.(match.(Regex("(Intercept)"), phenomes_i.traits))
        n = length(phenomes_i.entries)
        t = length(phenomes_i.traits)
        if sum(bool_intercept) == 1
            idx_intercept = findall(bool_intercept)[1]
            idx_ϕ = findall(.!bool_intercept)
            traits = repeat([""], t - 1)
            ϕ::Matrix{Union{Missing,Float64}} = fill(0.0, n, t - 1)
            μ::Matrix{Bool} = fill(true, n, t - 1)
            for (i, j) in enumerate(idx_ϕ)
                traits[i] = phenomes_i.traits[j]
                ϕ[:, i] = phenomes_i.phenotypes[:, idx_intercept] + phenomes_i.phenotypes[:, j]
                μ[:, i] = phenomes_i.mask[:, j]
            end
            phenomes_i.traits = traits
            phenomes_i.phenotypes = ϕ
            phenomes_i.mask = μ
        else
            phenomes_i.traits = phenomes_i.traits
            phenomes_i.phenotypes = phenomes_i.phenotypes
            phenomes_i.mask = phenomes_i.mask
        end
        if i == 1
            phenomes = clone(phenomes_i)
        else
            phenomes = merge(phenomes, phenomes_i)
        end
    end
    # Rename the traits to match the input TEBV if we have the same number of output traits
    if length(phenomes.traits) == length(tebv.traits)
        phenomes.traits = tebv.traits
    end
    # Output
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted."))
    end
    phenomes
end

"""
    plot(tebv::TEBV)

Create a visualization of True Estimated Breeding Values (TEBV) analysis results.

This function extracts phenomes from the TEBV object and generates a plot to visualize
the breeding value estimates.

# Arguments
- `tebv::TEBV`: A TEBV object containing the analysis results

# Returns
- A plot object representing the visualization of the phenomes data
"""
function plot(tebv::TEBV)
    # trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);
    # tebv = analyse(trials, max_levels=50, verbose=false);
    phenomes = extractphenomes(tebv)
    plot(phenomes)
end
