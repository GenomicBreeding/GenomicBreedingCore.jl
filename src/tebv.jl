"""
    clone(x::TEBV)::TEBV

Clone a TEBV object

## Example
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

Hash a TEBV struct using the traits, formualae and phenomes.
We deliberately excluded the models, df_BLUEs, and df_BLUPs for efficiency.

## Examples
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
    Base.:(==)(x::TEBV, y::TEBV)::Bool

Equality of TEBV structs using the hash function defined for TEBV structs.

## Examples
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

Check dimension compatibility of the fields of the TEBV struct

## Examples
```jldoctest; setup = :(using GBCore, StatsBase, MixedModels, DataFrames)
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

Count the number of entries, populations, and traits in the TEBV struct

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

Count the total number of factor levels across the column names provided.
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
    string2formula(x)

Macro to `Meta.parse` a string into a formula.
"""
macro string2formula(x)
    @eval(@formula($(Meta.parse(x))))
end


"""
    trialsmodelsfomulae!(df::DataFrame; trait::String, max_levels::Int64 = 100)::Vector{String}

Define formulae for the mixed models to fit on the tabularised `Trials` struct.
    - appends interaction effects intto `df`
    - returns:
        + a vector of formulae as strings
        + a vector of the total number of non-entry factor levels

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
        verbose::Bool=false)::Tuple{String, Any, DataFrame, DataFrame, Phenomes}

Fit univarite (one trait) linear mixed models to extract the effects of the entries the best fitting model.

We have the following guiding principles:

- Avoid over-parameterisation we'll have enough of that with the genomic prediction models
- We will fit mixed models with unstructure variance-covariance matrix of the random effects
- We prefer REML over ML
- We compare BLUEs vs BLUPs of entries

# Examples
```jldoctest; setup = :(using GBCore, StatsBase, DataFrames, MixedModels)
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> df::DataFrame = tabularise(trials);

julia> formulae, n_levels = trialsmodelsfomulae!(df; trait = "trait_1", max_levels = 10);

julia> idx_parallel_models::Vector{Int64} = findall(n_levels .<= (15));

julia> idx_iterative_models::Vector{Int64} = findall((n_levels .<= (15)) .!= true);

julia> formula_string, model, df_BLUEs, df_BLUPs, phenomes = analyse(df, formulae=formulae, idx_parallel_models=idx_parallel_models, idx_iterative_models=idx_iterative_models);

julia> length(phenomes.entries) == length(unique(df.entries))
true

julia> df_2 = df[(df.years .== df.years[1]) .&& (df.harvests .== df.harvests[1]) .&& (df.seasons .== df.seasons[1]) .&& (df.sites .== df.sites[1]) .&& (df.replications .== df.replications[1]), :];

julia> formula_string_2, model_2, df_BLUEs_2, df_BLUPs_2, phenomes_2 = analyse(df_2, formulae=["trait_1 ~ 1 + 1|entries"]);

julia> cor(phenomes_2.phenotypes[sortperm(phenomes_2.entries),1], df_2.trait_1[sortperm(df_2.entries)]) == 1.00
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
    # idx_parallel_models::Vector{Int64} = findall(n_levels .<= (1.5 * max_levels))
    # idx_iterative_models::Vector{Int64} = findall((n_levels .<= (1.5 * max_levels)) .!= true)
    # max_time_per_model = 60; verbose = true
    # Extract entries
    entries::Vector{String} = sort(unique(df[!, "entries"]))
    n::Int64 = length(entries)
    # Extract the trait
    trait::String = strip(split(formulae[1], "~")[1])
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
        model = MixedModel(f, df)
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
        model = MixedModel(f, df)
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
    if sum(.!isinf.(deviances)) == 0
        # No mixed model fitting possible
        return ("", missing, DataFrame(), DataFrame(), Phenomes(n = 0, t = 0))
    end
    model = models[argmin(deviances)]
    # Fixed effects
    df_BLUEs = DataFrame(coeftable(model))
    # Determine the effect of the first entry and add the intercept to the effects of the other entries
    idx_row = findall(match.(r"entries: ", df_BLUEs[!, "Name"]) .!= nothing)
    if length(idx_row) == (n - 1)
        df_BLUEs[idx_row, "Name"] = replace.(df_BLUEs[idx_row, "Name"], "entries: " => "")
        df_BLUEs[1, "Name"] = setdiff(entries, df_BLUEs[idx_row, "Name"])[1]
        df_BLUEs[2:end, "Coef."] .+= df_BLUEs[1, "Coef."]
    end
    # Random effects
    df_BLUPs = DataFrame(only(raneftables(model)))
    # Find non-spatial interaction effects with the entries
    idx_col::Vector{Int64} = findall(
        (match.(r"^entries$|blocks|rows|cols", names(df_BLUPs)) .== nothing) .||
        (match.(r"^\(Intercept\)$", names(df_BLUPs)) .!= nothing),
    )
    # Instantiate the Phenomes struct
    t::Int64 = length(idx_col)
    if t == 0
        if length(idx_row) == (n - 1)
            t = 1
        else
            # No BLUEs nor BLUPs
            return ("", missing, df_BLUEs, df_BLUPs, Phenomes(n = 0, t = 0))
        end
    end
    phenomes = Phenomes(n = length(entries), t = t)
    phenomes.mask .= true
    # If we have BLUPs (multiple columns or "multiple traits": i.e. nested effects)
    if length(idx_col) > 1
        phenomes.entries = df_BLUPs[:, :entries]
        phenomes.phenotypes = Matrix(df_BLUPs[:, idx_col])
        phenomes.traits = string.(trait, "|", names(df_BLUPs))[idx_col]
    end
    # If we have a single set of BLUPs for the entries, i.e. no nesting, then we add the intercept
    if length(idx_col) == 1
        phenomes.entries = df_BLUPs[:, :entries]
        phenomes.phenotypes = Matrix(df_BLUPs[:, idx_col]) .+ df_BLUEs[1, "Coef."] # The first row of df_BLUEs is the intercept
        phenomes.traits[1] = trait
    end
    # If we have BLUEs (1 column or "1 trait" only)
    if length(idx_row) == (n - 1)
        phenomes.entries = df_BLUEs[!, "Name"]
        phenomes.phenotypes[:, 1] = df_BLUEs[!, "Coef."]
        phenomes.traits[1] = trait
    end
    # Populate populations
    for (i, entry) in enumerate(phenomes.entries)
        idx = findall(df.entries .== entry)[1]
        phenomes.populations[i] = df.populations[idx]
    end
    # Output
    (formulae[argmin(deviances)], model, df_BLUEs, df_BLUPs, phenomes)
end


"""
    analyse(
        trials::Trials,
        formula_string::String = "";
        max_levels::Int64 = 100,
        max_time_per_model::Int64 = 60,
        verbose::Bool = true,
    )::TEBV

# Analyse trials

## Arguments
- `trials`: Trials struct 
- `max_levels`: maximum number of non-entry factor levels to include in the linear mixed models (default = 100)
- `max_time_per_model`: maximum time in seconds for fitting each linear mixed model (default = 60)
- `verbose`: Show trials analysis progress bar? (default = true)

## Outputs
- `TEBV` struct containing the trait names, the best fitting formulae, models, BLUEs, and BLUPs for each trait

## Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);

julia> tebv = analyse(trials, max_levels=50, verbose=false);

julia> tebv.traits
3-element Vector{String}:
 "trait_1"
 "trait_2"
 "trait_3"

julia> checkdims(tebv)
true
```
"""
function analyse(
    trials::Trials,
    formula_string::String = "";
    max_levels::Int64 = 100,
    max_time_per_model::Int64 = 60,
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
    for trait in unique(trials.traits)
        # trait = unique(trials.traits)[1]
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
            formula_string = replace(formula_string, split(formula_string, "~")[1] => trait)
            formulae = [formula_string]
            idx_parallel_models = []
            idx_iterative_models = [1]
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
    TEBV(
        traits = out_traits,
        formulae = out_formulae,
        models = out_models,
        df_BLUEs = out_df_BLUEs,
        df_BLUPs = out_df_BLUPs,
        phenomes = out_phenomes,
    )
end

"""
    extractphenomes(tebv::TEBV)::Phenomes

Extract Phenomes from TEBV

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
    phenomes.traits = tebv.phenomes[1].traits
    phenomes.entries = tebv.phenomes[1].entries
    phenomes.populations = tebv.phenomes[1].populations
    phenomes.phenotypes = tebv.phenomes[1].phenotypes
    phenomes.mask = tebv.phenomes[1].mask
    for i in eachindex(tebv.phenomes)
        # i = 2
        if sum(tebv.phenomes[i].traits .== phenomes.traits) > 0
            continue
        end
        phenomes = merge(phenomes, tebv.phenomes[i])
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted."))
    end
    phenomes
end

"""
    plot(tebv::TEBV)

Plot TEBV output by plotting the resulting Phenomes struct
"""
function plot(tebv::TEBV)
    # trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);
    # tebv = analyse(trials, max_levels=50, verbose=false);
    phenomes = extractphenomes(tebv)
    plot(phenomes)
end
