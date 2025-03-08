"""
    clone(x::Trials)::Trials

Create a deep copy of a `Trials` object, including all its fields.

This function performs a complete deep copy of the input `Trials` object,
ensuring that all nested data structures are also copied rather than referenced.

# Arguments
- `x::Trials`: The source `Trials` object to be cloned

# Returns
- `Trials`: A new `Trials` object containing copies of all data from the input

# Example
```jldoctest; setup = :(using GBCore)
julia> trials = Trials(n=2, t=2);

julia> copy_trials = clone(trials)
Trials(Union{Missing, Float64}[missing missing; missing missing], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""])
```
"""
function clone(x::Trials)::Trials
    y::Trials = Trials(n = length(x.entries), t = length(x.traits))
    y.entries = deepcopy(x.entries)
    y.phenotypes = deepcopy(x.phenotypes)
    y.traits = deepcopy(x.traits)
    y.years = deepcopy(x.years)
    y.seasons = deepcopy(x.seasons)
    y.harvests = deepcopy(x.harvests)
    y.sites = deepcopy(x.sites)
    y.replications = deepcopy(x.replications)
    y.blocks = deepcopy(x.blocks)
    y.rows = deepcopy(x.rows)
    y.cols = deepcopy(x.cols)
    y.entries = deepcopy(x.entries)
    y.populations = deepcopy(x.populations)
    y
end


"""
    Base.hash(x::Trials, h::UInt)::UInt

Compute a hash value for a `Trials` struct by recursively hashing all of its fields.

This method implements hash functionality for the `Trials` type, allowing `Trials` 
objects to be used as dictionary keys or in hash-based collections.

# Arguments
- `x::Trials`: The Trials struct to be hashed
- `h::UInt`: The hash value to be mixed with the object's hash

# Returns
- `UInt`: A hash value for the entire Trials struct

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials = Trials(n=2, t=2);

julia> typeof(hash(trials))
UInt64
```
"""
function Base.hash(x::Trials, h::UInt)::UInt
    hash(
        Trials,
        hash(
            x.phenotypes,
            hash(
                x.traits,
                hash(
                    x.years,
                    hash(
                        x.seasons,
                        hash(
                            x.harvests,
                            hash(
                                x.sites,
                                hash(
                                    x.replications,
                                    hash(x.blocks, hash(x.rows, hash(x.cols, hash(x.entries, hash(x.populations, h))))),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
end


"""
    ==(x::Trials, y::Trials)::Bool

Compare two `Trials` structs for equality by comparing their hash values.

Two `Trials` structs are considered equal if they have identical hash values, which implies
they have the same configuration parameters (number of trials `n` and time steps `t`).

# Arguments
- `x::Trials`: First Trials struct to compare
- `y::Trials`: Second Trials struct to compare

# Returns
- `Bool`: `true` if the Trials structs are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials_1 = trials = Trials(n=2, t=4);

julia> trials_2 = trials = Trials(n=2, t=4);

julia> trials_3 = trials = Trials(n=1, t=2);

julia> trials_1 == trials_2
true

julia> trials_1 == trials_3
false
```
"""
function Base.:(==)(x::Trials, y::Trials)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(trials::Trials)::Bool

Check dimension compatibility of all fields in a `Trials` struct.

This function verifies that the dimensions of all vector fields in the `Trials` struct are 
consistent with the size of the phenotypes matrix. Specifically, it checks:

- Number of traits (`t`) matches number of columns in phenotypes and length of traits vector
- Number of entries (`n`) matches number of rows in phenotypes and length of:
  * years
  * seasons
  * harvests
  * sites
  * replications
  * blocks
  * rows
  * cols
  * entries
  * populations

Returns `true` if all dimensions are compatible, `false` otherwise.

# Arguments
- `trials::Trials`: A Trials struct containing trial data

# Returns
- `Bool`: `true` if dimensions are compatible, `false` otherwise

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials = Trials(n=1, t=2);

julia> trials.entries = ["entry_1"]; trials.traits = ["trait_1", "trait_2"];

julia> checkdims(trials)
true

julia> trials.entries = ["entering_2_entries", "instead_of_just_1"];

julia> checkdims(trials)
false
```
"""
function checkdims(trials::Trials)::Bool
    n, t = size(trials.phenotypes)
    if (t != length(trials.traits)) ||
       (t != length(unique(trials.traits))) ||
       (n != length(trials.years)) ||
       (n != length(trials.seasons)) ||
       (n != length(trials.harvests)) ||
       (n != length(trials.sites)) ||
       (n != length(trials.replications)) ||
       (n != length(trials.blocks)) ||
       (n != length(trials.rows)) ||
       (n != length(trials.cols)) ||
       (n != length(trials.entries)) ||
       (n != length(trials.populations))
        return false
    end
    true
end

"""
    dimensions(trials::Trials)::Dict{String, Int64}

Calculate dimensional statistics of a `Trials` struct, returning a dictionary with counts of various elements.

# Arguments
- `trials::Trials`: A `Trials` struct containing trial data

# Returns
A `Dict{String, Int64}` with the following keys:
- `"n_traits"`: Number of unique traits
- `"n_years"`: Number of unique years
- `"n_seasons"`: Number of unique seasons
- `"n_harvests"`: Number of unique harvests
- `"n_sites"`: Number of unique sites
- `"n_replications"`: Number of unique replications
- `"n_blocks"`: Number of unique blocks
- `"n_rows"`: Number of unique rows
- `"n_cols"`: Number of unique columns
- `"n_entries"`: Number of unique entries
- `"n_populations"`: Number of unique populations
- `"n_total"`: Total number of phenotype observations (entries × traits)
- `"n_zeroes"`: Count of zero values in phenotypes
- `"n_missing"`: Count of missing values in phenotypes
- `"n_nan"`: Count of NaN values in phenotypes
- `"n_inf"`: Count of Inf values in phenotypes

# Throws
- `ArgumentError`: If the Trials struct dimensions are inconsistent

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials = Trials(n=1, t=2);

julia> trials.entries = ["entry_1"]; trials.traits = ["trait_1", "trait_2"];

julia> dimensions(trials)
Dict{String, Int64} with 16 entries:
  "n_zeroes"       => 0
  "n_harvests"     => 1
  "n_nan"          => 0
  "n_entries"      => 1
  "n_traits"       => 2
  "n_seasons"      => 1
  "n_rows"         => 1
  "n_blocks"       => 1
  "n_missing"      => 2
  "n_inf"          => 0
  "n_total"        => 2
  "n_replications" => 1
  "n_years"        => 1
  "n_sites"        => 1
  "n_cols"         => 1
  "n_populations"  => 1
```
"""
function dimensions(trials::Trials)::Dict{String,Int64}
    if !checkdims(trials)
        throw(ArgumentError("Trials struct is corrupted."))
    end
    idx_non_missing = .!ismissing.(trials.phenotypes)
    Dict(
        "n_traits" => length(unique(trials.traits)),
        "n_years" => length(unique(trials.years)),
        "n_seasons" => length(unique(trials.seasons)),
        "n_harvests" => length(unique(trials.harvests)),
        "n_sites" => length(unique(trials.sites)),
        "n_replications" => length(unique(trials.replications)),
        "n_blocks" => length(unique(trials.blocks)),
        "n_rows" => length(unique(trials.rows)),
        "n_cols" => length(unique(trials.cols)),
        "n_entries" => length(unique(trials.entries)),
        "n_populations" => length(unique(trials.populations)),
        "n_total" => prod(size(trials.phenotypes)),
        "n_zeroes" => sum(trials.phenotypes[idx_non_missing] .== 0.0),
        "n_missing" => sum(.!idx_non_missing),
        "n_nan" => sum(isnan.(trials.phenotypes[idx_non_missing])),
        "n_inf" => sum(isinf.(trials.phenotypes[idx_non_missing])),
    )
end

"""
    tabularise(trials::Trials)::DataFrame

Convert a Trials struct into a DataFrame representation for easier data manipulation and analysis.

# Arguments
- `trials::Trials`: A valid Trials struct containing experimental field trial data.

# Returns
- `DataFrame`: A DataFrame with the following columns:
  - `id`: Unique identifier for each trial observation
  - `years`: Year of the trial
  - `seasons`: Season identifier
  - `harvests`: Harvest identifier
  - `sites`: Location/site identifier
  - `replications`: Replication number
  - `blocks`: Block identifier
  - `rows`: Row position
  - `cols`: Column position
  - `entries`: Entry identifier
  - `populations`: Population identifier
  - Additional columns for each trait in `trials.traits`

# Throws
- `ArgumentError`: If the Trials struct dimensions are inconsistent

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> df = tabularise(trials);

julia> size(df)
(12800, 14)
```
"""
function tabularise(trials::Trials)::DataFrame
    # trials::Trials, _ = simulatetrials(genomes = simulategenomes());
    if !checkdims(trials)
        throw(ArgumentError("The Trials struct is corrupted."))
    end
    df_ids::DataFrame = DataFrame(;
        id = 1:length(trials.years),
        years = trials.years,
        seasons = trials.seasons,
        harvests = trials.harvests,
        sites = trials.sites,
        replications = trials.replications,
        blocks = trials.blocks,
        rows = trials.rows,
        cols = trials.cols,
        entries = trials.entries,
        populations = trials.populations,
    )
    df_phe::DataFrame = DataFrame(trials.phenotypes, :auto)
    rename!(df_phe, trials.traits)
    df_phe.id = 1:length(trials.years)
    df = innerjoin(df_ids, df_phe; on = :id)
    return df
end

"""
    extractphenomes(trials::Trials)::Phenomes

Convert a `Trials` struct into a `Phenomes` struct by extracting phenotypic values across different environments.

# Details
- Combines trait measurements with their environmental contexts
- Creates unique trait identifiers by combining trait names with environment variables
- Environment variables include: years, harvests, seasons, sites, and replications
- For single environment scenarios, trait names remain without environmental suffixes

# Arguments
- `trials::Trials`: A Trials struct containing phenotypic measurements across different environments

# Returns
- A Phenomes struct containing:
  - `phenotypes`: Matrix of phenotypic values (entries × traits)
  - `entries`: Vector of entry names
  - `populations`: Vector of population names
  - `traits`: Vector of trait names (with environmental contexts)

# Throws
- `ArgumentError`: If duplicate entries exist within year-harvest-season-site-replication combinations
- `ErrorException`: If dimensional validation fails during Phenomes construction

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> phenomes = extractphenomes(trials);

julia> size(phenomes.phenotypes)
(100, 384)
```
"""
function extractphenomes(trials::Trials)::Phenomes
    # trials::Trials, _ = simulatetrials(genomes = simulategenomes());
    df = tabularise(trials)
    df.id = string.(df.entries, "---X---", df.populations)
    ids = sort(unique(df.id))
    entries = [x[1] for x in split.(ids, "---X---")]
    populations = [x[2] for x in split.(ids, "---X---")]
    df.grouping = string.(df.years, "-", df.harvests, "-", df.seasons, "-", df.sites, "-", df.replications)
    base_traits = names(df)[12:(end-1)]
    traits::Vector{String} = []
    for trait_base in base_traits
        for env in sort(unique(df.grouping))
            push!(traits, string(trait_base, "|", env))
        end
    end
    phenomes = Phenomes(n = length(entries), t = length(traits))
    phenomes.traits = traits
    phenomes.entries = entries
    phenomes.populations = populations
    for trait_base in base_traits
        # trait_base = base_traits[1]
        tmp = try
            unstack(df, :id, :grouping, trait_base)
        catch
            throw(
                ArgumentError(
                    "You may have duplicate entries within year-harvest-season-site-replication combinations. " *
                    "These may possibly be controls. " *
                    "Please make sure each entry appears only once within these combinations.",
                ),
            )
        end
        for (i, id) in enumerate(ids)
            # i, id = 1, ids[1]
            idx_1 = findall(tmp.id .== id)[1]
            for (j, env) in enumerate(names(tmp))
                # j, env = 2, names(tmp)[2]
                if j == 1
                    # Skip ids column
                    continue
                end
                idx_2 = findall(phenomes.traits .== string(trait_base, "|", env))[1]
                phenomes.phenotypes[i, idx_2] = tmp[idx_1, j]
            end
        end
    end
    if length(unique(df.grouping)) == 1
        suffix = string("|", unique(df.grouping)[1])
        phenomes.traits = replace.(phenomes.traits, suffix => "")
    end
    if !checkdims(phenomes)
        throw(ErrorException("Error extracting Phenomes from Trials."))
    end
    phenomes
end

"""
    addcompositetrait(trials::Trials; composite_trait_name::String, formula_string::String)::Trials

Create a new composite trait by combining existing traits using a mathematical formula.

# Arguments
- `trials::Trials`: A Trials struct containing phenotypic data
- `composite_trait_name::String`: Name for the new composite trait
- `formula_string::String`: Mathematical formula defining how to combine existing traits

# Formula Syntax
The formula can include:
- Trait names (e.g., "trait_1", "trait_2")
- Mathematical operators: +, -, *, /, ^, %
- Functions: abs(), sqrt(), log(), log2(), log10()
- Parentheses for grouping operations

# Returns
- `Trials`: A new Trials struct with the added composite trait

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> trials_new = addcompositetrait(trials, composite_trait_name = "some_wild_composite_trait", formula_string = "trait_1");

julia> trials_new.phenotypes[:, end] == trials.phenotypes[:, 1]
true

julia> trials_new = addcompositetrait(trials, composite_trait_name = "some_wild_composite_trait", formula_string = "(trait_1^(trait_2/100)) + (trait_3/trait_1) - sqrt(abs(trait_2-trait_1)) + log(1.00 + trait_3)");

julia> trials_new.phenotypes[:, end] == (trials.phenotypes[:,1].^(trials.phenotypes[:,2]/100)) .+ (trials.phenotypes[:,3]./trials.phenotypes[:,1]) .- sqrt.(abs.(trials.phenotypes[:,2].-trials.phenotypes[:,1])) .+ log.(1.00 .+ trials.phenotypes[:,3])
true
```
"""
function addcompositetrait(trials::Trials; composite_trait_name::String, formula_string::String)::Trials
    # trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);
    # composite_trait_name = "some_wild_composite_trait";
    # formula_string = "((trait_1^(trait_2/100)) + trait_3) + sqrt(abs(log(1.00 / trait_1) - (trait_1 * (trait_2 + trait_3)) / (trait_2 - trait_3)^2))";
    # formula_string = "trait_1";
    df = tabularise(trials)
    formula_parsed_orig = deepcopy(formula_string)
    formula_parsed = deepcopy(formula_string)
    symbol_strings = ["=", "+", "-", "*", "/", "^", "%", "abs(", "sqrt(", "log(", "log2(", "log10(", "(", ")"]
    for s in symbol_strings
        formula_string = replace(formula_string, s => " ")
    end
    component_trait_names = unique(split(formula_string, " "))
    ϕ = Vector{Union{Missing,Float64}}(undef, size(df, 1))
    for i in eachindex(ϕ)
        # i = 1
        for var_name in component_trait_names
            # var_name = component_trait_names[2]
            if sum(names(df) .== var_name) == 0
                continue
            else
                formula_parsed = replace(formula_parsed, var_name => string(df[i, var_name]))
            end
        end
        ϕ[i] = @eval(@stringevaluation $(formula_parsed))
        # Reset the formula
        formula_parsed = deepcopy(formula_parsed_orig)
    end
    out = clone(trials)
    idx = findall(out.traits .== composite_trait_name)
    if length(idx) == 0
        push!(out.traits, composite_trait_name)
        out.phenotypes = hcat(out.phenotypes, ϕ)
    elseif length(idx) == 1
        out.phenotypes[:, idx] = ϕ
    else
        throw(ErrorException("Duplicate traits in trials, i.e. trait: " * composite_trait_name))
    end
    if !checkdims(out)
        throw(ErrorException("Error generating composite trait: `" * composite_trait_name * "`"))
    end
    out
end

"""
    plot(trials::Trials; nbins::Int64 = 10)::Nothing

Generate a comprehensive visualization of trial data through histograms, correlation heatmaps, and bar plots.

# Arguments
- `trials::Trials`: A Trials struct containing the trial data to be visualized
- `nbins::Int64=10`: Number of bins for the histogram plots (optional)

# Details
The function creates three types of plots:
1. Histograms for each trait within each population, showing the distribution of trait values
2. Correlation heatmaps between traits for each population
3. Bar plots showing mean trait values across different trial factors:
   - Years
   - Seasons
   - Harvests
   - Sites
   - Replications
   - Rows
   - Columns
   - Populations

Missing, NaN, or infinite values are automatically filtered out before plotting.

# Returns
- `Nothing`: The function displays plots but does not return any value

# Notes
- Requires valid trial data with non-zero variance for correlation plots
- Uses UnicodePlots for visualization in terminal
- Skips plotting for traits with insufficient data points

# Examples
```
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> GBCore.plot(trials);

```
"""
function plot(trials::Trials; nbins::Int64 = 10)
    # trials, _ = simulatetrials(genomes = simulategenomes()); nbins = 10;
    if !checkdims(trials)
        throw(ArgumentError("Phenomes struct is corrupted."))
    end
    for pop in unique(trials.populations)
        # pop = trials.populations[1]
        idx_trait_with_variance::Vector{Int64} = []
        for j in eachindex(trials.traits)
            # j = 1
            println("##############################################")
            println("Population: " * pop * " | Trait: " * trials.traits[j])
            idx = findall(
                (trials.populations .== pop) .&&
                .!ismissing.(trials.phenotypes[:, j]) .&&
                .!isnan.(trials.phenotypes[:, j]) .&&
                .!isinf.(trials.phenotypes[:, j]),
            )
            if length(idx) == 0
                println("All values are missing, NaN and/or infinities.")
                continue
            end
            ϕ::Vector{Float64} = trials.phenotypes[idx, j]
            if StatsBase.var(ϕ) > 1e-10
                append!(idx_trait_with_variance, j)
            end
            if length(ϕ) > 2
                plt = UnicodePlots.histogram(ϕ, title = trials.traits[j], vertical = false, nbins = nbins)
                display(plt)
            else
                println(string("Trait: ", trials.traits[j], " has ", length(ϕ), " non-missing data points."))
            end
        end
        # Build the correlation matrix
        idx_pop = findall(trials.populations .== pop)
        _, _, dist = try
            distances(
                slice(phenomes, idx_entries = idx_pop, idx_traits = idx_trait_with_variance),
                distance_metrics = ["correlation"],
            )
        catch
            println("Error in computing distances for the Trials struct.")
            continue
        end
        plt = UnicodePlots.heatmap(
            C;
            height = length(traits),
            width = length(traits),
            title = string(
                "Trait correlations: {",
                join(string.(collect(1:length(traits))) .* " => " .* traits, ", "),
                "}",
            ),
        )
        display(plt)
    end
    # Mean trait values across years, seasons, harvests, sites, replications, row, column, and populations
    df = tabularise(trials)
    for trait in trials.traits
        # trait = trials.traits[1]
        println("##############################################")
        println("Trait: " * trait)
        idx = findall(.!ismissing.(df[!, trait]) .&& .!isnan.(df[!, trait]) .&& .!isinf.(df[!, trait]))
        if length(idx) == 0
            println("All values are missing, NaN and/or infinities.")
            continue
        end
        for class in ["years", "seasons", "harvests", "sites", "replications", "rows", "cols", "populations"]
            # class = "years"
            agg = DataFrames.combine(DataFrames.groupby(df[idx, :], class), trait => mean)
            if sum(agg[!, 2] .>= 0) == 0
                # Skip empty aggregate
                continue
            end
            plt = UnicodePlots.barplot(agg[!, 1], agg[!, 2], title = class)
            display(plt)
        end
    end
    # Return nothing
    return nothing
end

"""
    slice(
        trials::Trials; 
        traits::Union{Nothing, Vector{String}} = nothing,
        populations::Union{Nothing, Vector{String}} = nothing,
        entries::Union{Nothing, Vector{String}} = nothing,
        years::Union{Nothing, Vector{String}} = nothing,
        harvests::Union{Nothing, Vector{String}} = nothing,
        seasons::Union{Nothing, Vector{String}} = nothing,
        sites::Union{Nothing, Vector{String}} = nothing,
        replications::Union{Nothing, Vector{String}} = nothing,
        blocks::Union{Nothing, Vector{String}} = nothing,
        rows::Union{Nothing, Vector{String}} = nothing,
        cols::Union{Nothing, Vector{String}} = nothing,
    )::Trials

Create a subset of a `Trials` struct by filtering its components based on specified criteria.

# Arguments
- `trials::Trials`: The source trials data structure to be sliced
- `traits::Vector{String}`: Selected trait names to include
- `populations::Vector{String}`: Selected population names to include
- `entries::Vector{String}`: Selected entry names to include
- `years::Vector{String}`: Selected years to include
- `harvests::Vector{String}`: Selected harvest identifiers to include
- `seasons::Vector{String}`: Selected seasons to include
- `sites::Vector{String}`: Selected site names to include
- `replications::Vector{String}`: Selected replication identifiers to include
- `blocks::Vector{String}`: Selected block identifiers to include
- `rows::Vector{String}`: Selected row identifiers to include
- `cols::Vector{String}`: Selected column identifiers to include

All arguments except `trials` are optional. When an argument is not provided (i.e., `nothing`), 
all values for that category are included in the slice.

# Returns
- A new `Trials` struct containing only the selected data

# Throws
- `ArgumentError`: If invalid names are provided for any category or if no data remains after filtering
- `DimensionMismatch`: If the resulting sliced trials structure has inconsistent dimensions
- `ArgumentError`: If the input trials structure is corrupted

# Examples
```jldoctest; setup = :(using GBCore)
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> sliced_trials = slice(trials, traits=trials.traits[2:3], years=[unique(trials.years)[1]], seasons=unique(trials.seasons)[2:3]);

julia> dimensions(sliced_trials)
Dict{String, Int64} with 16 entries:
  "n_zeroes"       => 0
  "n_harvests"     => 2
  "n_nan"          => 0
  "n_entries"      => 100
  "n_traits"       => 2
  "n_seasons"      => 2
  "n_blocks"       => 10
  "n_rows"         => 10
  "n_missing"      => 0
  "n_inf"          => 0
  "n_total"        => 6400
  "n_replications" => 2
  "n_years"        => 1
  "n_sites"        => 4
  "n_cols"         => 20
  "n_populations"  => 1
```
"""
function slice(
    trials::Trials;
    traits::Union{Nothing,Vector{String}} = nothing,
    populations::Union{Nothing,Vector{String}} = nothing,
    entries::Union{Nothing,Vector{String}} = nothing,
    years::Union{Nothing,Vector{String}} = nothing,
    harvests::Union{Nothing,Vector{String}} = nothing,
    seasons::Union{Nothing,Vector{String}} = nothing,
    sites::Union{Nothing,Vector{String}} = nothing,
    replications::Union{Nothing,Vector{String}} = nothing,
    blocks::Union{Nothing,Vector{String}} = nothing,
    rows::Union{Nothing,Vector{String}} = nothing,
    cols::Union{Nothing,Vector{String}} = nothing,
)::Trials
    # trials, _ = simulatetrials(genomes = simulategenomes())
    # populations = entries = traits = years = harvests = seasons = sites = replications = blocks = rows = cols = nothing
    # Check arguments
    if !checkdims(trials)
        throw(ArgumentError("Phenomes struct is corrupted."))
    end
    idx, idx_traits = begin
        all_traits = sort(unique(trials.traits))
        all_populations = sort(unique(trials.populations))
        all_entries = sort(unique(trials.entries))
        all_years = sort(unique(trials.years))
        all_harvests = sort(unique(trials.harvests))
        all_seasons = sort(unique(trials.seasons))
        all_sites = sort(unique(trials.sites))
        all_replications = sort(unique(trials.replications))
        all_blocks = sort(unique(trials.blocks))
        all_rows = sort(unique(trials.rows))
        all_cols = sort(unique(trials.cols))
        populations = if isnothing(populations)
            all_populations
        else
            wrong_names = setdiff(populations, all_populations)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong populations provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(populations))
        end
        entries = if isnothing(entries)
            all_entries
        else
            wrong_names = setdiff(entries, all_entries)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong entries provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(entries))
        end
        years = if isnothing(years)
            all_years
        else
            wrong_names = setdiff(years, all_years)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong years provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(years))
        end
        harvests = if isnothing(harvests)
            all_harvests
        else
            wrong_names = setdiff(harvests, all_harvests)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong harvests provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(harvests))
        end
        seasons = if isnothing(seasons)
            all_seasons
        else
            wrong_names = setdiff(seasons, all_seasons)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong seasons provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(seasons))
        end
        sites = if isnothing(sites)
            all_sites
        else
            wrong_names = setdiff(sites, all_sites)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong sites provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(sites))
        end
        replications = if isnothing(replications)
            all_replications
        else
            wrong_names = setdiff(replications, all_replications)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong replications provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(replications))
        end
        blocks = if isnothing(blocks)
            all_blocks
        else
            wrong_names = setdiff(blocks, all_blocks)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong blocks provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(blocks))
        end
        rows = if isnothing(rows)
            all_rows
        else
            wrong_names = setdiff(rows, all_rows)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong rows provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(rows))
        end
        cols = if isnothing(cols)
            all_cols
        else
            wrong_names = setdiff(cols, all_cols)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong cols provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(cols))
        end
        idx = []
        for i in eachindex(trials.years)
            if (trials.populations[i] ∈ populations) &&
               (trials.entries[i] ∈ entries) &&
               (trials.years[i] ∈ years) &&
               (trials.harvests[i] ∈ harvests) &&
               (trials.seasons[i] ∈ seasons) &&
               (trials.sites[i] ∈ sites) &&
               (trials.replications[i] ∈ replications) &&
               (trials.blocks[i] ∈ blocks) &&
               (trials.rows[i] ∈ rows) &&
               (trials.cols[i] ∈ cols)
                append!(idx, i)
            end
        end
        if length(idx) < 1
            throw(ArgumentError("No data retained after filtering by years, populations, entries, etc."))
        end
        traits = if isnothing(traits)
            all_traits
        else
            wrong_names = setdiff(traits, all_traits)
            if length(wrong_names) > 0
                throw(ArgumentError("Wrong traits provided:\n\t‣ " * join(wrong_names, "\n\t‣ ")))
            end
            unique(sort(traits))
        end
        idx_traits = [findall(trials.traits .== trait)[1] for trait in traits]
        (idx, idx_traits)
    end
    sliced_trials = Trials(n = length(idx), t = length(traits))
    sliced_trials.traits = trials.traits[idx_traits]
    sliced_trials.phenotypes = trials.phenotypes[idx, idx_traits]
    sliced_trials.populations = trials.populations[idx]
    sliced_trials.entries = trials.entries[idx]
    sliced_trials.years = trials.years[idx]
    sliced_trials.harvests = trials.harvests[idx]
    sliced_trials.seasons = trials.seasons[idx]
    sliced_trials.sites = trials.sites[idx]
    sliced_trials.replications = trials.replications[idx]
    sliced_trials.blocks = trials.blocks[idx]
    sliced_trials.rows = trials.rows[idx]
    sliced_trials.cols = trials.cols[idx]
    ### Check dimensions
    if !checkdims(sliced_trials)
        throw(DimensionMismatch("Error slicing the genome."))
    end
    # Output
    return sliced_trials
end
