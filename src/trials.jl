"""
    clone(x::Trials)::Trials

Clone a Trials object

## Example
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

Hash a Trials struct.

## Examples
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
    Base.:(==)(x::Trials, y::Trials)::Bool

Equality of Trials structs using the hash function defined for Trials structs.

## Examples
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

Check dimension compatibility of the fields of the Trials struct

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

Count the number of entries, populations, and traits in the Trials struct

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

Export the Trials structs into a DataFrames.DataFrame struct

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

Extract Phenomes from Trials struct.
Each trait-by-environment variables combination make up the traits in the resulting Phenomes struct.
The trait names start with trait name in Trials suffixed by the trait-by-environment variables combinations.
If there is a single environment variables combination, then no suffix is added to the trait name.

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

Add a composite trait from existing traits in a Trials struct

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
    plot(trials::Trials)::Nothing

Plot histogram/s of the trait value/s and a heatmap of trait correlations across the entire trial.
Additionally, plot mean trait values per year, season, harvest, site, replications, row, column, and population for each trait.

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
        Φ = trials.phenotypes[idx_pop, idx_trait_with_variance]
        traits = trials.traits[idx_trait_with_variance]
        C::Matrix{Float64} = fill(0.0, length(traits), length(traits))
        for ti in eachindex(traits)
            for tj in eachindex(traits)
                idx = findall(
                    .!ismissing.(Φ[:, ti]) .&&
                    .!ismissing.(Φ[:, tj]) .&&
                    .!isnan.(Φ[:, ti]) .&&
                    .!isnan.(Φ[:, tj]) .&&
                    .!isinf.(Φ[:, ti]) .&&
                    .!isinf.(Φ[:, tj]),
                )
                C[ti, tj] = StatsBase.cor(Φ[idx, ti], Φ[idx, tj])
            end
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

Slice a Trials struct by specifing indexes of entries and traits

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
