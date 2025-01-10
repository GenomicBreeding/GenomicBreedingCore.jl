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

julia> trials.entries = ["entry_1"];

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
    plot(trials::Trials)::Nothing

Plot histogram/s of the trait value/s and a heatmap of trait correlations across the entire trial.
Additionally, plot mean trait values per year, season, harvest, site, replications, row, column, and population for each trait.

# Examples
```
julia> trials, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false);

julia> plot(trials);

```
"""
function plot(trials::Trials; nbins::Int64 = 10)
    # trials, _ = simulatetrials(genomes = simulategenomes()); nbins = 10; 
    if !checkdims(trials)
        throw(ArgumentError("Phenomes struct is corrupted."))
    end
    for pop in unique(trials.populations)
        # pop = trials.populations[1]
        idx_pop = findall(trials.populations .== pop)
        idx_trait_with_variance::Vector{Int64} = []
        for j in eachindex(trials.traits)
            # j = 1
            println("##############################################")
            println("Population: " * pop * " | Trait: " * trials.traits[j])
            ϕ::Vector{Float64} = trials.phenotypes[.!ismissing.(trials.phenotypes[:, j][idx_pop]), j]
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
        # View correlation between traits using scatter plots
        C::Matrix{Float64} = StatsBase.cor(trials.phenotypes[idx_pop, idx_trait_with_variance])
        plt = UnicodePlots.heatmap(
            C;
            height = length(trials.traits),
            width = length(trials.traits),
            title = string(
                "Trait correlations: {",
                join(
                    string.(collect(1:length(idx_trait_with_variance))) .* " => " .*
                    trials.traits[idx_trait_with_variance],
                    ", ",
                ),
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
        for class in ["years", "seasons", "harvests", "sites", "replications", "rows", "cols", "populations"]
            agg = DataFrames.combine(DataFrames.groupby(df, class), trait => mean)
            plt = UnicodePlots.barplot(agg[!, 1], agg[!, 2], title = class)
            display(plt)
        end
    end
    # Return nothing
    return nothing
end
