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

# # TODO: plot raw data
# function plot(trials::Trials)::Bool
#     # trials, _ = simulatetrials(genomes = simulategenomes())
#     df::DataFrame = tabularise(trials)
#     cor(trials.phenotypes)
#     plt = corrplot(trials.phenotypes)

#     false
# end
