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
