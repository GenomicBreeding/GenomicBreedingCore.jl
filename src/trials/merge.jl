"""
    merge(trials::Trials, other::Trials; conflict_resolution::Tuple{Float64,Float64}=(0.5, 0.5), verbose::Bool=true)::Trials

Merge two `Trials` structs into a single combined `Trials` struct.

# Arguments
- `trials::Trials`: The first Trials struct to merge
- `other::Trials`: The second Trials struct to merge
- `conflict_resolution::Tuple{Float64,Float64}`: Weights for resolving conflicts when the same trait measurement exists in both trials. 
  Default is (0.5, 0.5), meaning equal weights. Must sum to 1.0
- `verbose::Bool`: Whether to print progress information. Default is `true`

# Returns
- `::Trials`: A new merged Trials struct containing combined data from both input trials

# Details
The function performs an outer join of the two trial datasets, combining them based on their identifying columns 
(years, seasons, harvests, sites, replications, blocks, rows, cols, entries, populations).

For traits that exist in both trials:
- If a measurement exists in only one trial, that value is used
- If measurements exist in both trials, they are combined using weighted average based on conflict_resolution weights

# Throws
- `ArgumentError`: If either input Trials struct is corrupted (invalid dimensions)
- `ArgumentError`: If conflict_resolution is not a 2-element tuple summing to 1.0
- `ArgumentError`: If the resulting merged Trials struct is corrupted

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> trials_all, _ = simulatetrials(genomes = simulategenomes(verbose=false), verbose=false); 

julia> trials = slice(trials_all, entries=sort(unique(trials_all.entries))[1:(end-1)], traits=trials_all.traits[1:(end-1)]); 

julia> other = slice(trials_all, entries=sort(unique(trials_all.entries))[2:end], traits=trials_all.traits[2:end]);

julia> trials_merged = merge(trials, other, conflict_resolution=(0.5, 0.5), verbose=false);

julia> dimensions(trials_merged)["n_missing"] > dimensions(trials_all)["n_missing"]
true

julia> trials = slice(trials_all, entries=sort(unique(trials_all.entries))[1:50]); 

julia> other = slice(trials_all, entries=sort(unique(trials_all.entries))[51:end]);

julia> trials_merged = merge(trials, other, conflict_resolution=(0.5, 0.5), verbose=false);

julia> dimensions(trials_merged) == dimensions(trials_all)
true

julia> trials = slice(trials_all, traits=trials_all.traits[1:1]); 

julia> other = slice(trials_all, traits=trials_all.traits[2:end]);

julia> trials_merged = merge(trials, other, conflict_resolution=(0.5, 0.5), verbose=false);

julia> dimensions(trials_merged) == dimensions(trials_all)
true
```
"""
function Base.merge(
    trials::Trials,
    other::Trials;
    conflict_resolution::Tuple{Float64,Float64} = (0.5, 0.5),
    verbose::Bool = true,
)::Trials
    # trials_all::Trials, _ = simulatetrials(genomes = simulategenomes()); trials = slice(trials_all, entries=sort(unique(trials_all.entries))[1:(end-1)], traits=trials_all.traits[1:(end-1)]); other = slice(trials_all, entries=sort(unique(trials_all.entries))[2:end], traits=trials_all.traits[2:end]); conflict_resolution = (0.5, 0.5); verbose = true
    # Check arguments
    if !checkdims(trials) && !checkdims(other)
        throw(ArgumentError("Both Trials structs are corrupted ☹."))
    end
    if !checkdims(trials)
        throw(ArgumentError("The first Trials struct is corrupted ☹."))
    end
    if !checkdims(other)
        throw(ArgumentError("The second Trials struct is corrupted ☹."))
    end
    if (length(conflict_resolution) != 2) && (sum(conflict_resolution) != 1.00)
        throw(ArgumentError("We expect `conflict_resolution` 2 be a 2-item tuple which sums up to exactly 1.00."))
    end
    # Merge the 2 Trials structs
    df_1 = begin
        df_1 = tabularise(trials)
        df_1[:, findall(names(df_1) .!= "id")]
    end
    df_2 = begin
        df_2 = tabularise(other)
        df_2[:, findall(names(df_2) .!= "id")]
    end
    id_cols =
        ["years", "seasons", "harvests", "sites", "replications", "blocks", "rows", "cols", "entries", "populations"]
    ids_1 = [join(x, "|") for x in eachrow(df_1[:, id_cols])]
    ids_2 = [join(x, "|") for x in eachrow(df_2[:, id_cols])]
    common_ids = intersect(ids_1, ids_2)
    common_traits = intersect(trials.traits, other.traits)
    df_merged = if length(common_traits) == 0
        outerjoin(df_1, df_2, on = id_cols)
    else
        df_tmp = outerjoin(
            df_1[:, [!(x ∈ common_traits) for x in names(df_1)]],
            df_2[:, [!(x ∈ common_traits) for x in names(df_2)]],
            on = id_cols,
        )
        for trait in common_traits
            # trait = common_traits[1]
            df_1_x_2 = outerjoin(
                df_1[:, vcat(id_cols, trait)],
                df_2[:, vcat(id_cols, trait)],
                on = id_cols,
                makeunique = true,
            )
            rename!(df_1_x_2, Symbol(trait) => Symbol(trait * "_1"), Symbol(trait * "_1") => Symbol(trait * "_2"))
            ϕ = []
            @inbounds for i = 1:nrow(df_1_x_2)
                # i = 1
                x = df_1_x_2[i, "$(trait)_1"]
                y = df_1_x_2[i, "$(trait)_2"]
                z = if !ismissing(x) && !ismissing(y)
                    conflict_resolution[1] * x + conflict_resolution[2] * y
                elseif !ismissing(x)
                    x
                elseif !ismissing(y)
                    y
                else
                    missing
                end
                push!(ϕ, z)
            end
            df_tmp[!, Symbol(trait)] = ϕ
        end
        df_tmp
    end
    # Output
    trials_merged = Trials(n = nrow(df_merged), t = (ncol(df_merged)-length(id_cols)))
    trials_merged.traits = filter(x -> !(x ∈ id_cols), names(df_merged))
    trials_merged.years = df_merged.years
    trials_merged.seasons = df_merged.seasons
    trials_merged.harvests = df_merged.harvests
    trials_merged.sites = df_merged.sites
    trials_merged.replications = df_merged.replications
    trials_merged.blocks = df_merged.blocks
    trials_merged.rows = df_merged.rows
    trials_merged.cols = df_merged.cols
    trials_merged.entries = df_merged.entries
    trials_merged.populations = df_merged.populations
    trials_merged.phenotypes = Matrix(df_merged[:, trials_merged.traits])
    if !checkdims(trials_merged)
        throw(ArgumentError("The merged Trials struct is corrupted ☹."))
    end
    trials_merged
end
