"""
    slice(phenomes::Phenomes; idx_entries::Union{Nothing, Vector{Int64}}=nothing, idx_traits::Union{Nothing, Vector{Int64}}=nothing)::Phenomes

Create a new `Phenomes` object containing a subset of the original data by selecting specific entries and traits.

# Arguments
- `phenomes::Phenomes`: The original Phenomes object to slice
- `idx_entries::Union{Nothing, Vector{Int64}}=nothing`: Indices of entries to keep. If `nothing`, all entries are kept
- `idx_traits::Union{Nothing, Vector{Int64}}=nothing`: Indices of traits to keep. If `nothing`, all traits are kept

# Returns
- `Phenomes`: A new Phenomes object containing only the selected entries and traits

# Notes
- The function preserves the original structure while reducing dimensions
- Indices must be within valid ranges (1 to n_entries/n_traits)
- Duplicate indices are automatically removed
- The resulting object maintains all relationships between entries, populations, traits, and phenotypes

# Throws
- `ArgumentError`: If the input Phenomes struct is corrupted or if indices are out of bounds
- `DimensionMismatch`: If the slicing operation results in invalid dimensions

# Examples
```jldoctest; setup = :(using GBCore)
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = fill(0.0, 10,3);

julia> sliced_phenomes = slice(phenomes, idx_entries=collect(1:5); idx_traits=collect(2:3));

julia> dimensions(sliced_phenomes)
Dict{String, Int64} with 8 entries:
  "n_total"       => 10
  "n_zeroes"      => 10
  "n_nan"         => 0
  "n_entries"     => 5
  "n_traits"      => 2
  "n_inf"         => 0
  "n_populations" => 1
  "n_missing"     => 0
```
"""
function slice(
    phenomes::Phenomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_traits::Union{Nothing,Vector{Int64}} = nothing,
)::Phenomes
    # phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3); nbins = 10;
    # idx_entries::Vector{Int64}=collect(2:7); idx_traits::Vector{Int64}=collect(1:2);
    if !checkdims(phenomes)
        throw(ArgumentError("Phenomes struct is corrupted."))
    end
    phenomes_dims::Dict{String,Int64} = dimensions(phenomes)
    n_entries::Int64 = phenomes_dims["n_entries"]
    n_traits::Int64 = phenomes_dims["n_traits"]
    idx_entries = if isnothing(idx_entries)
        collect(1:n_entries)
    else
        if (minimum(idx_entries) < 1) || (maximum(idx_entries) > n_entries)
            throw(ArgumentError("We accept `idx_entries` from 1 to `n_entries` of `phenomes`."))
        end
        unique(sort(idx_entries))
    end
    idx_traits = if isnothing(idx_traits)
        collect(1:n_traits)
    else
        if (minimum(idx_traits) < 1) || (maximum(idx_traits) > n_traits)
            throw(ArgumentError("We accept `idx_traits` from 1 to `n_traits` of `phenomes`."))
        end
        unique(sort(idx_traits))
    end
    n, t = length(idx_entries), length(idx_traits)
    sliced_phenomes::Phenomes = Phenomes(n = n, t = t)
    for (i1, i2) in enumerate(idx_entries)
        sliced_phenomes.entries[i1] = phenomes.entries[i2]
        sliced_phenomes.populations[i1] = phenomes.populations[i2]
        for (j1, j2) in enumerate(idx_traits)
            if i1 == 1
                sliced_phenomes.traits[j1] = phenomes.traits[j2]
            end
            sliced_phenomes.phenotypes[i1, j1] = phenomes.phenotypes[i2, j2]
            sliced_phenomes.mask[i1, j1] = phenomes.mask[i2, j2]
        end
    end
    ### Check dimensions
    if !checkdims(sliced_phenomes)
        throw(DimensionMismatch("Error slicing the genome."))
    end
    # Output
    return sliced_phenomes
end


"""
    filter(phenomes::Phenomes)::Phenomes

Filter a Phenomes struct by removing rows (entries) and columns (traits) as indicated by the mask matrix. 
An entry or trait is removed if it contains at least one false value in the mask.

# Arguments
- `phenomes::Phenomes`: The Phenomes struct to be filtered, containing entries, populations, traits,
  phenotypes, and a boolean mask matrix.

# Returns
- `Phenomes`: A new Phenomes struct with filtered entries and traits, where the mask matrix is all true.

# Details
The function uses the mean of rows and columns in the mask matrix to identify which entries and traits
should be kept. Only entries and traits with a mean of 1.0 (all true values) are retained in the
filtered result.

# Examples
```jldoctest; setup = :(using GBCore)
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = fill(0.0, 10,3);

julia> phenomes.mask .= true; phenomes.mask[6:10, 1] .= false;
    
julia> filtered_phenomes = filter(phenomes);

julia> size(filtered_phenomes.phenotypes)
(5, 2)
```
"""
function Base.filter(phenomes::Phenomes)::Phenomes
    # phenomes = simulatephenomes(); phenomes.mask[1:10, 42:100] .= false;
    idx_entries = findall(mean(phenomes.mask, dims = 2)[:, 1] .== 1.0)
    idx_traits = findall(mean(phenomes.mask, dims = 1)[1, :] .== 1.0)
    filtered_phenomes::Phenomes = slice(phenomes, idx_entries = idx_entries; idx_traits = idx_traits)
    filtered_phenomes
end
