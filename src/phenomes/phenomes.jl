"""
    clone(x::Phenomes)::Phenomes

Create a deep copy of a `Phenomes` object, including all its fields.

This function performs a deep copy of the following fields:
- entries: Vector of entry names
- populations: Vector of population identifiers
- traits: Vector of trait names
- phenotypes: Matrix of phenotypic values
- mask: Matrix of boolean masks

Returns a new `Phenomes` object with identical structure but independent memory allocation.

# Arguments
- `x::Phenomes`: The source Phenomes object to be cloned

# Returns
- `Phenomes`: A new Phenomes object containing deep copies of all fields

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> phenomes = Phenomes(n=2, t=2);

julia> copy_phenomes = clone(phenomes)
Phenomes(["", ""], ["", ""], ["", ""], Union{Missing, Float64}[missing missing; missing missing], Bool[1 1; 1 1])
```
"""
function clone(x::Phenomes)::Phenomes
    out = Phenomes(n = length(x.entries), t = length(x.traits))
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        setfield!(out, field, deepcopy(getfield(x, field)))
    end
    out
end

"""
    Base.hash(x::Phenomes, h::UInt)::UInt

Compute a hash value for a `Phenomes` struct by recursively hashing its internal fields.

# Arguments
- `x::Phenomes`: The Phenomes struct to be hashed
- `h::UInt`: The hash value to be mixed with

# Returns
- `UInt`: A hash value for the entire Phenomes struct

# Note
This function is used for dictionary operations and computing hash-based data structures.
The hash is computed by combining hashes of all internal fields: entries, populations,
traits, phenotypes, and mask.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> phenomes = Phenomes(n=2, t=2);

julia> typeof(hash(phenomes))
UInt64
```
"""
function Base.hash(x::Phenomes, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    ==(x::Phenomes, y::Phenomes)::Bool

Compare two `Phenomes` structs for equality using their hash values.

This method implements equality comparison for `Phenomes` objects by comparing their hash values,
ensuring that two phenomes with identical structure and content are considered equal.

# Arguments
- `x::Phenomes`: First phenomes object to compare
- `y::Phenomes`: Second phenomes object to compare

# Returns
- `Bool`: `true` if the phenomes are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> phenomes_1 = phenomes = Phenomes(n=2, t=4);

julia> phenomes_2 = phenomes = Phenomes(n=2, t=4);

julia> phenomes_3 = phenomes = Phenomes(n=1, t=2);

julia> phenomes_1 == phenomes_2
true

julia> phenomes_1 == phenomes_3
false
```
"""
function Base.:(==)(x::Phenomes, y::Phenomes)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(y::Phenomes; verbose::Bool=false)::Bool

Verify dimensional compatibility between all fields of a Phenomes struct.

Checks if:
- Number of entries matches the number of rows in phenotypes matrix
- All entry names are unique
- Number of populations matches number of entries
- Number of traits matches number of columns in phenotypes matrix
- All trait names are unique
- Dimensions of mask matrix match phenotypes matrix

# Arguments
- `y::Phenomes`: A Phenomes struct containing phenotypic data
- `verbose::Bool=false`: If true, prints the dimensions of each field for debugging

# Returns
- `Bool`: `true` if all dimensions are compatible, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> y = Phenomes(n=2, t=2);

julia> checkdims(y)
false

julia> y.entries = ["entry_1", "entry_2"];

julia> y.traits = ["trait_1", "trait_2"];

julia> checkdims(y)
true
```
"""
function checkdims(y::Phenomes; verbose::Bool = false)::Bool
    if verbose
        @show size(y.phenotypes)
        @show length(y.entries)
        @show length(unique(y.entries))
        @show length(y.populations)
        @show length(unique(y.populations))
        @show length(y.traits)
        @show length(unique(y.traits))
        @show size(y.mask)
    end
    n, p = size(y.phenotypes)
    if (n != length(y.entries)) ||
       (n != length(unique(y.entries))) ||
       (n != length(y.populations)) ||
       (p != length(y.traits)) ||
       (p != length(unique(y.traits))) ||
       ((n, p) != size(y.mask))
        return false
    end
    true
end

"""
    dimensions(phenomes::Phenomes)::Dict{String, Int64}

Calculate various dimensional statistics of a `Phenomes` struct.

Returns a dictionary containing counts of:
- `"n_entries"`: unique entries in the dataset
- `"n_populations"`: unique populations
- `"n_traits"`: number of traits
- `"n_total"`: total number of phenotypic observations (entries × traits)
- `"n_zeroes"`: number of zero values in phenotypes
- `"n_missing"`: number of missing values
- `"n_nan"`: number of NaN values
- `"n_inf"`: number of infinite values

# Arguments
- `phenomes::Phenomes`: A Phenomes struct containing phenotypic data

# Returns
- `Dict{String,Int64}`: Dictionary with dimensional statistics

# Throws
- `ArgumentError`: If the Phenomes struct dimensions are inconsistent

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = fill(0.0, 10,3);

julia> dimensions(phenomes)
Dict{String, Int64} with 8 entries:
  "n_total"       => 30
  "n_zeroes"      => 30
  "n_nan"         => 0
  "n_entries"     => 10
  "n_traits"      => 3
  "n_inf"         => 0
  "n_populations" => 1
  "n_missing"     => 0
```
"""
function dimensions(phenomes::Phenomes)::Dict{String,Int64}
    if !checkdims(phenomes)
        throw(ArgumentError("Phenomes struct is corrupted ☹."))
    end
    idx_non_missing = .!ismissing.(phenomes.phenotypes)
    Dict(
        "n_entries" => length(unique(phenomes.entries)),
        "n_populations" => length(unique(phenomes.populations)),
        "n_traits" => length(phenomes.traits),
        "n_total" => prod(size(phenomes.phenotypes)),
        "n_zeroes" => sum(phenomes.phenotypes[idx_non_missing] .== 0.0),
        "n_missing" => sum(.!idx_non_missing),
        "n_nan" => sum(isnan.(phenomes.phenotypes[idx_non_missing])),
        "n_inf" => sum(isinf.(phenomes.phenotypes[idx_non_missing])),
    )
end

"""
    distances(
        phenomes::Phenomes; 
        distance_metrics::Vector{String}=["euclidean", "correlation", "mad", "rmsd", "χ²"],
        standardise_traits::Bool = false
    )::Tuple{Vector{String}, Vector{String}, Dict{String, Matrix{Float64}}}

Calculate pairwise distances/correlations between traits and entries in a phenotypic dataset.

# Arguments
- `phenomes::Phenomes`: A Phenomes struct containing phenotypic data
- `distance_metrics::Vector{String}`: Vector of distance metrics to compute. Valid options are:
  * "euclidean": Euclidean distance
  * "correlation": Pearson correlation coefficient
  * "mad": Mean absolute deviation
  * "rmsd": Root mean square deviation
  * "χ²": Chi-square distance
- `standardise_traits::Bool`: If true, standardizes traits to mean=0 and sd=1 before computing distances

# Returns
A tuple containing:
1. Vector of trait names
2. Vector of entry names
3. Dictionary mapping "{dimension}|{metric}" to distance matrices, where:
   * dimension ∈ ["traits", "entries"]
   * metric ∈ distance_metrics ∪ ["counts"]
   * "counts" matrices contain the number of non-missing pairs used in calculations

# Notes
- Pairs with fewer than 2 non-missing values result in -Inf distance values
- For correlation calculations, traits with near-zero variance (< 1e-7) are skipped
- χ² distance adds machine epsilon to denominator to avoid division by zero

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3); phenomes.phenotypes[2,2] = missing;

julia> (traits, entries, dist) = distances(phenomes, distance_metrics=["correlation", "χ²"]);

julia> sort(string.(keys(dist))) == ["entries|correlation", "entries|counts", "entries|χ²", "traits|correlation", "traits|counts", "traits|χ²"]
true

julia> C = dist["entries|correlation"]; C[diagind(C)] == repeat([1], length(phenomes.entries))
true

julia> dist["traits|counts"][:, 2] == dist["traits|counts"][2, :] == repeat([9], length(phenomes.traits))
true

julia> dist["entries|counts"][:, 2] == dist["entries|counts"][2, :] == repeat([2], length(phenomes.entries))
true
```
"""
function distances(
    phenomes::Phenomes;
    distance_metrics::Vector{String} = ["euclidean", "correlation", "mad", "rmsd", "χ²"],
    standardise_traits::Bool = false,
)::Tuple{Vector{String},Vector{String},Dict{String,Matrix{Float64}}}
    # phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3);
    # distance_metrics = ["euclidean", "correlation", "mad", "rmsd", "χ²"]; standardise_traits = true
    if !checkdims(phenomes)
        throw(ArgumentError("The phenomes struct is corrupted ☹."))
    end
    recognised_distance_metrics = ["euclidean", "correlation", "mad", "rmsd", "χ²"]
    unique!(distance_metrics)
    m = length(distance_metrics)
    if m < 1
        throw(
            ArgumentError(
                "Please supply at least 1 distance metric. Choose from:\n\t‣ " *
                join(recognised_distance_metrics, "\n\t‣ "),
            ),
        )
    end
    idx_unrecognised_distance_metrics = findall([sum(recognised_distance_metrics .== m) == 0 for m in distance_metrics])
    if length(idx_unrecognised_distance_metrics) > 0
        throw(
            ArgumentError(
                "Unrecognised metric/s:\n\t‣ " *
                join(distance_metrics[idx_unrecognised_distance_metrics], "\n\t‣ ") *
                "\nPlease choose from:\n\t‣ " *
                join(recognised_distance_metrics, "\n\t‣ "),
            ),
        )
    end
    # Standardise traits if requested
    Φ = if standardise_traits
        Φ = deepcopy(phenomes)
        for j = 1:size(Φ.phenotypes, 2)
            y = Φ.phenotypes[:, j]
            idx = findall(.!ismissing.(y) .&& .!isnan.(y) .&& .!isinf.(y))
            y = y[idx]
            Φ.phenotypes[idx, j] = (y .- mean(y)) ./ std(y)
        end
        Φ
    else
        deepcopy(phenomes)
    end
    # Instantiate vectors of matrices and metric names
    dimension::Vector{String} = [] # across traits and/or entries
    metric_names::Vector{String} = []
    matrices::Vector{Matrix{Float64}} = []
    # Number of traits and entries
    n = length(Φ.entries)
    t = length(Φ.traits)
    # Per trait first
    if t > 1
        y1::Vector{Union{Missing,Float64}} = fill(missing, n)
        y2::Vector{Union{Missing,Float64}} = fill(missing, n)
        counts = fill(0.0, t, t)
        for metric in distance_metrics
            # metric = distance_metrics[1]
            D::Matrix{Float64} = fill(-Inf, t, t)
            for i = 1:t
                # i = 1
                y1 .= Φ.phenotypes[:, i]
                bool1 = .!ismissing.(y1) .&& .!isnan.(y1) .&& .!isinf.(y1)
                for j = 1:t
                    # j = 3
                    # Make sure we have no missing, NaN or infinite values
                    y2 .= Φ.phenotypes[:, j]
                    bool2 = .!ismissing.(y2) .&& .!isnan.(y2) .&& .!isinf.(y2)
                    idx = findall(bool1 .&& bool2)
                    if length(idx) < 2
                        continue
                    end
                    # Count the number of elements used to estimate the distance
                    if metric == distance_metrics[1]
                        counts[i, j] = length(idx)
                    end
                    # Estimate the distance/correlation
                    if metric == "euclidean"
                        D[i, j] = sqrt(sum((y1[idx] - y2[idx]) .^ 2))
                    elseif metric == "correlation"
                        (var(y1[idx]) < 1e-7) || (var(y2[idx]) < 1e-7) ? continue : nothing
                        D[i, j] = cor(y1[idx], y2[idx])
                    elseif metric == "mad"
                        D[i, j] = mean(abs.(y1[idx] - y2[idx]))
                    elseif metric == "rmsd"
                        D[i, j] = sqrt(mean((y1[idx] - y2[idx]) .^ 2))
                    elseif metric == "χ²"
                        D[i, j] = sum((y1[idx] - y2[idx]) .^ 2 ./ (y2[idx] .+ eps(Float64)))
                    else
                        throw(
                            ErrorException(
                                "This should not happen as we checked for the validity of the distance_metrics above.",
                            ),
                        )
                    end
                end
            end
            push!(dimension, "traits")
            push!(metric_names, metric)
            push!(matrices, D)
        end
        push!(dimension, "traits")
        push!(metric_names, "counts")
        push!(matrices, counts)
    end
    # Finally per entry
    if n > 1
        ϕ1::Vector{Union{Missing,Float64}} = fill(missing, t)
        ϕ2::Vector{Union{Missing,Float64}} = fill(missing, t)
        counts = fill(0.0, n, n)
        for metric in distance_metrics
            # metric = distance_metrics[1]
            D::Matrix{Float64} = fill(-Inf, n, n)
            for i = 1:n
                # i = 1
                ϕ1 .= Φ.phenotypes[i, :]
                bool1 = .!ismissing.(ϕ1) .&& .!isnan.(ϕ1) .&& .!isinf.(ϕ1)
                for j = 1:n
                    # j = 4
                    # Make sure we have no missing, NaN or infinite values
                    ϕ2 .= Φ.phenotypes[j, :]
                    bool2 = .!ismissing.(ϕ2) .&& .!isnan.(ϕ2) .&& .!isinf.(ϕ2)
                    idx = findall(bool1 .&& bool2)
                    if length(idx) < 2
                        continue
                    end
                    # Count the number of elements used to estimate the distance
                    if metric == distance_metrics[1]
                        counts[i, j] = length(idx)
                    end
                    # Estimate the distance/correlation
                    if metric == "euclidean"
                        D[i, j] = sqrt(sum((ϕ1[idx] - ϕ2[idx]) .^ 2))
                    elseif metric == "correlation"
                        (var(ϕ1[idx]) < 1e-7) || (var(ϕ2[idx]) < 1e-7) ? continue : nothing
                        D[i, j] = cor(ϕ1[idx], ϕ2[idx])
                    elseif metric == "mad"
                        D[i, j] = mean(abs.(ϕ1[idx] - ϕ2[idx]))
                    elseif metric == "rmsd"
                        D[i, j] = sqrt(mean((ϕ1[idx] - ϕ2[idx]) .^ 2))
                    elseif metric == "χ²"
                        D[i, j] = sum((ϕ1[idx] - ϕ2[idx]) .^ 2 ./ (ϕ2[idx] .+ eps(Float64)))
                    else
                        throw(
                            ErrorException(
                                "This should not happen as we checked for the validity of the distance_metrics above.",
                            ),
                        )
                    end
                end
            end
            push!(dimension, "entries")
            push!(metric_names, metric)
            push!(matrices, D)
        end
        push!(dimension, "entries")
        push!(metric_names, "counts")
        push!(matrices, counts)
    end
    # Output
    if length(dimension) == 0
        throw(ErrorException("Phenomes struct is too sparse. No distance matrix was calculated."))
    end
    dist::Dict{String,Matrix{Float64}} = Dict()
    for i in eachindex(dimension)
        # i = 4
        key = string(dimension[i], "|", metric_names[i])
        dist[key] = matrices[i]
    end
    (Φ.traits, Φ.entries, dist)
end

"""
    plot(phenomes::Phenomes; nbins::Int64 = 10)::Nothing

Generate diagnostic plots for phenotypic data stored in a `Phenomes` struct.

# Arguments
- `phenomes::Phenomes`: A Phenomes struct containing phenotypic data
- `nbins::Int64=10`: Number of bins for the histograms (optional)

# Description
For each population in the dataset:
1. Creates histograms showing the distribution of each trait
2. Generates a heatmap of trait correlations for traits with non-zero variance

# Notes
- Skips traits with all missing, NaN, or infinite values
- Only includes traits with variance > 1e-10 in correlation analysis
- Requires at least 3 data points to generate a histogram
- Uses UnicodePlots for visualization in terminal

# Examples
```
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3);

julia> GenomicBreedingCore.plot(phenomes);

```
"""
function plot(phenomes::Phenomes; nbins::Int64 = 10)
    # phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3); nbins = 10;
    if !checkdims(phenomes)
        throw(ArgumentError("Phenomes struct is corrupted ☹."))
    end
    # View distributions per population per trait using histograms
    for pop in unique(phenomes.populations)
        # pop = phenomes.populations[1]
        idx_trait_with_variance::Vector{Int64} = []
        for j in eachindex(phenomes.traits)
            # j = 1
            println("##############################################")
            println("Population: " * pop * " | Trait: " * phenomes.traits[j])
            idx = findall(
                (phenomes.populations .== pop) .&&
                .!ismissing.(phenomes.phenotypes[:, j]) .&&
                .!isnan.(phenomes.phenotypes[:, j]) .&&
                .!isinf.(phenomes.phenotypes[:, j]),
            )
            if length(idx) == 0
                println("All values are missing, NaN and/or infinities.")
                continue
            end
            ϕ::Vector{Float64} = phenomes.phenotypes[idx, j]
            if StatsBase.var(ϕ) > 1e-10
                append!(idx_trait_with_variance, j)
            end
            if length(ϕ) > 2
                plt = UnicodePlots.histogram(ϕ, title = phenomes.traits[j], vertical = false, nbins = nbins)
                display(plt)
            else
                println(string("Trait: ", phenomes.traits[j], " has ", length(ϕ), " non-missing data points."))
            end
        end
        # Build the correlation matrix
        idx_pop = findall(phenomes.populations .== pop)
        _, _, dist = try
            distances(
                slice(phenomes, idx_entries = idx_pop, idx_traits = idx_trait_with_variance),
                distance_metrics = ["correlation"],
            )
        catch
            println("Error in computing distances for the Phenomes struct.")
            continue
        end
        C = try
            dist["traits|correlation"]
        catch
            continue
        end
        plt = UnicodePlots.heatmap(
            C;
            height = length(phenomes.traits),
            width = length(phenomes.traits),
            title = string(
                "Trait correlations: {",
                join(
                    string.(collect(1:length(idx_trait_with_variance))) .* " => " .*
                    phenomes.traits[idx_trait_with_variance],
                    ", ",
                ),
                "}",
            ),
        )
        display(plt)
    end
    # Return John Snow's knowledge.
    return nothing
end
