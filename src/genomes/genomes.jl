"""
    clone(x::Genomes)::Genomes

Create a deep copy of a `Genomes` object.

This function performs a deep clone of all fields in the `Genomes` object, including:
- entries
- populations 
- loci_alleles
- allele_frequencies
- mask

Returns a new `Genomes` instance with identical but independent data.

# Arguments
- `x::Genomes`: The source Genomes object to clone

# Returns
- `Genomes`: A new Genomes object containing deep copies of all fields

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = Genomes(n=2, p=2);

julia> copy_genomes = clone(genomes)
Genomes(["", ""], ["", ""], ["", ""], Union{Missing, Float64}[missing missing; missing missing], Bool[1 1; 1 1])
```
"""
function clone(x::Genomes)::Genomes
    out = Genomes(n = length(x.entries), p = length(x.loci_alleles))
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        setfield!(out, field, deepcopy(getfield(x, field)))
    end
    out
end


"""
    Base.hash(x::Genomes, h::UInt)::UInt

Compute a hash value for a `Genomes` struct.

This hash function considers three key components of the `Genomes` struct:
- entries
- populations
- loci_alleles

For performance reasons, `allele_frequencies` and `mask` fields are deliberately excluded 
from the hash computation.

# Arguments
- `x::Genomes`: The Genomes struct to hash
- `h::UInt`: The hash seed value

# Returns
- `UInt`: A hash value for the Genomes struct

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = Genomes(n=2, p=2);

julia> typeof(hash(genomes))
UInt64
```
"""
function Base.hash(x::Genomes, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        if field == :allele_frequencies || field == :mask
            continue
        end
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    ==(x::Genomes, y::Genomes)::Bool

Compare two `Genomes` structs for equality by comparing their hash values.

This method implements equality comparison for `Genomes` structs by utilizing their hash values,
ensuring that two genomes are considered equal if and only if they have identical structural
properties and content.

# Arguments
- `x::Genomes`: First Genomes struct to compare
- `y::Genomes`: Second Genomes struct to compare

# Returns
- `Bool`: `true` if the genomes are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes_1 = genomes = Genomes(n=2,p=4);

julia> genomes_2 = genomes = Genomes(n=2,p=4);

julia> genomes_3 = genomes = Genomes(n=1,p=2);

julia> genomes_1 == genomes_2
true

julia> genomes_1 == genomes_3
false
```
"""
function Base.:(==)(x::Genomes, y::Genomes)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(genomes::Genomes)::Bool

Check dimension compatibility of the fields in a `Genomes` struct.

Returns `true` if all dimensions are compatible, `false` otherwise.

# Arguments
- `genomes::Genomes`: A Genomes struct containing genomic data

# Details
Verifies that:
- Number of entries matches number of populations (n)
- Entry names are unique
- Number of loci alleles matches width of frequency matrix (p) 
- Locus-allele combinations are unique
- Entries are unique
- Dimensions of frequency matrix (n×p) match mask matrix dimensions

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = Genomes(n=2,p=4);

julia> checkdims(genomes)
false

julia> genomes.entries = ["entry_1", "entry_2"];

julia> genomes.loci_alleles = ["chr1\\t1\\tA|T\\tA", "chr1\\t2\\tC|G\\tG", "chr2\\t3\\tA|T\\tA", "chr2\\t4\\tG|T\\tG"];

julia> checkdims(genomes)
true
```
"""
function checkdims(genomes::Genomes)::Bool
    n, p = size(genomes.allele_frequencies)
    if (n != length(genomes.entries)) ||
       (n != length(unique(genomes.entries))) ||
       (n != length(genomes.populations)) ||
       (p != length(genomes.loci_alleles)) ||
       (p != length(unique(genomes.loci_alleles))) ||
       ((n, p) != size(genomes.mask))
        return false
    end
    true
end

"""
    dimensions(genomes::Genomes)::Dict{String, Int64}

Calculate various dimensional metrics of a Genomes struct.

Returns a dictionary containing the following metrics:
- `"n_entries"`: Number of unique entries/samples
- `"n_populations"`: Number of unique populations
- `"n_loci_alleles"`: Total number of loci-allele combinations
- `"n_chr"`: Number of chromosomes
- `"n_loci"`: Number of unique loci across all chromosomes
- `"max_n_alleles"`: Maximum number of alleles observed at any locus
- `"n_missing"`: Count of missing values in allele frequencies

# Arguments
- `genomes::Genomes`: A valid Genomes struct containing genetic data

# Returns
- `Dict{String,Int64}`: Dictionary containing dimensional metrics

# Throws
- `ArgumentError`: If the Genomes struct is corrupted (fails dimension check)

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> dimensions(genomes)
Dict{String, Int64} with 7 entries:
  "n_entries"      => 100
  "n_chr"          => 7
  "n_loci"         => 1000
  "n_loci_alleles" => 3000
  "n_populations"  => 1
  "n_missing"      => 0
  "max_n_alleles"  => 4
```
"""
function dimensions(genomes::Genomes)::Dict{String,Int64}
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    n_entries::Int64 = length(unique(genomes.entries))
    n_populations::Int64 = length(unique(genomes.populations))
    n_loci_alleles::Int64 = length(genomes.loci_alleles)
    n_loci::Int64 = 0
    n_chr::Int64 = 0
    max_n_alleles::Int64 = 0
    chr::String = ""
    pos::Int64 = 0
    for locus in genomes.loci_alleles
        # locus = genomes.loci_alleles[1]
        locus_ids::Vector{String} = split(locus, '\t')
        if n_loci == 0
            chr = locus_ids[1]
            pos = parse(Int64, locus_ids[2])
            max_n_alleles = length(split(locus_ids[3], '|'))
            n_chr += 1
            n_loci += 1
        else
            if ((chr == locus_ids[1]) && (pos != parse(Int64, locus_ids[2]))) || (chr != locus_ids[1])
                if chr != locus_ids[1]
                    n_chr += 1
                end
                chr = locus_ids[1]
                pos = parse(Int64, locus_ids[2])
                n_alleles = length(split(locus_ids[3], '|'))
                max_n_alleles = max_n_alleles < n_alleles ? n_alleles : max_n_alleles
                n_loci += 1
            end
        end
    end
    return Dict(
        "n_entries" => n_entries,
        "n_populations" => n_populations,
        "n_loci_alleles" => n_loci_alleles,
        "n_chr" => n_chr,
        "n_loci" => n_loci,
        "max_n_alleles" => max_n_alleles,
        "n_missing" => sum(ismissing.(genomes.allele_frequencies)),
    )
end

"""
    loci_alleles(genomes::Genomes; verbose::Bool = false)::Tuple{Vector{String},Vector{Int64},Vector{String}}

Extract chromosomes, positions, and alleles information from a `Genomes` object.

Returns a tuple of three vectors containing:
1. Chromosomes identifiers as strings
2. Base-pair positions as integers
3. Allele identifiers as strings

Each vector has length equal to the total number of loci-allele combinations in the genome.
The function processes the data in parallel using multiple threads for performance optimization.


# Arguments
- `genomes::Genomes`: A valid Genomes object containing loci and allele information
- `verbose::Bool = false`: If true, displays a progress bar during extraction

# Returns
- `Tuple{Vector{String},Vector{Int64},Vector{String}}`: A tuple containing three vectors:
    - chromosomes: Vector of chromosome identifiers
    - positions: Vector of base-pair positions
    - alleles: Vector of allele identifiers

# Throws
- `ArgumentError`: If the Genomes struct dimensions are invalid or corrupted

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> chromosomes, positions, alleles = loci_alleles(genomes);

julia> length(chromosomes), length(positions), length(alleles)
(3000, 3000, 3000)
```
"""
function loci_alleles(genomes::Genomes; verbose::Bool = false)::Tuple{Vector{String},Vector{Int64},Vector{String}}
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    p = length(genomes.loci_alleles)
    chromosomes = Vector{String}(undef, p)
    positions = Vector{Int64}(undef, p)
    alleles = Vector{String}(undef, p)
    if verbose
        pb = ProgressMeter.Progress(p, desc = "Extract locus-allele names")
    end
    Threads.@threads for i = 1:p
        locus = genomes.loci_alleles[i]
        locus_ids::Vector{String} = split(locus, '\t')
        chromosomes[i] = locus_ids[1]
        positions[i] = parse(Int64, locus_ids[2])
        alleles[i] = locus_ids[4]
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    return (chromosomes, positions, alleles)
end

"""
    loci(genomes::Genomes; verbose::Bool = false)::Tuple{Vector{String},Vector{Int64},Vector{Int64},Vector{Int64}}

Extract genomic positional information from a `Genomes` object, returning a tuple of vectors containing
chromosome names, positions, and locus boundary indices.

# Arguments
- `genomes::Genomes`: A Genomes object containing genomic data
- `verbose::Bool = false`: If true, displays a progress bar during computation

# Returns
A tuple containing four vectors:
- `chromosomes::Vector{String}`: Names of chromosomes
- `positions::Vector{Int64}`: Positions within chromosomes
- `loci_ini_idx::Vector{Int64}`: Starting indices for each locus
- `loci_fin_idx::Vector{Int64}`: Ending indices for each locus

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> chromosomes, positions, loci_ini_idx, loci_fin_idx = loci(genomes);

julia> length(chromosomes), length(positions), length(loci_ini_idx), length(loci_fin_idx)
(1000, 1000, 1000, 1000)
```
"""
function loci(genomes::Genomes; verbose::Bool = false)::Tuple{Vector{String},Vector{Int64},Vector{Int64},Vector{Int64}}
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    chromosomes::Vector{String} = []
    positions::Vector{Int64} = []
    loci_ini_idx::Vector{Int64} = []
    loci_fin_idx::Vector{Int64} = []
    # Cannot be multi-threaded as we need the loci-alleles sorted
    if verbose
        pb = ProgressMeter.Progress(
            length(genomes.loci_alleles),
            desc = "Extract loci identities iteratively (single-threaded)",
        )
    end
    for i in eachindex(genomes.loci_alleles)
        locus = genomes.loci_alleles[i]
        locus_ids::Vector{String} = split(locus, '\t')
        if i == 1
            push!(chromosomes, locus_ids[1])
            push!(positions, parse(Int64, locus_ids[2]))
            push!(loci_ini_idx, i)
        else
            if ((chromosomes[end] == locus_ids[1]) && (positions[end] != parse(Int64, locus_ids[2]))) ||
               (chromosomes[end] != locus_ids[1])
                push!(chromosomes, locus_ids[1])
                push!(positions, parse(Int64, locus_ids[2]))
                push!(loci_ini_idx, i)
                push!(loci_fin_idx, i - 1)
            end
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    if loci_fin_idx[end] < length(genomes.loci_alleles)
        push!(loci_fin_idx, length(genomes.loci_alleles))
    end
    return (chromosomes, positions, loci_ini_idx, loci_fin_idx)
end

"""
    distances(
        genomes::Genomes; 
        distance_metrics::Vector{String}=["euclidean", "correlation", "mad", "rmsd", "χ²"],
        idx_loci_alleles::Union{Nothing, Vector{Int64}} = nothing,
        include_loci_alleles::Bool = true,
        include_entries::Bool = true,
        include_counts::Bool = true,
        verbose::Bool = false
    )::Tuple{Vector{String},Vector{String},Dict{String,Matrix{Float64}}}

Calculate pairwise distances/similarity metrics between loci-alleles and entries in a `Genomes` object.

# Arguments
- `genomes::Genomes`: Input Genomes object
- `distance_metrics::Vector{String}`: Vector of distance metrics to calculate. Valid options:
  - "euclidean": Euclidean distance
  - "correlation": Pearson correlation coefficient 
  - "mad": Mean absolute deviation
  - "rmsd": Root mean square deviation 
  - "χ²": Chi-square distance
- `idx_loci_alleles::Union{Nothing, Vector{Int64}}`: Optional indices of loci-alleles to include. If nothing, randomly samples 100 loci-alleles.
- `include_loci_alleles::Bool`: Whether to calculate distances between loci-alleles. Defaults to true.
- `include_entries::Bool`: Whether to calculate distances between entries. Defaults to true.
- `include_counts::Bool`: Whether to include matrices showing number of valid pairs used. Defaults to true.
- `verbose::Bool`: Whether to show progress bars. Defaults to false.

# Returns
Tuple containing:
1. Vector of loci-allele names used
2. Vector of entry names  
3. Dictionary mapping "{dimension}|{metric}" to distance matrices, where:
   - dimension is either "loci_alleles" or "entries"
   - metric is one of the distance metrics or "counts" (number of valid pairs used)
   - matrices contain pairwise distances/correlations (-Inf where insufficient data)

# Details
- For loci-alleles, calculates distances between allele frequency profiles across entries
- For entries, calculates distances between entries based on their allele frequencies
- Requires at least 2 valid (non-missing, finite) pairs to calculate metrics
- Includes count matrices showing number of valid pairs used per calculation
- Multi-threaded implementation which uses indexing on pre-allocated vectors and matrices which should avoid data races

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> (loci_alleles_names, entries, dist) = distances(genomes, distance_metrics=["correlation", "χ²"]);

julia> sort(string.(keys(dist))) == ["entries|correlation", "entries|counts", "entries|χ²", "loci_alleles|correlation", "loci_alleles|counts", "loci_alleles|χ²"]
true

julia> C = dist["entries|correlation"]; C[diagind(C)] == repeat([1], length(genomes.entries))
true

julia> χ² = dist["loci_alleles|χ²"]; χ²[diagind(χ²)] == repeat([0.0], 100)
true
```
"""
function distances(
    genomes::Genomes;
    distance_metrics::Vector{String} = ["euclidean", "correlation", "mad", "rmsd", "χ²"],
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    include_loci_alleles::Bool = true,
    include_entries::Bool = true,
    include_counts::Bool = true,
    verbose::Bool = false,
)::Tuple{Vector{String},Vector{String},Dict{String,Matrix{Float64}}}
    # genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);
    # distance_metrics = ["euclidean", "correlation", "mad", "rmsd", "χ²"]; idx_loci_alleles = nothing
    # include_loci_alleles = true; include_entries = false; include_counts = false; verbose = true
    if !checkdims(genomes)
        throw(ArgumentError("The genomes struct is corrupted."))
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
    n_loci_alleles = length(genomes.loci_alleles)
    idx_loci_alleles = if isnothing(idx_loci_alleles)
        if verbose
            @warn "Randomly sampling 100 loci-alleles"
        end
        sort(sample(1:n_loci_alleles, 100, replace = false))
    else
        if (minimum(idx_loci_alleles) < 1) || (maximum(idx_loci_alleles) > n_loci_alleles)
            throw(ArgumentError("We accept `idx_loci_alleles` from 1 to `n_loci_alleles` of `genomes`."))
        end
        unique(sort(idx_loci_alleles))
    end
    # Instantiate vectors of matrices and metric names
    dimension::Vector{String} = [] # across loci_alleles and/or entries
    metric_names::Vector{String} = []
    matrices::Vector{Matrix{Float64}} = []
    # Number of traits and entries
    n = length(genomes.entries)
    t = length(idx_loci_alleles)
    # Per locus-allele first
    if include_loci_alleles && (t > 1)
        if include_counts
            counts = fill(0.0, t, t)
        end
        for metric in distance_metrics
            # metric = distance_metrics[2]
            # Use -Inf for correlation and Inf for the other distance metrics for NaN values
            D::Matrix{Float64} = if metric == "correlation"
                fill(-Inf, t, t)
            else
                fill(Inf, t, t)
            end
            if verbose
                pb = ProgressMeter.Progress(length(idx_loci_alleles), desc = "Calculate distances for loci-alleles")
            end
            # Multi-threaded distance calculation (no need for thread locking as we are accessing unique indexes)
            Threads.@threads for i in eachindex(idx_loci_alleles)
                # i = 1
                # println(i)
                ix = idx_loci_alleles[i]
                y1 = genomes.allele_frequencies[:, ix]
                bool1 = .!ismissing.(y1) .&& .!isnan.(y1) .&& .!isinf.(y1)
                for (j, jx) in enumerate(idx_loci_alleles)
                    # i = 1; j = 3; ix = idx_loci_alleles[i]; jx = idx_loci_alleles[j]
                    # println(j)
                    # Make sure we have no missing, NaN or infinite values
                    y2 = genomes.allele_frequencies[:, jx]
                    bool2 = .!ismissing.(y2) .&& .!isnan.(y2) .&& .!isinf.(y2)
                    idx = findall(bool1 .&& bool2)
                    if length(idx) < 2
                        continue
                    end
                    # Count the number of elements used to estimate the distance
                    if include_counts && (metric == distance_metrics[1])
                        counts[i, j] = length(idx)
                    end
                    # Estimate the distance/correlation
                    D[i, j] = if metric == "euclidean"
                        sqrt(sum((y1[idx] - y2[idx]) .^ 2))
                    elseif metric == "correlation"
                        (var(y1[idx]) < 1e-7) || (var(y2[idx]) < 1e-7) ? continue : nothing
                        cor(y1[idx], y2[idx])
                    elseif metric == "mad"
                        mean(abs.(y1[idx] - y2[idx]))
                    elseif metric == "rmsd"
                        sqrt(mean((y1[idx] - y2[idx]) .^ 2))
                    elseif metric == "χ²"
                        sum((y1[idx] - y2[idx]) .^ 2 ./ (y2[idx] .+ eps(Float64)))
                    else
                        throw(
                            ErrorException(
                                "This should not happen as we checked for the validity of the distance_metrics above.",
                            ),
                        )
                    end
                end
                if verbose
                    ProgressMeter.next!(pb)
                end
            end
            if verbose
                ProgressMeter.finish!(pb)
            end
            push!(dimension, "loci_alleles")
            push!(metric_names, metric)
            push!(matrices, D)
        end
        if include_counts
            push!(dimension, "loci_alleles")
            push!(metric_names, "counts")
            push!(matrices, counts)
        end
    end
    # Finally per entry
    if include_entries && (n > 1)
        if include_counts
            counts = fill(0.0, n, n)
        end
        for metric in distance_metrics
            # metric = distance_metrics[1]
            # Use -Inf for correlation and Inf for the other distance metrics for NaN values
            D::Matrix{Float64} = if metric == "correlation"
                fill(-Inf, n, n)
            else
                fill(Inf, n, n)
            end
            # Multi-threaded distance calculation (no need for thread locking as we are accessing unique indexes)
            Threads.@threads for i = 1:n
                ϕ1 = genomes.allele_frequencies[i, idx_loci_alleles]
                bool1 = .!ismissing.(ϕ1) .&& .!isnan.(ϕ1) .&& .!isinf.(ϕ1)
                for j = 1:n
                    # i = 1; j = 3
                    # Make sure we have no missing, NaN or infinite values
                    ϕ2 = genomes.allele_frequencies[j, idx_loci_alleles]
                    bool2 = .!ismissing.(ϕ2) .&& .!isnan.(ϕ2) .&& .!isinf.(ϕ2)
                    idx = findall(bool1 .&& bool2)
                    if length(idx) < 2
                        continue
                    end
                    # Count the number of elements used to estimate the distance
                    if include_counts && (metric == distance_metrics[1])
                        counts[i, j] = length(idx)
                    end
                    # Estimate the distance/correlation
                    if metric == "euclidean"
                        D[i, j] = sqrt(sum((ϕ1[idx] - ϕ2[idx]) .^ 2))
                    elseif metric == "correlation"
                        D[i, j] = if (var(ϕ1[idx]) < 1e-7) || (var(ϕ2[idx]) < 1e-7)
                            -Inf
                        else
                            cor(ϕ1[idx], ϕ2[idx])
                        end
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
        if include_counts
            push!(dimension, "entries")
            push!(metric_names, "counts")
            push!(matrices, counts)
        end
    end
    # Output
    if length(dimension) == 0
        throw(ErrorException("Genomes struct is too sparse. No distance matrix was calculated."))
    end
    distance_matrices::Dict{String,Matrix{Float64}} = Dict()
    for i in eachindex(dimension)
        # i = 4
        # println(i)
        key = string(dimension[i], "|", metric_names[i])
        distance_matrices[key] = matrices[i]
    end
    (genomes.loci_alleles[idx_loci_alleles], genomes.entries, distance_matrices)
end

"""
    plot(genomes::Genomes, seed::Int64 = 42)::Nothing

Generate visualization plots for allele frequencies in genomic data.

For each population in the dataset, creates three plots:
1. Histogram of per-entry allele frequencies
2. Histogram of mean allele frequencies per locus
3. Correlation heatmap of allele frequencies between loci

# Arguments
- `genomes::Genomes`: A Genomes struct containing allele frequency data
- `seed::Int64=42`: Random seed for reproducibility of sampling loci

# Returns
- `Nothing`: Displays plots but doesn't return any value

# Notes
- Uses up to 100 randomly sampled loci for visualization
- Handles missing values in the data
- Displays folded frequency spectra (both q and 1-q)
- Will throw ArgumentError if the Genomes struct is corrupted

# Examples
```
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> GenomicBreedingCore.plot(genomes)

```
"""
function plot(genomes::Genomes, seed::Int64 = 42)
    # genomes = simulategenomes(n=100, l=1_000, n_alleles=3, n_populations=3, μ_β_params=(2.0,2.0), sparsity=0.01, verbose=false); seed::Int64=42;
    # Per poulation, using min([250, p]) randomly sampled loci plot:
    #   (1) histogram of allele frequencies per entry,
    #   (2) histogram of mean allele frequencies per locus, and
    #   (3) correlation heatmap of allele frequencies.
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    rng::TaskLocalRNG = Random.seed!(seed)
    for pop in unique(genomes.populations)
        # pop = genomes.populations[1]
        println("##############################################")
        println("Population: " * pop)
        p = size(genomes.allele_frequencies, 2)
        idx_row::Vector{Int64} = findall(genomes.populations .== pop)
        idx_col::Vector{Int64} = StatsBase.sample(rng, 1:p, minimum([100, p]); replace = false, ordered = true)
        Q = genomes.allele_frequencies[idx_row, idx_col]
        q::Vector{Float64} = filter(!ismissing, reshape(Q, (length(idx_row) * length(idx_col), 1)))
        plt_1 = UnicodePlots.histogram(
            vcat(q, 1.00 .- q);
            title = string("Per entry allele frequencies (", pop, ")"),
            vertical = true,
            nbins = 50,
        )
        display(plt_1)
        # # Mean allele frequencies across entries unfolded
        μ_q::Vector{Float64} = fill(0.0, p)
        for j = 1:length(idx_col)
            μ_q[j] = mean(skipmissing(Q[:, j]))
        end
        plt_2 = UnicodePlots.histogram(
            vcat(μ_q, 1.00 .- μ_q);
            title = string("Mean allele frequencies (", pop, ")"),
            vertical = true,
            nbins = 50,
        )
        display(plt_2)
        # Correlation between allele frequencies
        _, _, dist = try
            distances(
                slice(genomes, idx_entries = idx_row, idx_loci_alleles = idx_col),
                distance_metrics = ["correlation"],
                idx_loci_alleles = collect(1:length(idx_col)),
            )
        catch
            println("Error in computing distances for the Genomes struct.")
            return nothing
        end
        C = try
            dist["loci_alleles|correlation"]
        catch
            continue
        end
        idx = []
        for i = 1:size(C, 1)
            # i = 1
            if sum(isinf.(C[i, :])) < length(idx_col)
                push!(idx, i)
            end
        end
        if length(idx) < 10
            return nothing
        end
        C = C[idx, idx]
        plt_3 = UnicodePlots.heatmap(
            C;
            height = size(C, 2),
            width = size(C, 2),
            zlabel = string("Pairwise loci correlation (", pop, ")"),
        )
        display(plt_3)
    end
    # Return nada!
    return nothing
end
