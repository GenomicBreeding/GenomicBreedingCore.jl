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

## Arguments
- `x::Genomes`: The source Genomes object to clone

## Returns
- `Genomes`: A new Genomes object containing deep copies of all fields

## Example
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore)
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
```jldoctest; setup = :(using GBCore, LinearAlgebra)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> (loci_alleles_names, entries, dist) = distances(genomes, distance_metrics=["correlation", "χ²"]);

julia> sort(string.(keys(dist))) == ["entries|correlation", "entries|counts", "entries|χ²", "loci_alleles|correlation", "loci_alleles|counts", "loci_alleles|χ²"]
true

julia> C = dist["entries|correlation"]; C[diagind(C)] == repeat([1], length(genomes.entries))
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
        # @warn "Randomly sampling 100 loci-alleles"
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
            D::Matrix{Float64} = fill(-Inf, t, t)
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
        ϕ1::Vector{Union{Missing,Float64}} = fill(missing, t)
        ϕ2::Vector{Union{Missing,Float64}} = fill(missing, t)
        if include_counts
            counts = fill(0.0, n, n)
        end
        for metric in distance_metrics
            # metric = distance_metrics[1]
            D::Matrix{Float64} = fill(-Inf, n, n)
            for i = 1:n
                ϕ1 .= genomes.allele_frequencies[i, idx_loci_alleles]
                bool1 = .!ismissing.(ϕ1) .&& .!isnan.(ϕ1) .&& .!isinf.(ϕ1)
                for j = 1:n
                    # i = 1; j = 3
                    # Make sure we have no missing, NaN or infinite values
                    ϕ2 .= genomes.allele_frequencies[j, idx_loci_alleles]
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

julia> GBCore.plot(genomes)

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

"""
    slice(
        genomes::Genomes; 
        idx_entries::Union{Nothing, Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing, Vector{Int64}} = nothing,
        verbose::Bool = false
    )::Genomes

Create a subset of a `Genomes` struct by selecting specific entries and loci-allele combinations.

# Arguments
- `genomes::Genomes`: The source genomic data structure to be sliced
- `idx_entries::Union{Nothing, Vector{Int64}}`: Indices of entries to keep. If `nothing`, all entries are kept
- `idx_loci_alleles::Union{Nothing, Vector{Int64}}`: Indices of loci-allele combinations to keep. If `nothing`, all loci-alleles are kept
- `verbose::Bool`: If true, displays a progress bar during slicing. Defaults to false

# Returns
- `Genomes`: A new `Genomes` struct containing only the selected entries and loci-allele combinations

# Performance Notes
- The function uses multi-threaded implementation for optimal performance
- Progress bar is available when `verbose=true` to monitor the slicing operation
- Memory efficient implementation that creates a new pre-allocated structure

# Behaviour
- Both index vectors are automatically sorted and deduplicated
- If both `idx_entries` and `idx_loci_alleles` are `nothing`, returns a clone of the input
- Maintains all relationships and structure of the original genomic data
- Preserves population assignments and allele frequencies for selected entries

# Validation
- Performs dimension checks on both input and output genomic structures
- Validates that all indices are within proper bounds
- Ensures data consistency throughout the slicing operation

# Throws
- `ArgumentError`: If input `Genomes` struct is corrupted or indices are out of bounds
- `DimensionMismatch`: If the resulting sliced genome has inconsistent dimensions

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> sliced_genomes = slice(genomes, idx_entries=collect(1:10), idx_loci_alleles=collect(1:300));

julia> dimensions(sliced_genomes)
Dict{String, Int64} with 7 entries:
  "n_entries"      => 10
  "n_chr"          => 1
  "n_loci"         => 100
  "n_loci_alleles" => 300
  "n_populations"  => 1
  "n_missing"      => 0
  "max_n_alleles"  => 4
```
"""
function slice(
    genomes::Genomes;
    idx_entries::Union{Nothing,Vector{Int64}} = nothing,
    idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
    verbose::Bool = false,
)::Genomes
    # genomes::Genomes = simulategenomes(); idx_entries::Vector{Int64}=sample(1:100, 10); idx_loci_alleles::Vector{Int64}=sample(1:10_000, 1000); verbose=true
    # Check genomes struct
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    # Return early if not slicing needed
    if isnothing(idx_entries) && isnothing(idx_loci_alleles)
        if verbose
            println("Slicing not needed as `idx_entries` and `idx_loci` are both unset or equal to `nothing`.")
        end
        return clone(genomes)
    end
    genomes_dims::Dict{String,Int64} = dimensions(genomes)
    n_entries::Int64 = genomes_dims["n_entries"]
    n_loci_alleles::Int64 = genomes_dims["n_loci_alleles"]
    idx_entries = if isnothing(idx_entries)
        collect(1:n_entries)
    else
        if (minimum(idx_entries) < 1) || (maximum(idx_entries) > n_entries)
            throw(ArgumentError("We accept `idx_entries` from 1 to `n_entries` of `genomes`."))
        end
        unique(sort(idx_entries))
    end
    idx_loci_alleles = if isnothing(idx_loci_alleles)
        collect(1:n_loci_alleles)
    else
        if (minimum(idx_loci_alleles) < 1) || (maximum(idx_loci_alleles) > n_loci_alleles)
            throw(ArgumentError("We accept `idx_loci_alleles` from 1 to `n_loci_alleles` of `genomes`."))
        end
        unique(sort(idx_loci_alleles))
    end
    n, p = length(idx_entries), length(idx_loci_alleles)
    sliced_genomes::Genomes = Genomes(n = n, p = p)
    if verbose
        pb = ProgressMeter.Progress(length(idx_entries), desc = "Slicing genomes")
    end
    # Multi-threaded copying of selected allele frequencies from genomes to sliced_genomes (no need for thread locking as we are accessing unique indexes)
    for (i1, i2) in enumerate(idx_entries)
        sliced_genomes.entries[i1] = genomes.entries[i2]
        sliced_genomes.populations[i1] = genomes.populations[i2]
        Threads.@threads for j1 in eachindex(idx_loci_alleles)
            j2 = idx_loci_alleles[j1]
            if i1 == 1
                sliced_genomes.loci_alleles[j1] = genomes.loci_alleles[j2]
            end
            sliced_genomes.allele_frequencies[i1, j1] = genomes.allele_frequencies[i2, j2]
            sliced_genomes.mask[i1, j1] = genomes.mask[i2, j2]
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    ### Check dimensions
    if !checkdims(sliced_genomes)
        throw(DimensionMismatch("Error slicing the genome."))
    end
    # Output
    return sliced_genomes
end


"""
    filter(genomes::Genomes; verbose::Bool = false)::Genomes

Filter a Genomes struct by removing entries and loci with missing data based on the mask matrix.

# Description
This function filters a Genomes struct by:
1. Removing rows (entries) where any column has a false value in the mask matrix
2. Removing columns (loci) where any row has a false value in the mask matrix

# Arguments
- `genomes::Genomes`: Input Genomes struct containing genetic data and a mask matrix
- `verbose::Bool`: Optional flag to control verbose output (default: false)

# Returns
- `Genomes`: A new filtered Genomes struct with complete data (no missing values)

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(verbose=false); genomes.mask[1:10, 42:100] .= false;
    
julia> filtered_genomes = filter(genomes);

julia> size(filtered_genomes.allele_frequencies)
(90, 9941)
```
"""
function Base.filter(genomes::Genomes; verbose::Bool = false)::Genomes
    # genomes = simulategenomes(); genomes.mask[1:10, 42:100] .= false;
    idx_entries = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
    idx_loci_alleles = findall(mean(genomes.mask, dims = 1)[1, :] .== 1.0)
    filtered_genomes::Genomes =
        slice(genomes, idx_entries = idx_entries; idx_loci_alleles = idx_loci_alleles, verbose = verbose)
    filtered_genomes
end

"""
    sparsities(genomes::Genomes) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate the sparsity (proportion of missing data) for each entry and locus in a `Genomes` object.

Returns a tuple of two vectors:
- First vector contains sparsity values for each entry (row-wise mean of missing values)
- Second vector contains sparsity values for each locus (column-wise mean of missing values)

The function processes the data in parallel using multiple threads for performance optimization.

# Arguments
- `genomes::Genomes`: A Genomes object containing allele frequency data with potentially missing values

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: A tuple containing:
    - Vector of entry sparsities (values between 0.0 and 1.0)
    - Vector of locus sparsities (values between 0.0 and 1.0)

# Example
```jldoctest; setup = :(using GBCore, StatsBase)
julia> genomes = simulategenomes(n=100, l=1_000, sparsity=0.25, verbose=false);

julia> entry_sparsities, locus_sparsities = sparsities(genomes);

julia> abs(0.25 - mean(entry_sparsities)) < 0.0001
true

julia> abs(0.25 - mean(locus_sparsities)) < 0.0001
true
```
"""
function sparsities(genomes::Genomes)::Tuple{Vector{Float64},Vector{Float64}}
    # No thread locking required as we are assigning values per index
    entry_sparsities::Array{Float64,1} = fill(0.0, length(genomes.entries))
    locus_sparsities::Array{Float64,1} = fill(0.0, length(genomes.loci_alleles))
    Threads.@threads for i in eachindex(entry_sparsities)
        entry_sparsities[i] = mean(Float64.(ismissing.(genomes.allele_frequencies[i, :])))
    end
    Threads.@threads for j in eachindex(locus_sparsities)
        locus_sparsities[j] = mean(Float64.(ismissing.(genomes.allele_frequencies[:, j])))
    end
    entry_sparsities, locus_sparsities
end

"""
    filter(
        genomes::Genomes,
        maf::Float64;
        max_entry_sparsity::Float64 = 0.0,
        max_locus_sparsity::Float64 = 0.0,
        max_prop_pc_varexp::Float64 = 1.00,
        max_entry_sparsity_percentile::Float64 = 0.90,
        max_locus_sparsity_percentile::Float64 = 0.50,
        chr_pos_allele_ids::Union{Nothing,Vector{String}} = nothing,
        verbose::Bool = false
    )::Genomes

Filter a Genomes struct based on multiple criteria.

# Arguments
- `genomes::Genomes`: Input genomic data structure
- `maf::Float64`: Minimum allele frequency threshold (required)
- `max_entry_sparsity::Float64`: Maximum allowed proportion of missing values per entry (default: 0.0)
- `max_locus_sparsity::Float64`: Maximum allowed proportion of missing values per locus (default: 0.0)
- `max_prop_pc_varexp::Float64`: Maximum proportion of variance explained by PC1 and PC2 for outlier detection. Set to `Inf` for no filtering by PCA. (default: 0.9)
- `max_entry_sparsity_percentile::Float64`: Percentile threshold for entry sparsity filtering (default: 0.90)
- `max_locus_sparsity_percentile::Float64`: Percentile threshold for locus sparsity filtering (default: 0.50)
- `chr_pos_allele_ids::Union{Nothing,Vector{String}}`: Optional vector of specific locus-allele combinations to retain, 
    formatted as tab-separated strings "chromosome\\tposition\\tallele"
- `verbose::Bool`: Whether to display progress bars during filtering (default: false)

# Returns
- `Genomes`: Filtered genomic data structure

# Details
Filters genomic data based on six criteria:
1. Minimum allele frequency (MAF)
2. Maximum entry sparsity (proportion of missing values per entry)
3. Maximum locus sparsity (proportion of missing values per locus)
4. Entry sparsity percentile threshold
5. Locus sparsity percentile threshold
6. PCA-based outlier detection using variance explained threshold
7. Specific locus-allele combinations (optional)

The percentile thresholds control how aggressively the sparsity filtering is applied:

- `max_entry_sparsity_percentile` (default 0.90):
  - Controls what proportion of entries to keep in each iteration
  - Higher values (e.g. 0.95) retain more entries but may require more iterations
  - Lower values (e.g. 0.75) remove more entries per iteration but may be too aggressive
  - Adjust lower if dataset has many very sparse entries
  - Adjust higher if trying to preserve more entries

- `max_locus_sparsity_percentile` (default 0.50): 
  - Controls what proportion of loci to keep in each iteration
  - Higher values (e.g. 0.75) retain more loci but may require more iterations
  - Lower values (e.g. 0.25) remove more loci per iteration
  - Typically set lower than entry percentile since loci are often more expendable
  - Adjust based on tolerance for missing data vs. desire to retain markers

The iterative filtering will stop when either:
- All sparsity thresholds are satisfied
- No further filtering is possible without violating minimum thresholds
- Dataset becomes too small for analysis

Note that each filtering iteration includes multithreaded sparsity calculation and multithreaded genomes struct slicing.

# Throws
- `ArgumentError`: If Genomes struct is corrupted, if MAF is outside [0,1], if sparsity thresholds are outside [0,1], 
    if percentile thresholds are outside [0,1], if max_prop_pc_varexp < 0, or if chr_pos_allele_ids format is invalid
- `ErrorException`: If filtering results in empty dataset or if PCA cannot be performed due to insufficient data

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, sparsity=0.01, verbose=false);

julia> filtered_genomes_1 = filter(genomes, 0.1);

julia> filtered_genomes_2 = filter(genomes, 0.1, chr_pos_allele_ids=genomes.loci_alleles[1:1000]);

julia> size(genomes.allele_frequencies)
(100, 3000)

julia> size(filtered_genomes_1.allele_frequencies)
(92, 500)

julia> size(filtered_genomes_2.allele_frequencies)
(92, 145)
```
"""
function Base.filter(
    genomes::Genomes,
    maf::Float64;
    max_entry_sparsity::Float64 = 0.0,
    max_locus_sparsity::Float64 = 0.0,
    max_prop_pc_varexp::Float64 = 0.90,
    max_entry_sparsity_percentile::Float64 = 0.90,
    max_locus_sparsity_percentile::Float64 = 0.50,
    chr_pos_allele_ids::Union{Nothing,Vector{String}} = nothing,
    verbose::Bool = false,
)::Genomes
    # genomes::Genomes = simulategenomes(n_populations=3, sparsity=0.01, seed=123456); maf=0.01; max_entry_sparsity=0.1; max_locus_sparsity = 0.25; max_prop_pc_varexp = 1.0; max_entry_sparsity_percentile = 0.9; max_locus_sparsity_percentile = 0.5
    # chr_pos_allele_ids = sample(genomes.loci_alleles, Int(floor(0.5*length(genomes.loci_alleles)))); sort!(chr_pos_allele_ids); verbose = true
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    if (maf < 0.0) || (maf > 1.0)
        throw(ArgumentError("We accept `maf` from 0.0 to 1.0."))
    end
    if (max_entry_sparsity < 0.0) || (max_entry_sparsity > 1.0)
        throw(ArgumentError("We accept `max_entry_sparsity` from 0.0 to 1.0."))
    end
    if (max_locus_sparsity < 0.0) || (max_locus_sparsity > 1.0)
        throw(ArgumentError("We accept `max_locus_sparsity` from 0.0 to 1.0."))
    end
    if (max_entry_sparsity_percentile < 0.0) || (max_entry_sparsity_percentile > 1.0)
        throw(ArgumentError("We accept `max_entry_sparsity_percentile` from 0.0 to 1.0."))
    end
    if (max_locus_sparsity_percentile < 0.0) || (max_locus_sparsity_percentile > 1.0)
        throw(ArgumentError("We accept `max_locus_sparsity_percentile` from 0.0 to 1.0."))
    end
    if max_prop_pc_varexp < 0.0
        throw(ArgumentError("We accept `max_prop_pc_varexp` from 0.0 to Inf."))
    end
    # Remove sparsest entries and then sparsest loci-alleles until no data remains or max_entry_sparsity and max_locus_sparsity are satisfied
    filtered_genomes = if (max_entry_sparsity < 1.0) || (max_locus_sparsity < 1.0)
        filtered_genomes = clone(genomes)
        entry_sparsities, locus_sparsities = sparsities(filtered_genomes)
        bool::Vector{Bool} = [
            (length(entry_sparsities) > 0) &&
            (length(locus_sparsities) > 0) &&
            (maximum(entry_sparsities) > max_entry_sparsity) &&
            (maximum(locus_sparsities) > max_locus_sparsity),
        ]
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("Iteration 0: ")
            @show dimensions(filtered_genomes)
            println("Maximum entry sparsity = $(maximum(entry_sparsities))")
            println("Maximum locus sparsity = $(maximum(locus_sparsities))")
        end
        iter = 0
        while bool[1]
            iter += 1
            # Start every iteration by removing the sparsest loci, i.e. above the 75th percentile
            filtered_genomes = begin
                m = maximum([max_locus_sparsity, quantile(locus_sparsities, max_locus_sparsity_percentile)])
                slice(filtered_genomes, idx_loci_alleles = findall(locus_sparsities .<= m))
            end
            entry_sparsities, locus_sparsities = sparsities(filtered_genomes)
            bool[1] =
                (length(entry_sparsities) > 0) &&
                (length(locus_sparsities) > 0) &&
                (maximum(entry_sparsities) > max_entry_sparsity) &&
                (maximum(locus_sparsities) > max_locus_sparsity)
            if !bool[1]
                break
            end
            # Finish each iteration by removing the sparsest entries, i.e. above the 75th percentile
            filtered_genomes = begin
                m = maximum([max_entry_sparsity, quantile(entry_sparsities, max_entry_sparsity_percentile)])
                slice(filtered_genomes, idx_entries = findall(entry_sparsities .<= m))
            end
            entry_sparsities, locus_sparsities = sparsities(filtered_genomes)
            bool[1] =
                (length(entry_sparsities) > 0) &&
                (length(locus_sparsities) > 0) &&
                (maximum(entry_sparsities) > max_entry_sparsity) &&
                (maximum(locus_sparsities) > max_locus_sparsity)
            if !bool[1]
                break
            end
            if verbose
                println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                println("Iteration $iter: ")
                @show dimensions(filtered_genomes)
                println("Maximum entry sparsity = $(maximum(entry_sparsities))")
                println("Maximum locus sparsity = $(maximum(locus_sparsities))")
            end
        end
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("Finished: ")
            @show dimensions(filtered_genomes)
            println("Maximum entry sparsity = $(maximum(entry_sparsities))")
            println("Maximum locus sparsity = $(maximum(locus_sparsities))")
        end
        # Check if we are retaining any entries and loci
        if (length(filtered_genomes.entries) == 0) || (length(filtered_genomes.loci_alleles) == 0)
            throw(
                ErrorException(
                    string(
                        "All entries and/or loci filtered out at maximum entry sparsity = ",
                        max_entry_sparsity,
                        ", and maximum locus sparsity = ",
                        max_locus_sparsity,
                        ".",
                    ),
                ),
            )
        end
        filtered_genomes
    end
    # Filter by minimum allele frequency (maf)
    filtered_genomes = if maf > 0.0
        bool_loci_alleles::Vector{Bool} = fill(false, length(filtered_genomes.loci_alleles))
        Threads.@threads for j in eachindex(bool_loci_alleles)
            q = mean(skipmissing(filtered_genomes.allele_frequencies[:, j]))
            bool_loci_alleles[j] = (q >= maf) && (q <= (1.0 - maf))
        end
        # Check if we are retaining any entries and loci
        if sum(bool_loci_alleles) == 0
            throw(ErrorException(string("All loci filtered out at minimum allele frequencies (maf) = ", maf, ".")))
        end
        slice(filtered_genomes, idx_loci_alleles = findall(bool_loci_alleles))
    end
    # @show dimensions(filtered_genomes)
    # Are we filtering out outlying loci-alleles?
    filtered_genomes = if !isinf(max_prop_pc_varexp)
        if verbose
            println(
                string(
                    "Filtering-out outlier loci-alleles, i.e. with PC1 and PC2 lying outside ",
                    max_prop_pc_varexp * 100,
                    "% of the total variance explained by PC1 and PC2.",
                ),
            )
        end
        # Extract non-missing loci-alleles across all entries
        n, p = size(filtered_genomes.allele_frequencies)
        μ = mean(filtered_genomes.allele_frequencies, dims = 1)
        σ = std(filtered_genomes.allele_frequencies, dims = 1)
        idx = findall(.!ismissing.(μ[1, :]) .&& (σ[1, :] .> 0.0))
        if length(idx) < 10
            throw(
                ErrorException(
                    "There are less than 10 loci-alleles left after removing fixed and missing values across all entries. We cannot proceed with filtering-out outlying loci-alleles. Please consider setting `max_prop_pc_varexp = Inf` to skip this filtering step.",
                ),
            )
        end
        G = filtered_genomes.allele_frequencies[:, idx]
        # Standardize the allele frequencies per locus-allele in preparation for PCA and for filtering by variance explained threshold
        μ = mean(G, dims = 1)
        σ = std(G, dims = 1)
        G = (G .- μ) ./ σ
        # PCA and filtering by a fraction of the sum of the proportion of variance explained by the first 2 PCs
        M = MultivariateStats.fit(PCA, G')
        variance_explained_pc1_pc2 = sum((M.prinvars./sum(M.prinvars))[1:2])
        max_pc = variance_explained_pc1_pc2 * max_prop_pc_varexp
        pc1 = M.proj[:, 1]
        pc2 = M.proj[:, 2]
        bool_outliers = (
            ((pc1 .> +max_pc) .&& (pc2 .> +max_pc)) .||
            ((pc1 .> +max_pc) .&& (pc2 .< -max_pc)) .||
            ((pc1 .< -max_pc) .&& (pc2 .> +max_pc)) .||
            ((pc1 .< -max_pc) .&& (pc2 .< -max_pc))
        )
        if sum(bool_outliers) == 0
            if verbose
                println("All loci passed the PCA filtering.")
            end
            filtered_genomes
        else
            loci_alleles_outliers = filtered_genomes.loci_alleles[idx[bool_outliers]]
            if verbose
                println(string("Removing ", length(loci_alleles_outliers), " loci-alleles."))
            end
            idx_loci_alleles = findall([!(x ∈ loci_alleles_outliers) for x in filtered_genomes.loci_alleles])
            slice(filtered_genomes, idx_loci_alleles = idx_loci_alleles, verbose = verbose)
        end
    end
    # Are we filtering using a list of loci-allele combination names?
    filtered_genomes = if !isnothing(chr_pos_allele_ids) && (length(chr_pos_allele_ids) > 0)
        # Parse and make sure the input loci-allele combination names are valid
        requested_chr_pos_allele_ids = begin
            chr::Vector{String} = fill("", length(chr_pos_allele_ids))
            pos::Vector{Int64} = fill(0, length(chr_pos_allele_ids))
            ale::Vector{String} = fill("", length(chr_pos_allele_ids))
            if verbose
                pb = ProgressMeter.Progress(length(chr_pos_allele_ids), desc = "Parsing loci-allele combination names")
            end
            # Multi-threaded loci-allele names (no need for thread locking as we are accessing unique indexes)
            Threads.@threads for i in eachindex(chr_pos_allele_ids)
                ids = split(chr_pos_allele_ids[i], "\t")
                if length(ids) < 3
                    throw(
                        ArgumentError(
                            string(
                                "We expect the first two elements of each item in `chr_pos_allele_ids` to be the chromosome name, and position, while the last element is the allele id which are all delimited by tabs. See the element ",
                                i,
                                ": ",
                                chr_pos_allele_ids[i],
                            ),
                        ),
                    )
                end
                chr[i] = ids[1]
                pos[i] = try
                    Int64(parse(Float64, ids[2]))
                catch
                    throw(
                        ArgumentError(
                            string(
                                "We expect the second element of each item in `chr_pos_allele_ids` to be the position (Int64). See the element ",
                                i,
                                ": ",
                                chr_pos_allele_ids[i],
                            ),
                        ),
                    )
                end
                ale[i] = ids[end]
                if verbose
                    ProgressMeter.next!(pb)
                end
            end
            if verbose
                ProgressMeter.finish!(pb)
            end
            sort(string.(chr, "\t", pos, "\t", ale))
        end
        # Extract the loci-allele combination names from the filtered_genomes struct
        chromosomes, positions, alleles = loci_alleles(filtered_genomes)
        available_chr_pos_allele_ids = string.(chromosomes, "\t", positions, "\t", alleles)
        # Find the loci-allele combination indices to retain
        bool_loci_alleles = fill(false, length(available_chr_pos_allele_ids))
        if verbose
            pb = ProgressMeter.Progress(
                length(available_chr_pos_allele_ids),
                desc = "Filtering loci-allele combination names",
            )
        end
        # Multi-threaded indexing (no need for thread locking as we are accessing unique indexes)
        Threads.@threads for i in eachindex(available_chr_pos_allele_ids)
            # i = 1
            bool_loci_alleles[i] = available_chr_pos_allele_ids[i] ∈ requested_chr_pos_allele_ids
            if verbose
                ProgressMeter.next!(pb)
            end
        end
        if verbose
            ProgressMeter.finish!(pb)
        end
        idx_loci_alleles = findall(bool_loci_alleles)
        # @show length(idx_loci_alleles)
        if length(idx_loci_alleles) == 0
            throw(
                ErrorException(
                    string(
                        "No loci retained after filtering using a list of loci-alleles combination names `loci_alleles::Union{Missing,Vector{String}}`",
                        " in addition to filtering by maf = ",
                        maf,
                        ", and maximum locus sparsity = ",
                        max_locus_sparsity,
                        ".",
                    ),
                ),
            )
        end
        slice(filtered_genomes, idx_loci_alleles = idx_loci_alleles, verbose = verbose)
    else
        filtered_genomes
    end
    # @show dimensions(filtered_genomes)
    # Output
    filtered_genomes
end

"""
    merge(
        genomes::Genomes,
        other::Genomes;
        conflict_resolution::Tuple{Float64,Float64} = (0.5, 0.5),
        verbose::Bool = true
    )::Genomes

Merge two Genomes structs by combining their entries and loci_alleles while resolving conflicts in allele frequencies.

# Arguments
- `genomes::Genomes`: First Genomes struct to merge
- `other::Genomes`: Second Genomes struct to merge
- `conflict_resolution::Tuple{Float64,Float64}`: Weights for resolving conflicts between allele frequencies (must sum to 1.0)
- `verbose::Bool`: If true, displays a progress bar during merging

# Returns
- `Genomes`: A new Genomes struct containing the merged data

# Details
The function performs the following operations:
1. Combines unique entries and loci_alleles from both input structs
2. Resolves population conflicts by concatenating conflicting values
3. For overlapping entries and loci:
   - If allele frequencies match, uses the existing value
   - If frequencies differ, applies weighted average using conflict_resolution
   - For missing values, uses available non-missing value
   - Resolves mask conflicts using weighted average

# Examples
```jldoctest; setup = :(using GBCore)
julia> n = 100; l = 5_000; n_alleles = 2;

julia> all = simulategenomes(n=n, l=l, n_alleles=n_alleles, verbose=false);

julia> genomes = slice(all, idx_entries=collect(1:Int(floor(n*0.75))), idx_loci_alleles=collect(1:Int(floor(l*(n_alleles-1)*0.75))));

julia> other = slice(all, idx_entries=collect(Int(floor(n*0.50)):n), idx_loci_alleles=collect(Int(floor(l*(n_alleles-1)*0.50)):l*(n_alleles-1)));

julia> merged_genomes = merge(genomes, other, conflict_resolution=(0.75, 0.25), verbose=false);

julia> size(merged_genomes.allele_frequencies)
(100, 5000)

julia> sum(ismissing.(merged_genomes.allele_frequencies))
123725
```
"""
function Base.merge(
    genomes::Genomes,
    other::Genomes;
    conflict_resolution::Tuple{Float64,Float64} = (0.5, 0.5),
    verbose::Bool = true,
)::Genomes
    # n = 100; l = 5_000; n_alleles = 2;
    # all = simulategenomes(n=n, l=l, n_alleles=n_alleles, sparsity=0.05, seed=123456);
    # genomes = slice(all, idx_entries=collect(1:Int(floor(n*0.75))), idx_loci_alleles=collect(1:Int(floor(l*(n_alleles-1)*0.75))));
    # other = slice(all, idx_entries=collect(Int(floor(n*0.50)):n), idx_loci_alleles=collect(Int(floor(l*(n_alleles-1)*0.50)):l*(n_alleles-1)));
    # conflict_resolution::Tuple{Float64,Float64} = (0.5,0.5); verbose::Bool = true
    # Check arguments
    if !checkdims(genomes) && !checkdims(other)
        throw(ArgumentError("Both Genomes structs are corrupted."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The first Genomes struct is corrupted."))
    end
    if !checkdims(other)
        throw(ArgumentError("The second Genomes struct is corrupted."))
    end
    if (length(conflict_resolution) != 2) && (sum(conflict_resolution) != 1.00)
        throw(ArgumentError("We expect `conflict_resolution` 2 be a 2-item tuple which sums up to exactly 1.00."))
    end
    # Instantiate the merged Genomes struct
    entries::Vector{String} = genomes.entries ∪ other.entries
    populations::Vector{String} = fill("", length(entries))
    loci_alleles::Vector{String} = genomes.loci_alleles ∪ other.loci_alleles
    allele_frequencies::Matrix{Union{Missing,Float64}} = fill(missing, (length(entries), length(loci_alleles)))
    mask::Matrix{Bool} = fill(false, (length(entries), length(loci_alleles)))
    out::Genomes = Genomes(n = length(entries), p = length(loci_alleles))
    # Merge and resolve conflicts in allele frequencies and mask
    if verbose
        pb = ProgressMeter.Progress(length(entries) * length(loci_alleles); desc = "Merging 2 Genomes structs: ")
    end
    idx_entry_1::Vector{Int} = []
    idx_entry_2::Vector{Int} = []
    bool_entry_1::Bool = false
    bool_entry_2::Bool = false
    idx_locus_allele_1::Vector{Int} = []
    idx_locus_allele_2::Vector{Int} = []
    bool_locus_allele_1::Bool = false
    bool_locus_allele_2::Bool = false
    for (i, entry) in enumerate(entries)
        # entry = entries[i]
        idx_entry_1 = findall(genomes.entries .== entry)
        idx_entry_2 = findall(other.entries .== entry)
        # We expect a maximum of 1 match per entry as we checked the Genomes structs
        bool_entry_1 = length(idx_entry_1) > 0
        bool_entry_2 = length(idx_entry_2) > 0
        if bool_entry_1 && bool_entry_2
            if genomes.populations[idx_entry_1[1]] == other.populations[idx_entry_2[1]]
                populations[i] = genomes.populations[idx_entry_1[1]]
            else
                populations[i] = string(
                    "CONFLICT (",
                    genomes.populations[idx_entry_1[1]]...,
                    ", ",
                    other.populations[idx_entry_2[1]]...,
                    ")",
                )
            end
        elseif bool_entry_1
            populations[i] = genomes.populations[idx_entry_1[1]]
        elseif bool_entry_2
            populations[i] = other.populations[idx_entry_2[1]]
        else
            continue # should never happen
        end
        for (j, locus_allele) in enumerate(loci_alleles)
            # locus_allele = loci_alleles[j]
            # We expect 1 locus-allele match as we checked the Genomes structs
            idx_locus_allele_1 = findall(genomes.loci_alleles .== locus_allele)
            idx_locus_allele_2 = findall(other.loci_alleles .== locus_allele)
            bool_locus_allele_1 = length(idx_locus_allele_1) > 0
            bool_locus_allele_2 = length(idx_locus_allele_2) > 0
            if bool_entry_1 && bool_locus_allele_1 && bool_entry_2 && bool_locus_allele_2
                q_1 = genomes.allele_frequencies[idx_entry_1[1], idx_locus_allele_1[1]]
                q_2 = other.allele_frequencies[idx_entry_2[1], idx_locus_allele_2[1]]
                m_1 = genomes.mask[idx_entry_1[1], idx_locus_allele_1[1]]
                m_2 = other.mask[idx_entry_2[1], idx_locus_allele_2[1]]
                if skipmissing(q_1) == skipmissing(q_2)
                    allele_frequencies[i, j] = q_1
                    mask[i, j] = m_1
                else
                    if !ismissing(q_1) && !ismissing(q_2)
                        allele_frequencies[i, j] = sum((q_1, q_2) .* conflict_resolution)
                    elseif !ismissing(q_1)
                        allele_frequencies[i, j] = q_1
                    else
                        allele_frequencies[i, j] = q_2
                    end
                    mask[i, j] = Bool(round(sum((m_1, m_2) .* conflict_resolution)))
                end
            elseif bool_entry_1 && bool_locus_allele_1
                allele_frequencies[i, j] = genomes.allele_frequencies[idx_entry_1[1], idx_locus_allele_1[1]]
                mask[i, j] = genomes.mask[idx_entry_1[1], idx_locus_allele_1[1]]
            elseif bool_entry_2 && bool_locus_allele_2
                allele_frequencies[i, j] = other.allele_frequencies[idx_entry_2[1], idx_locus_allele_2[1]]
                mask[i, j] = other.mask[idx_entry_2[1], idx_locus_allele_2[1]]
            else
                continue
            end
            if verbose
                next!(pb)
            end
        end
    end
    if verbose
        finish!(pb)
    end
    # Output
    out.entries = entries
    out.populations = populations
    out.loci_alleles = loci_alleles
    out.allele_frequencies = allele_frequencies
    out.mask = mask
    if !checkdims(out)
        throw(ErrorException("Error merging the 2 Genomes structs."))
    end
    out
end


"""
    merge(genomes::Genomes, phenomes::Phenomes; keep_all::Bool=true)::Tuple{Genomes,Phenomes}

Merge `Genomes` and `Phenomes` structs based on their entries, combining genomic and phenotypic data.

# Arguments
- `genomes::Genomes`: A struct containing genomic data including entries, populations, and allele frequencies
- `phenomes::Phenomes`: A struct containing phenotypic data including entries, populations, and phenotypes
- `keep_all::Bool=true`: If true, performs a union of entries; if false, performs an intersection

# Returns
- `Tuple{Genomes,Phenomes}`: A tuple containing:
    - A new `Genomes` struct with merged entries and corresponding genomic data
    - A new `Phenomes` struct with merged entries and corresponding phenotypic data

# Details
- Maintains dimensional consistency between input and output structs
- Handles population conflicts by creating a combined population name
- Preserves allele frequencies and phenotypic data for matched entries
- When `keep_all=true`, includes all entries from both structs
- When `keep_all=false`, includes only entries present in both structs

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=10, verbose=false);

julia> trials, effects = simulatetrials(genomes=slice(genomes, idx_entries=collect(1:5), idx_loci_alleles=collect(1:length(genomes.loci_alleles))), f_add_dom_epi=[0.90 0.05 0.05;], n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=2, verbose=false);

julia> phenomes = Phenomes(n=5, t=1);

julia> phenomes.entries = trials.entries[1:5];

julia> phenomes.populations = trials.populations[1:5];

julia> phenomes.traits = trials.traits;

julia> phenomes.phenotypes = trials.phenotypes[1:5, :];

julia> phenomes.mask .= true;

julia> genomes_merged_1, phenomes_merged_1 = merge(genomes, phenomes, keep_all=true);

julia> size(genomes_merged_1.allele_frequencies), size(phenomes_merged_1.phenotypes)
((10, 10000), (10, 1))

julia> genomes_merged_2, phenomes_merged_2 = merge(genomes, phenomes, keep_all=false);

julia> size(genomes_merged_2.allele_frequencies), size(phenomes_merged_2.phenotypes)
((5, 10000), (5, 1))
```
"""
function Base.merge(genomes::Genomes, phenomes::Phenomes; keep_all::Bool = true)::Tuple{Genomes,Phenomes}
    # genomes = simulategenomes(n=10, verbose=false);
    # trials, effects = simulatetrials(genomes=slice(genomes, idx_entries=collect(1:5), idx_loci_alleles=collect(1:length(genomes.loci_alleles))), f_add_dom_epi=[0.90 0.05 0.05;], n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=2, verbose=false);
    # phenomes = analyse(trials, max_levels=20, max_time_per_model=10, verbose=false).phenomes[1]; keep_all::Bool = false
    # Check input arguments
    if !checkdims(genomes) && !checkdims(phenomes)
        throw(ArgumentError("The Genomes and Phenomes structs are corrupted."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted."))
    end
    # Identify the entries to be included
    entries::Vector{String} = []
    if keep_all
        entries = genomes.entries ∪ phenomes.entries
    else
        entries = genomes.entries ∩ phenomes.entries
    end
    # Instantiate the output structs
    out_genomes::Genomes = Genomes(n = length(entries), p = length(genomes.loci_alleles))
    out_phenomes::Phenomes = Phenomes(n = length(entries), t = length(phenomes.traits))
    # Populate the loci, and trait information
    out_genomes.loci_alleles = genomes.loci_alleles
    out_phenomes.traits = phenomes.traits
    # Iterate across entries which guarantees the order of entries in both out_genomes and out_phenomes is the same
    for (i, entry) in enumerate(entries)
        out_genomes.entries[i] = entry
        out_phenomes.entries[i] = entry
        idx_1 = findall(genomes.entries .== entry)
        idx_2 = findall(phenomes.entries .== entry)
        # We expect a maximum of 1 match per entry as we checked the Genomes structs
        bool_1 = length(idx_1) > 0
        bool_2 = length(idx_2) > 0
        if bool_1 && bool_2
            if genomes.populations[idx_1[1]] == phenomes.populations[idx_2[1]]
                out_genomes.populations[i] = out_phenomes.populations[i] = genomes.populations[idx_1[1]]
            else
                out_genomes.populations[i] =
                    out_phenomes.populations[i] = string(
                        "CONFLICT (",
                        genomes.populations[idx_1[1]]...,
                        ", ",
                        phenomes.populations[idx_2[1]]...,
                        ")",
                    )
            end
            out_genomes.allele_frequencies[i, :] = genomes.allele_frequencies[idx_1, :]
            out_genomes.mask[i, :] = genomes.mask[idx_1, :]
            out_phenomes.phenotypes[i, :] = phenomes.phenotypes[idx_2, :]
            out_phenomes.mask[i, :] = phenomes.mask[idx_2, :]
        elseif bool_1
            out_genomes.populations[i] = out_phenomes.populations[i] = genomes.populations[idx_1[1]]
            out_genomes.allele_frequencies[i, :] = genomes.allele_frequencies[idx_1, :]
            out_genomes.mask[i, :] = genomes.mask[idx_1, :]
        elseif bool_2
            out_genomes.populations[i] = out_phenomes.populations[i] = phenomes.populations[idx_2[1]]
            out_phenomes.phenotypes[i, :] = phenomes.phenotypes[idx_2, :]
            out_phenomes.mask[i, :] = phenomes.mask[idx_2, :]
        else
            continue # should never happen
        end
    end
    # Outputs
    if !checkdims(out_genomes) || !checkdims(out_phenomes)
        throw(ErrorException("Error merging Genomes and Phenomes structs"))
    end
    out_genomes, out_phenomes
end
