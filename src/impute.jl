"""
    maskmissing!(genomes::Genomes; verbose::Bool = false)

Update the mask matrix for missing values in the genomes struct.

This function updates the mask matrix in a `Genomes` struct by marking positions where
allele frequencies are not missing. The mask is set to `true` for non-missing values
and `false` for missing values.

# Arguments
- `genomes::Genomes`: A Genomes struct containing genomic data including allele frequencies and mask matrix
- `verbose::Bool=false`: If true, displays a progress bar during computation

# Throws
- `ArgumentError`: If the dimensions in the Genomes struct are inconsistent

# Effects
- Modifies the `mask` field of the input `genomes` struct in-place

# Threads 
- Uses multi-threading for parallel computation across loci
- Uses a thread lock for safe concurrent access to shared memory

# Example
```jldoctest; setup = :(using GBCore, StatsBase)
julia> genomes = simulategenomes(n=10, sparsity=0.3, verbose=false);

julia> round(1.00 - mean(genomes.mask), digits=10)
0.0

julia> maskmissing!(genomes);

julia> round(1.00 - mean(genomes.mask), digits=10)
0.3
```
"""
function maskmissing!(genomes::Genomes; verbose::Bool = false)
    # genomes = simulategenomes(n=10, sparsity=0.3, verbose=true); verbose = true
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    thread_lock::ReentrantLock = ReentrantLock()
    if verbose
        pb = ProgressMeter.Progress(length(genomes.entries), desc = "Setting genomes mask matrix as non-missing values")
    end
    for i in eachindex(genomes.entries)
        Threads.@threads for j in eachindex(genomes.loci_alleles)
            @lock thread_lock genomes.mask[i, j] = !ismissing(genomes.allele_frequencies[i, j])
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
end

"""
    sparsities(genomes::Genomes; verbose::Bool = false) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate sparsity (proportion of missing values) for each entry and locus-allele in the genomes data.

# Arguments
- `genomes::Genomes`: A Genomes struct containing allele frequencies and mask data
- `verbose::Bool`: If true, displays progress bars and histograms during calculation (default: false)

# Returns
A tuple containing:
- First element: Vector of sparsity values per entry (row)
- Second element: Vector of sparsity values per locus-allele (column)

# Details
Sparsity is calculated as 1 - mean(mask), where mask contains 1s for non-missing values
and 0s for missing values. The function processes entries and locus-alleles in parallel
using multiple threads with thread-safe operations through ReentrantLock.

When verbose mode is enabled:
- Displays progress bars during calculation
- Shows histograms of sparsity distributions for both entries and locus-alleles
- Prints descriptive messages about the calculation progress

# Throws
- `ArgumentError`: If the dimensions in the Genomes struct are inconsistent

# Examples
```jldoctest; setup = :(using GBCore, StatsBase)
julia> genomes = simulategenomes(n=10, sparsity=0.3, verbose=false);

julia> sparsity_per_entry, sparsity_per_locus_allele = sparsities(genomes);

julia> round(mean(sparsity_per_entry), digits=10)
0.3

julia> round(mean(sparsity_per_locus_allele), digits=10)
0.3
```
"""
function sparsities(genomes::Genomes; verbose::Bool = false)::Tuple{Vector{Float64},Vector{Float64}}
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Set mask to non-missing data-points
    maskmissing!(genomes)
    n, p = size(genomes.allele_frequencies)
    sparsity_per_entry = Vector{Float64}(undef, n)
    sparsity_per_locus_allele = Vector{Float64}(undef, p)
    thread_lock::ReentrantLock = ReentrantLock()
    if verbose
        pb = ProgressMeter.Progress(n, desc = "Calculating sparsity per entry")
    end
    Threads.@threads for i = 1:n
        @lock thread_lock sparsity_per_entry[i] = 1.00 .- mean(genomes.mask[i, :])
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        println("Sparsity per entry")
        UnicodePlots.histogram(sparsity_per_entry)
    end
    if verbose
        pb = ProgressMeter.Progress(p, desc = "Calculating sparsity per locus-allele")
    end
    Threads.@threads for j = 1:p
        @lock thread_lock sparsity_per_locus_allele[j] = 1.00 .- mean(genomes.mask[:, j])
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        println("Sparsity per locus-allele")
        UnicodePlots.histogram(sparsity_per_locus_allele)
    end
    # Output
    (sparsity_per_entry, sparsity_per_locus_allele)
end

"""
    divideintomockscaffolds(genomes::Genomes; max_n_loci_per_chrom::Int64 = 100_000, verbose::Bool = false)::Vector{String}

Divide genomic loci into mock scaffolds based on a maximum number of loci per chromosome.

# Arguments
- `genomes::Genomes`: A Genomes struct containing genomic data
- `max_n_loci_per_chrom::Int64`: Maximum number of loci per chromosome (default: 100,000)
- `verbose::Bool`: If true, prints additional information during execution (default: false)

# Returns
- `Vector{String}`: A vector containing mock scaffold assignments for each locus

# Description
This function takes a Genomes struct and divides the loci into mock scaffolds based on the 
specified maximum number of loci per chromosome. It creates scaffold names in the format 
"mock_scaffold_X" where X is the scaffold number.

# Throws
- `ArgumentError`: If the Genomes struct dimensions are invalid or corrupted

# Example
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=10, sparsity=0.3, verbose=false);

julia> mock_scaffolds = divideintomockscaffolds(genomes, max_n_loci_per_chrom=100);

julia> sum(mock_scaffolds .== mock_scaffolds[1]) == Int64(length(genomes.loci_alleles) / 100)
true
```
"""
function divideintomockscaffolds(
    genomes::Genomes;
    max_n_loci_per_chrom::Int64 = 100_000,
    verbose::Bool = false,
)::Vector{String}
    # genomes = simulategenomes(n=10, sparsity=0.3, verbose=false); max_n_loci_per_chrom = 100; verbose = true
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Divide the loci into mock scaffolds
    p = length(genomes.loci_alleles)
    n_scaffolds = Int(ceil(p / max_n_loci_per_chrom))
    mock_scalfolds = fill("", p)
    if verbose
        pb = ProgressMeter.Progress(n_scaffolds, desc = "Defining $n_scaffolds mock scaffolds")
    end
    for i = 1:n_scaffolds
        # i = 1
        idx_ini = ((i - 1) * max_n_loci_per_chrom) + 1
        idx_fin = if i == n_scaffolds
            p
        else
            i * max_n_loci_per_chrom
        end
        mock_scalfolds[idx_ini:idx_fin] .= string("mock_scaffold_", i)
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Output
    mock_scalfolds
end

"""
    estimateld(genomes::Genomes; chromosomes::Union{Nothing, Vector{String}} = nothing, verbose::Bool=false)::Vector{Matrix{Float64}}

Calculate linkage disequilibrium (LD) matrices for each chromosome in the given genomic data.

# Arguments
- `genomes::Genomes`: A Genomes struct containing genomic data
- `chromosomes::Union{Nothing, Vector{String}}`: Optional vector of chromosome names to analyze. If nothing, all chromosomes in the data will be used
- `verbose::Bool`: If true, prints progress information during computation

# Returns
- `Vector{Matrix{Float64}}`: A vector of correlation matrices, one for each unique chromosome, containing pairwise LD values between loci

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=10, l=1_000, sparsity=0.3, verbose=false);

julia> LDs_all_chroms = estimateld(genomes);

julia> chrom, pos, allele = loci_alleles(genomes);

julia> mock_scaffolds = divideintomockscaffolds(genomes, max_n_loci_per_chrom=100);

julia> LDs_mock_chroms = estimateld(genomes, chromosomes=mock_scaffolds);

julia> length(LDs_all_chroms) == length(unique(chrom))
true

julia> length(LDs_mock_chroms) == Int(length(chrom) / 100)
true
```
"""
function estimateld(
    genomes::Genomes;
    chromosomes::Union{Nothing,Vector{String}} = nothing,
    verbose::Bool = false,
)::Vector{Matrix{Float64}}
    # genomes = simulategenomes(n=10, sparsity=0.3, verbose=false); chromosomes = nothing; verbose = true
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    chromosomes, chroms_uniq = if isnothing(chromosomes)
        chromosomes = [x[1] for x in split.(genomes.loci_alleles, "\t")]
        chromosomes, sort(unique(chromosomes))
    else
        chromosomes, sort(unique(chromosomes))
    end
    LDs = Vector{Matrix{Float64}}(undef, length(chroms_uniq))
    for (i, chrom) in enumerate(chroms_uniq)
        # i = 3; chrom = chromosomes[i]
        if verbose
            println(string("Chromosome or scaffold: ", chrom, " (", i, " of ", length(LDs), ")"))
        end
        idx_loci_alleles = findall(chromosomes .== chrom)
        if length(idx_loci_alleles) < 2
            continue
        end
        (_loci_alleles, _entries, corr) = distances(
            genomes,
            distance_metrics = ["correlation"],
            idx_loci_alleles = idx_loci_alleles,
            include_loci_alleles = true,
            include_counts = false,
            include_entries = false,
            verbose = verbose,
        )
        LDs[i] = corr["loci_alleles|correlation"]
        # Save the correlation matrix in case the run gets interrupted
        fname_LD_matrix = string("LD_matrix-", hash(genomes), "-", chrom, ".tmp.jld2")
        save(fname_LD_matrix, corr, compress = true)
    end
    # Output
    LDs
end

# TODO: implement AOPTIM with automatic arbitrary classification into mock scaffolds based on data size
function impute(genomes::Genomes; max_n_loci_per_chrom::Int64 = 100_000, verbose::Bool = false)::Genomes
    # genomes = simulategenomes(n=10, sparsity=0.3, verbose=false); max_n_loci_per_chrom = 100; verbose = true
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Instantiate the output Genomes struct
    n, p = size(genomes.allele_frequencies)
    out::Genomes = Genomes(n = n, p = p)
    # Divide the allele frequencies into mock scaffolds if we have more than 100,000 loci per scaffold for at least 1 scaffold
    chromosomes, positions, alleles = loci_alleles(genomes)
    max_m = 1
    for chrom in unique(chromosomes)
        m = sum(chromosomes .== chrom)
        max_m = if max_m < m
            m
        else
            max_m
        end
    end
    chromosomes = if max_m > max_n_loci_per_chrom
        # # Divide the loci into mock scaffolds
        divideintomockscaffolds(genomes, max_n_loci_per_chrom = max_n_loci_per_chrom, verbose = verbose)
    else
        chromosomes
    end
    # Estimate linkage disequilibrium (LD) between loci using Pearson's correlation per chromosome
    LDs::Vector{Matrix{Float64}} = estimateld(genomes, chromosomes = chromosomes, verbose = verbose)
    # TODO: 
    # distance estimation via MAE
    # simulate missing data per locus-allele with at least 1 missing data
    # optimise for minimum loci correlation and maximum pool distance per locus-allele
    # impute missing data per locus-allele
    # Output
    out
end
