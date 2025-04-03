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
```jldoctest; setup = :(using GenomicBreedingCore)
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
        throw(ArgumentError("Genomes struct is corrupted ☹."))
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
```jldoctest; setup = :(using GenomicBreedingCore)
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
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase)
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
    filterbysparsity(
        genomes::Genomes;
        max_entry_sparsity::Float64 = 0.0,
        max_locus_sparsity::Float64 = 0.0,
        max_entry_sparsity_percentile::Float64 = 0.90,
        max_locus_sparsity_percentile::Float64 = 0.50,
        verbose::Bool = false,
    )::Genomes

Filter genomic data by removing entries and loci with high sparsity.

# Arguments
- `genomes::Genomes`: A `Genomes` struct containing the genomic data.
- `max_entry_sparsity::Float64`: The maximum allowable sparsity for entries. Default is 0.0.
- `max_locus_sparsity::Float64`: The maximum allowable sparsity for loci. Default is 0.0.
- `max_entry_sparsity_percentile::Float64`: The percentile threshold for entry sparsity. Default is 0.90.
- `max_locus_sparsity_percentile::Float64`: The percentile threshold for locus sparsity. Default is 0.50.
- `verbose::Bool`: If `true`, prints detailed progress information during the filtering process. Default is `false`.

# Returns
- `Genomes`: A `Genomes` struct with filtered genomic data.

# Details
This function filters genomic data by iteratively removing entries and loci with high sparsity. The function performs the following steps:

1. **Input Validation**: Ensures that the `Genomes` struct is not corrupted and that the sparsity thresholds are within the valid range (0.0 to 1.0). Throws an `ArgumentError` if any argument is out of range.
2. **Calculate Sparsities**: Computes the sparsities of entries and loci in the genomic data.
3. **Initial Check**: Checks if the input `Genomes` struct passes all the filtering thresholds. If so, returns the original `Genomes` struct.
4. **Iterative Filtering**: Iteratively removes the sparsest loci and entries until the maximum allowable sparsity thresholds are met:
   - **Remove Sparsest Loci**: Removes loci with sparsity above the specified percentile threshold.
   - **Remove Sparsest Entries**: Removes entries with sparsity above the specified percentile threshold.
5. **Verbose Output**: If `verbose` is `true`, prints detailed progress information during each iteration.
6. **Final Check**: Ensures that there are remaining entries and loci after filtering. Throws an `ErrorException` if all entries or loci are filtered out.
7. **Output**: Returns the filtered `Genomes` struct.

# Notes
- The function uses percentile thresholds to iteratively remove the sparsest loci and entries.
- The `verbose` option provides additional insights into the filtering process by printing progress information.
- The function ensures that the filtered genomic data retains a minimum number of entries and loci.

# Throws
- `ArgumentError`: If the `Genomes` struct is corrupted.
- `ArgumentError`: If any of the sparsity thresholds are out of range.
- `ErrorException`: If all entries and/or loci are filtered out based on the sparsity thresholds.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, sparsity=0.01, verbose=false);

julia> filtered_genomes = filterbysparsity(genomes);

julia> size(genomes.allele_frequencies)
(100, 3000)

julia> size(filtered_genomes.allele_frequencies)
(92, 1239)
```
"""
function filterbysparsity(
    genomes::Genomes;
    max_entry_sparsity::Float64 = 0.0,
    max_locus_sparsity::Float64 = 0.0,
    max_entry_sparsity_percentile::Float64 = 0.90,
    max_locus_sparsity_percentile::Float64 = 0.50,
    verbose::Bool = false,
)::Genomes
    # genomes = simulategenomes(n=100, l=1_000, n_alleles=4, sparsity=0.25, verbose=true); max_entry_sparsity = 0.0; max_locus_sparsity = 0.0; max_entry_sparsity_percentile = 0.90; max_locus_sparsity_percentile = 0.50; verbose = true;
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted ☹."))
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
    # Calculate sparsities
    entry_sparsities, locus_sparsities = sparsities(genomes)
    # Instantiate the boolean vector for the while-loop to iteratively filter the Genomes struct
    bool::Vector{Bool} = [
        (length(entry_sparsities) > 0) &&
        (length(locus_sparsities) > 0) &&
        (maximum(entry_sparsities) > max_entry_sparsity) &&
        (maximum(locus_sparsities) > max_locus_sparsity),
    ]
    # Return early if the input Genomes struct passed all the filtering thresholds
    if !bool[1]
        if verbose
            println(
                "No filtering by sparsity necessary. All entries and loci passed the filtering thresholds (max_entry_sparsity=$max_entry_sparsity, max_locus_sparsity=$max_locus_sparsity, max_entry_sparsity_percentile=$max_entry_sparsity_percentile, and max_locus_sparsity_percentile=$max_locus_sparsity_percentile.",
            )
            @show dimensions(genomes)
        end
        return genomes
    end
    # Iteratively filter the Genomes struct
    if verbose
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("Iteration 0: ")
        @show dimensions(genomes)
        println("Maximum entry sparsity = $(maximum(entry_sparsities))")
        println("Maximum locus sparsity = $(maximum(locus_sparsities))")
    end
    iter = 0
    data_size = collect(size(genomes.allele_frequencies))
    while bool[1]
        iter += 1
        # Start every iteration by removing the sparsest loci
        genomes = begin
            m = maximum([max_locus_sparsity, quantile(locus_sparsities, max_locus_sparsity_percentile)])
            slice(genomes, idx_loci_alleles = findall(locus_sparsities .<= m))
        end
        entry_sparsities, locus_sparsities = sparsities(genomes)
        bool[1] =
            (length(entry_sparsities) > 0) &&
            (length(locus_sparsities) > 0) &&
            (maximum(entry_sparsities) > max_entry_sparsity) &&
            (maximum(locus_sparsities) > max_locus_sparsity)
        if !bool[1]
            break
        end
        # Finish each iteration by removing the sparsest entries
        genomes = begin
            m = maximum([max_entry_sparsity, quantile(entry_sparsities, max_entry_sparsity_percentile)])
            slice(genomes, idx_entries = findall(entry_sparsities .<= m))
        end
        entry_sparsities, locus_sparsities = sparsities(genomes)
        bool[1] =
            (length(entry_sparsities) > 0) &&
            (length(locus_sparsities) > 0) &&
            (maximum(entry_sparsities) > max_entry_sparsity) &&
            (maximum(locus_sparsities) > max_locus_sparsity)
        if !bool[1]
            break
        end
        if data_size == collect(size(genomes.allele_frequencies))
            if verbose
                n, p = data_size
                println(
                    "Dimensions fixed at $(n) entries and $(p) loci-alleles for the filtering thresholds (max_entry_sparsity=$max_entry_sparsity, max_locus_sparsity=$max_locus_sparsity, max_entry_sparsity_percentile=$max_entry_sparsity_percentile, and max_locus_sparsity_percentile=$max_locus_sparsity_percentile.",
                )
            end
            break
        else
            data_size[1] = size(genomes.allele_frequencies, 1)
            data_size[2] = size(genomes.allele_frequencies, 2)
        end
        if verbose
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("Iteration $iter: ")
            @show dimensions(genomes)
            println("Maximum entry sparsity = $(maximum(entry_sparsities))")
            println("Maximum locus sparsity = $(maximum(locus_sparsities))")
        end
    end
    if verbose
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        println("Finished: ")
        @show dimensions(genomes)
        println("Maximum entry sparsity = $(maximum(entry_sparsities))")
        println("Maximum locus sparsity = $(maximum(locus_sparsities))")
    end
    # Check if we are retaining any entries and loci
    if (length(genomes.entries) == 0) || (length(genomes.loci_alleles) == 0)
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
    # Output
    genomes
end

"""
    filterbymaf(
        genomes::Genomes;
        maf::Float64 = 0.01,
        verbose::Bool = false,
    )::Genomes

Filter genomic data by removing loci with minor allele frequencies (MAF) below a specified threshold.

# Arguments
- `genomes::Genomes`: A `Genomes` struct containing the genomic data.
- `maf::Float64`: The minimum allele frequency threshold. Default is 0.01.
- `verbose::Bool`: If `true`, prints detailed progress information during the filtering process. Default is `false`.

# Returns
- `Genomes`: A `Genomes` struct with filtered genomic data.

# Details
This function filters genomic data by removing loci with minor allele frequencies (MAF) below a specified threshold. The function performs the following steps:

1. **Input Validation**: Ensures that the `Genomes` struct is not corrupted and that the `maf` argument is within the valid range (0.0 to 1.0). Throws an `ArgumentError` if any argument is out of range.
2. **Early Return for maf = 0.0**: If `maf` is set to 0.0, the function returns the original `Genomes` struct without filtering.
3. **Calculate MAF**: Computes the mean allele frequency for each locus, skipping missing values.
4. **Filter Loci**: Identifies loci that pass the MAF threshold and retains only those loci:
   - If all loci pass the MAF threshold, the function returns the original `Genomes` struct.
   - If no loci pass the MAF threshold, the function throws an `ErrorException`.
5. **Verbose Output**: If `verbose` is `true`, prints detailed progress information during the filtering process.
6. **Output**: Returns the filtered `Genomes` struct.

# Notes
- The function uses multi-threading to compute the mean allele frequencies for each locus, improving performance on large datasets.
- The `verbose` option provides additional insights into the filtering process by printing progress information.
- The function ensures that the filtered genomic data retains a minimum number of loci.

# Throws
- `ArgumentError`: If the `Genomes` struct is corrupted.
- `ArgumentError`: If the `maf` argument is out of range.
- `ErrorException`: If all loci are filtered out based on the MAF threshold.

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> filtered_genomes = filterbymaf(genomes, maf=0.05);

julia> length(genomes.loci_alleles) >= length(filtered_genomes.loci_alleles)
true
```
"""
function filterbymaf(genomes::Genomes; maf::Float64 = 0.01, verbose::Bool = false)::Genomes
    # genomes = simulategenomes(n=100, l=1_000, n_alleles=4, sparsity=0.25, verbose=true); maf = 0.01; verbose = true;
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted ☹."))
    end
    if (maf < 0.0) || (maf > 1.0)
        throw(ArgumentError("We accept `maf` from 0.0 to 1.0."))
    end
    # Return early if maf = 0.0
    if maf == 0.0
        if verbose
            println("No filtering necessary. Minimum allele frequency = $maf.")
        end
        return genomes
    end
    # Instantiate indices of loci passing the maf threshold
    p = length(genomes.loci_alleles)
    bool_loci_alleles::Vector{Bool} = fill(false, p)
    if verbose
        println(string("Filtering genomes by maf=", maf))
    end
    Threads.@threads for j in eachindex(bool_loci_alleles)
        q = mean(skipmissing(genomes.allele_frequencies[:, j]))
        bool_loci_alleles[j] = (q >= maf) && (q <= (1.0 - maf))
    end
    # Return early if we are not filtering any loci
    if sum(bool_loci_alleles) == p
        if verbose
            println("No filtering by minimum allele frequency necessary. All loci passed the maf threshold ($maf).")
        end
        return genomes
    end
    # Check if we are retaining any entries and loci
    if sum(bool_loci_alleles) == 0
        throw(ErrorException(string("All loci filtered out at minimum allele frequencies (maf) = ", maf, ".")))
    end
    slice(genomes, idx_loci_alleles = findall(bool_loci_alleles), verbose = verbose)
end

"""
    filterbypca(
        genomes::Genomes;
        max_prop_pc_varexp::Float64 = 0.9,
        verbose::Bool = false,
    )::Genomes

Filter genomic data by removing outlier loci-alleles based on principal component analysis (PCA).

# Arguments
- `genomes::Genomes`: A `Genomes` struct containing the genomic data.
- `max_prop_pc_varexp::Float64`: The maximum proportion of variance explained by the first two principal components (PC1 and PC2). Default is 0.9.
- `verbose::Bool`: If `true`, prints detailed progress information during the filtering process. Default is `false`.

# Returns
- `Genomes`: A `Genomes` struct with filtered genomic data.

# Details
This function filters genomic data by removing outlier loci-alleles based on principal component analysis (PCA). The function performs the following steps:

1. **Input Validation**: Ensures that the `Genomes` struct is not corrupted and that the `max_prop_pc_varexp` argument is within the valid range (0.0 to Inf). Throws an `ArgumentError` if any argument is out of range.
2. **Early Return for max_prop_pc_varexp = Inf**: If `max_prop_pc_varexp` is set to Inf, the function returns the original `Genomes` struct without filtering.
3. **Extract Non-Missing Loci-Alleles**: Identifies loci-alleles that are non-missing and have non-zero variance across all entries.
4. **Standardize Allele Frequencies**: Standardizes the allele frequencies per locus-allele in preparation for PCA.
5. **Perform PCA**: Conducts PCA on the standardized allele frequencies and calculates the proportion of variance explained by the first two principal components (PC1 and PC2).
6. **Identify Outliers**: Identifies loci-alleles that are outliers based on the specified proportion of variance explained by PC1 and PC2.
7. **Filter Outliers**: Removes the identified outlier loci-alleles from the genomic data.
8. **Verbose Output**: If `verbose` is `true`, prints detailed progress information during the filtering process.
9. **Output**: Returns the filtered `Genomes` struct.

# Notes
- The function uses PCA to identify outlier loci-alleles based on the proportion of variance explained by the first two principal components.
- The `verbose` option provides additional insights into the filtering process by printing progress information.
- The function ensures that the filtered genomic data retains a minimum number of loci-alleles.

# Throws
- `ArgumentError`: If the `Genomes` struct is corrupted.
- `ArgumentError`: If the `max_prop_pc_varexp` argument is out of range.
- `ErrorException`: If there are less than 10 loci-alleles left after removing fixed and missing values across all entries.

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> filtered_genomes = filterbypca(genomes, max_prop_pc_varexp=0.9);

julia> length(filtered_genomes.loci_alleles) <= length(genomes.loci_alleles)
true
```
"""
function filterbypca(genomes::Genomes; max_prop_pc_varexp::Float64 = 0.9, verbose::Bool = false)::Genomes
    # genomes = simulategenomes(n=100, l=1_000, n_alleles=4, sparsity=0.25, verbose=true); max_prop_pc_varexp = 0.90; verbose = true;
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted ☹."))
    end
    if max_prop_pc_varexp < 0.0
        throw(ArgumentError("We accept `max_prop_pc_varexp` from 0.0 to Inf."))
    end
    # Return early if max_prop_pc_varexp = Inf
    if isinf(max_prop_pc_varexp)
        if verbose
            println(
                "No filtering by PCA necessary. Maximum proportion of variance explained by PC1 and PC2 = $max_prop_pc_varexp.",
            )
        end
        return genomes
    end
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
    n, p = size(genomes.allele_frequencies)
    μ = mean(genomes.allele_frequencies, dims = 1)
    σ = std(genomes.allele_frequencies, dims = 1)
    idx = findall(.!ismissing.(μ[1, :]) .&& (σ[1, :] .> 0.0))
    if length(idx) < 10
        throw(
            ErrorException(
                "There are less than 10 loci-alleles left after removing fixed and missing values across all entries. We cannot proceed with filtering-out outlying loci-alleles. Please consider setting `max_prop_pc_varexp = Inf` to skip this filtering step.",
            ),
        )
    end
    G = genomes.allele_frequencies[:, idx]
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
            println(
                string(
                    "No filtering necessary. All loci passed the PCA filtering. Variance explained by PC1 and PC2 = ",
                    round(variance_explained_pc1_pc2 * 100),
                    "% (with max pc1 = ",
                    maximum(pc1),
                    ", max pc2 = ",
                    maximum(pc2),
                    " and max_prop_pc_varexp set to ",
                    max_prop_pc_varexp,
                    ").",
                ),
            )
        end
        return genomes
    end
    loci_alleles_outliers = genomes.loci_alleles[idx[bool_outliers]]
    if verbose
        println(string("Removing ", length(loci_alleles_outliers), " loci-alleles."))
    end
    idx_loci_alleles = findall([!(x ∈ loci_alleles_outliers) for x in genomes.loci_alleles])
    slice(genomes, idx_loci_alleles = idx_loci_alleles, verbose = verbose)
end

"""
    filterbysnplist(
        genomes::Genomes;
        chr_pos_allele_ids::Union{Nothing,Vector{String}} = nothing,
        verbose::Bool = false,
    )::Genomes

Filter genomic data by retaining only the specified loci-allele combinations.

# Arguments
- `genomes::Genomes`: A `Genomes` struct containing the genomic data.
- `chr_pos_allele_ids::Union{Nothing, Vector{String}}`: A vector of loci-allele combination names in the format "chromosome\tposition\tallele". If `nothing`, no filtering is applied. Default is `nothing`.
- `verbose::Bool`: If `true`, prints detailed progress information during the filtering process. Default is `false`.

# Returns
- `Genomes`: A `Genomes` struct with filtered genomic data.

# Details
This function filters genomic data by retaining only the specified loci-allele combinations. The function performs the following steps:

1. **Input Validation**: Ensures that the `Genomes` struct is not corrupted. Throws an `ArgumentError` if the struct is corrupted.
2. **Early Return for No Loci-Allele Combinations**: If no loci-allele combination names are provided, the function returns the original `Genomes` struct without filtering.
3. **Parse Loci-Allele Combination Names**: Parses the input loci-allele combination names and ensures they are valid.
4. **Extract Available Loci-Allele Combinations**: Extracts the loci-allele combination names from the `Genomes` struct.
5. **Filter Loci-Alleles**: Identifies the indices of the loci-allele combinations to retain based on the provided list.
6. **Verbose Output**: If `verbose` is `true`, prints detailed progress information during the filtering process.
7. **Final Check**: Ensures that there are remaining loci-alleles after filtering. Throws an `ErrorException` if no loci-alleles are retained.
8. **Output**: Returns the filtered `Genomes` struct.

# Notes
- The function uses multi-threading to parse and filter the loci-allele combination names, improving performance on large datasets.
- The `verbose` option provides additional insights into the filtering process by printing progress information.
- The function ensures that the filtered genomic data retains a minimum number of loci-alleles.

# Throws
- `ArgumentError`: If the `Genomes` struct is corrupted.
- `ArgumentError`: If the loci-allele combination names are not in the expected format.
- `ErrorException`: If no loci-alleles are retained after filtering.

# Example
```jldoctest; setup = :(using GenomicBreedingCore, StatsBase)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> chr_pos_allele_ids = sample(genomes.loci_alleles, 100, replace=false); sort!(chr_pos_allele_ids);

julia> filtered_genomes = filterbysnplist(genomes, chr_pos_allele_ids=chr_pos_allele_ids);

julia> size(filtered_genomes.allele_frequencies)
(100, 100)
```
"""
function filterbysnplist(
    genomes::Genomes;
    chr_pos_allele_ids::Union{Nothing,Vector{String}} = nothing,
    verbose::Bool = false,
)::Genomes
    # genomes = simulategenomes(n_populations=3, sparsity=0.25, seed=123456); chr_pos_allele_ids = sample(genomes.loci_alleles, Int(floor(0.5*length(genomes.loci_alleles)))); sort!(chr_pos_allele_ids); verbose = true
    # Check arguments
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted ☹."))
    end
    # Return early if no loci-allele combination names are provided
    if isnothing(chr_pos_allele_ids) || (length(chr_pos_allele_ids) == 0)
        if verbose
            println("No filtering by SNP list necessary. No loci-allele combination names provided.")
        end
        return genomes
    end
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
    # Extract the loci-allele combination names from the genomes struct
    chromosomes, positions, alleles = loci_alleles(genomes)
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
    # Check if we are retaining any loci-allele combinations
    if sum(bool_loci_alleles) == 0
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
    # Return early if all loci-allele combination names belong to the SNP list
    if sum(bool_loci_alleles) == length(genomes.loci_alleles)
        if verbose
            println("No filtering by SNP list necessary. All loci-allele combination names belong to the SNP list.")
        end
        return genomes
    end
    # Output sliced genomes struct
    slice(genomes, idx_loci_alleles = findall(bool_loci_alleles), verbose = verbose)
end

"""
    filter(
        genomes::Genomes,
        maf::Float64;
        max_entry_sparsity::Float64 = 0.0,
        max_locus_sparsity::Float64 = 0.0,
        max_prop_pc_varexp::Float64 = 0.90,
        max_entry_sparsity_percentile::Float64 = 0.90,
        max_locus_sparsity_percentile::Float64 = 0.50,
        chr_pos_allele_ids::Union{Nothing,Vector{String}} = nothing,
        verbose::Bool = false,
    )::Tuple{Genomes, Dict{String,Vector{String}}}

Filter genomic data based on multiple criteria including sparsity, minor allele frequency (MAF), principal component analysis (PCA), and a list of loci-allele combinations.

# Arguments
- `genomes::Genomes`: A `Genomes` struct containing the genomic data.
- `maf::Float64`: The minimum allele frequency threshold.
- `max_entry_sparsity::Float64`: The maximum allowable sparsity for entries. Default is 0.0.
- `max_locus_sparsity::Float64`: The maximum allowable sparsity for loci. Default is 0.0.
- `max_prop_pc_varexp::Float64`: The maximum proportion of variance explained by the first two principal components (PC1 and PC2). Default is 0.90.
- `max_entry_sparsity_percentile::Float64`: The percentile threshold for entry sparsity. Default is 0.90.
- `max_locus_sparsity_percentile::Float64`: The percentile threshold for locus sparsity. Default is 0.50.
- `chr_pos_allele_ids::Union{Nothing, Vector{String}}`: A vector of loci-allele combination names in the format "chromosome\tposition\tallele". If `nothing`, no filtering is applied. Default is `nothing`.
- `verbose::Bool`: If `true`, prints detailed progress information during the filtering process. Default is `false`.

# Returns
- `Tuple{Genomes, Dict{String, Vector{String}}}`: A tuple containing:
  - A `Genomes` struct with filtered genomic data.
  - A dictionary of omitted loci-allele names categorized by the filtering criteria.

# DetailS
This function filters genomic data based on multiple criteria including sparsity, minor allele frequency (MAF), principal component analysis (PCA), and a list of loci-allele combinations. The function performs the following steps:

1. **Input Validation**: Ensures that the `Genomes` struct is not corrupted and that the filtering thresholds are within valid ranges. Throws an `ArgumentError` if any argument is out of range.
2. **Filter by Sparsity**: Removes the sparsest entries and loci-alleles until the maximum allowable sparsity thresholds are met.
3. **Filter by MAF**: Removes loci-alleles with minor allele frequencies below the specified threshold.
4. **Filter by PCA**: Removes outlier loci-alleles based on the proportion of variance explained by the first two principal components.
5. **Filter by SNP List**: Retains only the specified loci-allele combinations.
6. **Verbose Output**: If `verbose` is `true`, prints detailed progress information during each filtering step.
7. **Output**: Returns the filtered `Genomes` struct and a dictionary of omitted loci-allele names categorized by the filtering criteria.

# Notes
- The function uses multi-threading to improve performance on large datasets.
- The `verbose` option provides additional insights into the filtering process by printing progress information.
- The function ensures that the filtered genomic data retains a minimum number of entries and loci-alleles.

# Throws
- `ArgumentError`: If the `Genomes` struct is corrupted or any of the filtering thresholds are out of range.
- `ErrorException`: If no loci-alleles are retained after filtering.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, sparsity=0.01, verbose=false);

julia> filtered_genomes_1, omitted_loci_alleles_1 = filter(genomes, 0.1);

julia> filtered_genomes_2, omitted_loci_alleles_2 = filter(genomes, 0.1, chr_pos_allele_ids=genomes.loci_alleles[1:1000]);

julia> size(genomes.allele_frequencies)
(100, 3000)

julia> prod(size(filtered_genomes_1.allele_frequencies)) < prod(size(genomes.allele_frequencies))
true

julia> prod(size(filtered_genomes_2.allele_frequencies)) < prod(size(filtered_genomes_1.allele_frequencies))
true
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
)::Tuple{Genomes,Dict{String,Vector{String}}}
    # genomes = simulategenomes(n_populations=3, sparsity=0.01, seed=123456); maf=0.01; max_entry_sparsity=0.1; max_locus_sparsity = 0.25; max_prop_pc_varexp = 1.0; max_entry_sparsity_percentile = 0.9; max_locus_sparsity_percentile = 0.5
    # chr_pos_allele_ids = sample(genomes.loci_alleles, Int(floor(0.5*length(genomes.loci_alleles)))); sort!(chr_pos_allele_ids); verbose = true
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted ☹."))
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
    if (maf < 0.0) || (maf > 1.0)
        throw(ArgumentError("We accept `maf` from 0.0 to 1.0."))
    end
    if max_prop_pc_varexp < 0.0
        throw(ArgumentError("We accept `max_prop_pc_varexp` from 0.0 to Inf."))
    end
    # Instantiate dictionary of omitted loci-alleles names
    omitted_loci_alleles::Dict{String,Vector{String}} =
        Dict("by_sparsity" => [], "by_maf" => [], "by_pca" => [], "by_snplist" => [])
    # Remove sparsest entries and then sparsest loci-alleles until no data remains or max_entry_sparsity and max_locus_sparsity are satisfied
    filtered_genomes = filterbysparsity(
        genomes,
        max_entry_sparsity = max_entry_sparsity,
        max_locus_sparsity = max_locus_sparsity,
        max_entry_sparsity_percentile = max_entry_sparsity_percentile,
        max_locus_sparsity_percentile = max_locus_sparsity_percentile,
        verbose = verbose,
    )
    omitted_loci_alleles["by_sparsity"] = setdiff(genomes.loci_alleles, filtered_genomes.loci_alleles)
    # Filter by minimum allele frequency (maf)
    filtered_genomes = filterbymaf(filtered_genomes, maf = maf, verbose = verbose)
    omitted_loci_alleles["by_maf"] = setdiff(
        setdiff(genomes.loci_alleles, filtered_genomes.loci_alleles),
        reduce(vcat, values(omitted_loci_alleles)),
    )
    # Are we filtering out outlying loci-alleles?
    filtered_genomes = filterbypca(filtered_genomes, max_prop_pc_varexp = max_prop_pc_varexp, verbose = verbose)
    omitted_loci_alleles["by_pca"] = setdiff(
        setdiff(genomes.loci_alleles, filtered_genomes.loci_alleles),
        reduce(vcat, values(omitted_loci_alleles)),
    )
    # Are we filtering using a list of loci-allele combination names?
    filtered_genomes = filterbysnplist(filtered_genomes, chr_pos_allele_ids = chr_pos_allele_ids, verbose = verbose)
    omitted_loci_alleles["by_snplist"] = setdiff(
        setdiff(genomes.loci_alleles, filtered_genomes.loci_alleles),
        reduce(vcat, values(omitted_loci_alleles)),
    )
    # Output
    (filtered_genomes, omitted_loci_alleles)
end
