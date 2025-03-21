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
