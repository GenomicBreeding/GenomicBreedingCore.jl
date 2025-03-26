"""
    simulatechromstruct(;l::Int64, n_chroms::Int64, max_pos::Int64)

Generate chromosome structure parameters for genome simulation.

# Arguments
- `l::Int64`: Total number of loci to distribute across chromosomes (2 to 1e9)
- `n_chroms::Int64`: Number of chromosomes (1 to 1e6) 
- `max_pos::Int64`: Maximum genome size in base pairs (10 to 160e9)

# Returns
A tuple containing:
- `chrom_lengths::Vector{Int64}`: Vector of chromosome lengths in base pairs
- `chrom_loci_counts::Vector{Int64}`: Vector of loci counts per chromosome

# Examples
```jldoctest; setup = :(using GBCore)
julia> chrom_lengths, chrom_loci_counts = simulatechromstruct(l=10_000, n_chroms=7, max_pos=135_000_000)
([19285714, 19285714, 19285714, 19285714, 19285714, 19285714, 19285716], [1428, 1428, 1428, 1428, 1428, 1428, 1432])
```
"""
function simulatechromstruct(;l::Int64, n_chroms::Int64, max_pos::Int64)::Tuple{Vector{Int64}, Vector{Int64}}
    # Check Arguments
    if (l < 2) || (l > 1e9)
        throw(ArgumentError("We accept `l` from 2 to 1 billion."))
    end
    if (n_chroms < 1) || (n_chroms > 1e6)
        throw(ArgumentError("We accept `n_chroms` from 1 to 1 million."))
    end
    if (max_pos < 10) || (max_pos > 160e9)
        throw(ArgumentError("We accept `max_pos` from 10 to 160 billion (genome of *Tmesipteris oblanceolata*)."))
    end
    # Define maximum chromosome length (l1) and maximum number of loci per chromosome (l2)
    l1::Int64 = Int64(floor(max_pos / n_chroms))
    l2::Int64 = Int64(floor(l / n_chroms))
    chrom_lengths::Vector{Int64} = [
        if i < n_chroms
            l1
        elseif l1 * n_chroms < max_pos
            l1 + (max_pos - l1 * n_chroms)
        else
            l1
        end for i = 1:n_chroms
    ]
    chrom_loci_counts::Vector{Int64} = [
        if i < n_chroms
            l2
        elseif l2 * n_chroms < l
            l2 + (l - l2 * n_chroms)
        else
            l2
        end for i = 1:n_chroms
    ]
    # Output
    (
        chrom_lengths,
        chrom_loci_counts,
    )
end

"""
    simulateposandalleles(;
        chrom_lengths::Vector{Int64}, 
        chrom_loci_counts::Vector{Int64},
        n_alleles::Int64, 
        allele_choices::Vector{String} = ["A", "T", "C", "G", "D"],
        allele_weights::Weights{Float64,Float64,Vector{Float64}} = StatsBase.Weights([1.0, 1.0, 1.0, 1.0, 0.1] / sum([1.0, 1.0, 1.0, 1.0, 0.1])),
        rng::TaskLocalRNG = Random.GLOBAL_RNG
    ) -> Tuple{Vector{Vector{Int64}}, Vector{String}}

Simulates genomic positions and alleles for multiple chromosomes.

# Arguments
- `chrom_lengths::Vector{Int64}`: Vector containing the length of each chromosome
- `chrom_loci_counts::Vector{Int64}`: Vector containing the number of loci to generate for each chromosome
- `n_alleles::Int64`: Number of alleles to generate per locus
- `allele_choices::Vector{String}`: Vector of possible alleles to choose from (default: ["A", "T", "C", "G", "D"])
- `allele_weights::Weights`: Weights for sampling alleles (default: normalized weights favoring A,T,C,G over D)
- `rng::TaskLocalRNG`: Random number generator for reproducibility (default: global RNG)

# Returns
- `Tuple{Vector{Vector{Int64}}, Vector{String}}`: A tuple containing:
  1. A vector of vectors, where each inner vector contains the positions for a chromosome
  2. A vector of strings, where each string contains tab-separated locus information in the format:
     "chrom_N\\tposition\\tall_alleles\\tchosen_allele"

# Throws
- `ArgumentError`: If input argument lengths don't match or if invalid number of alleles is requested

# Example
```jldoctest; setup = :(using GBCore)
julia> chrom_lengths, chrom_loci_counts = simulatechromstruct(10_000, 7, 135_000_000);

julia> positions, loci_alleles = simulateposandalleles(chrom_lengths=chrom_lengths, chrom_loci_counts=chrom_loci_counts, n_alleles=2);

julia> length(positions) == length(chrom_lengths)
true

julia> length(loci_alleles) == sum(chrom_loci_counts)
true
```
"""
function simulateposandalleles(;
    chrom_lengths::Vector{Int64}, 
    chrom_loci_counts::Vector{Int64},
    n_alleles::Int64, 
    allele_choices::Vector{String} = ["A", "T", "C", "G", "D"],
    allele_weights::Weights{Float64,Float64,Vector{Float64}} = StatsBase.Weights([1.0, 1.0, 1.0, 1.0, 0.1] / sum([1.0, 1.0, 1.0, 1.0, 0.1])),
    rng::TaskLocalRNG = Random.GLOBAL_RNG,
)::Tuple{Vector{Vector{Int64}}, Vector{String}}
    # Check Arguments
    if length(chrom_lengths) != length(chrom_loci_counts)
        throw(ArgumentError("The length of `chrom_lengths` should be equal to the length of `chrom_loci_counts`."))
    end
    if length(allele_choices) != length(allele_weights)
        throw(ArgumentError("The length of `allele_choices` should be equal to the length of `allele_weights`."))
    end
    if length(allele_choices) < 2
        throw(ArgumentError("We expect at least 2 alleles in `allele_choices` and `allele_weights`."))
    end
    if (n_alleles < 2) || (n_alleles > length(allele_choices))
        throw(ArgumentError("We accept `n` from 2 to $(length(allele_choices)), which can be $(join(allele_choices, ", "))."))
    end
    # Define the positions per chromosome and the loci-alleles combinations across chromosomes
    n_chroms = length(chrom_lengths)
    p = sum(chrom_loci_counts) * (n_alleles - 1)
    positions::Vector{Vector{Int64}} = fill(Int64[], n_chroms)
    loci_alleles::Vector{String} = Vector{String}(undef, p)
    locus_counter::Int64 = 1
    for i = 1:n_chroms
        positions[i] = StatsBase.sample(rng, 1:chrom_lengths[i], chrom_loci_counts[i]; replace = false, ordered = true)
        for pos in positions[i]
            all_alleles::Vector{String} =
                StatsBase.sample(rng, allele_choices, allele_weights, n_alleles; replace = false, ordered = false)
            alleles::Vector{String} =
                StatsBase.sample(rng, all_alleles, n_alleles - 1; replace = false, ordered = false)
            for j in eachindex(alleles)
                loci_alleles[locus_counter] = join([string("chrom_", i), pos, join(all_alleles, "|"), alleles[j]], "\t")
                locus_counter += 1
            end
        end
    end
    # Output
    (
        positions,
        loci_alleles,
    )
end

"""
    simulatepopgroups(; n::Int64, n_populations::Int64)::Tuple{Vector{String}, Vector{Vector{Int64}}}

Simulate population groups by dividing a total number of samples into populations.

# Arguments
- `n::Int64`: Total number of samples (between 1 and 1 billion)
- `n_populations::Int64`: Number of populations to create (between 1 and n)

# Returns
A tuple containing:
- `Vector{String}`: Vector of population labels for each sample
- `Vector{Vector{Int64}}`: Vector of vectors containing indices for each population group

# Example
```jldoctest; setup = :(using GBCore)
julia> populations, idx_population_groupings = simulatepopgroups(n=100, n_populations=3);

julia> length(populations) == 100
true

julia> length(idx_population_groupings) == 3
true

julia> pops = fill("", 100); [pops[x] .= unique(populations)[i] for (i, x) in enumerate([x for x in (idx_population_groupings)])];

julia> pops == populations
true
```
"""
function simulatepopgroups(;
    n::Int64,
    n_populations::Int64,
)::Tuple{Vector{String}, Vector{Vector{Int64}}}
    # Check Arguments
    if (n < 1) || (n > 1e9)
        throw(ArgumentError("We accept `n` from 1 to 1 billion."))
    end
    if (n_populations < 1) || (n_populations > n)
        throw(ArgumentError("We accept `n_populations` from 1 to `n`."))
    end
    # Define population sizes
    population_sizes::Array{Int64} = [Int64(floor(n / n_populations)) for i = 1:n_populations]
    sum(population_sizes) < n ? population_sizes[end] += n - sum(population_sizes) : nothing
    # Set populations and the indices of entries per population
    populations::Array{String} = []
    idx_population_groupings::Vector{Vector{Int64}} = []
    for i = 1:n_populations
        append!(
            populations,
            [string("pop_", lpad(i, length(string(n_populations)), "0")) for _ = 1:population_sizes[i]],
        )
        if i == 1
            push!(idx_population_groupings, collect(1:population_sizes[i]))
        else
            push!(idx_population_groupings, collect((cumsum(population_sizes)[i-1]+1):cumsum(population_sizes)[i]))
        end
    end
    # Output
    (
        populations,
        idx_population_groupings,
    )
end

"""
    simulateldblocks(;
        chrom_positions::Vector{Int64}, 
        chrom_length::Int64, 
        ld_corr_50perc_kb::Int64,
        rel_dist_multiplier::Float64 = 2.0
    )::SparseMatrixCSC{Float64}

Simulate linkage disequilibrium (LD) blocks by generating a sparse variance-covariance matrix.

# Arguments
- `chrom_positions::Vector{Int64}`: Vector of positions on the chromosome for each locus
- `chrom_length::Int64`: Total length of the chromosome
- `ld_corr_50perc_kb::Int64`: Distance in kilobases at which the LD correlation decays to 50%
- `rel_dist_multiplier::Float64`: Multiplier for maximum relative distance to consider (default: 2.0)

# Returns
- `SparseMatrixCSC{Float64}`: A sparse variance-covariance matrix representing LD blocks

# Details
The function creates a variance-covariance matrix where the correlation between loci decays 
exponentially with distance. The decay rate is calculated to achieve 50% correlation at the 
specified distance (`ld_corr_50perc_kb`). 

For computational efficiency, correlations between loci are set to zero if:
1. The normalized distance is greater than rel_dist_multiplier * q, where q is the normalized 
   LD decay distance (ld_corr_50perc_kb / chrom_length)
2. The normalized distance is greater than 0.9 (90% of chromosome length)

The computation uses multi-threading with `Threads.@threads` to parallelize the calculation
of correlation values across loci positions.

# Throws
- `ArgumentError`: If number of loci exceeds chromosome length
- `ArgumentError`: If LD correlation distance exceeds chromosome length
- `ArgumentError`: If rel_dist_multiplier is less than 1.0

# Example
```jldoctest; setup = :(using GBCore)
julia> chrom_lengths, chrom_loci_counts = simulatechromstruct(10_000, 7, 135_000_000);

julia> positions, loci_alleles = simulateposandalleles(chrom_lengths=chrom_lengths, chrom_loci_counts=chrom_loci_counts, n_alleles=2);

julia> Σ = simulateldblocks(chrom_positions=positions[1], chrom_length=chrom_lengths[1], ld_corr_50perc_kb=1_000);

julia> size(Σ)
(1428, 1428)
```
"""
function simulateldblocks(;
    chrom_positions::Vector{Int64}, 
    chrom_length::Int64, 
    ld_corr_50perc_kb::Int64,
    rel_dist_multiplier::Float64 = 2.0,
)::SparseMatrixCSC{Float64}
    # Check Arguments
    n_loci = length(chrom_positions)
    if (n_loci > chrom_length)
        throw(
            ArgumentError(
                "The parameter `n_loci` ($n_loci) should be less than or equal to `chrom_length` ($chrom_length).",
            ),
        )
    end
    if ld_corr_50perc_kb > chrom_length
        throw(
            ArgumentError(
                "The parameter `ld_corr_50perc_kb` ($ld_corr_50perc_kb) should be less than or equal to `chrom_length` ($chrom_length).",
            ),
        )
    end
    if rel_dist_multiplier < 1.0
        throw(ArgumentError("The parameter `rel_dist_multiplier` should be greater than or equal to 1.0."))
    end
    # Define the LD decay with the variance-covariance matrix of a multivariate distribution from which allele frequencies will be sample later
    # Using a sparse matrix for computational efficiency as we will set covariances with relative distances greater than 0.5 to zero
    S::Matrix{Float64} = spzeros(n_loci, n_loci)
    q = (ld_corr_50perc_kb * 1_000) / chrom_length
    r::Float64 = log(2.0) / q # from f(x) = 0.5 = 1 / exp(r*x); where x = normalised distance between loci
    Threads.@threads for idx1 in 1:n_loci
        for idx2 in 1:n_loci
            # idx1 = 1; idx2 = n_loci
            dist = abs(chrom_positions[idx1] - chrom_positions[idx2]) / chrom_length
            # Keep covariances between far loci at zero
            if dist <= minimum([rel_dist_multiplier*q, 0.9])
                c = 1 / exp(r * dist)
                S[idx1, idx2] = c
                S[idx2, idx1] = c
            end
        end
    end
    # Output
    Σ::SparseMatrixCSC{Float64} = sparse(S)
    Σ
end

"""
    simulateperpopμΣ(; 
        Σ_base::SparseMatrixCSC{Float64}, 
        μ_β_params::Tuple{Float64,Float64} = (0.5, 0.5),
        rng::TaskLocalRNG = Random.GLOBAL_RNG
    ) -> Tuple{Vector{Float64}, SparseMatrixCSC{Float64}}

Simulate per-population mean allele frequencies and their variance-covariance matrix.

# Arguments
- `Σ_base::SparseMatrixCSC{Float64}`: Base variance-covariance matrix to be scaled
- `μ_β_params::Tuple{Float64,Float64}`: Parameters (α, β) for Beta distribution used to generate mean allele frequencies (default: (0.5, 0.5))
- `rng::TaskLocalRNG`: Random number generator for reproducibility (default: GLOBAL_RNG)

# Returns
- `Tuple{Vector{Float64}, SparseMatrixCSC{Float64}}`: A tuple containing:
  - Vector of mean allele frequencies (μ)
  - Scaled variance-covariance matrix (Σ)

# Details
The function:
1. Samples mean allele frequencies from a Beta distribution
2. Scales the variance-covariance matrix based on allele frequencies
3. Ensures positive definiteness of the resulting matrix through iterative adjustment

The variance scaling is performed such that loci closer to fixation (0.0 or 1.0) 
have lower variance, following population genetics expectations.

# Example
```jldoctest; setup = :(using GBCore)
julia> chrom_lengths, chrom_loci_counts = simulatechromstruct(10_000, 7, 135_000_000);

julia> positions, loci_alleles = simulateposandalleles(chrom_lengths=chrom_lengths, chrom_loci_counts=chrom_loci_counts, n_alleles=2);

julia> Σ_base = simulateldblocks(chrom_positions=positions[1], chrom_length=chrom_lengths[1], ld_corr_50perc_kb=1_000);

julia> μ, Σ = simulateperpopμΣ(Σ_base=Σ_base, μ_β_params=(0.5, 0.5));

julia> length(μ) == length(positions[1])
true

julia> size(Σ) == size(Σ_base)
true

julia> abs(sum(Σ .- Σ_base)) > 0.0
true
```
"""
function simulateperpopμΣ(;
    Σ_base::SparseMatrixCSC{Float64},
    μ_β_params::Tuple{Float64,Float64} = (0.5, 0.5),
    rng::TaskLocalRNG = Random.GLOBAL_RNG,
)::Tuple{Vector{Float64}, SparseMatrixCSC{Float64}}
    # Sample mean allele frequencies per population
    n_loci = size(Σ_base, 1)
    Beta_distibution = Distributions.Beta(μ_β_params[1], μ_β_params[2])
    μ::Vector{Float64} = rand(rng, Beta_distibution, n_loci)
    # Scale the variance-covariance matrix by the allele frequency means
    # such that the closer to fixation (closer to 0.0 or 1.0) the lower the variance
    idx_greater_than_half::Vector{Int64} = findall(μ .> 0.5)
    σ² = copy(μ)
    σ²[idx_greater_than_half] = 1.00 .- μ[idx_greater_than_half]
    Σ = Σ_base .* (σ² * σ²')
    # Make sure that the variance-covariance matrix is positive definite
    max_iter::Int64 = 10
    iter::Int64 = 1
    while !LinearAlgebra.isposdef(Σ) && (iter < max_iter)
        if iter == 1
            Σ[diagind(Σ)] .+= 1.0e-12
        end
        Σ[diagind(Σ)] .*= 10.0
        iter += 1
    end
    # Output
    (μ, Σ)
end

"""
    simulateallelefreqs!(
        allele_frequencies::Matrix{Union{Float64,Missing}}, 
        locus_counter::Vector{UInt},
        pb::Union{Nothing, ProgressMeter.Progress};
        μ::Vector{Float64}, 
        Σ::SparseMatrixCSC{Float64},
        n_alleles::Int64,
        idx_entries_per_population::Vector{Int64},
        rng::TaskLocalRNG = Random.GLOBAL_RNG,
    )::Nothing

Simulates allele frequencies for multiple loci and populations using a multivariate normal distribution.

# Arguments
- `allele_frequencies::Matrix{Union{Float64,Missing}}`: Matrix to store the simulated allele frequencies for all loci
- `locus_counter::Vector{UInt}`: Counter tracking the current locus position
- `pb::ProgressMeter.Progress`: Progress bar for tracking simulation progress
- `μ::Vector{Float64}`: Mean vector for the multivariate normal distribution
- `Σ::SparseMatrixCSC{Float64}`: Variance-covariance matrix for the multivariate normal distribution
- `n_alleles::Int64`: Number of alleles per locus
- `idx_entries_per_population::Vector{Int64}`: Indices for entries in each population
- `rng::TaskLocalRNG`: Random number generator (default: global RNG)

# Description
Simulates allele frequencies for multiple populations and loci using a multivariate normal distribution.
For each allele (except the last one) at each locus, frequencies are sampled from the specified 
distribution. For alleles after the first one, frequencies are adjusted to ensure they sum to 1.0
within each locus. All frequencies are bounded between 0.0 and 1.0.

# Returns
`Nothing`

# Throws
- `ArgumentError`: If length of allele_frequencies is less than or equal to length of μ
- `ArgumentError`: If length of μ does not match the dimensions of Σ
- `ArgumentError`: If maximum index in idx_entries_per_population exceeds number of entries

# Note
The last allele frequency for each locus is implicitly determined as 1 minus the sum of other allele 
frequencies to ensure frequencies sum to 1.0.

# Example
```jldoctest; setup = :(using GBCore)
julia> chrom_lengths, chrom_loci_counts = simulatechromstruct(10_000, 7, 135_000_000);

julia> positions, loci_alleles = simulateposandalleles(chrom_lengths=chrom_lengths, chrom_loci_counts=chrom_loci_counts, n_alleles=2);

julia> Σ_base = simulateldblocks(chrom_positions=positions[1], chrom_length=chrom_lengths[1], ld_corr_50perc_kb=1_000);

julia> μ, Σ = simulateperpopμΣ(Σ_base=Σ_base, μ_β_params=(0.5, 0.5));

julia> populations, idx_population_groupings = simulatepopgroups(n=100, n_populations=3);

julia> allele_frequencies::Matrix{Union{Missing, Float64}} = fill(missing, 100, 10_000); locus_counter::Vector{UInt} = [1]; pb = ProgressMeter.Progress(10_000);

julia> simulateallelefreqs!(allele_frequencies, locus_counter, pb, μ=μ, Σ=Σ, n_alleles=n_alleles, idx_entries_per_population=idx_population_groupings[1])

julia> sum(.!ismissing.(allele_frequencies[idx_population_groupings[1], 1:length(μ)])) == (length(idx_population_groupings[1]) * length(μ))
true
```
"""

"""
    simulateallelefreqs!(allele_frequencies, locus_counter, pb; μ, Σ, n_alleles, idx_entries_per_population, rng)

Simulate allele frequencies for multiple loci and populations using a multivariate normal distribution.

# Arguments
- `allele_frequencies::Matrix{Union{Missing, Float64}}`: Matrix to store simulated allele frequencies
- `locus_counter::Vector{UInt}`: Counter keeping track of current locus position
- `pb::Union{Nothing, ProgressMeter.Progress}`: Optional progress bar
- `μ::Vector{Float64}`: Mean vector for the multivariate normal distribution
- `Σ::SparseMatrixCSC{Float64}`: Variance-covariance matrix for the multivariate normal distribution
- `n_alleles::Int64`: Number of alleles per locus
- `idx_entries_per_population::Vector{Int64}`: Vector containing indices for each population
- `rng::TaskLocalRNG`: Random number generator (default: `Random.GLOBAL_RNG`)

# Returns
- `Nothing`

# Description
Simulates allele frequencies for multiple populations and loci using a multivariate normal distribution.
For each population and allele, it samples frequencies ensuring they sum to 1.0 across alleles at each locus.
The function bounds frequencies between 0.0 and 1.0 and updates the provided:
    - `allele_frequencies` matrix in-place,
    - `locus_counter` one-element vector to keep track of the current locus position, and
    - `pb` progress bar to track simulation progress, if it is not Nothing.

# Throws
- `ArgumentError`: If input dimensions are inconsistent:
  - If length of `allele_frequencies` is less than length of `μ`
  - If length of `μ` doesn't match size of `Σ`
  - If maximum index in `idx_entries_per_population` exceeds size of `allele_frequencies`

# Note
The last allele frequency for each locus is implicitly determined as 1 minus the sum of other allele 
frequencies to ensure frequencies sum to 1.0.

# Example
```jldoctest; setup = :(using GBCore)
julia> chrom_lengths, chrom_loci_counts = simulatechromstruct(10_000, 7, 135_000_000);

julia> positions, loci_alleles = simulateposandalleles(chrom_lengths=chrom_lengths, chrom_loci_counts=chrom_loci_counts, n_alleles=2);

julia> Σ_base = simulateldblocks(chrom_positions=positions[1], chrom_length=chrom_lengths[1], ld_corr_50perc_kb=1_000);

julia> μ, Σ = simulateperpopμΣ(Σ_base=Σ_base, μ_β_params=(0.5, 0.5));

julia> populations, idx_population_groupings = simulatepopgroups(n=100, n_populations=3);

julia> allele_frequencies::Matrix{Union{Missing, Float64}} = fill(missing, 100, 10_000); locus_counter::Vector{UInt} = [1]; pb = ProgressMeter.Progress(10_000);

julia> simulateallelefreqs!(allele_frequencies, locus_counter, pb, μ=μ, Σ=Σ, n_alleles=n_alleles, idx_entries_per_population=idx_population_groupings[1])

julia> sum(.!ismissing.(allele_frequencies[idx_population_groupings[1], 1:length(μ)])) == (length(idx_population_groupings[1]) * length(μ))
true
```
"""
function simulateallelefreqs!(
    allele_frequencies::Matrix{Union{Missing, Float64}}, 
    locus_counter::Vector{UInt},
    pb::Union{Nothing, ProgressMeter.Progress};
    μ::Vector{Float64}, 
    Σ::SparseMatrixCSC{Float64},
    n_alleles::Int64,
    idx_entries_per_population::Vector{Int64},
    rng::TaskLocalRNG = Random.GLOBAL_RNG,
)::Nothing
    # Check Arguments
    if length(allele_frequencies) <= length(μ)
        throw(ArgumentError("The length of `allele_frequencies` (for the entire genome) should be greater than or equal to the length of `μ` (for the current chromosome)."))
    end
    if length(μ) != size(Σ, 1)
        throw(ArgumentError("The length of `μ` should be equal to the number of loci in the variance-covariance matrix `Σ`."))
    end
    if maximum(idx_entries_per_population) > size(allele_frequencies, 1)
        throw(ArgumentError("The maximum index in `idx_entries_per_population` should be less than or equal to the number of entries in `allele_frequencies`."))
    end
    # Define the multivariate normal distribution
    mvnormal_distribution = Distributions.MvNormal(μ, PDMats.PDSparseMat(Σ))
    n_loci = length(μ)
    # Sample per allele per population
    for a = 1:(n_alleles-1)
        # Loop through each individual in the current population
        for j in idx_entries_per_population
            # Calculate indices for storing allele frequencies in the output matrix
            idx_ini = ((n_alleles - 1) * (locus_counter[1] - 1)) + a
            idx_fin = ((n_alleles - 1) * ((locus_counter[1] - 1) + (n_loci - 1))) + a
            # Sample allele frequencies from the multivariate normal distribution
            allele_freqs::Vector{Float64} = rand(rng, mvnormal_distribution)
            # For alleles after the first one, ensure frequencies sum to 1.0
            if a > 1
                # Initialize vector to store sum of previous allele frequencies
                sum_of_prev_allele_freqs::Vector{Float64} = fill(0.0, n_loci)
                # Sum up frequencies of all previous alleles
                for ap = 1:(a-1)
                    ap_idx_ini = ((n_alleles - 1) * (locus_counter[1] - 1)) + ap
                    ap_idx_fin = ((n_alleles - 1) * ((locus_counter[1] - 1) + (n_loci - 1))) + ap
                    x = allele_frequencies[j, range(ap_idx_ini, ap_idx_fin; step = (n_alleles - 1))]
                    sum_of_prev_allele_freqs = sum_of_prev_allele_freqs + x
                end
                # Current allele frequency is 1 minus sum of previous frequencies
                allele_freqs = 1 .- sum_of_prev_allele_freqs
            end
            # Bound frequencies between 0.0 and 1.0
            allele_freqs[allele_freqs.>1.0] .= 1.0
            allele_freqs[allele_freqs.<0.0] .= 0.0
            # Store frequencies in output matrix
            allele_frequencies[j, range(idx_ini, idx_fin; step = (n_alleles - 1))] = allele_freqs
            if !isnothing(pb)
                ProgressMeter.next!(pb)
            end
        end # entries per population
    end # alleles
end

"""
    simulategenomes(;
        n::Int64 = 100,
        n_populations::Int64 = 1, 
        l::Int64 = 10_000,
        n_chroms::Int64 = 7,
        n_alleles::Int64 = 2,
        max_pos::Int64 = 135_000_000,
        ld_corr_50perc_kb::Int64 = 1_000,
        rel_dist_multiplier::Float64 = 2.0,
        μ_β_params::Tuple{Float64,Float64} = (0.5, 0.5),
        sparsity::Float64 = 0.0,
        seed::Int64 = 42,
        verbose::Bool = true
    )::Genomes

Simulates genomic data with population structure and linkage disequilibrium.

# Arguments
- `n::Int64`: Number of entries/individuals to simulate (1 to 1e9)
- `n_populations::Int64`: Number of populations to simulate (1 to n)
- `l::Int64`: Number of loci to simulate (2 to 1e9)
- `n_chroms::Int64`: Number of chromosomes (1 to 1e6)
- `n_alleles::Int64`: Number of alleles per locus (2 to 5, representing A, T, C, G, D)
- `max_pos::Int64`: Maximum position in base pairs (10 to 160e9)
- `ld_corr_50perc_kb::Int64`: Distance in kb where correlation decay reaches 50%
- `rel_dist_multiplier::Float64`: Multiplier for maximum relative distance to consider in LD blocks
- `μ_β_params::Tuple{Float64,Float64}`: Shape parameters for Beta distribution of allele frequencies
- `sparsity::Float64`: Proportion of missing data to simulate (0.0 to 1.0)
- `seed::Int64`: Random seed for reproducibility
- `verbose::Bool`: Whether to show progress bar and final plot

# Returns
- `Genomes`: A struct containing:
  - entries: Vector of entry IDs
  - populations: Vector of population assignments
  - loci_alleles: Vector of locus-allele combinations
  - allele_frequencies: Matrix of allele frequencies
  - mask: Boolean matrix indicating valid data points

# Details
- Simulates genomic data by:
    + Generating chromosome lengths and loci positions
    + Assigning alleles to loci
    + Grouping entries into populations
    + Simulating allele frequencies with linkage disequilibrium using multivariate normal distribution
    + Adding optional sparsity (missing data)
- Chromosome lengths are distributed evenly, with any remainder added to last chromosome
- Loci positions are randomly sampled without replacement within each chromosome
- LD decay follows an exponential function: corr = 1/exp(r*d), where d is normalized distance
- Mean allele frequencies are sampled from Beta(α,β) distribution
- Population structure is implemented by sampling the mean allele frequencies per population
- For each entry and locus, allele frequencies with linkage disequilibrium are simulated by sampling a multivariate normal distribution per chromosome
- Missing data is randomly assigned if sparsity > 0

# Throws
- `ArgumentError`: If input parameters are outside acceptable ranges
- `DimensionMismatch`: If there's an error in the simulation process

# Examples
```jldoctest; setup = :(using GBCore, StatsBase, Random)
julia> genomes = simulategenomes(n=100, l=10_000, n_alleles=3, verbose=false);

julia> length(genomes.entries)
100

julia> length(genomes.populations)
100

julia> length(genomes.loci_alleles)
20000

julia> size(genomes.allele_frequencies)
(100, 20000)

julia> mean(ismissing.(genomes.allele_frequencies))
0.0

julia> rng::TaskLocalRNG = Random.seed!(123);

julia> idx = StatsBase.sample(rng, range(1, 20_000, step=2), 250, replace = false, ordered = true);

julia> correlations = StatsBase.cor(genomes.allele_frequencies[:, idx]);

julia> correlations[10,10] == 1.00
true

julia> correlations[10,10] > correlations[10,250]
true

julia> genomes = simulategenomes(n=100, l=10_000, n_alleles=3, sparsity=0.25, verbose=false);

julia> mean(ismissing.(genomes.allele_frequencies))
0.25
```
"""
function simulategenomes(;
    n::Int64 = 100,
    n_populations::Int64 = 1,
    l::Int64 = 10_000,
    n_chroms::Int64 = 7,
    n_alleles::Int64 = 2,
    max_pos::Int64 = 135_000_000,
    ld_corr_50perc_kb::Int64 = 1_000,
    rel_dist_multiplier::Float64 = 2.0,
    μ_β_params::Tuple{Float64,Float64} = (0.5, 0.5),
    sparsity::Float64 = 0.0,
    seed::Int64 = 42,
    verbose::Bool = true,
)::Genomes
    # n::Int64=100; n_populations::Int64 = 5; l::Int64=10_000; n_chroms::Int64=7;n_alleles::Int64=3; max_pos::Int64=135_000_000; ld_corr_50perc_kb::Int64=1_000; rel_dist_multiplier = 2.0; seed::Int64=42; μ_β_params::Tuple{Float64, Float64} = (0.5, 0.5); sparsity::Float64 = 0.25; verbose::Bool=true;
    # Parameter checks
    if (n < 1) || (n > 1e9)
        throw(ArgumentError("We accept `n` from 1 to 1 billion."))
    end
    if (n_populations < 1) || (n_populations > n)
        throw(ArgumentError("We accept `n_populations` from 1 to `n`."))
    end
    if (l < 2) || (l > 1e9)
        throw(ArgumentError("We accept `l` from 2 to 1 billion."))
    end
    if (n_chroms < 1) || (n_chroms > 1e6)
        throw(ArgumentError("We accept `n_chroms` from 1 to 1 million."))
    end
    if (n_alleles < 2) || (n_alleles > 5)
        throw(ArgumentError("We accept `n` from 2 to 5, which can be A, T, C, G, and D (for deletion)."))
    end
    if (max_pos < 10) || (max_pos > 160e9)
        throw(ArgumentError("We accept `max_pos` from 10 to 160 billion (genome of *Tmesipteris oblanceolata*)."))
    end
    if (ld_corr_50perc_kb > ceil(max_pos / n_chroms))
        throw(
            ArgumentError(
                "The parameter `ld_corr_50perc_kb` should be less than or equal to `ceil(max_pos/n_chroms)`.",
            ),
        )
    end
    if l > max_pos
        throw(ArgumentError("The parameter `l` should be less than or equal to `max_pos`."))
    end
    if n_chroms > l
        throw(ArgumentError("The parameter `n_chroms` should be less than or equal to `l`."))
    end
    # Instantiate the output struct
    p = l * (n_alleles - 1)
    genomes = Genomes(; n = n, p = p)
    # Instantiate the randomisation
    rng::TaskLocalRNG = Random.seed!(seed)
    # Simulate chromosome lengths and number of loci per chromosome
    chrom_lengths, chrom_loci_counts = simulatechromstruct(l=l, n_chroms=n_chroms, max_pos=max_pos)
    # Simulate loci-alleles combinations coordinates
    positions, loci_alleles = simulateposandalleles(
        chrom_lengths = chrom_lengths,
        chrom_loci_counts = chrom_loci_counts,
        n_alleles = n_alleles,
        rng = rng,
    )
    # Group entries into populations
    populations, idx_population_groupings = simulatepopgroups(n=n, n_populations=n_populations)
    # Simulate allele frequencies with linkage disequillibrium by sampling from a multivariate normal distribution with non-spherical variance-covariance matrix
    allele_frequencies::Matrix{Union{Float64,Missing}} = fill(missing, n, p)
    locus_counter::Vector{UInt} = [1]
    pb = if verbose
        ProgressMeter.Progress(n_chroms * n * (n_alleles - 1); desc = "Simulating allele frequencies: ")
    else
        nothing
    end
    for i = 1:n_chroms
        # Simulate linkage blocks
        # i  = 1
        Σ_base = simulateldblocks(
            chrom_positions = positions[i], 
            chrom_length = chrom_lengths[i], 
            ld_corr_50perc_kb = ld_corr_50perc_kb,
            rel_dist_multiplier = rel_dist_multiplier,
        )
        for k = 1:n_populations
            # k = 4
            # Sample mean allele frequencies per population, and scale the variance-covariance matrix by the allele frequency means
            μ, Σ = simulateperpopμΣ(Σ_base=Σ_base, μ_β_params=μ_β_params, rng=rng)
            # Update the allele frequency vector, locus_counter, and progress bar
            simulateallelefreqs!(
                allele_frequencies, 
                locus_counter,
                pb,
                μ = μ, 
                Σ = Σ,
                n_alleles = n_alleles,
                idx_entries_per_population = idx_population_groupings[k],
                rng = rng,
            )
        end # populations
        locus_counter[1] = locus_counter[1] + chrom_loci_counts[i]
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Populate the output struct
    genomes.entries = ["entry_" * lpad(i, length(string(n)), "0") for i = 1:n]
    genomes.populations = populations
    genomes.loci_alleles = loci_alleles
    genomes.allele_frequencies = allele_frequencies
    genomes.mask = fill(true, (n, p))
    # Simulate sparsity
    if sparsity > 0.0
        idx = sample(rng, 0:((n*p)-1), Int64(round(sparsity * n * p)); replace = false)
        idx_rows = (idx .% n) .+ 1
        idx_cols = Int.(floor.(idx ./ n)) .+ 1
        genomes.allele_frequencies[CartesianIndex.(idx_rows, idx_cols)] .= missing
    end
    if verbose
        plot(genomes)
    end
    ### Check dimensions
    if !checkdims(genomes)
        throw(DimensionMismatch("Error simulating genomes."))
    end
    # Output
    return (genomes)
end
