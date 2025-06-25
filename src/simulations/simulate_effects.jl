"""
    hash(x::SimulatedEffects, h::UInt)::UInt

Compute a hash value for a `SimulatedEffects` object.

This method implements custom hashing for `SimulatedEffects` by iterating through all fields
of the object and combining their hash values with the provided seed hash `h`.

# Arguments
- `x::SimulatedEffects`: The object to be hashed
- `h::UInt`: The hash seed value

# Returns
- `UInt`: The computed hash value

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> effects = SimulatedEffects();

julia> typeof(hash(effects))
UInt64
```
"""
function Base.hash(x::SimulatedEffects, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    Base.:(==)(x::SimulatedEffects, y::SimulatedEffects)::Bool

Defines equality comparison for SimulatedEffects structs by comparing their hash values.

This method overloads the == operator for SimulatedEffects type and determines if two
SimulatedEffects instances are equal by comparing their hash values rather than doing
a field-by-field comparison.

# Arguments
- `x::SimulatedEffects`: First SimulatedEffects instance to compare
- `y::SimulatedEffects`: Second SimulatedEffects instance to compare

# Returns
- `Bool`: true if the hash values of both instances are equal, false otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> effects_1 = SimulatedEffects();

julia> effects_2 = SimulatedEffects();

julia> effects_3 = SimulatedEffects(); effects_3.id[1] = "SOMETHING_ELSE";

julia> effects_1 == effects_2
true

julia> effects_1 == effects_3
false
```
"""
function Base.:(==)(x::SimulatedEffects, y::SimulatedEffects)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(effects::SimulatedEffects)::Bool

Check dimension compatibility of the fields of the SimulatedEffects struct.

# Arguments
- `effects::SimulatedEffects`: A SimulatedEffects struct containing various genetic and experimental effects

# Returns
- `Bool`: `true` if all dimensions are compatible, `false` otherwise

Verifies that:
- `id` has length 6
- `field_layout` has 4 columns 
- All following vectors have the same length (n):
  - `replications_x_site_x_harvest_x_season_x_year`
  - `blocks_x_site_x_harvest_x_season_x_year`
  - `rows_x_site_x_harvest_x_season_x_year`
  - `cols_x_site_x_harvest_x_season_x_year`
  - `additive_genetic`
  - `dominance_genetic`
  - `epistasis_genetic`
  - `additive_allele_x_site_x_harvest_x_season_x_year`
  - `dominance_allele_x_site_x_harvest_x_season_x_year`
  - `epistasis_allele_x_site_x_harvest_x_season_x_year`

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> effects = SimulatedEffects();

julia> typeof(hash(effects))
UInt64
```jldoctest; setup = :(using GenomicBreedingCore)
julia> effects = SimulatedEffects();

julia> checkdims(effects)
true

julia> effects.id = ["beaking_change"];

julia> checkdims(effects)
false
```
"""
function checkdims(effects::SimulatedEffects)::Bool
    n::Int64 = length(effects.replications_x_site_x_harvest_x_season_x_year)
    if (length(effects.id) != 6) ||
       (size(effects.field_layout, 2) != 4) ||
       (n != length(effects.blocks_x_site_x_harvest_x_season_x_year)) ||
       (n != length(effects.rows_x_site_x_harvest_x_season_x_year)) ||
       (n != length(effects.cols_x_site_x_harvest_x_season_x_year)) ||
       (n != length(effects.additive_genetic)) ||
       (n != length(effects.dominance_genetic)) ||
       (n != length(effects.epistasis_genetic)) ||
       (n != length(effects.additive_allele_x_site_x_harvest_x_season_x_year)) ||
       (n != length(effects.dominance_allele_x_site_x_harvest_x_season_x_year)) ||
       (n != length(effects.epistasis_allele_x_site_x_harvest_x_season_x_year))
        return false
    end
    return true
end

"""
    sum(effects::SimulatedEffects)::Vector{Float64}

Sum up all simulated effects to generate the simulated phenotype values. The function iterates through
all fields of the SimulatedEffects struct (except :id and :field_layout) and adds their values
element-wise to produce a vector of phenotypic values.

# Arguments
- `effects::SimulatedEffects`: A struct containing various genetic and environmental effects

# Returns
- `Vector{Float64}`: A vector containing the summed effects (phenotypic values)

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> effects = SimulatedEffects();

julia> sum(effects)
1-element Vector{Float64}:
 0.0

julia> effects.additive_genetic[1] = pi;

julia> sum(effects)
1-element Vector{Float64}:
 3.141592653589793
```
"""
function Base.sum(effects::SimulatedEffects)::Vector{Float64}
    ϕ::Vector{Float64} = fill(0.0, size(effects.additive_genetic))
    for name in fieldnames(SimulatedEffects)
        if (name == :id) || (name == :field_layout)
            continue
        end
        ϕ .+= getproperty(effects, name)
    end
    return ϕ
end

"""
    simulatecovariancespherical(p::Int64, σ²::Float64)::Matrix{Float64}

Simulate a spherical covariance matrix with constant variance σ² on the diagonal.

# Arguments
- `p::Int64`: Dimension of the covariance matrix
- `σ²::Float64`: Constant variance value for diagonal elements

# Returns
- `Matrix{Float64}`: p × p spherical covariance matrix

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> Σ = simulatecovariancespherical(7, 2.15);

julia> size(Σ)
(7, 7)

julia> Σ[diagind(Σ)] == fill(2.15, 7)
true
```
"""
function simulatecovariancespherical(p::Int64, σ²::Float64)::Matrix{Float64}
    # p=7; σ²=2.15
    Σ::Matrix{Float64} = zeros(p, p)
    Σ[diagind(Σ)] .= σ²
    Σ
end

"""
    simulatecovariancediagonal(p::Int64, σ²::Vector{Float64})::Matrix{Float64}

Simulate a diagonal covariance matrix with specified variances σ² on the diagonal.

# Arguments
- `p::Int64`: Dimension of the covariance matrix 
- `σ²::Vector{Float64}`: Vector of variance values for diagonal elements

# Returns
- `Matrix{Float64}`: p × p diagonal covariance matrix

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> Σ = simulatecovariancediagonal(7, rand(7));

julia> size(Σ)
(7, 7)

julia> var(Σ[diagind(Σ)]) > 0.0
true
```
"""
function simulatecovariancediagonal(p::Int64, σ²::Vector{Float64})::Matrix{Float64}
    # p=7; σ²=rand(p)
    if length(σ²) != p
        throw(ArgumentError("Length of σ² must match p."))
    end
    Σ::Matrix{Float64} = zeros(p, p)
    Σ[diagind(Σ)] = σ²
    Σ
end

"""
    simulatecovariancerandom(p::Int64, seed::Int64 = 42)::Matrix{Float64}

Generate a random positive semidefinite covariance matrix of size p × p.

# Arguments
- `p::Int64`: Dimension of the covariance matrix to generate
- `seed::Int64 = 42`: Random seed for reproducibility

# Returns
- `Matrix{Float64}`: A p × p positive semidefinite covariance matrix

# Details
The function generates a random covariance matrix by:
1. Creating a p × 1 random normal vector
2. Computing the outer product of this vector with itself
3. Inflating the diagonal elements to ensure positive definiteness

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> Σ = simulatecovariancerandom(7, 123);

julia> size(Σ)
(7, 7)

julia> (var(Σ) > 0.0) && (abs(det(Σ)) > 0.0)
true
```
"""
function simulatecovariancerandom(p::Int64, seed::Int64 = 42)::Matrix{Float64}
    # p=7; seed=42
    rng = Random.seed!(seed)
    r::Matrix{Float64} = randn(rng, p, 1)
    Σ::Matrix{Float64} = r * r'
    inflatediagonals!(Σ)
    Σ
end

"""
    simulatecovarianceautocorrelated(p::Int64, ρ::Float64 = 0.75)::Matrix{Float64}

Generate a p × p autocorrelated covariance matrix where the correlation between elements
decays exponentially with their distance.

# Arguments
- `p::Int64`: Size of the square covariance matrix
- `ρ::Float64 = 0.75`: Autocorrelation parameter, must be between -1 and 1

# Returns
- `Matrix{Float64}`: A p × p positive definite covariance matrix where Σᵢⱼ = ρ^(|i-j|)

# Throws
- `ArgumentError`: If ρ is not between -1 and 1

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> Σ = simulatecovarianceautocorrelated(7, 0.72);

julia> size(Σ)
(7, 7)

julia> (var(Σ) > 0.0) && (abs(det(Σ)) > 0.0)
true

julia> (Σ[:, 1] == Σ[1, :]) && (Σ[:, 2] == Σ[2, :])
true

julia> sum(diff(reverse(Σ[1, :])) .> 0.0) == 6
true
```
"""
function simulatecovarianceautocorrelated(p::Int64, ρ::Float64 = 0.75)::Matrix{Float64}
    # p=7; ρ=0.75
    if (ρ < -1.0) || (ρ > 1.0)
        throw(ArgumentError("ρ must be between -1 and 1"))
    end
    Σ::Matrix{Float64} = zeros(p, p)
    for i = 1:p
        for j = 1:p
            Σ[i, j] = ρ^(abs(i - j))
        end
    end
    inflatediagonals!(Σ)
    Σ
end

"""
    simulatecovariancekinship(p::Int64, genomes::Genomes)::Matrix{Float64}

Calculate a genomic relationship matrix (GRM) from simulated genomic data.

# Arguments
- `p::Int64`: Expected number of entries/individuals in the genomic data
- `genomes::Genomes`: Genomic data structure containing genetic information

# Returns
- `Matrix{Float64}`: A p × p genomic relationship matrix

# Throws
- `ArgumentError`: If the provided `p` doesn't match the number of entries in `genomes`
# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> Σ = simulatecovariancekinship(7, simulategenomes(n=7, l=1_000, verbose=false));

julia> size(Σ)
(7, 7)

julia> (var(Σ) > 0.0) && (abs(det(Σ)) > 0.0)
true
```
"""
function simulatecovariancekinship(p::Int64, genomes::Genomes)::Matrix{Float64}
    # p = 100; genomes=simulategenomes(n=p, l=2_000, n_alleles=3, verbose=false);
    if p != length(genomes.entries)
        throw(ArgumentError("simulatecovariancekinship: p must match the number of entries in genomes."))
    end
    grm = grmsimple(genomes)
    grm.genomic_relationship_matrix
end

"""
    simulateeffects(; p::Int64 = 2, q::Int64 = 1, λ::Float64 = 1.00, 
                    covar_details::Tuple{Function,Union{Float64,Vector{Float64},Int64,Genomes}} = (simulatecovariancespherical, 1.00),
                    seed::Int64 = 42)::Matrix{Float64}

Simulate correlated effects by sampling from a multivariate normal distribution.

This function generates a matrix of correlated effects by:
1. Sampling means (μ) from an exponential distribution with parameter λ
2. Creating a covariance matrix Σ using the specified covariance function and parameters
3. Drawing samples from MvNormal(μ, Σ)

# Arguments
- `p::Int64`: Number of correlated effects to simulate (default = 2)
- `q::Int64`: Number of times to simulate the correlated effects from the same distribution (default = 1)
- `λ::Float64`: Rate parameter of the exponential distribution for sampling means (default = 1.00)
- `covar_details::Tuple{Function,Union{Float64,Vector{Float64},Int64,Genomes}}`: Tuple containing:
    - First element: Covariance simulation function to use
    - Second element: Parameter(s) for the covariance function:
        * Float64 for spherical or autocorrelated covariance
        * Vector{Float64} for diagonal covariance
        * Int64 for random covariance seed
        * Genomes object for kinship covariance
- `seed::Int64`: Random number generator seed for reproducibility (default = 42)

# Returns
- `Matrix{Float64}`: A p × q matrix where each column represents a set of correlated effects

# Supported Covariance Functions
- `simulatecovariancespherical`: Spherical covariance with constant variance parameter 
- `simulatecovariancediagonal`: Diagonal covariance with vector of variances
- `simulatecovariancerandom`: Random covariance with seed parameter
- `simulatecovarianceautocorrelated`: Autocorrelated covariance with correlation parameter
- `simulatecovariancekinship`: Kinship-based covariance with Genomes object

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> θ₀ = simulateeffects();

julia> sum(abs.(θ₀)) > 0.0
true

julia> p = 10; q = 5;

julia> θ₁ = simulateeffects(p=p, q=q, λ=0.50, covar_details=(simulatecovariancediagonal, rand(p)));

julia> (size(θ₁) == (p, q)) && (θ₀ != θ₁)
true

julia> θ₂ = simulateeffects(p=p, q=q, λ=0.50, covar_details=(simulatecovariancerandom, 123));

julia> (size(θ₂) == (p, q)) && (θ₁ != θ₂)
true

julia> θ₃ = simulateeffects(p=p, q=q, λ=0.50, covar_details=(simulatecovarianceautocorrelated, 0.71));

julia> (size(θ₃) == (p, q)) && (θ₂ != θ₃)
true

julia> genomes = simulategenomes(n=p, l=1_000, n_alleles=3, verbose=false);

julia> θ₄ = simulateeffects(p=p, q=q, λ=0.50, covar_details=(simulatecovariancekinship, genomes));

julia> (size(θ₄) == (p, q)) && (θ₃ != θ₄)
true
```
"""
function simulateeffects(;
    p::Int64 = 2,
    q::Int64 = 1,
    λ::Float64 = 1.00,
    covar_details::Tuple{Any,Union{Float64,Int64,Vector{Float64},Genomes}} = (simulatecovariancespherical, 1.00),
    seed::Int64 = 42,
)::Matrix{Float64}
    # p::Int64 = 20; q::Int64 = 1; λ::Float64 = 1.00; seed::Int64 = 42;
    # covar_details = (simulatecovariancespherical, 1.12)
    if covar_details[1] == simulatecovariancespherical
        if !(covar_details[2] isa Float64)
            throw(ArgumentError("simulateeffects: covar_details[2] must be a Float64 for spherical covariance."))
        end
    elseif covar_details[1] == simulatecovariancediagonal
        if !(covar_details[2] isa Vector{Float64}) || (p != length(covar_details[2]))
            throw(
                ArgumentError(
                    "simulateeffects: covar_details[2] must be a Vector{Float64} of length p for diagonal covariance.",
                ),
            )
        end
    elseif covar_details[1] == simulatecovariancerandom
        if !(covar_details[2] isa Int64)
            throw(ArgumentError("simulateeffects: covar_details[2] must be an Int64 seed for random covariance."))
        end
    elseif covar_details[1] == simulatecovarianceautocorrelated
        if !(covar_details[2] isa Float64) || (covar_details[2] < -1.0) || (covar_details[2] > 1.0)
            throw(
                ArgumentError(
                    "simulateeffects: covar_details[2] must be a Float64 between -1 and 1 for autocorrelated covariance.",
                ),
            )
        end
    elseif covar_details[1] == simulatecovariancekinship
        if !(covar_details[2] isa Genomes)
            throw(ArgumentError("simulateeffects: covar_details[2] must be a Genomes object for kinship covariance."))
        end
    else
        throw(
            ArgumentError(
                string(
                    "simulateeffects: covar_details[1] must be one of the covariance simulation functions, i.e.\n\t‣ ",
                    join(
                        [
                            "simulatecovariancespherical",
                            "simulatecovariancediagonal",
                            "simulatecovariancerandom",
                            "simulatecovarianceautocorrelated",
                            "simulatecovariancekinship",
                        ],
                        "\n\t‣ ",
                    ),
                ),
            ),
        )
    end
    rng::TaskLocalRNG = Random.seed!(seed)
    μ_dist::Exponential = Distributions.Exponential(λ)
    μ::Vector{Float64} = rand(rng, μ_dist, p)
    Σ::Matrix{Float64} = covar_details[1](p, covar_details[2])
    dist::MvNormal = Distributions.MvNormal(μ, Σ)
    X::Matrix{Float64} = rand(rng, dist, q)
    return X
end

"""
    simulategenomiceffects(;
        genomes::Genomes,
        f_additive::Float64 = 0.01,
        f_dominance::Float64 = 0.10,
        f_epistasis::Float64 = 0.05,
        seed::Int64 = 42,
    )::Tuple{Matrix{Float64},Matrix{Float64}}

Simulate additive, dominance, and epistatic effects for multiple loci.

# Arguments
- `genomes::Genomes`: Genome struct containing `n` entries x `p` loci-alleles combinations
- `f_additive::Float64`: Proportion of loci with non-zero additive effects (default = 0.01)
- `f_dominance::Float64`: Proportion of additive loci with dominance effects (default = 0.10)
- `f_epistasis::Float64`: Proportion of additive loci with epistasis effects (default = 0.05)
- `seed::Int64`: Random seed for reproducibility (default = 42)

# Returns
- `Tuple{Matrix{Float64},Matrix{Float64}}`:
  + First matrix (n x 3): Additive, dominance and epistasis effects per entry
  + Second matrix (p x 3): Effects per locus-allele combination

# Details
The genetic effects are simulated using diagonal variance-covariance matrices:

1. For additive effects: Uses a diagonal matrix with random variances for each of the `a` loci 
   with max_n_alleles-1 allele effects per locus.

2. For dominance effects: Uses a diagonal matrix with random variances for each of the `d` loci,
   simulating one dominance effect per locus.

3. For epistasis effects: Uses a diagonal matrix with random variances for each of the `e` loci 
   with max_n_alleles-1 allele effects per locus. The final epistatic effects are computed by 
   multiplying allele frequencies and effects for all possible pairs of epistatic loci.

For all three types of effects, means are sampled from an exponential distribution (λ=1).

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> genomes::Genomes = simulategenomes(n=100, l=2_000, n_alleles=3, verbose=false);

julia> G, B = simulategenomiceffects(genomes=genomes, f_additive=0.05, f_dominance=0.75, f_epistasis=0.25);

julia> size.([G, B])
2-element Vector{Tuple{Int64, Int64}}:
 (100, 3)
 (4000, 3)

julia> sum(B .!= 0.0, dims=1)
1×3 Matrix{Int64}:
 200  75  50
```
"""
function simulategenomiceffects(;
    genomes::Genomes,
    f_additive::Float64 = 0.01,
    f_dominance::Float64 = 0.10,
    f_epistasis::Float64 = 0.05,
    seed::Int64 = 42,
)::Tuple{Matrix{Float64},Matrix{Float64}}
    # genomes::Genomes = simulategenomes(n=100, l=2_000, n_alleles=3, verbose=false); f_additive::Float64 = 0.01; f_dominance::Float64 = 0.10; f_epistasis::Float64 = 0.05; seed::Int64 = 42;
    # Argument checks
    if !checkdims(genomes)
        throw(ArgumentError("simulategenomiceffects: error in the genomes input"))
    end
    if (f_additive < 0.0) || (f_additive > 1.0)
        throw(ArgumentError("We accept `f_additive` from 0.00 to 1.00."))
    end
    if (f_dominance < 0.0) || (f_dominance > 1.0)
        throw(ArgumentError("We accept `f_dominance` from 0.00 to 1.00."))
    end
    if (f_epistasis < 0.0) || (f_epistasis > 1.0)
        throw(ArgumentError("We accept `f_epistasis` from 0.00 to 1.00."))
    end
    # Genomes dimensions
    genomes_dims::Dict{String,Int64} = dimensions(genomes)
    n::Int64 = genomes_dims["n_entries"]
    n_populations::Int64 = genomes_dims["n_populations"]
    p::Int64 = genomes_dims["n_loci_alleles"]
    l::Int64 = genomes_dims["n_loci"]
    max_n_alleles::Int64 = genomes_dims["max_n_alleles"]
    # Number of loci with additive, dominance, and epistasis allele effects (minimum values of 1, 0, and 0 or 2 (if there is non-zero loci with epistasis then we expect to have at least 2 loci interacting), respectively; Note that these are also imposed in the above arguments checks)
    a::Int64 = Int64(maximum([1, round(l * f_additive)]))
    d::Int64 = Int64(maximum([0, round(l * f_additive * f_dominance)]))
    e::Int64 = Int64(maximum([0, round(l * f_additive * f_epistasis)]))
    if e == 1
        e = 2 # if there is one epistatic locus then it should interact with at least one other locus
    end
    # Instantiate the output vectors
    α::Vector{Float64} = fill(0.0, p) # additive allele effects of the p loci-allele combinations
    δ::Vector{Float64} = fill(0.0, p) # dominance allele effects of the p loci-allele combinations
    ξ::Vector{Float64} = fill(0.0, p) # epistasis allele effects of the p loci-allele combinations
    # Set randomisation seed
    rng::TaskLocalRNG = Random.seed!(seed)
    # Define the loci coordinates with non-zero genetic effects
    idx_additive::Vector{Int64} = StatsBase.sample(rng, 1:l, a; replace = false, ordered = true)
    idx_dominance::Vector{Int64} = StatsBase.sample(rng, idx_additive, d; replace = false, ordered = true)
    idx_epistasis::Vector{Int64} = StatsBase.sample(rng, idx_additive, e; replace = false, ordered = true)
    # Sample additive allele effects from a multivariate normal distribution with non-spherical covariance matrix
    # Notes:
    #   - We are simulating effects on max_n_alleles - 1 alleles hence assuming the remaining allele has zero relative effect.
    #   - We are using a begin-end block to modularise the additive allele effects simulation and will do the same for the dominance and epistasis allele effects.
    additive_effects_per_entry::Vector{Float64} = begin
        # Simulate the additive allele effects
        A::Matrix{Float64} = simulateeffects(;
            p = a,
            q = (max_n_alleles - 1),
            covar_details = (simulatecovariancediagonal, rand(a)),
            seed = seed,
        )
        # Define the loci-alleles combination indexes corresponding to the additive allele loci
        idx_p_additive::Vector{Int64} = []
        for i = 1:(max_n_alleles-1)
            append!(idx_p_additive, (idx_additive * (max_n_alleles - 1)) .- (i - 1))
        end
        sort!(idx_p_additive)
        # Update the additive allele effects
        α[idx_p_additive] = reshape(A', (a * (max_n_alleles - 1), 1))
        # Additive effects per entry
        genomes.allele_frequencies * α
    end
    # Sample dominance allele effects from a multivariate normal distribution with non-spherical covariance matrix
    dominance_effects_per_entry::Vector{Float64} = begin
        # Simulate the dominance allele effects
        D::Matrix{Float64} =
            simulateeffects(; p = d, q = 1, covar_details = (simulatecovariancediagonal, rand(d)), seed = seed)
        # Define the loci-alleles combination indexes corresponding to the first allele per locus with a dominance effect
        idx_p_dominance = (idx_dominance * (max_n_alleles - 1)) .- 1
        sort!(idx_p_dominance)
        # Update the dominance allele effects
        δ[idx_p_dominance] = D[:, 1]
        # Dominance effects per entry
        genomes.allele_frequencies * δ
    end
    # Sample epistasis allele effects from a multivariate normal distribution with non-spherical covariance matrix
    # Notes:
    #   - We are simulating effects on max_n_alleles - 1 alleles hence assuming the remaining allele has zero relative effect.
    #   - Then we simulate the non-additive or epistasis allele effects by multiplying the allele frequencies of 2 epistasis loci and their effects.
    epistasis_effects_per_entry::Vector{Float64} = begin
        # Simulate the epistasis allele effects
        E::Matrix{Float64} = simulateeffects(;
            p = e,
            q = (max_n_alleles - 1),
            covar_details = (simulatecovariancediagonal, rand(e)),
            seed = seed,
        )
        # Define the loci-alleles combination indexes corresponding to the epistasis allele loci
        idx_p_epistasis::Vector{Int64} = []
        for i = 1:(max_n_alleles-1)
            append!(idx_p_epistasis, (idx_epistasis * (max_n_alleles - 1)) .- (i - 1))
        end
        sort!(idx_p_epistasis)
        # Update the epistasis allele effects
        ξ[idx_p_epistasis] = reshape(E', (e * (max_n_alleles - 1), 1))
        # Simulate the epistasis allele effects as the sum of the products over all possible pairs of epistatic alleles of their allele frequencies, and epistatic sllele effects
        epistasis_per_entry::Vector{Float64} = fill(0.0, n)
        for i = 1:(length(idx_p_epistasis)-1)
            idx_1 = idx_p_epistasis[i]
            for j = (i+1):length(idx_p_epistasis)
                idx_2 = idx_p_epistasis[j]
                epistasis_per_entry .+=
                    genomes.allele_frequencies[:, idx_1] .* genomes.allele_frequencies[:, idx_2] .* ξ[idx_1] .* ξ[idx_2]
            end
        end
        epistasis_per_entry
    end
    return (hcat(additive_effects_per_entry, dominance_effects_per_entry, epistasis_effects_per_entry), hcat(α, δ, ξ))
end
