"""
    inflatediagonals!(X::Matrix{Float64}; max_iter::Int64=1_000, verbose::Bool=false)::Nothing

Ensure matrix invertibility by iteratively inflating diagonal elements until the determinant is nonzero.

# Arguments
- `X::Matrix{Float64}`: Input square symmetric matrix to be modified in-place
- `max_iter::Int64=1_000`: Maximum number of iterations
- `verbose::Bool=false`: If true, prints information about the inflation process

# Details
The function adds progressively larger values to the diagonal elements until the matrix
becomes invertible (det(X) > eps(Float64)) or the maximum number of iterations is reached.
The initial inflation value ϵ is set to the maximum absolute value in the matrix and
increases slightly in each iteration.

# Throws
- `ArgumentError`: If the input matrix is not symmetric
- `ArgumentError`: If the input matrix is not square
- `ArgumentError`: If the input matrix contains NaN values
- `ArgumentError`: If the input matrix contains Inf values

# Returns
`Nothing`. The input matrix `X` is modified in-place.

# Example
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> x::Vector{Float64} = rand(10);

julia> X::Matrix{Float64} = x * x';

julia> inflatediagonals!(X);

julia> det(X) > eps(Float64)
true
```
"""
function inflatediagonals!(X::Matrix{Float64}; max_iter::Int64 = 1_000, verbose::Bool = false)::Nothing
    # Check arguments
    if !issymmetric(X)
        throw(ArgumentError("The input matrix is not symmetric."))
    end
    if size(X, 1) != size(X, 2)
        throw(ArgumentError("The input matrix is not square."))
    end
    if det(X) > eps(Float64)
        return nothing
    end
    if sum(isnan.(X)) > 0
        throw(ArgumentError("The input matrix contains NaN values."))
    end
    if sum(isinf.(X)) > 0
        throw(ArgumentError("The input matrix contains Inf values."))
    end
    iter = 0
    ϵ = maximum(abs.(X))
    ϵ_total = 0.0
    while (det(X) < eps(Float64)) && (iter < max_iter)
        iter += 1
        X[diagind(X)] .+= ϵ
        ϵ_total += ϵ
        ϵ *= (1.0 + eps(Float64))
    end
    if verbose
        println("Performed $iter iterations of diagonal inflation to ensure matrix invertibility (ϵ_total = $ϵ_total).")
    end
end

"""
    grmsimple(
        genomes::Genomes;
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        verbose::Bool = false
    )::Matrix{Float64}

Generate a simple genetic relationship matrix (GRM) from genomic data.

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to select specific entries/individuals
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to select specific loci/alleles
- `verbose::Bool`: If true, displays a heatmap visualization of the GRM

# Returns
- `Matrix{Float64}`: A symmetric positive definite genetic relationship matrix

# Details
The function computes a genetic relationship matrix by:
1. Converting genomic data to a numerical matrix
2. Computing GRM as G * G' / ncol(G)
3. Adding small positive values to diagonal elements if necessary to ensure matrix invertibility

# Notes
- The resulting matrix is always symmetric
- Diagonal elements may be slightly inflated to ensure matrix invertibility
- The matrix dimensions will be n×n where n is the number of entries/individuals

# Example
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> genomes = simulategenomes(l=1_000, verbose=false);

julia> grm = grmsimple(genomes);

julia> size(grm.genomic_relationship_matrix), issymmetric(grm.genomic_relationship_matrix)
((100, 100), true)

julia> det(grm.genomic_relationship_matrix) > eps(Float64)
true
```
"""
function grmsimple(genomes::Genomes; max_iter::Int64 = 1_000, verbose::Bool = false)::GRM
    # genomzes = simulategenomes(); max_iter = 1_000; verbose = true;
    # Check arguments while extracting the allele frequencies but first create a dummy phenomes struct
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted ☹."))
    end
    # Calculate a simple GRM
    n, p = size(genomes.allele_frequencies)
    grm = GRM(genomes.entries, genomes.loci_alleles, Matrix{Float64}(undef, n, n))
    Threads.@threads for i = 1:n
        for j = i:n
            # i = 1; j = 1;
            a = dot(genomes.allele_frequencies[i, :], genomes.allele_frequencies[j, :]) / p
            grm.genomic_relationship_matrix[i, j] = a
            grm.genomic_relationship_matrix[j, i] = a
        end
    end
    # Inflate the diagonals until invertible
    inflatediagonals!(grm.genomic_relationship_matrix, max_iter = max_iter, verbose = verbose)
    # Output
    if !checkdims(grm)
        throw(ErrorException("Error computing a simple GRM."))
    end
    if verbose
        UnicodePlots.heatmap(grm.genomic_relationship_matrix)
    end
    grm
end

"""
    grmploidyaware(
        genomes::Genomes;
        ploidy::Int64 = 2,
        idx_entries::Union{Nothing,Vector{Int64}} = nothing,
        idx_loci_alleles::Union{Nothing,Vector{Int64}} = nothing,
        verbose::Bool = false
    )::Matrix{Float64}

Generate a ploidy-aware genetic relationship matrix (GRM) based on the methods described in 
Bell et al. (2017) and VanRaden et al. (2008).

# Arguments
- `genomes::Genomes`: Genomic data structure containing allele frequencies
- `ploidy::Int64`: Number of chromosome copies in the organism (default: 2)
- `idx_entries::Union{Nothing,Vector{Int64}}`: Optional indices to select specific entries (default: nothing)
- `idx_loci_alleles::Union{Nothing,Vector{Int64}}`: Optional indices to select specific loci/alleles (default: nothing)
- `verbose::Bool`: If true, displays a heatmap of the resulting GRM (default: false)

# Returns
- `Matrix{Float64}`: A symmetric genetic relationship matrix with dimensions (n × n), where n is the number of entries

# Details
The function implements the following steps:
1. Extracts and processes genomic data
2. Calculates allele frequencies and centers the data
3. Computes the GRM using VanRaden's method
4. Ensures matrix invertibility by adding small values to the diagonal if necessary

# Note
The diagonal elements may be slightly inflated to ensure matrix invertibility for downstream analyses.

# Example
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> genomes = simulategenomes(l=1_000, verbose=false);

julia> grm = grmploidyaware(genomes);

julia> size(grm.genomic_relationship_matrix), issymmetric(grm.genomic_relationship_matrix)
((100, 100), true)

julia> det(grm.genomic_relationship_matrix) > eps(Float64)
true
```
"""
function grmploidyaware(genomes::Genomes; ploidy::Int64 = 2, max_iter::Int64 = 1_000, verbose::Bool = false)::GRM
    # genomes = simulategenomes(); ploidy = 2; max_iter = 1_000; verbose = true;
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted ☹."))
    end
    # Calculate GRM via Bell et al (2017) and VanRaden et al (2008)
    n, _p = size(genomes.allele_frequencies)
    grm = GRM(genomes.entries, genomes.loci_alleles, Matrix{Float64}(undef, n, n))
    q = mean(genomes.allele_frequencies, dims = 1)[1, :]
    G_star = ploidy .* (genomes.allele_frequencies .- 0.5)
    # q_star = ploidy .* (q .- 0.5)
    # Z = G_star .- q'
    d = (ploidy * sum(q .* (1 .- q)))
    Threads.@threads for i = 1:n
        for j = i:n
            # i = 1; j = 1;
            a = dot((G_star[i, :] - q), (G_star[j, :] - q)) / d
            grm.genomic_relationship_matrix[i, j] = a
            grm.genomic_relationship_matrix[j, i] = a
        end
    end
    # Inflate the diagonals until invertible
    inflatediagonals!(grm.genomic_relationship_matrix, max_iter = max_iter, verbose = verbose)
    # Output
    if verbose
        UnicodePlots.heatmap(grm.genomic_relationship_matrix)
    end
    grm
end
