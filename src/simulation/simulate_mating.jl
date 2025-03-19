"""
    histallelefreqs(genomes::Genomes)::Nothing

Plot a histogram of allele frequencies from a Genomes object.

# Arguments
- `genomes::Genomes`: A Genomes object containing allele frequency data in its `allele_frequencies` field

# Returns
- `Nothing`: Displays a histogram plot and returns nothing

# Description
Creates and displays a vertical histogram of non-missing allele frequencies using UnicodePlots.
The histogram shows frequency distribution in the range [0,1] with 50 bins.

# Example
```julia
julia> genomes = simulategenomes(n=100, l=10_000, n_alleles=3, n_populations=3, verbose=false);

julia> histallelefreqs(genomes)
```
"""
function histallelefreqs(genomes::Genomes)::Nothing
    q::Vector{Float64} =
        filter(!ismissing, reshape(genomes.allele_frequencies, (prod(size(genomes.allele_frequencies)), 1)))
    plt_histogram = UnicodePlots.histogram(q; vertical = true, xlim = (0.0, 1.0), nbins = 50)
    display(plt_histogram)
    return nothing
end

"""
    simulatemating(;
        parent_genomes::Genomes,
        n_generations::Int = 1,
        pop_size_per_gen::Vector{Int64} = [100],
        seed::Int64 = 42,
        verbose::Bool = false
    )::Vector{Genomes}

Simulates mating processes across multiple generations using a multivariate normal distribution approach.

# Arguments
- `parent_genomes::Genomes`: Initial parent genomic information containing allele frequencies
- `n_generations::Int`: Number of generations to simulate (default: 1)
- `pop_size_per_gen::Vector{Int64}`: Vector of population sizes for each generation (default: [100])
- `seed::Int64`: Random seed for reproducibility (default: 42)
- `verbose::Bool`: Whether to print progress messages (default: false)

# Returns
- `Vector{Genomes}`: Vector of genomes across generations

# Description
This function simulates genetic inheritance across generations by:
1. Sampling progeny allele frequencies using multivariate normal distribution
2. Enforcing biological constraints (frequencies between 0 and 1)
3. Normalizing frequencies for multiallelic loci
4. Displaying frequency histograms for each generation

The simulation maintains allele frequency correlations within chromosomes and handles
multiallelic loci by ensuring their frequencies sum to 1.

# Throws
- `ArgumentError`: If parent genomes contain missing values or invalid dimensions

# Example
```jldoctest; setup = :(using GBCore)
julia> parent_genomes = simulategenomes(n=5, l=10_000, n_alleles=3, verbose=false);

julia> great_great_offspring_genomes = simulatemating(parent_genomes=parent_genomes, n_generations=3, pop_size_per_gen=[10, 20, 30]);

julia> [length(x.entries) for x in great_great_offspring_genomes] == [5, 10, 20, 30]
true
```
"""
function simulatemating(;
    parent_genomes::Genomes,
    n_generations::Int = 1,
    pop_size_per_gen::Vector{Int64} = [100],
    seed::Int64 = 42,
    verbose::Bool = false,
)::Vector{Genomes}
    # parent_genomes = simulategenomes(n=5, μ_β_params=(0.5,0.5)); n_generations::Int = 10; pop_size_per_gen::Vector{Int64} = [100]; seed::Int64 = 42; verbose = true
    # parent_genomes = simulategenomes(n=5, μ_β_params=(2.0,2.0)); n_generations::Int = 10; pop_size_per_gen::Vector{Int64} = [100]; seed::Int64 = 42; verbose = true
    # Check input arguments
    if !checkdims(parent_genomes)
        throw(ArgumentError("Error in the parents' genomes input"))
    end
    n, p, n_missing::Int64 = begin
        genomes_dims::Dict{String,Int64} = dimensions(parent_genomes)
        genomes_dims["n_entries"], genomes_dims["n_loci_alleles"], genomes_dims["n_missing"]
    end
    if n_missing > 0
        throw(
            ArgumentError(
                "We expect no missing values in the allele frequencies of the parents. Please consider filtering them out or imputing.",
            ),
        )
    end
    # If the vector of population sizes per generation is less than the requested number of generations the we replicate
    # Also, add the number of parents at generation 0
    if length(pop_size_per_gen) < n_generations
        pop_size_per_gen =
            vcat(n, repeat(pop_size_per_gen; outer = Int(ceil(n_generations / length(pop_size_per_gen)))))
    else
        pop_size_per_gen = vcat(n, pop_size_per_gen)
    end
    # Set randomisation seed
    rng::TaskLocalRNG = Random.seed!(seed)
    # Extract loci names
    chromosomes_per_locus_allele::Vector{String}, _, _ = loci_alleles(parent_genomes)
    chromosomes_per_locus::Vector{String}, _, loci_ini_idx::Vector{Int64}, loci_fin_idx::Vector{Int64} =
        loci(parent_genomes)
    unique_chromosomes::Vector{String} = unique(chromosomes_per_locus)
    # Instantiate output genomes
    genomes_across_generations::Vector{Genomes} = [parent_genomes]
    for t = 1:n_generations
        # We're using t+1 indexes because we added the number of parents at genration 0
        progeny_genomes = Genomes(; n = pop_size_per_gen[t+1], p = length(parent_genomes.loci_alleles))
        progeny_genomes.entries = [string("progeny_t", t, "_", i) for i = 1:pop_size_per_gen[t+1]]
        progeny_genomes.loci_alleles = parent_genomes.loci_alleles
        push!(genomes_across_generations, progeny_genomes)
    end
    # Iterate across generations
    if verbose
        println("Allele frequency distribution of the $n parents:")
        histallelefreqs(parent_genomes)
        pb = ProgressMeter.Progress(n_generations, desc = "Simulating $n_generations generations: ")
    end
    for t = 2:length(genomes_across_generations)
        # t = 2
        for chr in unique_chromosomes
            # chr = unique_chromosomes[1]
            idx_loci_alleles::Vector{Int64} = findall(chromosomes_per_locus_allele .== chr)
            idx_loci::Vector{Int64} = findall(chromosomes_per_locus .== chr)
            # Extract the allele frequencies from the previous generation
            allele_freqs::Matrix{Float64} = genomes_across_generations[t-1].allele_frequencies[:, idx_loci_alleles]
            μ::Vector{Float64} = mean(allele_freqs; dims = 1)[1, :]
            Σ::Matrix{Float64} = StatsBase.cov(allele_freqs)
            max_iter::Int64 = 10
            iter::Int64 = 1
            while !isposdef(Σ) && (iter < max_iter)
                if iter == 1
                    Σ[diagind(Σ)] .+= 1.0e-12
                end
                Σ[diagind(Σ)] .*= 10.0
                iter += 1
            end
            # Define the multivariate normal distribution
            mvnormal_distribution = Distributions.MvNormal(μ, Σ)
            # Sample the progeny allele frequencies
            progeny_allele_freqs::Matrix{Float64} = rand(rng, mvnormal_distribution, pop_size_per_gen[t])'
            # Restrict allele frequencies between zero and one
            progeny_allele_freqs[progeny_allele_freqs.>1.0] .= 1.0
            progeny_allele_freqs[progeny_allele_freqs.<0.0] .= 0.0
            # Make sure allele frequencies sum up to one for multiallelic loci
            for j in eachindex(loci_ini_idx[idx_loci])
                # j = 1
                # println(j)
                idx_ini = loci_ini_idx[idx_loci][j]
                idx_fin = loci_fin_idx[idx_loci][j]
                a = idx_fin - idx_ini
                sum_of_prev_allele_freqs::Vector{Float64} = fill(0.0, size(progeny_allele_freqs, 1))
                for k = j:(j+a)
                    # k = idx_fin
                    sum_of_prev_allele_freqs = sum_of_prev_allele_freqs + progeny_allele_freqs[:, k]
                end
                idx_overloaded = findall(sum_of_prev_allele_freqs .> 1.0)
                # Rescale so that the allele frequencies sum up to one
                progeny_allele_freqs[idx_overloaded, j:(j+a)] ./= sum_of_prev_allele_freqs[idx_overloaded]
            end
            # Update the progenies' genomes
            genomes_across_generations[t].allele_frequencies[:, idx_loci_alleles] = progeny_allele_freqs
        end
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
        for t = 1:length(genomes_across_generations)
            println("Generation $(t - 1):")
            @show dimensions(genomes_across_generations[t])
            histallelefreqs(genomes_across_generations[t])
        end
    end
    # Output
    genomes_across_generations
end
