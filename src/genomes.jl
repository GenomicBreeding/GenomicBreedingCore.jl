"""
    clone(x::Genomes)::Genomes

Clone a Genomes object

## Example
```jldoctest; setup = :(using GBCore)
julia> genomes = Genomes(n=2, p=2);

julia> copy_genomes = clone(genomes)
Genomes(["", ""], ["", ""], ["", ""], Union{Missing, Float64}[missing missing; missing missing], Bool[1 1; 1 1])
```
"""
function clone(x::Genomes)::Genomes
    y::Genomes = Genomes(n = length(x.entries), p = length(x.loci_alleles))
    y.entries = deepcopy(x.entries)
    y.populations = deepcopy(x.populations)
    y.loci_alleles = deepcopy(x.loci_alleles)
    y.allele_frequencies = deepcopy(x.allele_frequencies)
    y.mask = deepcopy(x.mask)
    y
end


"""
    Base.hash(x::Genomes, h::UInt)::UInt

Hash a Genomes struct using the entries, populations and loci_alleles.
We deliberately excluded the allele_frequencies, and mask for efficiency.

## Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = Genomes(n=2, p=2);

julia> typeof(hash(genomes))
UInt64
```
"""
function Base.hash(x::Genomes, h::UInt)::UInt
    # hash(Genomes, hash(x.entries, hash(x.populations, hash(x.loci_alleles, hash(x.allele_frequencies, hash(x.mask, h))))))
    hash(Genomes, hash(x.entries, hash(x.populations, hash(x.loci_alleles, h))))
end


"""
    Base.:(==)(x::Genomes, y::Genomes)::Bool

Equality of Genomes structs using the hash function defined for Genomes structs.

## Examples
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

Check dimension compatibility of the fields of the Genomes struct

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

Count the number of entries, populations, loci-alleles combination, loci, maximum number of alleles per locus, and number of missing data in the Genomes struct

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
    loci_alleles(genomes::Genomes)::Tuple{Vector{String},Vector{Int64},Vector{String}}

Extract chromosomes, positions, and alleles across loci-allele combinations

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> chromosomes, positions, alleles = loci_alleles(genomes);

julia> length(chromosomes), length(positions), length(alleles)
(3000, 3000, 3000)
```
"""
function loci_alleles(genomes::Genomes)::Tuple{Vector{String},Vector{Int64},Vector{String}}
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    chromosomes::Vector{String} = []
    positions::Vector{Int64} = []
    alleles::Vector{String} = []
    for locus in genomes.loci_alleles
        # locus = genomes.loci_alleles[1]
        locus_ids::Vector{String} = split(locus, '\t')
        push!(chromosomes, locus_ids[1])
        push!(positions, parse(Int64, locus_ids[2]))
        push!(alleles, locus_ids[4])
    end
    return (chromosomes, positions, alleles)
end

"""
    loci(genomes::Genomes)::Tuple{Vector{String},Vector{Int64},Vector{Int64},Vector{Int64}}

Extract chromosome names, positions, start and end indexes of each locus across loci

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> chromosomes, positions, loci_ini_idx, loci_fin_idx = loci(genomes);

julia> length(chromosomes), length(positions), length(loci_ini_idx), length(loci_fin_idx)
(1000, 1000, 1000, 1000)
```
"""
function loci(genomes::Genomes)::Tuple{Vector{String},Vector{Int64},Vector{Int64},Vector{Int64}}
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    chromosomes::Vector{String} = []
    positions::Vector{Int64} = []
    loci_ini_idx::Vector{Int64} = []
    loci_fin_idx::Vector{Int64} = []
    idx::Int64 = 0
    for locus in genomes.loci_alleles
        # locus = genomes.loci_alleles[1]
        idx += 1
        locus_ids::Vector{String} = split(locus, '\t')
        if length(chromosomes) == 0
            push!(chromosomes, locus_ids[1])
            push!(positions, parse(Int64, locus_ids[2]))
            push!(loci_ini_idx, idx)
        else
            if ((chromosomes[end] == locus_ids[1]) && (positions[end] != parse(Int64, locus_ids[2]))) ||
               (chromosomes[end] != locus_ids[1])
                push!(chromosomes, locus_ids[1])
                push!(positions, parse(Int64, locus_ids[2]))
                push!(loci_ini_idx, idx)
                push!(loci_fin_idx, idx - 1)
            end
        end
    end
    if loci_fin_idx[end] < length(genomes.loci_alleles)
        push!(loci_fin_idx, length(genomes.loci_alleles))
    end
    return (chromosomes, positions, loci_ini_idx, loci_fin_idx)
end

"""
    distances(
        genomes::Genomes; 
        distance_metrics::Vector{String}=["euclidean", "correlation", "mad", "rmsd", "χ²"]
        idx_loci_alleles::Union{Nothing, Vector{Int64}} = nothing,
    )::Tuple{Vector{String}, Vector{String}, Dict{String, Matrix{Float64}}}

Estimate pairwise distances between loci_alleles and entries. 
Sparsity leading to less than 2 pairs will yield -Inf values in the resulting matrices. 
Matrices with how many pairs were used to estimate the distance and correlation matrices are also included as well as the locus-allele and entry names.

```jldoctest; setup = :(using GBCore, LinearAlgebra)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> (loci_alleles, entries, dist) = distances(genomes, distance_metrics=["correlation", "χ²"]);

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
)::Tuple{Vector{String},Vector{String},Dict{String,Matrix{Float64}}}
    # genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);
    # distance_metrics = ["euclidean", "correlation", "mad", "rmsd", "χ²"]; idx_loci_alleles = nothing
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
    if t > 1
        y1::Vector{Union{Missing,Float64}} = fill(missing, n)
        y2::Vector{Union{Missing,Float64}} = fill(missing, n)
        counts = fill(0.0, t, t)
        for metric in distance_metrics
            # metric = distance_metrics[1]
            D::Matrix{Float64} = fill(-Inf, t, t)
            for (i, ix) in enumerate(idx_loci_alleles)
                y1 .= genomes.allele_frequencies[:, ix]
                bool1 = .!ismissing.(y1) .&& .!isnan.(y1) .&& .!isinf.(y1)
                for (j, jx) in enumerate(idx_loci_alleles)
                    # i = 1; j = 3; ix = idx_loci_alleles[i]; jx = idx_loci_alleles[j]
                    # Make sure we have no missing, NaN or infinite values
                    y2 .= genomes.allele_frequencies[:, jx]
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
            push!(dimension, "loci_alleles")
            push!(metric_names, metric)
            push!(matrices, D)
        end
        push!(dimension, "loci_alleles")
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
        throw(ErrorException("Genomes struct is too sparse. No distance matrix was calculated."))
    end
    dist::Dict{String,Matrix{Float64}} = Dict()
    for i in eachindex(dimension)
        # i = 4
        key = string(dimension[i], "|", metric_names[i])
        dist[key] = matrices[i]
    end
    (genomes.loci_alleles[idx_loci_alleles], genomes.entries, dist)
end

"""
    plot(genomes::Genomes)::Nothing

Plot allele frequencies

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
        C = dist["loci_alleles|correlation"]
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
    )::Genomes

Slice a Genomes struct by specifing indixes of entries and loci-allele combinations

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> sliced_genomes = slice(genomes, idx_entries=collect(1:10); idx_loci_alleles=collect(1:300));

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
)::Genomes
    # genomes::Genomes = simulategenomes(); idx_entries::Vector{Int64}=sample(1:100, 10); idx_loci_alleles::Vector{Int64}=sample(1:10_000, 1000);
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
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
    for (i1, i2) in enumerate(idx_entries)
        sliced_genomes.entries[i1] = genomes.entries[i2]
        sliced_genomes.populations[i1] = genomes.populations[i2]
        for (j1, j2) in enumerate(idx_loci_alleles)
            if i1 == 1
                sliced_genomes.loci_alleles[j1] = genomes.loci_alleles[j2]
            end
            sliced_genomes.allele_frequencies[i1, j1] = genomes.allele_frequencies[i2, j2]
            sliced_genomes.mask[i1, j1] = genomes.mask[i2, j2]
        end
    end
    ### Check dimensions
    if !checkdims(sliced_genomes)
        throw(DimensionMismatch("Error slicing the genome."))
    end
    # Output
    return sliced_genomes
end


"""
    filter(genomes::Genomes)::Genomes

Filter a Genomes struct using its mask matrix where all rows and columns with at least one false value are excluded

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(verbose=false); genomes.mask[1:10, 42:100] .= false;
    
julia> filtered_genomes = filter(genomes);

julia> size(filtered_genomes.allele_frequencies)
(90, 9941)
```
"""
function Base.filter(genomes::Genomes)::Genomes
    # genomes = simulategenomes(); genomes.mask[1:10, 42:100] .= false;
    idx_entries = findall(mean(genomes.mask, dims = 2)[:, 1] .== 1.0)
    idx_loci_alleles = findall(mean(genomes.mask, dims = 1)[1, :] .== 1.0)
    filtered_genomes::Genomes = slice(genomes, idx_entries = idx_entries; idx_loci_alleles = idx_loci_alleles)
    filtered_genomes
end


"""
    filter(
        genomes::Genomes;
        maf::Float64,
        max_entry_sparsity::Float64 = 0.0,
        max_locus_sparsity::Float64 = 0.0,
        chr_pos_allele_ids::Union{Missing,Vector{String}} = missing,
    )::Genomes

Filter a Genomes struct by minimum allele frequency, entries sparsity, loci sparsity, and/or vector of locus-allele names

# Examples
```jldoctest; setup = :(using GBCore)
julia> genomes = simulategenomes(n=100, l=1_000, n_alleles=4, verbose=false);

julia> filtered_genomes_1 = filter(genomes, 0.1);

julia> filtered_genomes_2 = filter(genomes, 0.1, chr_pos_allele_ids=genomes.loci_alleles[1:1000]);

julia> size(genomes.allele_frequencies)
(100, 3000)

julia> size(filtered_genomes_1.allele_frequencies)
(100, 1236)

julia> size(filtered_genomes_2.allele_frequencies)
(100, 388)
```
"""
function Base.filter(
    genomes::Genomes,
    maf::Float64;
    max_entry_sparsity::Float64 = 0.0,
    max_locus_sparsity::Float64 = 0.0,
    chr_pos_allele_ids::Union{Missing,Vector{String}} = missing,
)::Genomes
    # genomes::Genomes = simulategenomes(sparsity=0.01, seed=123456); maf=0.01; max_entry_sparsity=0.1; max_locus_sparsity = 0.25
    # chr_pos_allele_ids = sample(genomes.loci_alleles, Int(floor(0.5*length(genomes.loci_alleles)))); sort!(chr_pos_allele_ids)
    if !checkdims(genomes)
        throw(ArgumentError("Genomes struct is corrupted."))
    end
    if (maf < 0.0) || (maf > 1.0)
        throw(ArgumentError("We accept `maf` from 0.0 to 1.0."))
    end
    # Filter by entry sparsity, locus sparsity and minimum allele frequency thresholds
    entry_sparsities::Array{Float64,1} = fill(0.0, length(genomes.entries))
    mean_frequencies::Array{Float64,1} = fill(0.0, length(genomes.loci_alleles))
    locus_sparsities::Array{Float64,1} = fill(0.0, length(genomes.loci_alleles))
    for i in eachindex(entry_sparsities)
        entry_sparsities[i] = mean(Float64.(ismissing.(genomes.allele_frequencies[i, :])))
    end
    for j in eachindex(mean_frequencies)
        mean_frequencies[j] = mean(skipmissing(genomes.allele_frequencies[:, j]))
        locus_sparsities[j] = mean(Float64.(ismissing.(genomes.allele_frequencies[:, j])))
    end
    idx_entries::Array{Int64,1} = findall((entry_sparsities .<= max_entry_sparsity))
    idx_loci_alleles::Array{Int64,1} = findall(
        (mean_frequencies .>= maf) .&& (mean_frequencies .<= (1.0 - maf)) .&& (locus_sparsities .<= max_locus_sparsity),
    )
    # Check if we are retaining any entries and loci
    if (length(idx_entries) == 0) && (length(idx_loci_alleles) == 0)
        throw(
            ErrorException(
                string(
                    "All entries and loci filtered out at maximum entry sparsity = ",
                    max_entry_sparsity,
                    ", minimum allele frequencies (maf) = ",
                    maf,
                    ", and maximum locus sparsity = ",
                    max_locus_sparsity,
                    ".",
                ),
            ),
        )
    end
    if length(idx_entries) == 0
        throw(ErrorException(string("All entries filtered out at maximum entry sparsity = ", max_entry_sparsity, ".")))
    end
    if length(idx_loci_alleles) == 0
        throw(
            ErrorException(
                string(
                    "All loci filtered out at minimum allele frequencies (maf) = ",
                    maf,
                    ", and maximum locus sparsity = ",
                    max_locus_sparsity,
                    ".",
                ),
            ),
        )
    end
    # Are we filtering using a list of loci-allele combination names?
    if !ismissing(chr_pos_allele_ids) && (length(chr_pos_allele_ids) > 0)
        requested_chr, requested_pos, requested_all = begin
            chr::Vector{String} = fill("", length(chr_pos_allele_ids))
            pos::Vector{Int64} = fill(0, length(chr_pos_allele_ids))
            ale::Vector{String} = fill("", length(chr_pos_allele_ids))
            ids::Vector{String} = []
            for i in eachindex(chr_pos_allele_ids)
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
                    parse(Int64, ids[2])
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
            end
            chr, pos, ale
        end
        chromosomes, positions, alleles = loci_alleles(genomes)
        idx_loci_alleles = filter(
            i -> (
                sum(
                    (chromosomes[i] .== requested_chr) .&&
                    (positions[i] .== requested_pos) .&&
                    (alleles[i] .== requested_all),
                ) > 0
            ),
            idx_loci_alleles,
        )
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
    end
    # Output
    filtered_genomes::Genomes = slice(genomes, idx_entries = idx_entries; idx_loci_alleles = idx_loci_alleles)
    filtered_genomes
end


"""
    merge(
        genomes::Genomes,
        other::Genomes;
        conflict_resolution::Tuple{Float64,Float64} = (0.5, 0.5),
        verbose::Bool = true,
    )::Genomes

Merge two Genomes structs using a tuple of conflict resolution weights

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

Merge a Genomes struct with a Phenomes struct using union or intersection

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
