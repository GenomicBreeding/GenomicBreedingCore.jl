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

julia> chroms_uniq, LDs_all_chroms = estimateld(genomes);

julia> chrom, pos, allele = loci_alleles(genomes);

julia> mock_scaffolds = divideintomockscaffolds(genomes, max_n_loci_per_chrom=100);

julia> mock_scaffolds_uniq, LDs_mock_scaffolds = estimateld(genomes, chromosomes=mock_scaffolds);

julia> length(LDs_all_chroms) == length(chroms_uniq) == length(unique(chrom))
true

julia> length(LDs_mock_scaffolds) == length(mock_scaffolds_uniq) == Int(length(chrom) / 100)
true
```
"""
function estimateld(
    genomes::Genomes;
    chromosomes::Union{Nothing,Vector{String}} = nothing,
    verbose::Bool = false,
)::Tuple{Vector{String},Vector{Matrix{Float64}}}
    # genomes = simulategenomes(n=10, sparsity=0.3, verbose=false); chromosomes = nothing; verbose = true
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    # Define the chromosomes from which LD will be estimated separately
    chromosomes = if isnothing(chromosomes)
        [x[1] for x in split.(genomes.loci_alleles, "\t")]
    else
        chromosomes
    end
    chroms_uniq = sort(unique(chromosomes))
    # Check if the expected LD output JLD2 file already exists from previous runs
    fname_LD_matrix_final = string("LD_matrices_per_chrom-", hash(genomes), ".jld2")
    if isfile(fname_LD_matrix_final)
        if verbose
            println(string("Outpult LDs JLD2 file found. Loading ", fname_LD_matrix_final, "."))
        end
        return (chroms_uniq, JLD2.load(fname_LD_matrix_final)["LDs"])
    end
    # Instantiate the output vector of LD matrices
    LDs = Vector{Matrix{Float64}}(undef, length(chroms_uniq))
    # Check if we can resume previous run in the same directory
    fnames = readdir()
    fnames = fnames[(.!isnothing.(
        match.(Regex("^LD_matrix-"), fnames)
    )).&&(.!isnothing.(match.(Regex(".tmp.jld2\$"), fnames)))]
    # Define the chromosomes with previously estimated LD matrices
    chroms_finished::Vector{String} = if length(fnames) > 0
        # Resume
        chroms = [replace(split(x, "-")[end], ".tmp.jld2" => "") for x in fnames]
        chroms_finished = if length(unique(chroms)) != length(chroms)
            # Start anew if we have duplicate LD files for the same chromosomes in the current directory just to be sure
            []
        else
            chroms
        end
        for i in eachindex(fnames)
            fname = fnames[i]
            idx = findall(chroms_uniq .== chroms_finished[i])[1]
            LDs[idx] = JLD2.load(fname)["loci_alleles|correlation"]
        end
        chroms_finished
    else
        # Start anew
        []
    end
    if verbose
        println(
            string(
                "There are ",
                length(chroms_finished),
                " chromosome with LD matrices previously computed (",
                length(chroms_finished),
                " of ",
                length(LDs),
                ")",
            ),
        )
    end
    # Iteratively estimate LD per chromosome where LD estimation is itself multi-threaded
    for (i, chrom) in enumerate(chroms_uniq)
        # i = 3; chrom = chromosomes[i]
        if verbose
            println(string("Esimtating LD of chromosome: ", chrom, " (", i, " of ", length(LDs), ")"))
        end
        if chrom ∈ chroms_finished
            if verbose
                println(string("Using previously computed LD of chromosome: ", chrom))
            end
            continue
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
        JLD2.save(fname_LD_matrix, corr)
    end
    # Save output as backup
    JLD2.save(fname_LD_matrix_final, Dict("LDs" => LDs))
    # Clean-up
    fnames = readdir()
    fnames = fnames[(.!isnothing.(
        match.(Regex("^LD_matrix-"), fnames)
    )).&&(.!isnothing.(match.(Regex(".tmp.jld2\$"), fnames)))]
    rm.(fnames)
    # Output
    (chroms_uniq, LDs)
end

function estimatedistances(
    genomes::Genomes;
    LD::Matrix{Float64},
    idx_focal_locus::Int64,
    idx_loci_alleles_per_chrom::Vector{Int64},
    min_loci_corr::Float64,
    min_l_loci::Int64,
    verbose::Bool = false,
)::Matrix{Float64}
    # Define the global indexes of the most linked loci to the focal locus
    bool_linked_loci = LD[:, idx_focal_locus] .>= min_loci_corr
    idx_linked_loci = if sum(bool_linked_loci) < min_l_loci
        idx_loci_sort = sortperm(LD[:, idx_focal_locus], rev = true)
        sort(idx_loci_alleles_per_chrom[idx_loci_sort[1:min_l_loci]])
    else
        idx_loci_alleles_per_chrom[bool_linked_loci]
    end
    # Estimate pairwise distances between entries using mean absolute difference in allele frequencies of loci most linked to the focal locus
    (_loci_alleles, _entries, mad) = distances(
        genomes,
        distance_metrics = ["mad"],
        idx_loci_alleles = idx_linked_loci,
        include_loci_alleles = false,
        include_counts = false,
        include_entries = true,
        verbose = verbose,
    )
    mad["entries|mad"]
end

function knni(;
    qs::Vector{Float64},
    d::Vector{Float64},
    max_entries_dist::Float64 = 0.1,
    min_k_entries::Int64 = 2,
)::Float64
    if length(qs) != length(d)
        throw(
            ArgumentError(
                "The non-missing allele frequencies (`qs`) " *
                "at the focal locus is incompatible with the distances vector (`d`). " *
                "We expect `qs = genomes.allele_frequencies[idx_entries_not_missing, j]` and " *
                "`d = D[i, idx_entries_not_missing].",
            ),
        )
    end
    # Find the allele frequencies of the nearest neighbours with non-missing data at the focal locus
    bool_nearest_entries = d .<= max_entries_dist
    idx_nearest_entries = if sum(bool_nearest_entries) < min_k_entries
        idx_entries_sort = sortperm(d, rev = false)
        idx_entries_sort[1:minimum([min_k_entries, length(idx_entries_sort)])]
    else
        findall(bool_nearest_entries)
    end
    # Impute missing data using the weighted means of these nearest neighbours
    Σd = sum(d[idx_nearest_entries] .+ eps(Float64))
    d = if isinf(Σd)
        1.00 / length(idx_nearest_entries)
    else
        # Weigh each nearest neighbour allele frequency based on their distances
        (d[idx_nearest_entries] .+ eps(Float64)) ./ sum(d[idx_nearest_entries] .+ eps(Float64))
    end
    q̄ = sum(qs[idx_nearest_entries] .* d)
    q̄
end

function knnioptim(
    genomes::Genomes;
    j::Int64,
    idx_focal_locus::Int64,
    idx_loci_alleles_per_chrom::Vector{Int64},
    idx_entries_not_missing::Vector{Int64},
    LD::Matrix{Float64},
    n_reps::Int64 = 2,
    optim_n::Int64 = 10,
    min_l_loci::Int64 = 10,
    min_k_entries::Int64 = 2,
    verbose::Bool = false,
)::Dict{String,Float64}
    if length(idx_entries_not_missing) < 2
        throw(ArgumentError("We expect at least 2 entries with non-missing data at the focal locus."))
    end
    optim_min_loci_corr = range(0.0, 1.0, length = optim_n)
    optim_max_entries_dist = range(0.0, 1.0, length = optim_n)
    optim_params = Matrix{Float64}(undef, optim_n^2, 3) # each collumn corresponds to (1) min_loci_corr, (2) max_entries_dist, (3) mae
    for idx_ld in eachindex(optim_min_loci_corr)
        # idx_ld = 5
        min_loci_corr = optim_min_loci_corr[idx_ld]
        # Estimate pairwise distances between entries using mean absolute difference in allele frequencies of loci most linked to the focal locus
        D = estimatedistances(
            genomes,
            LD = LD,
            idx_focal_locus = idx_focal_locus,
            idx_loci_alleles_per_chrom = idx_loci_alleles_per_chrom,
            min_loci_corr = min_loci_corr,
            min_l_loci = min_l_loci,
            verbose = false,
        )
        for idx_dist in eachindex(optim_max_entries_dist)
            # idx_dist = 5
            max_entries_dist = optim_max_entries_dist[idx_dist]
            mae::Float64 = 0.0
            idx_entries_not_missing_and_sim_missing = sample(idx_entries_not_missing, minimum([length(idx_entries_not_missing)-1, n_reps]), replace = false)
            idx_entries_not_missing_and_not_sim_missing = idx_entries_not_missing[.!(
                x ∈ idx_entries_not_missing_and_sim_missing for x in idx_entries_not_missing
            )]
            for i in idx_entries_not_missing_and_sim_missing
                # i = idx_entries_not_missing_and_sim_missing[1]
                qs::Vector{Float64} = genomes.allele_frequencies[idx_entries_not_missing_and_not_sim_missing, j]
                d = D[i, idx_entries_not_missing_and_not_sim_missing]
                q̄ = knni(qs = qs, d = d, max_entries_dist = max_entries_dist, min_k_entries = min_k_entries)
                q = genomes.allele_frequencies[i, j]
                mae += abs(q - q̄)
            end
            mae /= length(idx_entries_not_missing_and_sim_missing)
            optim_params[(idx_ld-1)*optim_n+idx_dist, :] = [min_loci_corr, max_entries_dist, mae]
        end
    end
    # Output
    optim_idx = argmin(optim_params[:, 3])
    params = Dict(
        "min_loci_corr" => optim_params[optim_idx, 1],
        "max_entries_dist" => optim_params[optim_idx, 2],
        "mae" => optim_params[optim_idx, 3],
    )
    if verbose
        MAE = Matrix{Float64}(undef, optim_n, optim_n)
        for idx_ld in eachindex(optim_min_loci_corr)
            for idx_dist in eachindex(optim_max_entries_dist)
                MAE[idx_ld, idx_dist] = optim_params[(idx_ld-1)*optim_n+idx_dist, 3]
            end
        end
        UnicodePlots.heatmap(MAE, xfact = 1, yfact = 1, xlabel = "Distance (mad)", ylabel = "LD (corr)", zlabel = "MAE")
        @show params
    end
    params
end

function impute(
    genomes::Genomes;
    max_n_loci_per_chrom::Int64 = 100_000,
    n_reps::Int64 = 2,
    optim_n::Int64 = 10,
    min_l_loci::Int64 = 10,
    min_k_entries::Int64 = 2,
    verbose::Bool = false,
)::Tuple{Genomes, Float64}
    # genomes = simulategenomes(n=10, sparsity=0.3, verbose=false); max_n_loci_per_chrom = 100; n_reps = 2; optim_n = 10; min_l_loci = 10; min_k_entries = 2; verbose = true
    # Check input arguments
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted."))
    end
    n, p = size(genomes.allele_frequencies)
    if n < 2
        throw(ArgumentError("The number of entries in the genomes struct is less than 2."))
    end
    if p / max_n_loci_per_chrom < 10
        throw(
            ArgumentError(
                string(
                    "The number of loci per chromosome is less than 10, i.e. ",
                    Int64(p / max_n_loci_per_chrom),
                    ". Please consider increasing `max_n_loci_per_chrom` (currently set to ",
                    max_n_loci_per_chrom,
                    ").",
                ),
            ),
        )
    end
    if (n_reps < 1) || (n_reps > n)
        throw(ArgumentError("The `n_reps` is expected to be between 1 and $n."))
    end
    if (min_l_loci < 10) || (min_l_loci > Int64(p / max_n_loci_per_chrom))
        throw(
            ArgumentError(
                string("The `min_l_loci` is expected to be between 10 and ", Int64(p / max_n_loci_per_chrom), "."),
            ),
        )
    end
    if (min_k_entries < 2) || (min_k_entries > n)
        throw(ArgumentError("The `min_k_entries` is expected to be between 2 and $n."))
    end
    # Instantiate the output Genomes struct
    out::Genomes = clone(genomes)
    # Divide the allele frequencies into mock scaffolds if we have more than 100,000 loci per scaffold for at least 1 scaffold
    chromosomes, _positions, _alleles = loci_alleles(genomes)
    max_m = 1
    min_m = Inf
    for chrom in unique(chromosomes)
        m = sum(chromosomes .== chrom)
        max_m = max_m < m ? m : max_m
        min_m = min_m > m ? m : min_m
    end
    chromosomes = if max_m > max_n_loci_per_chrom
        # # Divide the loci into mock scaffolds
        divideintomockscaffolds(genomes, max_n_loci_per_chrom = max_n_loci_per_chrom, verbose = verbose)
    else
        chromosomes
    end
    # Estimate linkage disequilibrium (LD) between loci using Pearson's correlation per chromosome
    chroms_uniq, LDs = estimateld(genomes, chromosomes = chromosomes, verbose = verbose)
    # Impute per locus-allele per chromosome
    if verbose
        pb = ProgressMeter.Progress(length(p), desc="Imputing using the imputef algorithm")
    end
    mae_expected::Vector{Union{Missing, Float64}} = fill(missing, p)
    for k in eachindex(LDs)
        # k = 1
        LD = LDs[k]
        chrom = chroms_uniq[k]
        idx_loci_alleles_per_chrom = findall(chromosomes .== chrom)
        Threads.@threads for idx_focal_locus in eachindex(idx_loci_alleles_per_chrom)
            # idx_focal_locus = 1 # Local index of the focal locus in the current chromosome
            # Define the global index of the focal locus
            j = idx_loci_alleles_per_chrom[idx_focal_locus]
            # Identify entries with missing data at the focal locus
            bool_entries_missing = ismissing.(genomes.allele_frequencies[:, j])
            if sum(bool_entries_missing) == 0
                continue
            end
            idx_entries_missing = findall(bool_entries_missing)
            idx_entries_not_missing = findall(.!bool_entries_missing)
            # Skip imputation if the number of non-missing entries at the focal locus is less than `min_k_entries`
            if length(idx_entries_not_missing) < min_k_entries
                continue
            end
            # Optimise for `min_loci_corr` and `max_entries_dist`
            params = knnioptim(
                genomes,
                j = j,
                idx_focal_locus = idx_focal_locus,
                idx_loci_alleles_per_chrom = idx_loci_alleles_per_chrom,
                idx_entries_not_missing = idx_entries_not_missing,
                LD = LD,
                n_reps = n_reps,
                optim_n = optim_n,
                min_l_loci = min_l_loci,
                min_k_entries = min_k_entries,
                verbose = false,
            )
            # Compute the distances between entries using the optimum `min_loci_corr`
            D = estimatedistances(
                genomes,
                LD = LD,
                idx_focal_locus = idx_focal_locus,
                idx_loci_alleles_per_chrom = idx_loci_alleles_per_chrom,
                min_loci_corr = params["min_loci_corr"],
                min_l_loci = min_l_loci,
                verbose = false,
            )
            # Impute missing data using the weighted means using the optimum `max_entries_dist`
            for i in idx_entries_missing
                # i = idx_entries_missing[1]
                qs::Vector{Float64} = genomes.allele_frequencies[idx_entries_not_missing, j]
                d = D[i, idx_entries_not_missing]
                q̄ = knni(qs = qs, d = d, max_entries_dist = params["max_entries_dist"], min_k_entries = min_k_entries)
                out.allele_frequencies[i, j] = q̄
            end
            # Update the expected MAE
            mae_expected[j] = params["mae"]
            if verbose
                ProgressMeter.next!(pb)
            end
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    # Output
    if !checkdims(out)
        throw(ErrorException("Error imputing"))
    end
    idx_non_missing_mae = findall(.!ismissing.(mae_expected))
    (
        out,
        mean(mae_expected[idx_non_missing_mae])
    )
end
