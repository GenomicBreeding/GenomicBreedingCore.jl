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
```jldoctest; setup = :(using GenomicBreedingCore)
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
        throw(ArgumentError("Both Genomes structs are corrupted ☹."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The first Genomes struct is corrupted ☹."))
    end
    if !checkdims(other)
        throw(ArgumentError("The second Genomes struct is corrupted ☹."))
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
```jldoctest; setup = :(using GenomicBreedingCore)
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
        throw(ArgumentError("The Genomes and Phenomes structs are corrupted ☹."))
    end
    if !checkdims(genomes)
        throw(ArgumentError("The Genomes struct is corrupted ☹."))
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted ☹."))
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
