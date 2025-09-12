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
1. If the loci_alleles are identical and there are no overlapping entries, performs a quick merge:
   - Concatenates entries, populations, allele frequencies, and mask without conflict resolution.
2. Combines unique entries and loci_alleles from both input structs
3. Resolves population conflicts by concatenating conflicting values
4. For overlapping entries and loci:
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
    # Check if quick merging of entries is possible
    if (sort(genomes.loci_alleles) == sort(other.loci_alleles)) &&
       (length(intersect(genomes.entries, other.entries)) == 0)
        # If the loci_alleles are the same and there are no common entries, we can quickly merge
        if verbose
            println("Quick merging of 2 Genomes structs with no common entries and identical loci_alleles.")
        end
        out = Genomes(n = length(genomes.entries) + length(other.entries), p = length(genomes.loci_alleles))
        out.entries = vcat(genomes.entries, other.entries)
        out.populations = vcat(genomes.populations, other.populations)
        idx_1 = sortperm(genomes.loci_alleles)
        idx_2 = sortperm(other.loci_alleles)
        out.loci_alleles = genomes.loci_alleles[idx_1]
        out.allele_frequencies = vcat(genomes.allele_frequencies[:, idx_1], other.allele_frequencies[:, idx_2])
        out.mask = vcat(genomes.mask[:, idx_1], other.mask[:, idx_2])
        if !checkdims(out)
            throw(ErrorException("Error merging the 2 Genomes structs."))
        end
        return out
    end
    # Check if quick merging of loci is possible
    if (sort(genomes.entries) == sort(other.entries)) &&
       (length(intersect(genomes.loci_alleles, other.loci_alleles)) == 0)
        # If the entries are the same and there are no common loci_alleles, we can quickly merge
        if verbose
            println("Quick merging of 2 Genomes structs with no common loci_alleles and identical entries.")
        end
        out = Genomes(n = length(genomes.entries), p = length(genomes.loci_alleles) + length(other.loci_alleles))
        out.loci_alleles = vcat(genomes.loci_alleles, other.loci_alleles)
        out.populations = genomes.populations
        idx_1 = sortperm(genomes.entries)
        idx_2 = sortperm(other.entries)
        out.entries = genomes.entries
        out.allele_frequencies = hcat(genomes.allele_frequencies[idx_1, :], other.allele_frequencies[idx_2, :])
        out.mask = hcat(genomes.mask[idx_1, :], other.mask[idx_2, :])
        if !checkdims(out)
            throw(ErrorException("Error merging the 2 Genomes structs."))
        end
        return out
    end
    # Instantiate the merged Genomes struct
    entries::Vector{String} = genomes.entries ∪ other.entries
    loci_alleles::Vector{String} = genomes.loci_alleles ∪ other.loci_alleles
    n = length(entries)
    p = length(loci_alleles)
    out::Genomes = Genomes(n = n, p = p)
    out.entries = entries
    out.loci_alleles = loci_alleles
    # Vectors of booleans to speed up the merging process
    bool_entries_1::Vector{Bool} = begin
        x = zeros(Bool, n)
        x[indexin(genomes.entries, entries)] .= true
        x
    end
    bool_entries_2::Vector{Bool} = begin
        x = zeros(Bool, n)
        x[indexin(other.entries, entries)] .= true
        x
    end
    bool_loci_alleles_1::Vector{Bool} = begin
        x = zeros(Bool, p)
        x[indexin(genomes.loci_alleles, loci_alleles)] .= true
        x
    end
    bool_loci_alleles_2::Vector{Bool} = begin
        x = zeros(Bool, p)
        x[indexin(other.loci_alleles, loci_alleles)] .= true
        x
    end
    # Vectors of indices in both genomes and counters to complements the vectors of booleans above during merging
    idx_entries_1_in_1::Vector{Int} = filter(x -> !isnothing(x), indexin(entries, genomes.entries))
    idx_entries_2_in_2::Vector{Int} = filter(x -> !isnothing(x), indexin(entries, other.entries))
    idx_loci_alleles_1_in_1::Vector{Int} = filter(x -> !isnothing(x), indexin(loci_alleles, genomes.loci_alleles))
    idx_loci_alleles_2_in_2::Vector{Int} = filter(x -> !isnothing(x), indexin(loci_alleles, other.loci_alleles))
    counter_entries_1::Int64 = 0
    counter_entries_2::Int64 = 0
    # Merge and resolve conflicts in allele frequencies and mask
    if verbose
        pb = ProgressMeter.Progress(length(entries) * length(loci_alleles); desc = "Merging 2 Genomes structs: ")
    end
    @inbounds for i = 1:n
        # i = 50
        i_1 = if bool_entries_1[i]
            counter_entries_1 += 1
            idx_entries_1_in_1[counter_entries_1]
        else
            0
        end
        i_2 = if bool_entries_2[i]
            counter_entries_2 += 1
            idx_entries_2_in_2[counter_entries_2]
        else
            0
        end
        out.populations[i] = if bool_entries_1[i] && bool_entries_2[i]
            join(unique([genomes.populations[i_1], other.populations[i_2]]), ", ")
        elseif bool_entries_1[i]
            genomes.populations[i_1]
        else
            other.populations[i_2]
        end
        counter_loci_alleles_1::Int64 = 0
        counter_loci_alleles_2::Int64 = 0
        @inbounds for j = 1:p
            # j = 1
            # j = 2_500
            j_1 = if bool_loci_alleles_1[j]
                counter_loci_alleles_1 += 1
                idx_loci_alleles_1_in_1[counter_loci_alleles_1]
            else
                0
            end
            j_2 = if bool_loci_alleles_2[j]
                counter_loci_alleles_2 += 1
                idx_loci_alleles_2_in_2[counter_loci_alleles_2]
            else
                0
            end
            out.allele_frequencies[i, j] =
                if bool_entries_1[i] && bool_entries_2[i] && bool_loci_alleles_1[j] && bool_loci_alleles_2[j]
                    (genomes.allele_frequencies[i_1, j_1] * conflict_resolution[1]) +
                    (other.allele_frequencies[i_2, j_2] * conflict_resolution[2])
                elseif bool_entries_1[i] && bool_loci_alleles_1[j]
                    genomes.allele_frequencies[i_1, j_1]
                elseif bool_entries_2[i] && bool_loci_alleles_2[j]
                    other.allele_frequencies[i_2, j_2]
                else
                    missing
                end
            if verbose
                next!(pb)
            end
        end
    end
    if verbose
        finish!(pb)
    end
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
