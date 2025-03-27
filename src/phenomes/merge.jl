"""
    merge(
        phenomes::Phenomes,
        other::Phenomes;
        conflict_resolution::Tuple{Float64,Float64} = (0.5, 0.5),
        verbose::Bool = true
    )::Phenomes

Merge two `Phenomes` structs into a single combined struct, handling overlapping entries and traits.

# Arguments
- `phenomes::Phenomes`: The first Phenomes struct to merge
- `other::Phenomes`: The second Phenomes struct to merge
- `conflict_resolution::Tuple{Float64,Float64}`: Weights for resolving conflicts between overlapping values (must sum to 1.0)
- `verbose::Bool`: Whether to display a progress bar during merging

# Returns
- `Phenomes`: A new merged Phenomes struct containing all entries and traits from both input structs

# Details
The merge operation combines:
- All unique entries from both structs
- All unique traits from both structs
- Phenotype values and masks, using weighted averaging for conflicts
- Population information, marking conflicts with a "CONFLICT" prefix

For overlapping entries and traits:
- Identical values are preserved as-is
- Different values are combined using the weights specified in `conflict_resolution`
- Missing values are handled by using the available non-missing value
- Population conflicts are marked in the format "CONFLICT (pop1, pop2)"

# Throws
- `ArgumentError`: If either Phenomes struct is corrupted (invalid dimensions)
- `ArgumentError`: If conflict_resolution weights don't sum to 1.0 or aren't a 2-tuple
- `ErrorException`: If the merging operation produces an invalid result

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> all = Phenomes(n=10, t=3); all.entries = string.("entry_", 1:10); all.traits = ["A", "B", "C"]; all.phenotypes = rand(10,3);

julia> phenomes = slice(all, idx_entries=collect(1:7), idx_traits=[1,2]);

julia> other = slice(all, idx_entries=collect(5:10), idx_traits=[2,3]);

julia> merged_phenomes = merge(phenomes, other, conflict_resolution=(0.75, 0.25), verbose=false);

julia> size(merged_phenomes.phenotypes)
(10, 3)

julia> sum(ismissing.(merged_phenomes.phenotypes))
7
```
"""
function Base.merge(
    phenomes::Phenomes,
    other::Phenomes;
    conflict_resolution::Tuple{Float64,Float64} = (0.5, 0.5),
    verbose::Bool = true,
)::Phenomes
    # Check arguments
    if !checkdims(phenomes) && !checkdims(other)
        throw(ArgumentError("Both Phenomes structs are corrupted."))
    end
    if !checkdims(phenomes)
        throw(ArgumentError("The first Phenomes struct is corrupted."))
    end
    if !checkdims(other)
        throw(ArgumentError("The second Phenomes struct is corrupted."))
    end
    if (length(conflict_resolution) != 2) && (sum(conflict_resolution) != 1.00)
        throw(ArgumentError("We expect `conflict_resolution` 2 be a 2-item tuple which sums up to exactly 1.00."))
    end
    # Instantiate the merged Phenomes struct
    entries::Vector{String} = phenomes.entries ∪ other.entries
    populations::Vector{String} = fill("", length(entries))
    traits::Vector{String} = phenomes.traits ∪ other.traits
    phenotypes::Matrix{Union{Missing,Float64}} = fill(missing, (length(entries), length(traits)))
    mask::Matrix{Bool} = fill(false, (length(entries), length(traits)))
    out::Phenomes = Phenomes(n = length(entries), t = length(traits))
    # Merge and resolve conflicts in allele frequencies and mask
    if verbose
        pb = ProgressMeter.Progress(length(entries) * length(traits); desc = "Merging 2 Phenomes structs: ")
    end
    idx_entry_1::Vector{Int} = []
    idx_entry_2::Vector{Int} = []
    bool_entry_1::Bool = false
    bool_entry_2::Bool = false
    idx_trait_1::Vector{Int} = []
    idx_trait_2::Vector{Int} = []
    bool_trait_1::Bool = false
    bool_trait_2::Bool = false
    for (i, entry) in enumerate(entries)
        # entry = entries[i]
        idx_entry_1 = findall(phenomes.entries .== entry)
        idx_entry_2 = findall(other.entries .== entry)
        # We expect a maximum of 1 match per entry as we checked the Phenomes structs
        bool_entry_1 = length(idx_entry_1) > 0
        bool_entry_2 = length(idx_entry_2) > 0
        if bool_entry_1 && bool_entry_2
            if phenomes.populations[idx_entry_1[1]] == other.populations[idx_entry_2[1]]
                populations[i] = phenomes.populations[idx_entry_1[1]]
            else
                populations[i] = string(
                    "CONFLICT (",
                    phenomes.populations[idx_entry_1[1]]...,
                    ", ",
                    other.populations[idx_entry_2[1]]...,
                    ")",
                )
            end
        elseif bool_entry_1
            populations[i] = phenomes.populations[idx_entry_1[1]]
        elseif bool_entry_2
            populations[i] = other.populations[idx_entry_2[1]]
        else
            continue # should never happen
        end
        for (j, trait) in enumerate(traits)
            # trait = traits[j]
            # We expect 1 locus-allele match as we checked the Phenomes structs
            idx_trait_1 = findall(phenomes.traits .== trait)
            idx_trait_2 = findall(other.traits .== trait)
            bool_trait_1 = length(idx_trait_1) > 0
            bool_trait_2 = length(idx_trait_2) > 0
            if bool_entry_1 && bool_trait_1 && bool_entry_2 && bool_trait_2
                q_1 = phenomes.phenotypes[idx_entry_1[1], idx_trait_1[1]]
                q_2 = other.phenotypes[idx_entry_2[1], idx_trait_2[1]]
                m_1 = phenomes.mask[idx_entry_1[1], idx_trait_1[1]]
                m_2 = other.mask[idx_entry_2[1], idx_trait_2[1]]
                if skipmissing(q_1) == skipmissing(q_2)
                    phenotypes[i, j] = q_1
                    mask[i, j] = m_1
                else
                    if !ismissing(q_1) && !ismissing(q_2)
                        phenotypes[i, j] = sum((q_1, q_2) .* conflict_resolution)
                    elseif !ismissing(q_1)
                        phenotypes[i, j] = q_1
                    else
                        phenotypes[i, j] = q_2
                    end
                    mask[i, j] = Bool(round(sum((m_1, m_2) .* conflict_resolution)))
                end
            elseif bool_entry_1 && bool_trait_1
                phenotypes[i, j] = phenomes.phenotypes[idx_entry_1[1], idx_trait_1[1]]
                mask[i, j] = phenomes.mask[idx_entry_1[1], idx_trait_1[1]]
            elseif bool_entry_2 && bool_trait_2
                phenotypes[i, j] = other.phenotypes[idx_entry_2[1], idx_trait_2[1]]
                mask[i, j] = other.mask[idx_entry_2[1], idx_trait_2[1]]
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
    out.traits = traits
    out.phenotypes = phenotypes
    out.mask = mask
    if !checkdims(out)
        throw(ErrorException("Error merging the 2 Phenomes structs."))
    end
    out
end

"""
    tabularise(phenomes::Phenomes)::DataFrame

Convert a `Phenomes` struct into a tabular format as a `DataFrame`.

The resulting DataFrame contains the following columns:
- `id`: Integer index for each entry
- `entries`: Entry identifiers
- `populations`: Population assignments
- Additional columns for each trait in `phenomes.traits`

# Arguments
- `phenomes::Phenomes`: A valid Phenomes struct containing phenotypic data

# Returns
- `DataFrame`: A DataFrame with entries as rows and traits as columns

# Throws
- `ArgumentError`: If the Phenomes struct dimensions are inconsistent

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = fill(0.0, 10,3);

julia> tabularise(phenomes)
10×6 DataFrame
 Row │ id     entries   populations  A         B         C        
     │ Int64  String    String       Float64?  Float64?  Float64? 
─────┼────────────────────────────────────────────────────────────
   1 │     1  entry_1   pop_1             0.0       0.0       0.0
   2 │     2  entry_2   pop_1             0.0       0.0       0.0
   3 │     3  entry_3   pop_1             0.0       0.0       0.0
   4 │     4  entry_4   pop_1             0.0       0.0       0.0
   5 │     5  entry_5   pop_1             0.0       0.0       0.0
   6 │     6  entry_6   pop_1             0.0       0.0       0.0
   7 │     7  entry_7   pop_1             0.0       0.0       0.0
   8 │     8  entry_8   pop_1             0.0       0.0       0.0
   9 │     9  entry_9   pop_1             0.0       0.0       0.0
  10 │    10  entry_10  pop_1             0.0       0.0       0.0
```
"""
function tabularise(phenomes::Phenomes)::DataFrame
    # phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3);
    if !checkdims(phenomes)
        throw(ArgumentError("Phenomes struct is corrupted."))
    end
    df_ids::DataFrame =
        DataFrame(; id = 1:length(phenomes.entries), entries = phenomes.entries, populations = phenomes.populations)
    df_phe::DataFrame = DataFrame(phenomes.phenotypes, :auto)
    rename!(df_phe, phenomes.traits)
    df_phe.id = 1:length(phenomes.entries)
    df = innerjoin(df_ids, df_phe; on = :id)
    return df
end

"""
    @stringevaluation(x)

Parse and evaluate a string expression at compile time.

# Arguments
- `x`: A string containing a Julia expression to be parsed.

# Returns
- The parsed expression as an `Expr` object ready for evaluation.
"""
macro stringevaluation(x)
    Meta.parse(string("$(x)"))
end

"""
    addcompositetrait(phenomes::Phenomes; composite_trait_name::String, formula_string::String)::Phenomes

Create a new composite trait by combining existing traits using mathematical operations.

# Arguments
- `phenomes::Phenomes`: A Phenomes struct containing the original trait data
- `composite_trait_name::String`: Name for the new composite trait
- `formula_string::String`: Mathematical formula describing how to combine existing traits. 
  Supports traits as variables and the following operations:
  * Basic arithmetic: +, -, *, /, ^, %
  * Functions: abs(), sqrt(), log(), log2(), log10()
  * Parentheses for operation precedence

# Returns
- `Phenomes`: A new Phenomes struct with the composite trait added as the last column

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3);

julia> phenomes_new = addcompositetrait(phenomes, composite_trait_name = "some_wild_composite_trait", formula_string = "A");

julia> phenomes_new.phenotypes[:, end] == phenomes.phenotypes[:, 1]
true

julia> phenomes_new = addcompositetrait(phenomes, composite_trait_name = "some_wild_composite_trait", formula_string = "(A^B) + (C/A) - sqrt(abs(B-A)) + log(1.00 + C)");

julia> phenomes_new.phenotypes[:, end] == (phenomes.phenotypes[:,1].^phenomes.phenotypes[:,2]) .+ (phenomes.phenotypes[:,3]./phenomes.phenotypes[:,1]) .- sqrt.(abs.(phenomes.phenotypes[:,2].-phenomes.phenotypes[:,1])) .+ log.(1.00 .+ phenomes.phenotypes[:,3])
true
```
"""
function addcompositetrait(phenomes::Phenomes; composite_trait_name::String, formula_string::String)::Phenomes
    # phenomes = Phenomes(n=10, t=3); phenomes.entries = string.("entry_", 1:10); phenomes.populations .= "pop_1"; phenomes.traits = ["A", "B", "C"]; phenomes.phenotypes = rand(10,3);
    # composite_trait_name = "some_wild_composite_trait";
    # formula_string = "((A^B) + C) + sqrt(abs(log(1.00 / A) - (A * (B + C)) / (B - C)^2))";
    # formula_string = "A";
    df = tabularise(phenomes)
    formula_parsed_orig = deepcopy(formula_string)
    formula_parsed = deepcopy(formula_string)
    symbol_strings = ["=", "+", "-", "*", "/", "^", "%", "abs(", "sqrt(", "log(", "log2(", "log10(", "(", ")"]
    for s in symbol_strings
        formula_string = replace(formula_string, s => " ")
    end
    component_trait_names = unique(split(formula_string, " "))
    ϕ = Vector{Union{Missing,Float64}}(undef, size(df, 1))
    for i in eachindex(ϕ)
        # i = 1
        for var_name in component_trait_names
            # var_name = component_trait_names[2]
            if sum(names(df) .== var_name) == 0
                continue
            else
                formula_parsed = replace(formula_parsed, var_name => string(df[i, var_name]))
            end
        end
        ϕ[i] = @eval(@stringevaluation $(formula_parsed))
        # Reset the formula
        formula_parsed = deepcopy(formula_parsed_orig)
    end
    out = clone(phenomes)
    idx = findall(out.traits .== composite_trait_name)
    if length(idx) == 0
        push!(out.traits, composite_trait_name)
        out.phenotypes = hcat(out.phenotypes, ϕ)
    elseif length(idx) == 1
        out.phenotypes[:, idx] = ϕ
    else
        throw(ErrorException("Duplicate traits in phenomes, i.e. trait: " * composite_trait_name))
    end
    out.mask = hcat(out.mask, ones(size(out.mask, 1)))
    if !checkdims(out)
        throw(ErrorException("Error generating composite trait: `" * composite_trait_name * "`"))
    end
    out
end
