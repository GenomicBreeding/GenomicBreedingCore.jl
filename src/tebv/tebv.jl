"""
    clone(x::TEBV)::TEBV

Create a deep copy of a TEBV (Trial-Estimated Breeding Value) object.

Returns a new TEBV instance with all fields deeply copied from the input object,
ensuring complete independence between the original and cloned objects.

# Arguments
- `x::TEBV`: The source TEBV object to be cloned

# Returns
- `TEBV`: A new TEBV object containing deep copies of all fields from the input

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, MixedModels, DataFrames)
julia> tebv = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> copy_tebv = clone(tebv);

julia> copy_tebv.traits == tebv.traits
true

julia> copy_tebv.phenomes == tebv.phenomes
true
```
"""
function clone(x::TEBV)::TEBV
    Φ::TEBV = TEBV(
        traits = deepcopy(x.traits),
        formulae = deepcopy(x.formulae),
        models = deepcopy(x.models),
        df_BLUEs = deepcopy(x.df_BLUEs),
        df_BLUPs = deepcopy(x.df_BLUPs),
        phenomes = deepcopy(x.phenomes),
    )
    Φ
end


"""
    Base.hash(x::TEBV, h::UInt)::UInt

Calculate a hash value for a TEBV (Trial-Estimated Breeding Value) struct.

This method implements hashing for TEBV objects by combining the hash values of selected fields:
- traits: Vector of trait names
- formulae: Vector of formula strings
- phenomes: Vector of Phenomes objects

Note: For performance reasons, the following fields are deliberately excluded from the hash calculation:
- models
- df_BLUEs
- df_BLUPs

# Arguments
- `x::TEBV`: The TEBV struct to be hashed
- `h::UInt`: The hash value to be mixed with the object's hash

# Returns
- `UInt`: A unique hash value for the TEBV object

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, MixedModels, DataFrames)
julia> tebv = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> typeof(hash(tebv))
UInt64
```
"""
function Base.hash(x::TEBV, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        if field == :df_BLUEs || field == :df_BLUPs || field == :models
            continue
        end
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    ==(x::TEBV, y::TEBV)::Bool

Compare two TEBV (Trial-Estimated Breeding Values) objects for equality.

This method implements equality comparison for TEBV structs by comparing their hash values.
Two TEBV objects are considered equal if they have identical values for all their fields:
traits, formulae, models, df_BLUEs, df_BLUPs, and phenomes.

# Arguments
- `x::TEBV`: First TEBV object to compare
- `y::TEBV`: Second TEBV object to compare

# Returns
- `Bool`: `true` if the TEBV objects are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, MixedModels, DataFrames)
julia> tebv_1 = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> tebv_2 = clone(tebv_1);

julia> tebv_3 = TEBV(traits=["SOMETHING_ELSE"], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> tebv_1 == tebv_2
true

julia> tebv_1 == tebv_3
false
```
"""
function Base.:(==)(x::TEBV, y::TEBV)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(y::TEBV)::Bool

Check if all fields in the TEBV struct have compatible dimensions. The function verifies that
the length of all arrays in the TEBV struct match the number of traits.

# Arguments
- `tebv::TEBV`: A TEBV (Trial-estimated Breeding Values) struct containing traits,
  formulae, models, BLUEs, BLUPs, and phenomes.

# Returns
- `Bool`: Returns `true` if all fields have matching dimensions (equal to the number of traits),
  `false` otherwise.

# Details
The function checks if the following fields have the same length as `traits`:
- formulae
- unique models
- unique BLUEs DataFrames
- unique BLUPs DataFrames
- unique phenomes

# Examples
julia> tebv = TEBV(traits=[""], formulae=[""], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> checkdims(tebv)
true
```
"""
function checkdims(tebv::TEBV)::Bool
    t = length(tebv.traits)
    if (t != length(tebv.formulae)) ||
       (t != length(unique(tebv.models))) ||
       (t != length(unique(tebv.df_BLUEs))) ||
       (t != length(unique(tebv.df_BLUPs))) ||
       (t != length(unique(tebv.phenomes)))
        return false
    end
    true
end

"""
    dimensions(tebv::TEBV)::Dict{String, Int64}

Calculate various dimensional metrics for a TEBV (Trial-Estimated Breeding Values) struct.

# Arguments
- `tebv::TEBV`: A TEBV struct containing traits, formulae, models, BLUEs, BLUPs, and phenomes data

# Returns
A dictionary containing the following counts:
- `"n_entries"`: Number of unique entries across all phenomes
- `"n_populations"`: Number of unique populations across all phenomes
- `"n_traits"`: Number of traits in the TEBV struct
- `"n_total"`: Total number of observations across all traits
- `"n_zeroes"`: Total number of zero values across all traits
- `"n_missing"`: Total number of missing values across all traits
- `"n_nan"`: Total number of NaN values across all traits
- `"n_inf"`: Total number of Infinite values across all traits

# Throws
- `ArgumentError`: If the TEBV struct dimensions are inconsistent or corrupted

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, MixedModels, DataFrames)
julia> tebv = TEBV(traits=["trait_1"], formulae=["trait_1 ~ 1 + 1|entries"], models=[MixedModel(@formula(y~1+(1|x)), DataFrame(y=1, x=1))], df_BLUEs=[DataFrame(x=1)], df_BLUPs=[DataFrame(x=1)], phenomes=[Phenomes(n=1,t=1)]);

julia> dimensions(tebv)
Dict{String, Int64} with 8 entries:
  "n_total"       => 1
  "n_zeroes"      => 0
  "n_nan"         => 0
  "n_entries"     => 1
  "n_traits"      => 1
  "n_inf"         => 0
  "n_populations" => 1
  "n_missing"     => 1
```
"""
function dimensions(tebv::TEBV)::Dict{String,Int64}
    if !checkdims(tebv)
        throw(ArgumentError("TEBV struct is corrupted ☹."))
    end
    entries = tebv.phenomes[1].entries
    populations = tebv.phenomes[1].populations
    n_traits = length(tebv.traits)
    n_total = 0
    n_zeroes = 0
    n_missing = 0
    n_nan = 0
    n_inf = 0
    for i = 1:n_traits
        # i, trait = 1, tebv.traits[1]
        phenomes = tebv.phenomes[i]
        entries = unique(vcat(entries, tebv.phenomes[1].entries))
        populations = unique(vcat(populations, tebv.phenomes[1].populations))
        dims = dimensions(phenomes)
        n_total += dims["n_total"]
        n_zeroes += dims["n_zeroes"]
        n_missing += dims["n_missing"]
        n_nan += dims["n_nan"]
        n_inf += dims["n_inf"]
    end
    Dict(
        "n_entries" => length(entries),
        "n_populations" => length(populations),
        "n_traits" => n_traits,
        "n_total" => n_total,
        "n_zeroes" => n_zeroes,
        "n_missing" => n_missing,
        "n_nan" => n_nan,
        "n_inf" => n_inf,
    )
end

"""
    extractphenomes(tebv::TEBV)::Phenomes

Extract phenotypic values from a Trial-Estimated Breeding Value (TEBV) object.

This function processes phenotypic data from a TEBV object, handling intercept effects
and merging multiple phenomes if present. It performs the following operations:

1. Validates input TEBV dimensions
2. Processes intercept effects if present by:
   - Identifying intercept terms
   - Combining intercept values with trait effects
   - Adjusting trait names and phenotypic values accordingly
3. Merges multiple phenomes if present
4. Renames traits to match input TEBV traits if dimensions align
5. Validates output Phenomes dimensions

# Arguments
- `tebv::TEBV`: A Trial Estimated Breeding Value object containing phenotypic data

# Returns
- `Phenomes`: A Phenomes object containing processed phenotypic values

# Throws
- `ArgumentError`: If input TEBV or output Phenomes dimensions are invalid

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);

julia> tebv = analyse(trials, max_levels=50, verbose=false);

julia> phenomes = extractphenomes(tebv);

julia> phenomes.traits == ["trait_1", "trait_2", "trait_3"]
true
```
"""
function extractphenomes(tebv::TEBV)::Phenomes
    # trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);
    # tebv = analyse(trials, max_levels=50, verbose=false);
    if !checkdims(tebv)
        throw(ArgumentError("The TEBV struct is corrupted ☹."))
    end
    phenomes = Phenomes(n = length(tebv.phenomes[1].entries), t = 1)
    for i in eachindex(tebv.phenomes)
        # i = 2
        phenomes_i = clone(tebv.phenomes[i])
        # Add intercept effects if present
        bool_intercept = .!isnothing.(match.(Regex("(Intercept)"), phenomes_i.traits))
        n = length(phenomes_i.entries)
        t = length(phenomes_i.traits)
        if sum(bool_intercept) == 1
            idx_intercept = findall(bool_intercept)[1]
            idx_ϕ = findall(.!bool_intercept)
            traits = repeat([""], t - 1)
            ϕ::Matrix{Union{Missing,Float64}} = fill(0.0, n, t - 1)
            μ::Matrix{Bool} = fill(true, n, t - 1)
            for (i, j) in enumerate(idx_ϕ)
                traits[i] = phenomes_i.traits[j]
                ϕ[:, i] = phenomes_i.phenotypes[:, idx_intercept] + phenomes_i.phenotypes[:, j]
                μ[:, i] = phenomes_i.mask[:, j]
            end
            phenomes_i.traits = traits
            phenomes_i.phenotypes = ϕ
            phenomes_i.mask = μ
        else
            phenomes_i.traits = phenomes_i.traits
            phenomes_i.phenotypes = phenomes_i.phenotypes
            phenomes_i.mask = phenomes_i.mask
        end
        if i == 1
            phenomes = clone(phenomes_i)
        else
            phenomes = merge(phenomes, phenomes_i)
        end
    end
    # Rename the traits to match the input TEBV if we have the same number of output traits
    if length(phenomes.traits) == length(tebv.traits)
        phenomes.traits = tebv.traits
    end
    # Output
    if !checkdims(phenomes)
        throw(ArgumentError("The Phenomes struct is corrupted ☹."))
    end
    phenomes
end

"""
    plot(tebv::TEBV)

Create a visualization of True Estimated Breeding Values (TEBV) analysis results.

This function extracts phenomes from the TEBV object and generates a plot to visualize
the breeding value estimates.

# Arguments
- `tebv::TEBV`: A TEBV object containing the analysis results

# Returns
- A plot object representing the visualization of the phenomes data
"""
function plot(tebv::TEBV)
    # trials, _simulated_effects = simulatetrials(genomes = simulategenomes(n=10, verbose=false), n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=10, verbose=false);
    # tebv = analyse(trials, max_levels=50, verbose=false);
    phenomes = extractphenomes(tebv)
    plot(phenomes)
end
