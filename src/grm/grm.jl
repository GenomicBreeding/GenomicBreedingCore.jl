"""
    clone(x::GRM)::GRM

Create a deep copy of a GRM (Genomic Relationship Matrix) object.

Creates a new GRM object with deep copies of all fields from the input object.
The clone function ensures that modifications to the cloned object do not affect 
the original object.

# Arguments
- `x::GRM`: The GRM object to be cloned 

# Returns
- `GRM`: A new GRM object containing deep copies of the entries, loci_alleles, 
         and genomic_relationship_matrix fields from the input

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm = GRM(string.(["entries_1", "entries_2"]), string.(["chr1\t123\tA|T\tA", "chr1\t456\tC|G\tG"]), Float64.(rand(2,2)));

julia> copy_grm = clone(grm)
GRM("replication_1", "fold_1", GRM("", ["", ""], [0.0, 0.0], "", [""], [""], [0.0], [0.0], Dict("" => 0.0), nothing), ["population_1"], ["entry_1"], [0.0], [0.0], Dict("" => 0.0))
```
"""
function clone(x::GRM)::GRM
    GRM(
        deepcopy(x.entries),
        deepcopy(x.loci_alleles),
        clone(x.genomic_relationship_matrix),
    )
end

"""
    Base.hash(x::GRM, h::UInt)::UInt

Compute a hash value for a GRM (Cross-Validation) struct.

This method defines how GRM structs should be hashed, which is useful for
using GRM objects in hash-based collections like Sets or as Dict keys.

# Arguments
- `x::GRM`: The GRM struct to be hashed
- `h::UInt`: The hash value to be mixed with the new hash

# Returns
- `UInt`: A hash value for the GRM struct

# Implementation Details
The hash is computed by combining the following fields:
- replication
- fold
- grm
- validation_populations
- validation_entries
- validation_y_true
- validation_y_pred
- metrics

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm = GRM(n=1, l=2);

julia> grm = GRM("replication_1", "fold_1", grm, ["population_1"], ["entry_1"], [0.0], [0.0], grm.metrics);

julia> typeof(hash(grm))
UInt64
```
"""
function Base.hash(x::GRM, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    Base.:(==)(x::GRM, y::GRM)::Bool

Compare two GRM (Cross-Validation) structs for equality.

This method overloads the equality operator (`==`) for GRM structs by comparing their hash values.
Two GRM structs are considered equal if they have identical values for all fields.

# Arguments
- `x::GRM`: First GRM struct to compare
- `y::GRM`: Second GRM struct to compare

# Returns
- `Bool`: `true` if the GRM structs are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm = GRM(n=1, l=2);

julia> cv_1 = GRM("replication_1", "fold_1", grm, ["population_1"], ["entry_1"], [0.0], [0.0], grm.metrics);

julia> cv_2 = clone(cv_1);

julia> cv_3 = clone(cv_1); cv_3.replication = "other_replication";

julia> cv_1 == cv_2
true

julia> cv_1 == cv_3
false
```
"""
function Base.:(==)(x::GRM, y::GRM)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(grm::GRM)::Bool

Check dimension compatibility of the fields of the GRM struct.

The function verifies that:
- The grm object dimensions are valid
- The number of validation populations matches the number of validation entries
- The number of validation true values matches the number of validation predictions
- The number of metrics matches the number of metrics in the grm object

Returns:
- `true` if all dimensions are compatible
- `false` if any dimension mismatch is found

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm = GRM(n=1, l=2);

julia> grm = GRM("replication_1", "fold_1", grm, ["population_1"], ["entry_1"], [0.0], [0.0], grm.metrics);

julia> checkdims(grm)
true

julia> grm.validation_y_true = [0.0, 0.0];

julia> checkdims(grm)
false
```
"""
function checkdims(grm::GRM)::Bool
    n = length(grm.entries)
    if (n != size(grm.genomic_relationship_matrix, 1)) || (n != size(grm.genomic_relationship_matrix, 2))
        return false
    end
    true
end