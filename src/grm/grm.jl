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

julia> copy_grm = clone(grm);

julia> (copy_grm.entries == grm.entries) && (copy_grm.loci_alleles == grm.loci_alleles) && (copy_grm.genomic_relationship_matrix == grm.genomic_relationship_matrix)
true
```
"""
function clone(x::GRM)::GRM
    GRM(deepcopy(x.entries), deepcopy(x.loci_alleles), deepcopy(x.genomic_relationship_matrix))
end

"""
    Base.hash(x::GRM, h::UInt)::UInt

Compute a hash value for a GRM (Genomic Relationship Matrix) struct.

This method defines how GRM structs should be hashed, making them usable in 
hash-based collections like Sets or as Dict keys. The hash is computed by 
iteratively combining the hash values of all fields in the struct.

# Arguments
- `x::GRM`: The GRM struct to be hashed
- `h::UInt`: The initial hash value to be combined with the struct's hash

# Returns
- `UInt`: A combined hash value for the entire GRM struct

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm = GRM(string.(["entries_1", "entries_2"]), string.(["chr1\t123\tA|T\tA", "chr1\t456\tC|G\tG"]), Float64.(rand(2,2)));

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

Compare two GRM structs for equality.

Overloads the equality operator (`==`) for GRM structs by comparing their hash values.
Two GRM structs are considered equal if they have identical values for all their fields.

# Arguments
- `x::GRM`: First GRM struct to compare
- `y::GRM`: Second GRM struct to compare

# Returns
- `Bool`: `true` if the GRM structs are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm_1 = GRM(string.(["entries_1", "entries_2"]), string.(["chr1\t123\tA|T\tA", "chr1\t456\tC|G\tG"]), Float64.(rand(2,2)));

julia> grm_2 = clone(grm_1);

julia> grm_3 = GRM(string.(["entries_1", "entries_2"]), string.(["chr1\t123\tA|T\tA", "chr1\t456\tC|G\tG"]), Float64.(rand(2,2)));

julia> grm_1 == grm_2
true

julia> grm_1 == grm_3
false
```
"""
function Base.:(==)(x::GRM, y::GRM)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(grm::GRM; verbose::Bool=false)::Bool

Check dimension compatibility of the GRM (Genomic Relationship Matrix) struct fields.

# Arguments
- `grm::GRM`: A Genomic Relationship Matrix struct containing entries and relationship matrix
- `verbose::Bool=false`: If true, prints the dimensions of each field for debugging

# Returns
- `true` if the number of entries matches the dimensions of the genomic relationship matrix
- `false` if there is a mismatch between the number of entries and matrix dimensions

# Details
The function verifies that:
1. The number of entries equals the number of rows in the genomic relationship matrix
2. The number of entries equals the number of columns in the genomic relationship matrix
(The genomic relationship matrix should be square with dimensions matching the number of entries)

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> grm = GRM(string.(["entries_1", "entries_2"]), string.(["chr1\t123\tA|T\tA", "chr1\t456\tC|G\tG"]), Float64.(rand(2,2)));

julia> checkdims(grm)
true

julia> grm.entries = ["dummy_entry"];

julia> checkdims(grm)
false
```
"""
function checkdims(grm::GRM; verbose::Bool = false)::Bool
    if verbose
        @show length(grm.entries)
        @show size(grm.genomic_relationship_matrix)
    end
    n = length(grm.entries)
    if (n != size(grm.genomic_relationship_matrix, 1)) || (n != size(grm.genomic_relationship_matrix, 2))
        return false
    end
    true
end
