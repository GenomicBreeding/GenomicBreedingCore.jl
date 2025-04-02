"""
    clone(x::BLR)::BLR

Create a deep copy of a BLR object, duplicating all its fields.

This function performs a deep clone of the input BLR object, ensuring that all nested
structures and arrays are also copied, preventing any shared references between the
original and the cloned object.

# Arguments
- `x::BLR`: The BLR object to be cloned

# Returns
- `BLR`: A new BLR object with identical but independent values

# Fields copied
- `entries::Vector{String}`: Names or identifiers for the observations
- `coefficient_names::Vector{String}`: Names of the model coefficients/effects
- `y::Vector{Float64}`: Response/dependent variable vector
- `Xs::Dict{String, Matrix{Union{Bool, Float64}}}`: Design matrices for factors and numeric matrix for the other covariates
- `coefficients::Vector{Float64}`: Estimated coefficients/effects
- `ŷ::Vector{Float64}`: Fitted/predicted values
- `ϵ::Vector{Float64}`: Residuals (y - ŷ)
- `Σs::Dict{String, Union{Matrix{Float64}, UniformScaling{Float64}}}`: Variance-covariance matrices

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> blr = BLR(n=1, p=1);

julia> copy_blr = clone(blr)
BLR([""], [""], [0.0], Dict{String, Matrix{Union{Bool, Float64}}}("intercept" => [false;;]), [0.0], [0.0], [0.0], Dict{String, Union{UniformScaling{Float64}, Matrix{Float64}}}("σ²" => UniformScaling{Float64}(1.0)))
```
"""
function clone(x::BLR)::BLR
    out = BLR(n = length(x.entries), p = length(x.coefficient_names))
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        setfield!(out, field, deepcopy(getfield(x, field)))
    end
    out
end


"""
    Base.hash(x::BLR, h::UInt)::UInt

Calculate a hash value for a `BLR` struct.

This method implements hashing for the `BLR` type by combining the hashes of its fields,
excluding the "Xs" and "Σs" fields. Each field's hash is combined sequentially with the
input hash value.

# Arguments
- `x::BLR`: The BLR struct to be hashed
- `h::UInt`: The initial hash value to be mixed with

# Returns
- `UInt`: The computed hash value

# Notes
- The fields "Xs" and "Σs" are explicitly excluded from the hash computation
- Fields are processed in the order defined in the struct

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> blr = BLR(n=1, p=1);

julia> typeof(hash(blr))
UInt64
```
"""
function Base.hash(x::BLR, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        if field ∈ ["Xs", "Σs"]
            continue
        end
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    Base.:(==)(x::BLR, y::BLR)::Bool

Compare two `BLR` structs for equality based on their hash values.

This method defines equality comparison for `BLR` structs by comparing their hash values.
Two `BLR` structs are considered equal if they have identical hash values, which means
they have the same values for all their fields, except the computationally expensive Xs, and Σs fields.

# Arguments
- `x::BLR`: First BLR struct to compare
- `y::BLR`: Second BLR struct to compare

# Returns
- `Bool`: `true` if the BLR structs are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> blr_1 = BLR(n=1, p=4);

julia> blr_2 = BLR(n=1, p=4);

julia> blr_3 = BLR(n=1, p=1);

julia> blr_1 == blr_2
true

julia> blr_1 == blr_3
false
```
"""
function Base.:(==)(x::BLR, y::BLR)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(blr::BLR)::Bool

Check dimension compatibility of the internal fields of a `BLR` struct.

This function verifies that all vector and matrix fields in the `BLR` struct have compatible dimensions:
- Length of `entries`, `y`, `ȳ`, and `ϵ` must be equal (denoted as `n`)
- Length of `coefficients` and `coefficient_names` must be equal (denoted as `p`)
- Matrix `Xs` must have dimensions `n × p`
- Matrix `Σs` must have dimensions `p × p`

Returns `true` if all dimensions are compatible, `false` otherwise.

# Arguments
- `blr::BLR`: The BLR struct to check dimensions for

# Returns
- `Bool`: `true` if dimensions are compatible, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> blr = BLR(n=1, p=4);

julia> checkdims(blr)
true

julia> blr.coefficient_names["dummy_X"] = ["dummy_coef"];

julia> checkdims(blr)
false
```
"""
function checkdims(blr::BLR)::Bool
    n = length(blr.entries)
    p = sum([length(c) for (_, c) in blr.coefficients])
    if (n != length(blr.y)) ||
       (n != length(blr.ŷ)) ||
       (n != length(blr.ϵ)) ||
       prod([(n != size(X, 1)) for (_, X) in blr.Xs]) ||
       (p != sum([length(c) for (_, c) in blr.coefficient_names])) ||
       p != sum([size(X, 2) for (_, X) in blr.Xs]) ||
       p != sum([!isa(Σ, UniformScaling) ? size(Σ, 1) : k=="σ²" ? 1 : size(blr.Xs[k], 2) for (k, Σ) in blr.Σs]) ||
       p != sum([!isa(Σ, UniformScaling) ? size(Σ, 2) : k=="σ²" ? 1 : size(blr.Xs[k], 2) for (k, Σ) in blr.Σs]) ||
       !("intercept" ∈ string.(keys(blr.Xs))) ||
       !("σ²" ∈ string.(keys(blr.Σs))) ||
       !("intercept" ∈ string.(keys(blr.coefficients))) ||
       !("intercept" ∈ string.(keys(blr.coefficient_names)))
        return false
    end
    true
end


function extractXb(blr::BLR)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}}
    # Check argument
    if !checkdims(blr)
        throw(ArgumentError("The BLR struct is corrupted."))
    end
    components = string.(keys(blr.Xs))
    X::Matrix{Float64} = reshape(blr.Xs["intercept"], length(blr.Xs["intercept"]), 1)
    b::Vector{Float64} = blr.coefficients["intercept"]
    b_labels::Vector{String} = blr.coefficient_names["intercept"]
    if length(components) == 1
        return (X, b, b_labels)
    end
    for c in components
        if c == "intercept"
            continue
        end
        X = hcat(X, blr.Xs[c])
        b = vcat(b, blr.coefficients[c])
        b_labels = vcat(b_labels, blr.coefficient_names[c])
    end
    (X, b, b_labels)
end