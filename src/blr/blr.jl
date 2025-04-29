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
```jldoctest; setup = :(using GenomicBreedingCore, LinearAlgebra)
julia> blr = BLR(n=1, p=1);

julia> copy_blr = clone(blr)
BLR([""], Dict{String, Matrix{Union{Bool, Float64}}}("intercept" => [true;;]), Dict{String, Union{Nothing, UniformScaling{Float64}, Matrix{Float64}}}("σ²" => UniformScaling{Float64}(1.0)), Dict("intercept" => [0.0]), Dict("intercept" => ["intercept"]), [0.0], [0.0], [0.0], 0×0 DataFrame)
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
       p != sum([k == "σ²" ? 1 : size(blr.Xs[k], 2) for (k, Σ) in blr.Σs]) ||
       p != sum([k == "σ²" ? 1 : size(blr.Xs[k], 2) for (k, Σ) in blr.Σs]) ||
       !("intercept" ∈ string.(keys(blr.Xs))) ||
       !("σ²" ∈ string.(keys(blr.Σs))) ||
       !("intercept" ∈ string.(keys(blr.coefficients))) ||
       !("intercept" ∈ string.(keys(blr.coefficient_names))) ||
       !((ncol(blr.diagnostics) == 4) || (ncol(blr.diagnostics) == 0))
        return false
    end
    true
end


"""
    extractXb(blr::BLR)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}}

Extract design matrix X, coefficients b, and coefficient labels from a BLR (Bayesian Linear Regression) model.

# Arguments
- `blr::BLR`: A BLR struct containing model components

# Returns
A tuple containing:
- `X::Matrix{Float64}`: The design matrix combining all predictors
- `b::Vector{Float64}`: Vector of estimated coefficients
- `b_labels::Vector{String}`: Vector of coefficient labels/names

# Throws
- `ArgumentError`: If the dimensions in the BLR struct are inconsistent

# Details
Extracts and combines the design matrices, coefficients and their labels from the BLR model components.
The intercept is handled separately and placed first, followed by other variance components.

# Note
The function assumes the BLR struct has valid "intercept" components and optional additional variance components.

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> blr = BLR(n=1, p=4);

julia> X, b, b_labels = extractXb(blr);

julia> size(X) == (1, 4)
true

julia> length(b) == length(b_labels)
true
```
"""
function extractXb(blr::BLR)::Tuple{Matrix{Float64},Vector{Float64},Vector{String}}
    # Check argument
    if !checkdims(blr)
        throw(ArgumentError("The BLR struct is corrupted."))
    end
    variance_components = filter(x -> x != "intercept", string.(keys(blr.Xs)))
    X::Matrix{Float64} = reshape(blr.Xs["intercept"], length(blr.Xs["intercept"]), 1)
    b::Vector{Float64} = blr.coefficients["intercept"]
    b_labels::Vector{String} = blr.coefficient_names["intercept"]
    for c in variance_components
        X = hcat(X, blr.Xs[c])
        b = vcat(b, blr.coefficients[c])
        b_labels = vcat(b_labels, blr.coefficient_names[c])
    end
    (X, b, b_labels)
end

"""
    dimensions(blr::BLR)::Dict{String, Any}

Calculate various dimensional properties of a Bayesian Linear Regression (BLR) model.

# Arguments
- `blr::BLR`: A Bayesian Linear Regression model structure

# Returns
A dictionary containing the following keys:
- `"n_rows"`: Number of observations
- `"n_coefficients"`: Total number of coefficients across all components
- `"n_variance_components"`: Number of variance components
- `"n_entries"`: Number of unique entries
- `"coeff_per_varcomp"`: Dictionary mapping variance component names to their number of coefficients
- `"varex_per_varcomp"`: Dictionary mapping variance component names to their variance explained (normalized)

# Details
The function performs the following:
1. Validates BLR structure dimensions
2. Calculates coefficients per variance component
3. Computes variance explained for each component and normalizes by total variance
4. Returns dimensional summary as a dictionary

# Throws
- `ArgumentError`: If the BLR structure dimensions are inconsistent

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> blr = BLR(n=10, p=6, var_comp = Dict("entries" => 5, "σ²" => 1));

julia> dimensions(blr)
Dict{String, Any} with 6 entries:
  "n_coefficients"        => 6
  "n_variance_components" => 2
  "varex_per_varcomp"     => Dict("entries"=>0.333333, "σ²"=>0.666667)
  "n_entries"             => 1
  "n_rows"                => 10
  "coeff_per_varcomp"     => Dict("entries"=>5.0, "σ²"=>10.0)
```
"""
function dimensions(blr::BLR)::Dict{String,Any}
    # Check argument
    if !checkdims(blr)
        throw(ArgumentError("BLR struct is corrupted ☹."))
    end
    coeff_per_varcomp::Dict{String,Float64} = Dict()
    varex_per_varcomp::Dict{String,Float64} = Dict()
    for (k, v) in blr.Σs
        # k = string.(keys(blr.Σs))[1]; v = blr.Σs[k]
        coeff_per_varcomp[k] = if k == "σ²"
            length(blr.entries)
        elseif isa(v, UniformScaling{Float64})
            size(blr.Xs[k], 2)
        else
            size(v, 1)
        end
        varex_per_varcomp[k] = if isa(v, UniformScaling{Float64})
            v.λ * coeff_per_varcomp[k]
        else
            sum(diag(v))
        end
    end
    # Normalise variance explained by the total variance
    total_variance = sum([v for (_, v) in varex_per_varcomp])
    for (k, v) in varex_per_varcomp
        varex_per_varcomp[k] = v / total_variance
    end
    Dict(
        "n_rows" => length(blr.entries),
        "n_coefficients" => sum([length(c) for (_, c) in blr.coefficients]),
        "n_variance_components" => length(blr.Σs),
        "n_entries" => length(unique(blr.entries)),
        "coeff_per_varcomp" => coeff_per_varcomp,
        "varex_per_varcomp" => varex_per_varcomp,
    )
end
