"""
    clone(x::Fit)::Fit

Clone a Fit object

## Example
```jldoctest; setup = :(using GBCore)
julia> fit = Fit(n=1, l=2);

julia> copy_fit = clone(fit)
Fit("", ["", ""], [0.0, 0.0], "", [""], [""], [0.0], [0.0], Dict("" => 0.0))
```
"""
function clone(x::Fit)::Fit
    y::Fit = Fit(n = length(x.y_true), l = length(x.b_hat_labels))
    y.model = deepcopy(x.model)
    y.b_hat_labels = deepcopy(x.b_hat_labels)
    y.b_hat = deepcopy(x.b_hat)
    y.trait = deepcopy(x.trait)
    y.entries = deepcopy(x.entries)
    y.populations = deepcopy(x.populations)
    y.metrics = deepcopy(x.metrics)
    y.y_true = deepcopy(x.y_true)
    y.y_pred = deepcopy(x.y_pred)
    y
end


"""
    Base.hash(x::Fit, h::UInt)::UInt

Hash a Fit struct using the entries, populations and loci_alleles.
We deliberately excluded the allele_frequencies, and mask for efficiency.

## Examples
```jldoctest; setup = :(using GBCore)
julia> fit = Fit(n=1, l=2);

julia> typeof(hash(fit))
UInt64
```
"""
function Base.hash(x::Fit, h::UInt)::UInt
    hash(
        Fit,
        hash(
            x.model,
            hash(
                x.b_hat,
                hash(x.trait, hash(x.entries, hash(x.populations, hash(x.metrics, hash(x.y_true, hash(x.y_pred, h)))))),
            ),
        ),
    )
end


"""
    Base.:(==)(x::Fit, y::Fit)::Bool

Equality of Fit structs using the hash function defined for Fit structs.

## Examples
```jldoctest; setup = :(using GBCore)
julia> fit_1 = Fit(n=1, l=4);

julia> fit_2 = Fit(n=1, l=4);

julia> fit_3 = Fit(n=1, l=2);

julia> fit_1 == fit_2
true

julia> fit_1 == fit_3
false
```
"""
function Base.:(==)(x::Fit, y::Fit)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(fit::Fit)::Bool

Check dimension compatibility of the fields of the Fit struct

# Examples
```jldoctest; setup = :(using GBCore)
julia> fit = Fit(n=1, l=4);

julia> checkdims(fit)
true

julia> fit.b_hat_labels = ["chr1\\t1\\tA|T\\tA"];

julia> checkdims(fit)
false
```
"""
function checkdims(fit::Fit)::Bool
    n = length(fit.entries)
    l = length(fit.b_hat)
    if (n != length(fit.populations)) ||
       (n != length(fit.y_true)) ||
       (n != length(fit.y_pred)) ||
       (l != length(fit.b_hat_labels))
        return false
    end
    true
end
