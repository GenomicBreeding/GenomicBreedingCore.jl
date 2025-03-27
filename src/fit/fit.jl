"""
    clone(x::Fit)::Fit

Create a deep copy of a Fit object, duplicating all its fields.

This function performs a deep clone of the input Fit object, ensuring that all nested
structures and arrays are also copied, preventing any shared references between the
original and the cloned object.

# Arguments
- `x::Fit`: The Fit object to be cloned

# Returns
- `Fit`: A new Fit object with identical but independent values

# Fields copied
- `model`: The statistical model
- `b_hat_labels`: Labels for the estimated parameters
- `b_hat`: Estimated parameters
- `trait`: The trait being analyzed
- `entries`: Entry identifiers
- `populations`: Population identifiers
- `metrics`: Performance metrics
- `y_true`: Observed values
- `y_pred`: Predicted values

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=1, l=2);

julia> copy_fit = clone(fit)
Fit("", ["", ""], [0.0, 0.0], "", [""], [""], [0.0], [0.0], Dict("" => 0.0), nothing)
```
"""
function clone(x::Fit)::Fit
    out = Fit(n = length(x.y_true), l = length(x.b_hat_labels))
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        setfield!(out, field, deepcopy(getfield(x, field)))
    end
    out
end


"""
    Base.hash(x::Fit, h::UInt)::UInt

Calculate a hash value for a `Fit` struct.

This method implements hashing for the `Fit` type by combining the hashes of its core components
in a specific order. The hash is computed using the following fields:
- model
- b_hat (estimated effects)
- trait
- entries
- populations
- metrics
- y_true (observed values)
- y_pred (predicted values)

# Arguments
- `x::Fit`: The Fit struct to be hashed
- `h::UInt`: The hash value to be mixed with

# Returns
- `UInt`: The computed hash value

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=1, l=2);

julia> typeof(hash(fit))
UInt64
```
"""
function Base.hash(x::Fit, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    Base.:(==)(x::Fit, y::Fit)::Bool

Compare two `Fit` structs for equality based on their hash values.

This method defines equality comparison for `Fit` structs by comparing their hash values.
Two `Fit` structs are considered equal if they have identical hash values, which means
they have the same values for all their fields.

# Arguments
- `x::Fit`: First Fit struct to compare
- `y::Fit`: Second Fit struct to compare

# Returns
- `Bool`: `true` if the Fit structs are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
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

Check dimension compatibility of the internal fields of a `Fit` struct.

This function verifies that all vector fields in the `Fit` struct have compatible dimensions:
- Length of `entries`, `populations`, `y_true`, and `y_pred` must be equal (denoted as `n`)
- Length of `b_hat` and `b_hat_labels` must be equal (denoted as `l`)

Returns `true` if all dimensions are compatible, `false` otherwise.

# Arguments
- `fit::Fit`: The Fit struct to check dimensions for

# Returns
- `Bool`: `true` if dimensions are compatible, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
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

"""
    plot(fit::Fit, distribution::Any=[TDist(1), Normal()][2], α::Float64=0.05)

Generate diagnostic plots for genetic association analysis results.

# Arguments
- `fit::Fit`: A Fit object containing the association analysis results, specifically the `b_hat` field with effect sizes
- `distribution::Any`: The null distribution for p-value calculation. Defaults to Normal distribution
- `α::Float64`: Significance level for multiple testing correction (Bonferroni). Defaults to 0.05

# Returns
Displays three plots:
- Histogram showing the distribution of effect sizes
- Manhattan plot showing -log10(p-values) across loci with Bonferroni threshold
- Q-Q plot comparing observed vs expected -log10(p-values)

# Examples
```
julia> distribution = [TDist(1), Normal()][2];

julia> fit = Fit(n=100, l=10_000); fit.b_hat = rand(distribution, 10_000);  α = 0.05;

julia> GenomicBreedingCore.plot(fit);
```
"""
function plot(fit::Fit, distribution::Any = [TDist(1), Normal()][2], α::Float64 = 0.05)
    # distribution::Any=[TDist(1), Normal()][2];
    # fit = Fit(n=100, l=10_000); fit.b_hat = rand(distribution, 10_000);  α::Float64=0.05;
    l = length(fit.b_hat)
    p1 = UnicodePlots.histogram(fit.b_hat, title = "Distribution of " * string(distribution) * " values")
    # Manhattan plot
    pval = 1 .- cdf.(distribution, abs.(fit.b_hat))
    lod = -log10.(pval)
    threshold = -log10(α / l)
    p2 = UnicodePlots.scatterplot(lod, title = "Manhattan plot", xlabel = "Loci-alleles", ylabel = "-log10(pval)")
    UnicodePlots.lineplot!(p2, [0, l], [threshold, threshold])
    # QQ plot
    lod_expected = reverse(-log10.(collect(range(0, 1, l))))
    p3 = UnicodePlots.scatterplot(sort(lod), lod_expected, xlabel = "Observed LOD", ylabel = "Expected LOD")
    UnicodePlots.lineplot!(p3, [0, lod_expected[end-1]], [0, lod_expected[end-1]])
    @show p1
    @show p2
    @show p3
end

"""
    tabularise(fit::Fit, metric::String = "cor")::DataFrame

Convert a Fit struct into a DataFrame for easier data manipulation and analysis.

# Arguments
- `fit::Fit`: A Fit struct containing model results and parameters
- `metric::String = "cor"`: The metric to extract from fit.metrics dictionary (default: "cor")

# Returns
- `DataFrame`: A DataFrame with the following columns:
  - `model`: The model name
  - `trait`: The trait name
  - `population`: Semicolon-separated string of unique population names
  - `metric`: The specified metric value from fit.metrics
  - `b_hat_labels`: Labels for the effect sizes
  - `b_hat`: Estimated effect sizes

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=100, l=10_000); fit.b_hat = rand(10_000); fit.model="some_model"; fit.trait="some_trait"; 

julia> fit.metrics = Dict("cor" => rand(), "rmse" => rand()); fit.populations .= "pop_1";

julia> df = tabularise(fit);

julia> size(df)
(10000, 6)
```
"""
function tabularise(fit::Fit, metric::String = "cor")::DataFrame
    # fit = Fit(n=100, l=10_000); fit.b_hat = rand(10_000); fit.model="some_model"; fit.trait="some_trait"; fit.metrics = Dict("cor" => rand(), "rmse" => rand()); fit.populations .= "pop_1";
    # metric = "cor"
    df = DataFrame(
        model = fit.model,
        trait = fit.trait,
        population = join(sort(unique(fit.populations)), ";"),
        metric = fit.metrics[metric],
        b_hat_labels = fit.b_hat_labels,
        b_hat = fit.b_hat,
    )
    rename!(df, :metric => Symbol(metric))
    df
end
