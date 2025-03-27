"""
    clone(x::CV)::CV

Create a deep copy of a CV (cross-validation) object.

Creates a new CV object with deep copies of all fields from the input object.
The clone function ensures that modifications to the cloned object do not affect 
the original object.

# Arguments
- `x::CV`: The CV object to be cloned

# Returns
- `CV`: A new CV object containing deep copies of all fields from the input

# Example

Clone a CV object

## Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=1, l=2);

julia> cv = CV("replication_1", "fold_1", fit, ["population_1"], ["entry_1"], [0.0], [0.0], fit.metrics);

julia> copy_cv = clone(cv)
CV("replication_1", "fold_1", Fit("", ["", ""], [0.0, 0.0], "", [""], [""], [0.0], [0.0], Dict("" => 0.0), nothing), ["population_1"], ["entry_1"], [0.0], [0.0], Dict("" => 0.0))
```
"""
function clone(x::CV)::CV
    CV(
        deepcopy(x.replication),
        deepcopy(x.fold),
        clone(x.fit),
        deepcopy(x.validation_populations),
        deepcopy(x.validation_entries),
        deepcopy(x.validation_y_true),
        deepcopy(x.validation_y_pred),
        deepcopy(x.metrics),
    )
end

"""
    Base.hash(x::CV, h::UInt)::UInt

Compute a hash value for a CV (Cross-Validation) struct.

This method defines how CV structs should be hashed, which is useful for
using CV objects in hash-based collections like Sets or as Dict keys.

# Arguments
- `x::CV`: The CV struct to be hashed
- `h::UInt`: The hash value to be mixed with the new hash

# Returns
- `UInt`: A hash value for the CV struct

# Implementation Details
The hash is computed by combining the following fields:
- replication
- fold
- fit
- validation_populations
- validation_entries
- validation_y_true
- validation_y_pred
- metrics

# Example
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=1, l=2);

julia> cv = CV("replication_1", "fold_1", fit, ["population_1"], ["entry_1"], [0.0], [0.0], fit.metrics);

julia> typeof(hash(cv))
UInt64
```
"""
function Base.hash(x::CV, h::UInt)::UInt
    for field in fieldnames(typeof(x))
        # field = fieldnames(typeof(x))[1]
        h = hash(getfield(x, field), h)
    end
    h
end


"""
    Base.:(==)(x::CV, y::CV)::Bool

Compare two CV (Cross-Validation) structs for equality.

This method overloads the equality operator (`==`) for CV structs by comparing their hash values.
Two CV structs are considered equal if they have identical values for all fields.

# Arguments
- `x::CV`: First CV struct to compare
- `y::CV`: Second CV struct to compare

# Returns
- `Bool`: `true` if the CV structs are equal, `false` otherwise

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=1, l=2);

julia> cv_1 = CV("replication_1", "fold_1", fit, ["population_1"], ["entry_1"], [0.0], [0.0], fit.metrics);

julia> cv_2 = clone(cv_1);

julia> cv_3 = clone(cv_1); cv_3.replication = "other_replication";

julia> cv_1 == cv_2
true

julia> cv_1 == cv_3
false
```
"""
function Base.:(==)(x::CV, y::CV)::Bool
    hash(x) == hash(y)
end


"""
    checkdims(cv::CV)::Bool

Check dimension compatibility of the fields of the CV struct.

The function verifies that:
- The fit object dimensions are valid
- The number of validation populations matches the number of validation entries
- The number of validation true values matches the number of validation predictions
- The number of metrics matches the number of metrics in the fit object

Returns:
- `true` if all dimensions are compatible
- `false` if any dimension mismatch is found

# Examples
```jldoctest; setup = :(using GenomicBreedingCore)
julia> fit = Fit(n=1, l=2);

julia> cv = CV("replication_1", "fold_1", fit, ["population_1"], ["entry_1"], [0.0], [0.0], fit.metrics);

julia> checkdims(cv)
true

julia> cv.validation_y_true = [0.0, 0.0];

julia> checkdims(cv)
false
```
"""
function checkdims(cv::CV)::Bool
    n = length(cv.validation_populations)
    if !checkdims(cv.fit) ||
       (length(cv.validation_entries) != n) ||
       (length(cv.validation_y_true) != n) ||
       (length(cv.validation_y_pred) != n) ||
       (length(cv.metrics) != length(cv.fit.metrics))
        return false
    end
    true
end


"""
    tabularise(cvs::Vector{CV})::Tuple{DataFrame,DataFrame}

Convert a vector of CV (Cross-Validation) structs into two DataFrames containing metrics and predictions.

# Arguments
- `cvs::Vector{CV}`: Vector of CV structs containing cross-validation results

# Returns
- `Tuple{DataFrame,DataFrame}`: A tuple of two DataFrames:
  1. `df_across_entries`: Contains aggregated metrics across entries with columns:
     - `training_population`: Semicolon-separated list of training populations
     - `validation_population`: Semicolon-separated list of validation populations
     - `trait`: Name of the trait
     - `model`: Name of the model used
     - `replication`: Replication identifier
     - `fold`: Fold identifier
     - `training_size`: Number of entries in training set
     - `validation_size`: Number of entries in validation set
     - Additional columns for each metric (e.g., `cor`, `rmse`)

  2. `df_per_entry`: Contains per-entry predictions with columns:
     - `training_population`: Training population identifier
     - `validation_population`: Validation population identifier
     - `entry`: Entry identifier
     - `trait`: Name of the trait
     - `model`: Name of the model used
     - `replication`: Replication identifier
     - `fold`: Fold identifier
     - `y_true`: True values
     - `y_pred`: Predicted values

# Throws
- `ArgumentError`: If input vector is empty or if any CV struct is corrupted

# Notes
- Warns if there are empty CV structs resulting from insufficient training sizes or fixed traits
- Metrics are extracted from the `metrics` dictionary in each CV struct
- Population identifiers are sorted and joined with semicolons when multiple populations exist

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, DataFrames)
julia> fit_1 = Fit(n=1, l=2); fit_1.metrics = Dict("cor" => 0.0, "rmse" => 1.0); fit_1.trait = "trait_1";

julia> cv_1 = CV("replication_1", "fold_1", fit_1, ["population_1"], ["entry_1"], [0.0], [0.0], fit_1.metrics);

julia> fit_2 = Fit(n=1, l=2); fit_2.metrics = Dict("cor" => 1.0, "rmse" => 0.0); fit_2.trait = "trait_1";

julia> cv_2 = CV("replication_2", "fold_2", fit_2, ["population_2"], ["entry_2"], [0.0], [0.0], fit_2.metrics);

julia> cvs = [cv_1, cv_2];

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> names(df_across_entries)
10-element Vector{String}:
 "training_population"
 "validation_population"
 "trait"
 "model"
 "replication"
 "fold"
 "training_size"
 "validation_size"
 "cor"
 "rmse"

julia> df_across_entries[!, [:cor, :rmse]]
2×2 DataFrame
 Row │ cor      rmse    
     │ Float64  Float64 
─────┼──────────────────
   1 │     0.0      1.0
   2 │     1.0      0.0

julia> names(df_per_entry)
9-element Vector{String}:
 "training_population"
 "validation_population"
 "entry"
 "trait"
 "model"
 "replication"
 "fold"
 "y_true"
 "y_pred"

julia> df_per_entry[!, [:entry, :y_true, :y_pred]]
2×3 DataFrame
 Row │ entry    y_true   y_pred  
     │ String   Float64  Float64 
─────┼───────────────────────────
   1 │ entry_1      0.0      0.0
   2 │ entry_2      0.0      0.0
```
"""
function tabularise(cvs::Vector{CV})::Tuple{DataFrame,DataFrame}
    # genomes = GenomicBreedingCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GenomicBreedingCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
    # phenomes = extractphenomes(trials);
    # cvs, notes = cvbulk(genomes=genomes, phenomes=phenomes, models = [ols, ridge, lasso], n_replications=2, n_folds=2);
    # Check arguments
    c = length(cvs)
    if c < 1
        throw(ArgumentError("Input vector of CV structs is empty."))
    end
    # Extract the names of the genomic prediction cross-validation accuracy metrics
    metric_names = string.(keys(cvs[1].metrics))
    # Extract the metrics calculated across entries per trait, replication, fold and model
    df_across_entries = DataFrames.DataFrame(
        training_population = fill("", c),
        validation_population = fill("", c),
        trait = fill("", c),
        model = fill("", c),
        replication = fill("", c),
        fold = fill("", c),
        training_size = fill(0, c),
        validation_size = fill(0, c),
    )
    for metric in metric_names
        df_across_entries[!, metric] = fill(0.0, c)
    end
    # At the same time extract individual entry predictions
    training_populations::Vector{String} = []
    validation_populations::Vector{String} = []
    entries::Vector{String} = []
    traits::Vector{String} = []
    models::Vector{String} = []
    replications::Vector{String} = []
    folds::Vector{String} = []
    y_trues::Vector{Float64} = []
    y_preds::Vector{Float64} = []
    idx_non_missing_across_entries = []
    for i = 1:c
        # i = 1
        # println(i)
        if !checkdims(cvs[i])
            throw(ArgumentError("The CV struct at index " * string(i) * " is corrupted."))
        end
        df_across_entries.training_population[i] = join(sort(unique(cvs[i].fit.populations)), ";")
        df_across_entries.validation_population[i] = join(sort(unique(cvs[i].validation_populations)), ";")
        df_across_entries.trait[i] = cvs[i].fit.trait
        df_across_entries.model[i] = cvs[i].fit.model
        df_across_entries.replication[i] = cvs[i].replication
        df_across_entries.fold[i] = cvs[i].fold
        df_across_entries.training_size[i] = length(cvs[i].fit.entries)
        df_across_entries.validation_size[i] = length(cvs[i].validation_entries)
        if cvs[i].metrics == Dict("" => 0.0)
            continue
        end
        for metric in metric_names
            # metric = metric_names[1]
            # println(metric)
            df_across_entries[i, metric] = cvs[i].metrics[metric]
        end
        n = length(cvs[i].validation_entries)
        append!(training_populations, repeat([join(sort(unique(cvs[i].fit.populations)), ";")], n))
        append!(validation_populations, cvs[i].validation_populations)
        append!(entries, cvs[i].validation_entries)
        append!(traits, repeat([cvs[i].fit.trait], n))
        append!(models, repeat([cvs[i].fit.model], n))
        append!(replications, repeat([cvs[i].replication], n))
        append!(folds, repeat([cvs[i].fold], n))
        append!(y_trues, cvs[i].validation_y_true)
        append!(y_preds, cvs[i].validation_y_pred)
        append!(idx_non_missing_across_entries, i)
    end
    if length(idx_non_missing_across_entries) < nrow(df_across_entries)
        @warn "You have empty CV structs resulting from training size/s less than 5 and/or fixed traits."
        df_across_entries = df_across_entries[idx_non_missing_across_entries, :]
    end
    # Metrics per entry
    df_per_entry = DataFrames.DataFrame(
        training_population = training_populations,
        validation_population = validation_populations,
        entry = entries,
        trait = traits,
        model = models,
        replication = replications,
        fold = folds,
        y_true = y_trues,
        y_pred = y_preds,
    )
    # Output
    (df_across_entries, df_per_entry)
end

"""
    summarise(cvs::Vector{CV})::Tuple{DataFrame,DataFrame}

Summarize cross-validation results from a vector of CV structs into two DataFrames.

# Returns
- A tuple containing two DataFrames:
  1. Summary DataFrame with mean metrics across entries, replications, and folds
     - Contains means and standard deviations of correlation coefficients
     - Includes average training and validation set sizes
     - Grouped by training population, validation population, trait, and model
  2. Entry-level DataFrame with phenotype prediction statistics
     - Contains true phenotype values, predicted means (μ), and standard deviations (σ)
     - Grouped by training population, validation population, trait, model, and entry

# Arguments
- `cvs::Vector{CV}`: Vector of CV structs containing cross-validation results

# Notes
- Validates dimensions of input CV structs before processing
- Handles missing values in phenotype predictions

# Throws
- `ArgumentError`: If any CV struct in the input vector has inconsistent dimensions

# Examples
```jldoctest; setup = :(using GenomicBreedingCore, DataFrames)
julia> fit_1 = Fit(n=1, l=2); fit_1.metrics = Dict("cor" => 0.0, "rmse" => 1.0); fit_1.trait = "trait_1";

julia> cv_1 = CV("replication_1", "fold_1", fit_1, ["population_1"], ["entry_1"], [0.0], [0.0], fit_1.metrics);

julia> fit_2 = Fit(n=1, l=2); fit_2.metrics = Dict("cor" => 1.0, "rmse" => 0.0); fit_2.trait = "trait_1";

julia> cv_2 = CV("replication_2", "fold_2", fit_2, ["population_2"], ["entry_2"], [0.0], [0.0], fit_2.metrics);

julia> cvs = [cv_1, cv_2];

julia> df_summary, df_summary_per_entry = summarise(cvs);

julia> size(df_summary)
(2, 8)

julia> size(df_summary_per_entry)
(2, 8)
```
"""
function summarise(cvs::Vector{CV})::Tuple{DataFrame,DataFrame}
    # fit_1 = Fit(n = 1, l = 2); fit_1.trait = "trait_1"
    # fit_1.metrics = Dict("cor" => 0.0, "rmse" => 1.0)
    # cv_1 = CV("replication_1", "fold_1", fit_1, ["population_1"], ["entry_1"], [0.0], [0.0], fit_1.metrics)
    # fit_2 = Fit(n = 1, l = 2); fit_2.trait = "trait_2"
    # fit_2.metrics = Dict("cor" => 1.0, "rmse" => 0.0)
    # cv_2 = CV("replication_2", "fold_2", fit_2, ["population_2"], ["entry_2"], [0.0], [0.0], fit_2.metrics)
    # cvs = [cv_1, cv_2]
    # Check arguments
    for (i, cv) in enumerate(cvs)
        if !checkdims(cv)
            throw(ArgumentError("The element number " * string(i) * " in the vector of CV structs is corrupted."))
        end
    end
    # Tabularise
    df_across_entries, df_per_entry = tabularise(cvs)
    # Summarise across entries, reps and folds
    df_summary = combine(
        groupby(df_across_entries, [:training_population, :validation_population, :trait, :model]),
        [:cor => mean, :cor => std, :training_size => mean, :validation_size => mean],
    )
    # Mean and standard deviation of phenotype predictions per entry
    df_summary_per_entry =
        combine(groupby(df_per_entry, [:training_population, :validation_population, :trait, :model, :entry])) do g
            idx = findall(.!ismissing.(g.y_true) .&& .!ismissing.(g.y_pred))
            y_true = mean(g.y_true)
            μ = mean(g.y_pred[idx])
            σ = std(g.y_pred[idx])
            return (y_true = y_true, μ = μ, σ = σ)
        end
    (df_summary, df_summary_per_entry)
end
