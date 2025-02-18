"""
    clone(x::CV)::CV

Clone a CV object

## Example
```jldoctest; setup = :(using GBCore)
julia> fit = Fit(n=1, l=2);

julia> cv = CV("replication_1", "fold_1", fit, ["population_1"], ["entry_1"], [0.0], [0.0], fit.metrics);

julia> copy_cv = clone(cv)
CV("replication_1", "fold_1", Fit("", ["", ""], [0.0, 0.0], "", [""], [""], [0.0], [0.0], Dict("" => 0.0)), ["population_1"], ["entry_1"], [0.0], [0.0], Dict("" => 0.0))
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

Hash a CV struct using the entries, populations and loci_alleles.
We deliberately excluded the allele_frequencies, and mask for efficiency.

## Examples
```jldoctest; setup = :(using GBCore)
julia> fit = Fit(n=1, l=2);

julia> cv = CV("replication_1", "fold_1", fit, ["population_1"], ["entry_1"], [0.0], [0.0], fit.metrics);

julia> typeof(hash(cv))
UInt64
```
"""
function Base.hash(x::CV, h::UInt)::UInt
    hash(
        CV,
        hash(
            x.replication,
            hash(
                x.fold,
                hash(
                    x.fit,
                    hash(
                        x.validation_populations,
                        hash(
                            x.validation_entries,
                            hash(x.validation_y_true, hash(x.validation_y_pred, hash(x.metrics, h))),
                        ),
                    ),
                ),
            ),
        ),
    )
end


"""
    Base.:(==)(x::CV, y::CV)::Bool

Equality of CV structs using the hash function defined for CV structs.

## Examples
```jldoctest; setup = :(using GBCore)
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

Check dimension compatibility of the fields of the CV struct

# Examples
```jldoctest; setup = :(using GBCore)
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

Export a vector of CV structs into data frames of metrics across entries and per validation entry

# Examples
```jldoctest; setup = :(using GBCore, DataFrames)
julia> fit_1 = Fit(n=1, l=2); fit_1.metrics = Dict("cor" => 0.0, "rmse" => 1.0); fit_1.trait = "trait_1";

julia> cv_1 = CV("replication_1", "fold_1", fit_1, ["population_1"], ["entry_1"], [0.0], [0.0], fit_1.metrics);

julia> fit_2 = Fit(n=1, l=2); fit_2.metrics = Dict("cor" => 1.0, "rmse" => 0.0); fit_2.trait = "trait_1";

julia> cv_2 = CV("replication_2", "fold_2", fit_2, ["population_2"], ["entry_2"], [0.0], [0.0], fit_2.metrics);

julia> cvs = [cv_1, cv_2];

julia> df_across_entries, df_per_entry = tabularise(cvs);

julia> names(df_across_entries)
8-element Vector{String}:
 "training_population"
 "validation_population"
 "trait"
 "model"
 "replication"
 "fold"
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
    # genomes = GBCore.simulategenomes(n=300, verbose=false); genomes.populations = StatsBase.sample(string.("pop_", 1:3), length(genomes.entries), replace=true);
    # trials, _ = GBCore.simulatetrials(genomes=genomes, n_years=1, n_seasons=1, n_harvests=1, n_sites=1, n_replications=1, verbose=false);
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
        @warn "Oh naur! This should not have happend!"
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

Summarise a vector of CV structs into:

- a data frame of mean metrics, and
- a data frame of mean and standard deviation of phenotype predictions per entry, trait and model.

# Examples
```jldoctest; setup = :(using GBCore, DataFrames)
julia> fit_1 = Fit(n=1, l=2); fit_1.metrics = Dict("cor" => 0.0, "rmse" => 1.0); fit_1.trait = "trait_1";

julia> cv_1 = CV("replication_1", "fold_1", fit_1, ["population_1"], ["entry_1"], [0.0], [0.0], fit_1.metrics);

julia> fit_2 = Fit(n=1, l=2); fit_2.metrics = Dict("cor" => 1.0, "rmse" => 0.0); fit_2.trait = "trait_1";

julia> cv_2 = CV("replication_2", "fold_2", fit_2, ["population_2"], ["entry_2"], [0.0], [0.0], fit_2.metrics);

julia> cvs = [cv_1, cv_2];

julia> df_summary, df_summary_per_entry = summarise(cvs);

julia> size(df_summary)
(2, 6)

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
        [[:cor] => mean, [:cor] => std],
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
