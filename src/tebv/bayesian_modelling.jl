function analyse(
    trials::Trials,
    traits::Vector{String};
    GRM::Union{Matrix{Float64}, UniformScaling} = I, 
    other_covariates::Union{String, Nothing} = nothing,
    verbose::Bool = false)::TEBV
    # genomes = simulategenomes(n=500, l=10_000)
    # trials, simulated_effects = simulatetrials(genomes = genomes, n_years=10, n_seasons=2, n_harvests=1, n_sites=2, n_replications=3)
    # GRM::Union{Matrix{Float64}, UniformScaling} = I
    # other_covariates::Union{String, Nothing} = nothing
    # verbose::Bool = true
    # Check arguments
    if !checkdims(trials)
        error("The Trials struct is corrupted.")
    end
    if length(traits) > 0
        for trait in traits
            if !(trait ∈ trials.traits)
                throw(ArgumentError("The `traits` ($traits) argument is not a trait in the Trials struct."))
            end
        end
    end
    if !isnothing(other_covariates)
        if !(other_covariates ∈ trials.traits)
            throw(ArgumentError("The `other_covariates` ($other_covariates) argument is not a trait in the Trials struct."))
        end
    end
    # Tabularise the trials data
    df = tabularise(trials)
    # Make sure the blocks, rows, and columns are specific to each site
    df.blocks = string.(df.sites, "|", df.blocks)
    df.rows = string.(df.sites, "|", df.rows)
    df.cols = string.(df.sites, "|", df.cols)
    # Define the traits
    traits = if length(traits) == 0
        trials.traits
    else
        traits
    end
    # Identify non-fixed factors
    factors_all::Vector{String} = ["years", "seasons", "harvests", "sites", "blocks", "rows", "cols", "entries"]
    factors::Vector{String} = []
    for f in factors_all
        # f = factors_all[1]
        if length(unique(df[!, f])) > 1
            push!(factors, f)
        end
    end
    # If both blocks and rows are present, then we remove blocks as they are expected to be redundant
    if ("blocks" ∈ factors) && ("rows" ∈ factors)
        factors = filter(x -> x != "blocks", factors)
    end
    # Iterate per trait
    for trait in traits
        # trait = traits[1]
        # Define the formula for the model
        formula_string = string(trait, " ~ 1 + ", join(factors, " * "))
        formula_full = @eval(@string2formula($formula_string))



        f0 = @eval(@string2formula($(replace(formulae[i], "y" => "0"))))
        _, col_labels = coefnames(apply_schema(f0, schema(f0, df)))
        size(df)
        length(col_labels)
        X = modelmatrix(formula, df)
    end
end