module GenomicBreedingCore

using Random
using DataFrames
using LinearAlgebra
using StatsBase
using Distributions, SparseArrays, PDMats
using MixedModels, Turing
using MultivariateStats
using UnicodePlots, Plots
using JLD2
using ProgressMeter
using Suppressor
using Lux

# Structs
include("all_structs.jl")
# Genomic data
include("genomes/genomes.jl")
include("genomes/filter.jl")
include("genomes/merge.jl")
include("genomes/impute.jl")
# Phenomic data
include("phenomes/phenomes.jl")
include("phenomes/filter.jl")
include("phenomes/merge.jl")
# Trials data: single- or multi- environments, years, seasons and harvests
include("trials/trials.jl")
include("trials/filter.jl")
# Trial-estimated breeding values
include("tebv/tebv.jl")
include("tebv/linear_mixed_modelling.jl")
include("tebv/bayesian_modelling.jl")
# Linear and non-linear genotype-to-phenotype models struct
include("fit/fit.jl")
# Genotype-to-phenotype model cross-validation struct
include("cv/cv.jl")
# Genomic relationship matrix struct
include("grm/grm.jl")
include("grm/calc.jl")
# Simulations: genomes, trials, and mating
include("simulations/simulate_effects.jl")
include("simulations/simulate_genomes.jl")
include("simulations/simulate_trials.jl")
include("simulations/simulate_mating.jl")

export AbstractGB, Genomes, Phenomes, Trials, SimulatedEffects, TEBV, Fit, CV, GRM
export clone, hash, ==
export checkdims, dimensions, loci_alleles, loci, distances, plot, tabularise, summarise
export slice, sparsities, filter, filterbysparsity, filterbymaf, filterbypca, filterbysnplist
export simulatechromstruct,
    simulateposandalleles, simulatepopgroups, simulateldblocks, simulateperpopμΣ, simulateallelefreqs!
export simulategenomes, simulateeffects, simulategenomiceffects, simulatetrials
export histallelefreqs, simulatemating
export countlevels, @string2formula, trialsmodelsfomulae!, analyse, extractphenomes
export @stringevaluation, addcompositetrait
export maskmissing!, divideintomockscaffolds, estimateld, estimatedistances, knni, knnioptim, impute
export inflatediagonals!, grmsimple, grmploidyaware

end
