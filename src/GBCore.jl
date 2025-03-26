module GBCore

using Random
using DataFrames
using LinearAlgebra
using StatsBase
using Distributions, SparseArrays, PDMats
using MixedModels, Metida
using MultivariateStats
using UnicodePlots, Plots
using JLD2
using ProgressMeter
using Suppressor
using Lux # for Fit deep learning model

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
# Trial-estimated breeding values: extraction of BLUPs of entries
include("tebv/tebv.jl")
include("tebv/linear_mixed_modelling.jl")
# Linear and non-linear genotype-to-phenotype models struct
include("fit/fit.jl")
# Genotype-to-phenotype model cross-validation struct
include("cv/cv.jl")
# Simulations: genomes, trials, and mating
include("simulation/simulate_effects.jl")
include("simulation/simulate_genomes.jl")
include("simulation/simulate_trials.jl")
include("simulation/simulate_mating.jl")

export AbstractGB, Genomes, Phenomes, Trials, SimulatedEffects, TEBV, Fit, CV
export clone, hash, ==
export checkdims, dimensions, loci_alleles, loci, distances, plot, tabularise, summarise
export slice, sparsities, filter, filterbysparsity, filterbymaf, filterbypca, filterbysnplist
export simulatechromstruct, simulateposandalleles, simulatepopgroups, simulateldblocks, simulateperpopμΣ, simulateallelefreqs!, simulategenomes
export simulateeffects, simulategenomiceffects, simulatetrials
export histallelefreqs, simulatemating
export countlevels, @string2formula, trialsmodelsfomulae!, analyse, extractphenomes
export @stringevaluation, addcompositetrait
export maskmissing!, divideintomockscaffolds, estimateld, estimatedistances, knni, knnioptim, impute

end
