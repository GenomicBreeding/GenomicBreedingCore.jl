module GBCore

using Random
using DataFrames
using LinearAlgebra
using StatsBase
using Distributions
using MixedModels, Metida
using MultivariateStats
using UnicodePlots, Plots
using JLD2
using ProgressMeter
using Suppressor
using Lux # for Fit deep learning model

include("all_structs.jl")
include("genomes.jl")
include("phenomes.jl")
include("impute.jl")
include("trials.jl")
include("tebv.jl")
include("fit.jl")
include("cv.jl")

include("simulation/simulate_effects.jl")
include("simulation/simulate_genomes.jl")
include("simulation/simulate_trials.jl")
include("simulation/simulate_mating.jl")

export AbstractGB, Genomes, Phenomes, Trials, SimulatedEffects, TEBV, Fit, CV
export clone, hash, ==
export checkdims, dimensions, loci_alleles, loci, distances, plot, slice, sparsities, filter, tabularise, summarise
export simulategenomes, simulateeffects, simulategenomiceffects, simulatetrials
export histallelefreqs, simulatemating
export countlevels, @string2formula, trialsmodelsfomulae!, analyse, extractphenomes
export @stringevaluation, addcompositetrait
export maskmissing!, divideintomockscaffolds, estimateld, estimatedistances, knni, knnioptim, impute

end
