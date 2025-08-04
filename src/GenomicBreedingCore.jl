module GenomicBreedingCore

using Random
using DataFrames
using LinearAlgebra
using StatsBase, Clustering
using Distributions, SparseArrays, PDMats
using StatsModels, MixedModels
using Turing, MCMCDiagnosticTools, Zygote, ReverseDiff
using MultivariateStats
using Distances, CovarianceEstimation
using UnicodePlots
using JLD2
using ProgressMeter
using Suppressor
using Lux, Optimisers
using LuxCUDA, CUDA
# using LuxCUDA, AMDGPU, oneAPI # GPU support (Metal is for MacOS)
using ScatteredInterpolation

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
include("tebv/lmm.jl")
include("tebv/bayes.jl")
include("tebv/dl.jl")
# Linear and non-linear genotype-to-phenotype models struct
include("fit/fit.jl")
# Bayesian linear regression fit struct
include("blr/blr.jl")
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

export AbstractGB, Genomes, Phenomes, Trials, SimulatedEffects, DLModel, BLR, TEBV, Fit, CV, GRM
export clone, hash, ==
export checkdims, dimensions, loci_alleles, loci, distances, plot, tabularise, summarise, aggregateharvests
export slice,
    sparsities, filter, filterbysparsity, filterbymaf, filterbypca, filterbysnplist, removemissnaninf, extracteffects
export simulatechromstruct,
    simulateposandalleles, simulatepopgroups, simulateldblocks, simulateperpopμΣ, simulateallelefreqs!
export simulategenomes,
    simulatecovariancespherical,
    simulatecovariancediagonal,
    simulatecovariancerandom,
    simulatecovarianceautocorrelated,
    simulatecovariancekinship,
    simulateeffects,
    simulategenomiceffects,
    simulatetrials
export histallelefreqs, simulatemating
export countlevels, @string2formula, trialsmodelsfomulae!, analyse, extractphenomes
export extractXb,
    checkandfocalterms, instantiateblr, turingblr, extractmodelinputs, turingblrmcmc!, removespatialeffects!
export @stringevaluation, addcompositetrait
export maskmissing!, divideintomockscaffolds, estimateld, estimatedistances, knni, knnioptim, impute
export inflatediagonals!, grmsimple, grmploidyaware
# Experimental:
export analyseviaBLR
export makex, prepinputs, prepmodel, goodnessoffit, checkinputs, trainNN, makexnew, extracteffects, extractcovariances, optimNN, analyseviaNN

end
