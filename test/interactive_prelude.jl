using Pkg
Pkg.activate(".")
try
    Pkg.update()
catch
    nothing
end
using GenomicBreedingCore
using Random
using DataFrames
using LinearAlgebra
using StatsBase
using Distributions, SparseArrays, PDMats
using StatsModels, MixedModels
using Turing, MCMCDiagnosticTools, Zygote, ReverseDiff
using MultivariateStats
using Distances, CovarianceEstimation
using UnicodePlots
using ProgressMeter
using Suppressor
using JLD2
using Lux, Optimisers
using LuxCUDA, CUDA
# using LuxCUDA, AMDGPU, oneAPI # GPU support (Metal is for MacOS)
