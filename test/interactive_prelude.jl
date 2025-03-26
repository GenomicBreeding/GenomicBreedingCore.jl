using Pkg
Pkg.activate(".")
try
    Pkg.update()
catch
    nothing
end
using GBCore
using Random
using DataFrames
using LinearAlgebra
using StatsBase
using Distributions, SparseArrays, PDMats
using MixedModels, Metida
using MultivariateStats
using UnicodePlots, Plots
using ProgressMeter
using Suppressor
using JLD2
using Lux # for Fit deep learning model
