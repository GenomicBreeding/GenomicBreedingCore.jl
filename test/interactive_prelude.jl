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
using Distributions
using MixedModels
using UnicodePlots, Plots
using ProgressMeter
using Suppressor
using Suppressor
