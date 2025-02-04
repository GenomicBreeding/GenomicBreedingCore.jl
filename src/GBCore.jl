module GBCore

using Random
using DataFrames
using LinearAlgebra
using StatsBase
using Distributions
using MixedModels
using UnicodePlots, Plots
using ProgressMeter
using Suppressor
using PrecompileTools: @compile_workload

include("all_structs.jl")
include("genomes.jl")
include("phenomes.jl")
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
export checkdims, dimensions, loci_alleles, loci, distances, plot, slice, filter, tabularise, summarise
export simulategenomes, simulateeffects, simulategenomiceffects, simulatetrials
export countlevels, @string2formula, trialsmodelsfomulae!, analyse, extractphenomes
export @stringevaluation, addcompositetrait

# Precompile (does this improve start-up time significantly?)
# @compile_workload begin
#     n = 10
#     genomes = simulategenomes(n = n)
#     genomes_copy = clone(genomes)
#     genomes_copy.entries[1] = "other_entry"
#     genomes_copy.loci_alleles[1] = "other_locus_allele"
#     genomes == genomes
#     genomes == genomes_copy
#     trials, effects = simulatetrials(
#         genomes = genomes,
#         f_add_dom_epi = [
#             0.50 0.25 0.13
#             0.90 0.00 0.00
#         ],
#         n_years = 1,
#         n_seasons = 1,
#         n_harvests = 1,
#         n_sites = 1,
#         n_replications = 2,
#     )
#     phenomes = Phenomes(n = n, t = 2)
#     phenomes.entries = trials.entries[1:n]
#     phenomes.populations = trials.populations[1:n]
#     phenomes.traits = trials.traits
#     phenomes.phenotypes = trials.phenotypes[1:n, :]
#     phenomes.mask .= true
#     phenomes_copy = clone(phenomes)
#     phenomes_copy.entries[1] = "other_entry"
#     phenomes_copy.traits[1] = "other_trait"
#     phenomes == phenomes
#     phenomes == phenomes_copy
#     merged_genomes, merged_phenomes = merge(genomes, phenomes)
#     tebv = analyse(trials, max_levels = 10)
#     fit = Fit(n = 1, l = 2)
#     fit_copy = clone(fit)
#     fit == fit
#     checkdims(fit)
# end



end
