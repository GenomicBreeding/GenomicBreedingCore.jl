using GBCore
using Test
using Documenter

Documenter.doctest(GBCore)

@testset "GBCore.jl" begin
    genomes = simulategenomes(n = 123, l = 5_000, verbose = false)
    trials, effects = GBCore.simulatetrials(
        genomes = genomes,
        n_years = 1,
        n_seasons = 1,
        n_harvests = 1,
        n_sites = 1,
        n_replications = 3,
        f_add_dom_epi = [0.1 0.01 0.01;],
        verbose = false,
    )
    tebv = analyse(trials, max_time_per_model = 1, verbose = false)
    phenomes = extractphenomes(tebv)
    @test isa(genomes, Genomes)
    @test isa(trials, Trials)
    @test isa(effects[1], SimulatedEffects)
    @test isa(tebv, TEBV)
    @test isa(phenomes, Phenomes)
    # Test Plotting which are not tested in the docs because copying the plots into the docstring is too impractical
    @test isnothing(GBCore.plot(genomes))
    @test isnothing(GBCore.plot(trials))
    @test isnothing(GBCore.plot(tebv))
    @test isnothing(GBCore.plot(phenomes))
end
