using GBCore
using Test
using Documenter

Documenter.doctest(GBCore)

@testset "GBCore.jl" begin
    genomes = Genomes(n = 2, p = 10)
    phenomes = Phenomes(n = 2, t = 10)
    trials = Trials(n = 2, t = 10)
    @test isa(genomes, Genomes)
    @test isa(phenomes, Phenomes)
    @test isa(trials, Trials)
end
