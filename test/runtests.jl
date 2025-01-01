using GBCore
using Test
using Documenter

Documenter.doctest(GBCore)

@testset "GBCore.jl" begin
    genomes = Genomes(n = 2, p = 10)
    @test isa(genomes, Genomes)
end
