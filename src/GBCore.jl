module GBCore

using Random
using DataFrames
using StatsBase
using MixedModels
using UnicodePlots
using PrecompileTools: @compile_workload

include("all_structs.jl")
include("genomes.jl")
include("phenomes.jl")
include("trials.jl")

export Genomes, Phenomes, Trials, SimulatedEffects, TEBV
export deepcopy, hash, ==
export checkdims, dimensions, loci_alleles, loci, plot, slice, filter
export tabularise


# Precompile
@compile_workload begin
    genomes = Genomes(n = 2, p = 4)
    genomes.entries = ["entry_1", "entry_2"]
    genomes.loci_alleles = ["chr1\t1\tA|T\tA", "chr1\t2\tC|G\tG", "chr2\t3\tA|T\tA", "chr2\t4\tG|T\tG"]
    genomes.allele_frequencies = [0.50 0.25 0.12 0.6; 0.45 0.20 0.10 0.05]
    other = deepcopy(genomes)
    other.allele_frequencies[1] = 0.10
    hash(genomes)
    genomes == genomes
    genomes != other
    checkdims(genomes)
    dimensions(genomes)
    chromsomes, positions, alleles = loci_alleles(genomes)
    chromsomes, positions, loci_ini_idx, loci_fin_idx = loci(genomes)
    plot(genomes)
    sliced_genomes = slice(genomes, idx_entries = [1]; idx_loci_alleles = [1, 3])


end



end
