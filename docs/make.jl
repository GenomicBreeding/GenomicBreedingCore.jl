using GenomicBreedingCore
using Documenter

DocMeta.setdocmeta!(GenomicBreedingCore, :DocTestSetup, :(using GenomicBreedingCore); recursive = true)

makedocs(;
    modules = [GenomicBreedingCore],
    authors = "jeffersonparil@gmail.com",
    sitename = "GenomicBreedingCore.jl",
    format = Documenter.HTML(;
        canonical = "https://GenomicBreeding.github.io/GenomicBreedingCore.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 1000000,
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/GenomicBreeding/GenomicBreedingCore.jl", devbranch = "main")
