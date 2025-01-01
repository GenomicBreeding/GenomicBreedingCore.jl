using GBCore
using Documenter

DocMeta.setdocmeta!(GBCore, :DocTestSetup, :(using GBCore); recursive = true)

makedocs(;
    modules = [GBCore],
    authors = "jeffersonparil@gmail.com",
    sitename = "GBCore.jl",
    format = Documenter.HTML(;
        canonical = "https://jeffersonfparil.github.io/GBCore.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/jeffersonfparil/GBCore.jl", devbranch = "main")
