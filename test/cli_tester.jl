using Pkg;
Pkg.update();
using JuliaFormatter
Pkg.activate(".")
# Format
format(".")
# Test
include("runtests.jl")
