# GenomicBreedingCore

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://genomicbreeding.github.io/GenomicBreedingCore.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://genomicbreeding.github.io/GenomicBreedingCore.jl/dev/)
[![Build Status](https://github.com/GenomicBreeding/GenomicBreedingCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GenomicBreeding/GenomicBreedingCore.jl/actions/workflows/CI.yml?query=branch%3Amain)

Core library for GenomicBreeding.jl which includes simulation functions.

## Dev stuff:

### REPL prelude

```shell
julia --project=. --threads=2,1 --load test/interactive_prelude.jl
```

### Format and test

```shell
time julia --project=. --threads=2 test/cli_tester.jl
```

### Quick search and replace across the projects

```shell
find GenomicBreeding*/ -type f -name "*.jl" -exec sed -i 's/harvest/measurement/g' {} +
```