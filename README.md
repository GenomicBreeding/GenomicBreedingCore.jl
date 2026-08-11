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
julia --project=. -e 'using Pkg; Pkg.instantiate()' # For a fresh Julia installation
time julia --project=. --threads=2 test/cli_tester.jl
```

### Quick search and replace across the projects

```shell
find GenomicBreeding*/ -type f -name "*.jl" -exec sed -i 's/harvest/measurement/g' {} +
```

### Force stable docs on new release:

First create a new release and if the stable don't get updated then try the following:

```shell
TAG=v0.3.0 # should match the release tag you created on Github
git fetch origin tag $TAG
git show $TAG --no-patch
git checkout main
git commit --allow-empty -m "Rebuild docs"
git push
```