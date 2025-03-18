# GBCore

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://genomicbreeding.github.io/GBCore.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://genomicbreeding.github.io/GBCore.jl/dev/)
[![Build Status](https://github.com/GenomicBreeding/GBCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/GenomicBreeding/GBCore.jl/actions/workflows/CI.yml?query=branch%3Amain)

Core library for GenomicBreeding.jl which includes simulation functions.

## Dev stuff:

### REPL prelude

```shell
julia --threads 2,1 --load test/interactive_prelude.jl
```

### Format and test

```shell
time julia --threads 2,1 test/cli_tester.jl
```
