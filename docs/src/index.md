```@meta
CurrentModule = GenomicBreedingCore
```

# GenomicBreedingCore.jl

A Julia framework for genomic breeding, quantitative genetics, simulation, breeding value estimation, genomic prediction, cross-validation, and multi-environment trial analysis. It provides a complete workflow from genomic data simulation through phenotype analysis and genomic selection. 

---

# Features

- Genomic data management
- Phenotypic data management
- Multi-environment trial analysis
- Breeding value estimation (TEBV)
- Genomic prediction
- Cross-validation workflows
- Genomic relationship matrix (GRM) generation
- Missing data imputation
- Genome and trait simulation
- Mating simulation
- Neural-network based prediction
- Bayesian and mixed-model infrastructure 

---

# Architecture

```text
GenomicBreedingCore
│
├── Struct Definitions
│   └── all_structs.jl
│
├── Genomic Data
│   ├── genomes.jl
│   ├── filter.jl
│   ├── merge.jl
│   └── impute.jl
│
├── Phenomic Data
│   ├── phenomes.jl
│   ├── filter.jl
│   └── merge.jl
│
├── Trials
│   ├── trials.jl
│   ├── filter.jl
│   └── merge.jl
│
├── Breeding Value Estimation
│   ├── tebv.jl
│   └── lmm.jl
│
├── Genomic Prediction
│   └── fit.jl
│
├── Cross Validation
│   └── cv.jl
│
├── Genomic Relationship Matrices
│   ├── grm.jl
│   └── calc.jl
│
└── Simulations
    ├── simulate_effects.jl
    ├── simulate_genomes.jl
    ├── simulate_trials.jl
    └── simulate_mating.jl
```

---

# Core Types

## AbstractGB

Base abstract type from which all major package objects inherit.

```julia
abstract type AbstractGB end
```

---

## Genomes

Stores genomic marker information.

### Fields

```julia
entries
populations
loci_alleles
allele_frequencies
allele_frequencies_homologous_chroms
mask
```

### Example

```julia
genomes = Genomes(n=100, p=1000)
```

---

## Phenomes

Stores phenotype measurements.

### Fields

```julia
entries
populations
traits
phenotypes
mask
```

### Example

```julia
phenomes = Phenomes(n=100, t=10)
```

---

## Trials

Stores raw field-trial information.

Includes:

```text
Years
Seasons
Measurements
Sites
Replications
Blocks
Rows
Columns
Entries
Populations
Phenotypes
```

---

## TEBV

Trial Estimated Breeding Values.

Contains:

```text
Traits
Formulas
Mixed models
BLUE tables
BLUP tables
Breeding-value phenomes
```

---

## Fit

Stores genomic prediction model outputs.

Contains:

```text
Model
Marker effects
Predictions
Observed values
Metrics
Trait
```

---

## CV

Cross-validation result object.

Contains:

```text
Replication
Fold
Validation predictions
Metrics
Fit object
```

---

## GRM

Genomic Relationship Matrix.

Contains:

```text
Entries
Loci
Relationship matrix
```

---

## DLModel

Deep-learning model container.

Tracks:

```text
Model
Training progress
Predictions
Statistics
Features
Validation metrics
```

---

## SimulatedEffects

Stores simulated:

```text
Additive effects
Dominance effects
Epistasis
Year effects
Season effects
Site effects
Spatial effects
G×E interactions
```

---

# Quick Start

## 1. Simulate Genomic Data

```julia
using GenomicBreedingCore

genomes = simulategenomes(
    n = 100,
    l = 1000,
    n_alleles = 3,
    verbose = false
)

dimensions(genomes)
```

Expected output:

```julia
Dict(
    "n_entries" => 100,
    "n_loci" => 1000
)
```

---

## 2. Quality Control

Remove markers with low minor allele frequency.

```julia
filtered_genomes = filterbymaf(
    genomes,
    maf = 0.05
)
```

Remove missing-marker loci and samples.

```julia
filtered_genomes = filterbysparsity(
    filtered_genomes,
    max_entry_sparsity = 0.05,
    max_locus_sparsity = 0.05
)
```

---

## 3. Impute Missing Data

```julia
imputed_genomes, mae = impute(
    filtered_genomes,
    verbose = false
)

println(mae)
```

---

## 4. Generate Trials

```julia
trials, effects = simulatetrials(
    genomes = imputed_genomes,
    n_years = 2,
    n_seasons = 2,
    n_measurements = 1,
    n_sites = 3,
    n_replications = 2,
    verbose = false
)
```

---

## 5. Estimate Breeding Values

```julia
tebv = analyse(
    trials,
    verbose = false
)
```

Access breeding values:

```julia
tebv.phenomes[1]
```

---

## 6. Construct a GRM

Simple genomic relationship matrix:

```julia
grm = grmsimple(genomes)
```

Ploidy-aware GRM:

```julia
grm = grmploidyaware(
    genomes,
    ploidy = 2
)
```

---

## 7. Visualise Data

Genomic data:

```julia
plot(genomes)
```

Phenotypes:

```julia
plot(phenomes)
```

Prediction results:

```julia
plot(fit)
```

---

# Example Workflow

The following example demonstrates a complete breeding pipeline.

```julia
using GenomicBreedingCore

#
# STEP 1 – Simulate genomes
#
genomes = simulategenomes(
    n = 300,
    l = 5000,
    n_alleles = 3,
    verbose = false
)

#
# STEP 2 – Filter markers
#
genomes = filterbymaf(
    genomes,
    maf = 0.05
)

genomes = filterbysparsity(
    genomes,
    max_entry_sparsity = 0.05,
    max_locus_sparsity = 0.05
)

#
# STEP 3 – Impute missing values
#
genomes, mae = impute(
    genomes,
    verbose = false
)

println("Expected imputation MAE = $mae")

#
# STEP 4 – Simulate field trials
#
trials, effects = simulatetrials(
    genomes = genomes,
    n_years = 2,
    n_seasons = 2,
    n_measurements = 1,
    n_sites = 4,
    n_replications = 2,
    verbose = false
)

#
# STEP 5 – Estimate breeding values
#
tebv = analyse(
    trials,
    verbose = false
)

#
# STEP 6 – Extract breeding values
#
phenomes = tebv.phenomes[1]

#
# STEP 7 – Build GRM
#
grm = grmploidyaware(
    genomes,
    ploidy = 2
)

#
# STEP 8 – Inspect outputs
#
println(dimensions(genomes))
println(dimensions(phenomes))
println(size(grm.genomic_relationship_matrix))
```

---

# Common Operations

## Slice Data

### Genomes

```julia
subset = slice(
    genomes,
    idx_entries = 1:50,
    idx_loci_alleles = 1:1000
)
```

### Phenomes

```julia
subset = slice(
    phenomes,
    idx_entries = 1:50,
    idx_traits = 1:5
)
```

---

## Merge Data

### Merge Genomes

```julia
combined = merge(
    genomes1,
    genomes2
)
```

### Merge Phenomes

```julia
combined = merge(
    phenomes1,
    phenomes2
)
```

### Merge Genomes and Phenomes

```julia
genomes2, phenomes2 = merge(
    genomes,
    phenomes
)
```

---

## Create Composite Traits

```julia
phenomes2 = addcompositetrait(
    phenomes,
    composite_trait_name = "SelectionIndex",
    formula_string = "(Yield * 0.7) + (Protein * 0.3)"
)
```

---

# Distance Calculations

## Genomic Distances

```julia
loci_names, entries, dist = distances(
    genomes,
    distance_metrics = ["correlation", "χ²"]
)
```

Supported metrics:

```text
euclidean
correlation
covariance
mad
rmsd
χ²
```

---

## Phenotypic Distances

```julia
traits, entries, dist = distances(
    phenomes,
    distance_metrics = ["correlation"]
)
```

---

# Simulation Functions

## Simulate Effects

```julia
effects = simulateeffects(
    p = 10,
    q = 3
)
```

---

## Simulate Genomic Effects

```julia
G, B = simulategenomiceffects(
    genomes = genomes,
    f_additive = 0.05,
    f_dominance = 0.10,
    f_epistasis = 0.05
)
```

---

## Simulate Mating

```julia
offspring = simulatemating(...)
```

---

# Exported Types

```julia
Genomes
Phenomes
Trials
TEBV
Fit
CV
GRM
DLModel
SimulatedEffects
```

# Exported Workflows

```julia
analyse
simulategenomes
simulatetrials
simulatemating

filter
slice
merge

impute

grmsimple
grmploidyaware

addcompositetrait
plot
```

---

# Recommended Workflow

```text
Simulate genomes
      ↓
Quality control
      ↓
Imputation
      ↓
Create trials
      ↓
Estimate breeding values
      ↓
Construct GRM
      ↓
Fit prediction models
      ↓
Cross validation
      ↓
Selection decisions
```

Documentation for [GenomicBreedingCore](https://github.com/GenomicBreeding/GenomicBreedingCore.jl).

```@index
```

```@autodocs
Modules = [GenomicBreedingCore]
```
