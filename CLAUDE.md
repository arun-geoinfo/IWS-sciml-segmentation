# IWS_sciml — Claude Code Instructions

## Stack
- **Language:** Julia 1.12+ with Lux.jl (neural networks), CUDA.jl (GPU), Zygote.jl (AD)
- **Task:** Physics-informed semantic segmentation of ocean internal waves from SAR imagery
- **Data:** Zenodo Oceanic Phenomena V2 (5,011 images, 256x256, 16-bit GeoTIFF + pixel masks)
- **Hardware:** RTX 2000 Ada 8GB VRAM, 32GB RAM

## Project layout
```
src/
  config/          # Configuration (device, defaults)
  dataloaders/     # Data loading and batching
  models/          # Neural network architectures (Lux.jl)
  training/        # Training loops and losses
  evaluation/      # Metrics, ROC, confusion matrices
  filters/         # Physics-based filtering (FFT, Radon)
  preprocessing/   # Data preprocessing
  inference/       # Inference and benchmarking
  visualization/   # Plotting utilities
specs/             # Component specification files (YAML)
test/              # Tests — behavior IDs map to spec files
data/              # Datasets (gitignored)
```

## How to run
```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific test file
julia --project=. test/runtests.jl

# Run the project gate (lint + tests + traceability)
bash scripts/gate.sh

# Train
julia --project=. src/training/train.jl

# Evaluate
julia --project=. src/evaluation/evaluate.jl
```

## Spec files
- Live in `specs/` as YAML, one per pipeline component
- Every behavior has an ID (e.g., `DL-001`, `SEG-003`)
- Tests reference behavior IDs via `@testset "DL-001: ..."` naming

## Definition of done
1. Spec file exists and is current for the component
2. All behavior IDs in the spec have at least one passing test
3. `scripts/gate.sh` passes (tests green, no orphan IDs)
4. No test deleted, skipped, or weakened to make it pass

## Rules
- Every feature starts with its spec file
- Never delete or skip tests or spec behaviors to make a build green
- Physics losses must have testable invariants (not just "looks right")
- Keep functions under CC 10
- Prefer editing existing files over creating new ones
