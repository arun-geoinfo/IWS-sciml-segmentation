# run_pipeline.jl

# -------------------------------
# Force CUDA.jl to use internal binaries and suppress system CUDA warnings
# -------------------------------
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "true"
ENV["JULIA_CUDA_VERBOSE"] = "0"

# Prevent the system CUDA libraries from loading
ENV["LD_LIBRARY_PATH"] = ""

# -------------------------------
# Include all your pipeline stages
# -------------------------------
include("src/preprocessing/preprocess_data.jl")
include("src/dataloaders/dataloader.jl")
include("src/models/sar_cnn_model.jl")
include("src/train.jl")
include("src/evaluate.jl")
