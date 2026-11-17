# Pluto-compatible default config - NO CONST DECLARATIONS
PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

# Default to CPU - we'll detect GPU in the notebook
BACKEND = :cpu
DEVICE_FUNC = identity

# Simple function to get device (no CUDA loading)
function get_device_info()
    # We'll detect GPU in the notebook, not here
    return identity, :cpu
end