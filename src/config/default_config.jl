
module DefaultConfig

include("device_config.jl")
using .DeviceConfig

# ========================================
# 🎛️ USER-CONFIGURABLE COMPUTE MODE
# ========================================
# Options: :auto, :cpu, :gpu, :hybrid
const COMPUTE_MODE = :gpu  # ⬅️ CHANGE THIS TO SELECT MODE

# Get device based on mode (uses updated DeviceConfig)
const DEVICE, BACKEND = DeviceConfig.get_device(COMPUTE_MODE)

# ========================================
# Project Paths
# ========================================
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const RAW_TRAIN_IMAGES = joinpath(PROJECT_ROOT, "data", "raw", "train")
const RAW_TEST_IMAGES  = joinpath(PROJECT_ROOT, "data", "raw", "test")
const TRAIN_CSV = joinpath(PROJECT_ROOT, "data", "raw", "train.csv")
const TEST_CSV  = joinpath(PROJECT_ROOT, "data", "raw", "test.csv")
const PROC_TRAIN = joinpath(PROJECT_ROOT, "data", "processed", "train")
const PROC_TEST  = joinpath(PROJECT_ROOT, "data", "processed", "test")
const MODEL_SAVE_DIR = "models/"
# ========================================
# Image & Model Settings
# ========================================
const TARGET_SIZE = (256, 256)
const CHANNELS = 1
const NUM_CLASSES = 2
const RNG_SEED = 42

# ========================================
# Training Hyperparameters
# ========================================
const BATCH_SIZE = 128  # ⬅️ Adjust for GPU memory
const LEARNING_RATE = 3e-4
const EPOCHS = 30
const PATIENCE = 5  # Early stopping
const GPU_MEMORY_SAFETY_MARGIN = 0.2

# ========================================
# Display Configuration
# ========================================
function print_config()
    println("="^60)
    println("⚙️  CONFIGURATION")
    println("="^60)
    println("Compute Backend: $BACKEND")
    println("Compute Mode:    $COMPUTE_MODE")
    println("Device:          $(typeof(DEVICE))")
    println("Batch Size:      $BATCH_SIZE")
    println("Learning Rate:   $LEARNING_RATE")
    println("Image Size:      $TARGET_SIZE")
    println("="^60)
end

end  # module