module DefaultConfig

# Project root
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

# Data paths

const RAW_TRAIN_IMAGES = joinpath(PROJECT_ROOT, "data", "raw", "train")
const RAW_TEST_IMAGES  = joinpath(PROJECT_ROOT, "data", "raw", "test")




# Correct paths for your current structure
const TRAIN_CSV = joinpath(PROJECT_ROOT, "data", "raw", "train.csv")
const TEST_CSV  = joinpath(PROJECT_ROOT, "data", "raw", "test.csv")

const PROC_TRAIN = joinpath(PROJECT_ROOT, "data", "processed", "train")
const PROC_TEST  = joinpath(PROJECT_ROOT, "data", "processed", "test")

# Image settings
const TARGET_SIZE = (256, 256)
const CHANNELS = 1

# Lux settings
const NUM_CLASSES = 2
const RNG_SEED = 42

end
