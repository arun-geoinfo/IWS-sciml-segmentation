# src/dataloaders/lux_dataloader.jl
using Images, FileIO, Random, FilePathsBase, ImageTransformations

# include local config
include("../config/default_config.jl")
using .DefaultConfig

"""
    DataLoader(proc_dir::String; batch_size=32, rng_seed=42, shuffle=true, augment_fn=nothing)

Creates a lazy, Lux-compatible data loader.

# Arguments
- `proc_dir`: path to processed train/test folder
- `batch_size`: number of images per batch
- `rng_seed`: deterministic seed
- `shuffle`: randomize batches every epoch
- `augment_fn`: optional function to apply augmentations on the fly

# Returns
- An iterable yielding `(X, y)` batches
"""
mutable struct DataLoader
    image_paths::Vector{String}
    labels::Vector{Int}
    batch_size::Int
    rng_seed::Int
    shuffle::Bool
    augment_fn::Function
end

# -------------------------
# Constructor
# -------------------------
function DataLoader(proc_dir::String; batch_size=32, rng_seed=42, shuffle=true, augment_fn=nothing)
    class0_files = readdir(joinpath(proc_dir, "class0"); join=true)
    class1_files = readdir(joinpath(proc_dir, "class1"); join=true)

    image_paths = vcat(class0_files, class1_files)
    labels = vcat(fill(0, length(class0_files)), fill(1, length(class1_files)))

    # optional augmentation default: identity
    augment_fn = augment_fn === nothing ? x->x : augment_fn

    return DataLoader(image_paths, labels, batch_size, rng_seed, shuffle, augment_fn)
end

# -------------------------
# Iterator interface
# -------------------------
Base.iterate(dl::DataLoader, state=1) = _next_batch(dl, state)

function _next_batch(dl::DataLoader, state::Int)
    N = length(dl.image_paths)
    if state > N
        return nothing
    end

    # Shuffle at start of epoch
    if state == 1 && dl.shuffle
        Random.seed!(dl.rng_seed)
        perm = shuffle(1:N)
        dl.image_paths = dl.image_paths[perm]
        dl.labels = dl.labels[perm]
    end

    # Compute batch indices
    batch_end = min(state + dl.batch_size - 1, N)
    batch_idxs = state:batch_end

    # Lazy load images





# inside _next_batch
X_batch = Array{Float32, 4}(undef, DefaultConfig.TARGET_SIZE[1], DefaultConfig.TARGET_SIZE[2], 1, length(batch_idxs))

for (i, idx) in enumerate(batch_idxs)
    img = load(dl.image_paths[idx])
    img_gray = Gray.(img)
    img_resized = imresize(img_gray, DefaultConfig.TARGET_SIZE)
    X_batch[:, :, 1, i] = Float32.(img_resized) ./ 255   # assign to batch array
end

y_batch = Float32.(dl.labels[batch_idxs])





    return (X_batch, y_batch), batch_end + 1
end

# ===== Test block =====
if abspath(PROGRAM_FILE) == @__FILE__  # only runs if script is executed directly
    train_loader = DataLoader(DefaultConfig.PROC_TRAIN; batch_size=8)
    test_loader  = DataLoader(DefaultConfig.PROC_TEST; batch_size=8, shuffle=false)

    for (X, y) in train_loader
        println("Batch X size: ", size(X))
        println("Batch y size: ", size(y))
        break  # just first batch
    end
end
