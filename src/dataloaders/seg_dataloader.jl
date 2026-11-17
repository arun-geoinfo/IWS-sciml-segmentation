# ============================================================================
# seg_dataloader.jl — Segmentation DataLoader for Zenodo Oceanic Phenomena
#
# Loads GeoTIFF images + PNG label masks, filters for a target class (IW=12),
# splits into train/val/test, batches in WHCN layout for Lux.jl.
#
# Spec: specs/data_loader.yaml (DL-001 through DL-006)
# ============================================================================

module SegDataLoader

using Images, FileIO, Random, Statistics
using ImageTransformations

export SegData, create_loaders, SegBatchIterator,
       dataset_size, get_split_sizes

# ============================================================================
# Configuration
# ============================================================================
const IW_CLASS_INDEX = 12  # pixel value for Internal Waves in label masks

# ============================================================================
# Data container
# ============================================================================
struct SegData
    image_paths::Vector{String}
    mask_paths::Vector{String}
    target_size::Tuple{Int,Int}
    target_class::Int
end

# ============================================================================
# DL-001: Discover matching image-mask pairs
# ============================================================================
"""
    discover_pairs(data_dir; image_subdir, label_subdir)

Find all matching image-mask pairs by filename stem.
Images are .tiff in Image_Geo/, masks are .png in label/.
"""
function discover_pairs(data_dir::String;
                        image_subdir::String = "Image_Geo",
                        label_subdir::String = "label")
    img_dir = joinpath(data_dir, image_subdir)
    lbl_dir = joinpath(data_dir, label_subdir)

    isdir(img_dir) || error("Image directory not found: $img_dir")
    isdir(lbl_dir) || error("Label directory not found: $lbl_dir")

    # Build stem -> path maps
    img_files = filter(f -> endswith(f, ".tiff") || endswith(f, ".tif"), readdir(img_dir))
    lbl_files = filter(f -> endswith(f, ".png"), readdir(lbl_dir))

    img_stems = Dict(splitext(f)[1] => joinpath(img_dir, f) for f in img_files)
    lbl_stems = Dict(splitext(f)[1] => joinpath(lbl_dir, f) for f in lbl_files)

    # Match by stem
    common_stems = sort(collect(intersect(keys(img_stems), keys(lbl_stems))))

    image_paths = [img_stems[s] for s in common_stems]
    mask_paths  = [lbl_stems[s] for s in common_stems]

    return image_paths, mask_paths
end

# ============================================================================
# DL-006: Filter for images containing the target class
# ============================================================================
"""
    filter_by_class(image_paths, mask_paths, target_class)

Keep only pairs where the mask contains at least one pixel of target_class.
"""
function filter_by_class(image_paths::Vector{String},
                         mask_paths::Vector{String},
                         target_class::Int)
    keep = Int[]
    for i in eachindex(mask_paths)
        mask = load_mask_raw(mask_paths[i])
        if any(==(UInt8(target_class)), mask)
            push!(keep, i)
        end
    end
    return image_paths[keep], mask_paths[keep]
end

# ============================================================================
# DL-002: Load and preprocess image (16-bit TIFF -> Float32 [0,1])
# ============================================================================
"""
    load_image(path, target_size) -> Array{Float32,3}

Load a 16-bit GeoTIFF, normalize to [0,1] Float32, resize, return as (H,W,1).
"""
function load_image(path::String, target_size::Tuple{Int,Int})
    img = load(path)

    # Convert to grayscale Float32
    img_gray = Gray.(img)
    img_f32 = Float32.(img_gray)

    # Resize if needed
    if size(img_f32) != target_size
        img_f32 = Float32.(imresize(img_f32, target_size))
    end

    # Normalize to [0,1] — the images are already loaded as normalized by Images.jl
    # (FileIO/Images maps UInt16 range to [0,1] automatically)
    # Clamp to be safe
    img_f32 = clamp.(img_f32, 0.0f0, 1.0f0)

    # Reshape to (H, W, 1)
    return reshape(img_f32, target_size[1], target_size[2], 1)
end

# ============================================================================
# DL-003: Load mask and binarize for target class
# ============================================================================
"""
    load_mask_raw(path) -> Matrix{UInt8}

Load the raw PNG label mask as UInt8 matrix (class indices).
"""
function load_mask_raw(path::String)
    img = load(path)
    # Palette PNGs load as IndirectArray — extract raw index values directly
    # The .index field holds UInt8 class indices (0=background, 12=IW, etc.)
    if hasproperty(img, :index)
        return Matrix{UInt8}(getfield(img, :index))
    end
    # Fallback for grayscale masks
    return UInt8.(round.(Float32.(Gray.(img)) .* 255))
end

"""
    load_mask_binary(path, target_class, target_size) -> Array{Float32,3}

Load mask, binarize (target_class -> 1.0, else -> 0.0), return as (H,W,1).
"""
function load_mask_binary(path::String, target_class::Int,
                          target_size::Tuple{Int,Int})
    raw = load_mask_raw(path)

    # Binarize: target class -> 1.0, everything else -> 0.0
    binary = Float32.(raw .== UInt8(target_class))

    # Resize with nearest-neighbor (preserve binary values)
    if size(binary) != target_size
        binary = Float32.(imresize(binary, target_size, method=Constant()))
        # Re-binarize after resize to ensure clean 0/1
        binary = Float32.(binary .>= 0.5f0)
    end

    return reshape(binary, target_size[1], target_size[2], 1)
end

# ============================================================================
# DL-004: Train/val/test split
# ============================================================================
"""
    split_data(image_paths, mask_paths, ratios; seed)

Split into train/val/test with no overlap. Returns 3 tuples of (img_paths, mask_paths).
"""
function split_data(image_paths::Vector{String},
                    mask_paths::Vector{String},
                    ratios::Tuple{Float64,Float64,Float64};
                    seed::Int = 42)
    @assert length(ratios) == 3
    @assert abs(sum(ratios) - 1.0) < 1e-6 "Split ratios must sum to 1.0"
    @assert length(image_paths) == length(mask_paths)

    N = length(image_paths)
    rng = MersenneTwister(seed)
    perm = randperm(rng, N)

    n_train = round(Int, ratios[1] * N)
    n_val   = round(Int, ratios[2] * N)
    # Test gets the remainder to avoid off-by-one
    n_test  = N - n_train - n_val

    idx_train = perm[1:n_train]
    idx_val   = perm[n_train+1:n_train+n_val]
    idx_test  = perm[n_train+n_val+1:end]

    return (
        (image_paths[idx_train], mask_paths[idx_train]),  # train
        (image_paths[idx_val],   mask_paths[idx_val]),    # val
        (image_paths[idx_test],  mask_paths[idx_test])    # test
    )
end

# ============================================================================
# Batch iterator
# ============================================================================
mutable struct SegBatchIterator
    image_paths::Vector{String}
    mask_paths::Vector{String}
    target_size::Tuple{Int,Int}
    target_class::Int
    batch_size::Int
    shuffle::Bool
    seed::Int
    indices::Vector{Int}
    epoch::Int
    gpu_transfer::Union{Function,Nothing}  # optional: x -> cu(x)
end

function SegBatchIterator(image_paths::Vector{String},
                          mask_paths::Vector{String};
                          target_size::Tuple{Int,Int} = (256, 256),
                          target_class::Int = IW_CLASS_INDEX,
                          batch_size::Int = 8,
                          shuffle::Bool = true,
                          seed::Int = 42,
                          gpu_transfer::Union{Function,Nothing} = nothing)
    N = length(image_paths)
    indices = collect(1:N)
    if shuffle
        rng = MersenneTwister(seed)
        shuffle!(rng, indices)
    end
    return SegBatchIterator(image_paths, mask_paths, target_size,
                            target_class, batch_size, shuffle, seed,
                            indices, 0, gpu_transfer)
end

"""Reset and reshuffle for a new epoch."""
function new_epoch!(iter::SegBatchIterator)
    iter.epoch += 1
    if iter.shuffle
        rng = MersenneTwister(iter.seed + iter.epoch)
        shuffle!(rng, iter.indices)
    end
end

Base.length(iter::SegBatchIterator) = ceil(Int, length(iter.indices) / iter.batch_size)

function Base.iterate(iter::SegBatchIterator, state::Int=1)
    N = length(iter.indices)
    state > N && return nothing

    batch_end = min(state + iter.batch_size - 1, N)
    batch_idx = iter.indices[state:batch_end]
    B = length(batch_idx)

    h, w = iter.target_size

    # Allocate batch arrays (WHCN for Lux — but Julia is column-major so H,W,C,N)
    X_batch = Array{Float32}(undef, h, w, 1, B)
    Y_batch = Array{Float32}(undef, h, w, 1, B)

    for (j, i) in enumerate(batch_idx)
        X_batch[:, :, :, j] = load_image(iter.image_paths[i], iter.target_size)
        Y_batch[:, :, :, j] = load_mask_binary(iter.mask_paths[i], iter.target_class,
                                                iter.target_size)
    end

    # DL-005: GPU transfer if available
    if iter.gpu_transfer !== nothing
        X_batch = iter.gpu_transfer(X_batch)
        Y_batch = iter.gpu_transfer(Y_batch)
    end

    return (X_batch, Y_batch), batch_end + 1
end

# ============================================================================
# High-level API: create_loaders
# ============================================================================
"""
    create_loaders(data_dir; target_class, target_size, batch_size, split_ratios, seed, gpu_transfer)

One-call setup: discover pairs, filter by class, split, return 3 iterators.
"""
function create_loaders(data_dir::String;
                        target_class::Int = IW_CLASS_INDEX,
                        target_size::Tuple{Int,Int} = (256, 256),
                        batch_size::Int = 8,
                        split_ratios::Tuple{Float64,Float64,Float64} = (0.7, 0.15, 0.15),
                        seed::Int = 42,
                        gpu_transfer::Union{Function,Nothing} = nothing)
    # DL-001: Discover pairs
    image_paths, mask_paths = discover_pairs(data_dir)
    println("Discovered $(length(image_paths)) image-mask pairs")

    # DL-006: Filter for target class
    image_paths, mask_paths = filter_by_class(image_paths, mask_paths, target_class)
    println("After IW filter: $(length(image_paths)) images with class $target_class")

    # DL-004: Split
    (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) =
        split_data(image_paths, mask_paths, split_ratios; seed=seed)

    println("Split: train=$(length(train_imgs)), val=$(length(val_imgs)), test=$(length(test_imgs))")

    # Create iterators
    train_loader = SegBatchIterator(train_imgs, train_masks;
                                    target_size, target_class, batch_size,
                                    shuffle=true, seed, gpu_transfer)
    val_loader = SegBatchIterator(val_imgs, val_masks;
                                  target_size, target_class, batch_size,
                                  shuffle=false, seed, gpu_transfer)
    test_loader = SegBatchIterator(test_imgs, test_masks;
                                   target_size, target_class, batch_size,
                                   shuffle=false, seed, gpu_transfer)

    return train_loader, val_loader, test_loader
end

# ============================================================================
# Utilities
# ============================================================================
dataset_size(iter::SegBatchIterator) = length(iter.indices)

function get_split_sizes(train, val, test)
    return (train=dataset_size(train), val=dataset_size(val), test=dataset_size(test))
end

end # module SegDataLoader
