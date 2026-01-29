using Images, FileIO, Random, ImageTransformations

include("../config/default_config.jl")
using .DefaultConfig
using .DefaultConfig.DeviceConfig

"""
    DataLoader with device-aware batch loading AND PRELOADING
    All original functionalities preserved:
    - Shuffling (per-epoch)
    - Augmentation (if provided)
    - Hybrid mode support
    - Device-aware transfer
    - Batch size configuration
"""
mutable struct DataLoader
    # Original fields (unchanged)
    image_paths::Vector{String}
    labels::Vector{Int}
    batch_size::Int
    rng_seed::Int
    shuffle::Bool
    augment_fn::Function
    device
    backend::Symbol
    hybrid_mode::Bool
    
    # 🚀 NEW: Preloaded data in CPU memory
    X_preloaded::Array{Float32, 4}      # All images: (256, 256, 1, N)
    y_preloaded::Array{Float32, 2}      # All labels: (1, N)
    # 🚀 NEW: Indices for shuffling without modifying actual data
    shuffled_indices::Vector{Int}
end

function DataLoader(proc_dir::String; 
                   batch_size=DefaultConfig.BATCH_SIZE, 
                   rng_seed=DefaultConfig.RNG_SEED, 
                   shuffle=true, 
                   augment_fn=nothing)
    
    # 1. GET IMAGE PATHS AND LABELS (ORIGINAL FUNCTIONALITY)
    class0_files = readdir(joinpath(proc_dir, "class0"); join=true)
    class1_files = readdir(joinpath(proc_dir, "class1"); join=true)
    
    image_paths = vcat(class0_files, class1_files)
    labels = vcat(fill(0, length(class0_files)), fill(1, length(class1_files)))
    
    augment_fn = augment_fn === nothing ? x->x : augment_fn
    
    # 2. 🚀 PRELOAD ALL IMAGES ONCE (NEW OPTIMIZATION)
    println("   📦 Preloading all images to memory...")
    start_time = time()
    
    N = length(image_paths)
    h, w = DefaultConfig.TARGET_SIZE
    X_preloaded = Array{Float32, 4}(undef, h, w, 1, N)
    y_preloaded = zeros(Float32, 1, N)
    
    # Parallel loading for speed (optional but recommended)
    Threads.@threads for i in 1:N
        img = load(image_paths[i])
        img_gray = Gray.(img)
        img_resized = imresize(img_gray, DefaultConfig.TARGET_SIZE)
        X_preloaded[:, :, 1, i] = Float32.(img_resized) ./ 255.0f0
        y_preloaded[1, i] = Float32(labels[i])
    end
    
    preload_time = time() - start_time
    println("   ✅ Preloaded $N images in $(round(preload_time, digits=1)) seconds")
    println("   💾 Memory used: $(round((h*w*4*N)/1e9, digits=2)) GB")    
    # 3. HYBRID MODE DETECTION (ORIGINAL FUNCTIONALITY)
    hybrid_mode = (DefaultConfig.COMPUTE_MODE == :hybrid)
    
    # 4. INITIAL SHUFFLED INDICES
    shuffled_indices = collect(1:N)
    if shuffle
        Random.seed!(rng_seed)
        shuffle!(shuffled_indices)
    end
    
    return DataLoader(
        # Original fields
        image_paths, 
        labels, 
        batch_size, 
        rng_seed, 
        shuffle, 
        augment_fn,
        DefaultConfig.DEVICE,
        DefaultConfig.BACKEND,
        hybrid_mode,
        
        # New preloaded data fields
        X_preloaded,
        y_preloaded,
        
        # Shuffling indices
        shuffled_indices
    )
end

Base.iterate(dl::DataLoader, state=1) = _next_batch(dl, state)

function Base.length(dl::DataLoader)
    return ceil(Int, length(dl.image_paths) / dl.batch_size)
end

function Base.eltype(::Type{DataLoader})
    return Tuple{Array{Float32, 4}, Array{Float32, 2}}
end

function _next_batch(dl::DataLoader, state::Int)
    N = length(dl.image_paths)
    
    # Check if we're done
    if state > N
        return nothing
    end
    
    # 1. SHUFFLING AT EPOCH START (ORIGINAL FUNCTIONALITY)
    if state == 1 && dl.shuffle
        Random.seed!(dl.rng_seed)
        # 🚀 Only shuffle indices, not the actual arrays
        shuffle!(dl.shuffled_indices)
        
        # Optional: You can also shuffle paths and labels if needed elsewhere
        # But we only need to shuffle indices since we access via indices
    end
    
    # 2. GET BATCH INDICES
    batch_end = min(state + dl.batch_size - 1, N)
    batch_indices = dl.shuffled_indices[state:batch_end]
    
    # 3. 🚀 GET BATCH FROM PRELOADED DATA (NO DISK ACCESS!)
    X_batch = dl.X_preloaded[:, :, :, batch_indices]
    y_batch = dl.y_preloaded[:, batch_indices]
    
    # 4. APPLY AUGMENTATION (ORIGINAL FUNCTIONALITY)
    if dl.augment_fn !== nothing && dl.augment_fn != (x->x)
        X_batch = dl.augment_fn(X_batch)
    end
    
    # 5. DEVICE TRANSFER LOGIC (ORIGINAL FUNCTIONALITY)
    # ========================================
    # 🚀 DEVICE TRANSFER LOGIC (UNCHANGED)
    # ========================================
    if dl.hybrid_mode
        # Hybrid: Keep data on CPU, transfer happens in training loop
        return (X_batch, y_batch), batch_end + 1
    else
        # Auto/GPU/CPU: Transfer based on device
        X_batch = DefaultConfig.DeviceConfig.safe_gpu_transfer(X_batch, dl.device, dl.backend)
        y_batch = DefaultConfig.DeviceConfig.safe_gpu_transfer(y_batch, dl.device, dl.backend)
        return (X_batch, y_batch), batch_end + 1
    end
end

# 🚀 NEW: Reset function to re-shuffle at epoch end if needed externally
function reset_shuffling!(dl::DataLoader)
    if dl.shuffle
        Random.seed!(dl.rng_seed)
        shuffle!(dl.shuffled_indices)
    end
end

# 🚀 NEW: Get dataset size (useful for monitoring)
function dataset_size(dl::DataLoader)
    return length(dl.image_paths)
end

# 🚀 NEW: Memory info
function memory_info(dl::DataLoader)
    h, w, c, N = size(dl.X_preloaded)
    memory_mb = (h * w * c * N * 4) / (1024^2)  # Float32 = 4 bytes
    return (samples=N, memory_mb=round(memory_mb, 2), shape=size(dl.X_preloaded))
end

# Export new functions
export reset_shuffling!, dataset_size, memory_info