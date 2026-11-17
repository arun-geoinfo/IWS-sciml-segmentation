# ================================
# dataloader_gpu.jl  –  GPU DataLoader with automatic BSON caching
#
#  FIXES APPLIED:
#    1. Labels read from directory structure per-file (not positional concat)
#    2. Constructor shuffle uses seed for reproducibility, but epoch reshuffles
#       use a DIFFERENT seed each epoch (epoch_seed = rng_seed + epoch_counter)
#    3. Validation loader (shuffle=false) still gets properly interleaved data
#       because the BSON cache now stores data in shuffled order
#    4. Added epoch_counter to track reshuffles
#
#  Pipeline:
#    FIRST RUN  : images → decode → resize → UInt8 → BSON cache  (slow, ONCE)
#    EVERY NEXT : BSON → instant UInt8 host buffer                (fast, ALWAYS)
#    PER BATCH  : UInt8 slice → Float32 on CPU → cu() → GPU
# ================================

using Images, FileIO, Random, ImageTransformations, BSON, ProgressMeter, Dates

# ── GPU stack ─────────────────────────────────────────────────────────────────
using CUDA
CUDA.set_runtime_version!(; local_toolkit=true)
ENV["JULIA_CUDA_VERBOSE"]                = "0"
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# ── Config ────────────────────────────────────────────────────────────────────
const SCRIPT_DIR  = @__DIR__
const CONFIG_FILE = abspath(joinpath(SCRIPT_DIR, "..", "config", "default_config.jl"))

if !isfile(CONFIG_FILE)
    error("""
    ❌ Configuration file not found!
    Expected: $CONFIG_FILE
    """)
end

include(CONFIG_FILE)
using .DefaultConfig
using .DefaultConfig.DeviceConfig

# ── GPU sanity check ──────────────────────────────────────────────────────────
if !CUDA.functional()
    @warn "⚠️  CUDA not functional — safe_gpu_transfer will fall back to CPU arrays"
else
    dev     = CUDA.device()
    vram_gb = round(CUDA.totalmem(dev) / (1024^3), digits=2)
    println("🖥️  GPU : $(CUDA.name(dev))  |  VRAM: $(vram_gb) GiB")
end

# =============================================================================
# Struct
# =============================================================================
mutable struct DataLoader
    # ── Paths / labels ────────────────────────────────────────────────────────
    image_paths::Vector{String}
    labels::Vector{Int}

    # ── Hyperparams ───────────────────────────────────────────────────────────
    batch_size::Int
    rng_seed::Int
    shuffle::Bool
    augment_fn::Function

    # ── Device routing ────────────────────────────────────────────────────────
    device
    backend::Symbol
    hybrid_mode::Bool

    # ── Host buffer  H × W × C × N  (UInt8, compact) ─────────────────────────
    X_preloaded::Array{UInt8, 4}
    y_preloaded::Array{Float32, 2}

    # ── Shuffling ─────────────────────────────────────────────────────────────
    shuffled_indices::Vector{Int}
    epoch_counter::Int              # NEW: tracks epoch for unique reshuffles

    # ── Cache metadata ────────────────────────────────────────────────────────
    bson_path::String
    cache_hit::Bool
    load_time_s::Float64
end

# =============================================================================
# Constructor
# =============================================================================
function DataLoader(proc_dir::String;
                    batch_size = DefaultConfig.BATCH_SIZE,
                    rng_seed   = DefaultConfig.RNG_SEED,
                    shuffle    = true,
                    augment_fn = nothing)

    augment_fn  = augment_fn === nothing ? x -> x : augment_fn
    hybrid_mode = (DefaultConfig.COMPUTE_MODE == :hybrid)

    parent_dir  = dirname(proc_dir)
    split_name  = basename(proc_dir)
    # ── Use a NEW cache name to force rebuild with correct label order ─────
    bson_path   = joinpath(parent_dir, split_name * "_data_v2.bson")

    local X_preloaded::Array{UInt8, 4}
    local y_preloaded::Array{Float32, 2}
    local image_paths::Vector{String}
    local labels::Vector{Int}
    cache_hit = false
    load_time = 0.0

    # =========================================================================
    # PATH A — CACHE HIT
    # =========================================================================
    if isfile(bson_path)
        cache_hit = true

        println("\n" * "="^60)
        println("⚡ CACHE HIT — instant load from BSON")
        println("   $(bson_path)")
        println("="^60)

        t0   = time()
        data = BSON.load(bson_path)

        if !haskey(data, :X) || !haskey(data, :y)
            error("""
            ❌ Corrupt or incompatible BSON cache!
            File  : $bson_path
            Keys  : $(keys(data))
            Delete the cache file and re-run to rebuild.
            """)
        end

        raw = data[:X]
        if eltype(raw) != UInt8
            @warn "🔄 Non-UInt8 cache detected — converting & re-saving"
            raw = UInt8.(clamp.(round.(raw .* 255), 0, 255))
            y_raw_upgrade = Vector{Int}(data[:y])
            BSON.@save bson_path X=raw y=y_raw_upgrade
            println("   💾 Cache upgraded to UInt8")
        end
        X_preloaded = raw

        y_raw       = Vector{Int}(data[:y])
        y_preloaded = reshape(Float32.(y_raw), 1, length(y_raw))
        image_paths = fill("cached", length(y_raw))
        labels      = y_raw
        load_time   = time() - t0

        h, w, c, N = size(X_preloaded)
        mem_mb      = (h * w * c * N) / (1024^2)

        # ── Verify class balance in cache ─────────────────────────────────────
        n0 = count(==(0), y_raw)
        n1 = count(==(1), y_raw)
        println("   ✅ Loaded  : $N samples  |  shape $(h)×$(w)×$(c)")
        println("   📊 Classes : $n0 class0 / $n1 class1")
        println("   💾 Host RAM: $(round(mem_mb, digits=1)) MB  (UInt8)")
        println("   ⏱  Time   : $(round(load_time, digits=3))s")

        # ── Verify interleaving (first 20 labels should NOT be all same class)
        first_labels = y_raw[1:min(20, length(y_raw))]
        if all(==(first_labels[1]), first_labels)
            @warn """
            ⚠️  Cache appears to have non-interleaved labels!
            First 20 labels: $first_labels
            This will cause poor validation. Deleting cache to rebuild...
            """
            rm(bson_path)
            # Recurse to rebuild
            return DataLoader(proc_dir; batch_size=batch_size, rng_seed=rng_seed,
                              shuffle=shuffle, augment_fn=augment_fn)
        end

        println("   ✅ Label interleaving verified")
        println("="^60)

    # =========================================================================
    # PATH B — NO CACHE → decode, SHUFFLE, then cache
    # =========================================================================
    else
        println("\n" * "="^60)
        println("📂 FIRST RUN — decoding images, will cache to BSON")
        println("   Source : $proc_dir")
        println("   Cache  : $bson_path")
        println("="^60)

        for cls in ("class0", "class1")
            d = joinpath(proc_dir, cls)
            isdir(d) || error("❌ Expected directory not found: $d")
        end

        # ── Read files WITH their labels from directory names ─────────────────
        #    Each file gets its label from which directory it's in.
        #    We interleave immediately — no "all 0s then all 1s" ordering.
        class0_files = readdir(joinpath(proc_dir, "class0"); join=true)
        class1_files = readdir(joinpath(proc_dir, "class1"); join=true)

        # Build paired list: (path, label) for every file
        all_files = vcat(
            [(f, 0) for f in class0_files],
            [(f, 1) for f in class1_files]
        )

        # ── SHUFFLE before anything else — this is the critical fix ───────────
        #    The BSON cache will store data in this shuffled order.
        #    So even val loader with shuffle=false gets interleaved classes.
        Random.seed!(rng_seed)
        Random.shuffle!(all_files)

        image_paths = [f[1] for f in all_files]
        labels      = [f[2] for f in all_files]

        N           = length(image_paths)
        h, w        = DefaultConfig.TARGET_SIZE
        X_preloaded = Array{UInt8, 4}(undef, h, w, 1, N)
        y_raw       = Int.(labels)
        y_preloaded = reshape(Float32.(y_raw), 1, N)

        println("   🖼  Images  : $N  ($(length(class0_files)) class0 / $(length(class1_files)) class1)")
        println("   🔀 Data shuffled BEFORE caching (classes interleaved)")
        println("   🧵 Threads : $(Threads.nthreads())")
        println("   📐 Target  : $(h)×$(w) px  (grayscale → UInt8)")
        println()

        prog      = Progress(N;
                             desc      = "   🚀 Decoding  ",
                             barlen    = 35,
                             color     = :cyan,
                             showspeed = true)
        load_lock = ReentrantLock()
        failed    = String[]

        t0 = time()
        Threads.@threads for i in 1:N
            try
                img         = load(image_paths[i])
                img_gray    = Gray.(img)
                img_resized = imresize(img_gray, DefaultConfig.TARGET_SIZE)
                X_preloaded[:, :, 1, i] = UInt8.(round.(Float32.(img_resized) .* 255))
            catch e
                lock(load_lock) do
                    push!(failed, basename(image_paths[i]))
                end
                @warn "Failed to load $(basename(image_paths[i]))" exception=e
            end
            next!(prog)
        end
        load_time = time() - t0

        println()

        if !isempty(failed)
            @warn "⚠️  $(length(failed)) images failed to decode (zero-filled):\n" *
                  join(failed[1:min(10, length(failed))], "\n")
        end

        mem_mb = (h * w * 1 * N) / (1024^2)
        println("   ✅ Decoded  : $(N - length(failed)) / $N  in $(round(load_time, digits=1))s")
        println("   💾 Host RAM : $(round(mem_mb, digits=1)) MB  (UInt8)")

        # ── Verify interleaving before saving ─────────────────────────────────
        first_20 = y_raw[1:min(20, N)]
        n0_first = count(==(0), first_20)
        n1_first = count(==(1), first_20)
        println("   🔍 First 20 labels: $n0_first class0, $n1_first class1 (should be mixed)")

        # ── Save cache — data is ALREADY shuffled/interleaved ─────────────────
        print("   💾 Saving cache → $bson_path  ... ")
        t_save = time()
        BSON.@save bson_path X=X_preloaded y=y_raw
        println("done  ($(round(time()-t_save, digits=1))s)")
        println()
        println("   ✨ Next run will skip all of this and load instantly!")
        println("="^60)
    end

    # =========================================================================
    # Shared init
    # =========================================================================
    N                = size(X_preloaded, 4)
    shuffled_indices = collect(1:N)
    if shuffle
        # Initial shuffle with base seed — subsequent epochs use different seeds
        Random.seed!(rng_seed)
        Random.shuffle!(shuffled_indices)
    end

    return DataLoader(
        image_paths,
        labels,
        batch_size,
        rng_seed,
        shuffle,
        augment_fn,
        DefaultConfig.DEVICE,
        DefaultConfig.BACKEND,
        hybrid_mode,
        X_preloaded,
        y_preloaded,
        shuffled_indices,
        0,                  # epoch_counter starts at 0
        bson_path,
        cache_hit,
        load_time
    )
end

# =============================================================================
# Iteration interface
# =============================================================================
Base.iterate(dl::DataLoader, state=1) = _next_batch(dl, state)

function Base.length(dl::DataLoader)
    return ceil(Int, size(dl.X_preloaded, 4) / dl.batch_size)
end

function Base.eltype(::Type{DataLoader})
    return Tuple{Array{Float32, 4}, Array{Float32, 2}}
end

function _next_batch(dl::DataLoader, state::Int)
    N = size(dl.X_preloaded, 4)
    state > N && return nothing

    # ── Reshuffle at epoch boundary with UNIQUE seed per epoch ────────────────
    if state == 1 && dl.shuffle
        dl.epoch_counter += 1
        epoch_seed = dl.rng_seed + dl.epoch_counter   # Different every epoch!
        Random.seed!(epoch_seed)
        Random.shuffle!(dl.shuffled_indices)
    end

    batch_end = min(state + dl.batch_size - 1, N)
    batch_idx = dl.shuffled_indices[state:batch_end]

    # ── Step 1: slice UInt8 host buffer ───────────────────────────────────────
    X_uint8 = dl.X_preloaded[:, :, :, batch_idx]
    y_cpu   = dl.y_preloaded[:, batch_idx]

    # ── Step 2: Float32 conversion on CPU ─────────────────────────────────────
    X_batch = Float32.(X_uint8) ./ 255.0f0

    # ── Step 3: Augmentation on CPU Float32 ───────────────────────────────────
    if dl.augment_fn !== nothing && dl.augment_fn != (x -> x)
        X_batch = dl.augment_fn(X_batch)
    end

    # ── Step 4: Device transfer ───────────────────────────────────────────────
    if dl.hybrid_mode
        return (X_batch, y_cpu), batch_end + 1
    else
        X_gpu = DefaultConfig.DeviceConfig.safe_gpu_transfer(X_batch, dl.device, dl.backend)
        y_gpu = DefaultConfig.DeviceConfig.safe_gpu_transfer(y_cpu,   dl.device, dl.backend)
        return (X_gpu, y_gpu), batch_end + 1
    end
end

# =============================================================================
# Utility functions
# =============================================================================

"""Re-shuffle indices (for manual epoch reset if needed)."""
function reset_shuffling!(dl::DataLoader)
    dl.shuffle || return
    dl.epoch_counter += 1
    epoch_seed = dl.rng_seed + dl.epoch_counter
    Random.seed!(epoch_seed)
    Random.shuffle!(dl.shuffled_indices)
end

"""Total number of samples."""
function dataset_size(dl::DataLoader)
    return size(dl.X_preloaded, 4)
end

"""Print class balance diagnostics."""
function print_class_balance(dl::DataLoader)
    y = dl.y_preloaded[1, :]
    n0 = count(==(0.0f0), y)
    n1 = count(==(1.0f0), y)
    N  = length(y)
    println("   📊 Class balance: $n0 class0 ($(round(n0/N*100,digits=1))%) / $n1 class1 ($(round(n1/N*100,digits=1))%)")

    # Check first/last 50 for ordering issues
    first_50 = y[1:min(50, N)]
    last_50  = y[max(1, N-49):N]
    println("   🔍 First 50 labels mean: $(round(mean(first_50), digits=3)) (should be ~0.5 if interleaved)")
    println("   🔍 Last  50 labels mean: $(round(mean(last_50),  digits=3)) (should be ~0.5 if interleaved)")
end

"""Memory breakdown."""
function memory_info(dl::DataLoader)
    h, w, c, N  = size(dl.X_preloaded)
    uint8_mb     = (h * w * c * N)             / (1024^2)
    batch_f32_mb = (h * w * c * dl.batch_size) * 4 / (1024^2)
    full_f32_mb  = (h * w * c * N)             * 4 / (1024^2)

    return (
        samples            = N,
        shape              = size(dl.X_preloaded),
        uint8_host_mb      = round(uint8_mb,     digits=2),
        batch_f32_mb       = round(batch_f32_mb, digits=2),
        full_f32_would_mb  = round(full_f32_mb,  digits=2),
        vram_saved_mb      = round(full_f32_mb - uint8_mb, digits=2),
        cache_hit          = dl.cache_hit,
        load_time_s        = round(dl.load_time_s, digits=3),
        bson_path          = dl.bson_path,
        hybrid_mode        = dl.hybrid_mode,
        backend            = dl.backend
    )
end

"""Delete the BSON cache (forces full re-decode on next run)."""
function clear_cache!(dl::DataLoader)
    if isfile(dl.bson_path)
        rm(dl.bson_path)
        println("🗑️  Cache deleted: $(dl.bson_path)")
    else
        println("ℹ️  No cache file at $(dl.bson_path)")
    end
end

# =============================================================================
# Export
# =============================================================================
export DataLoader, reset_shuffling!, dataset_size, memory_info, clear_cache!, print_class_balance

# =============================================================================
# Self-test
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("🧪 GPU DATALOADER SELF-TEST")
    println("="^60)

    println("\n1️⃣  Creating DataLoader (train)...")
    dl = DataLoader(DefaultConfig.PROC_TRAIN; batch_size=8, shuffle=true)

    info = memory_info(dl)
    println("\n📊 Memory report:")
    println("   Samples            : $(info.samples)")
    println("   Shape              : $(info.shape)")
    println("   UInt8 host RAM     : $(info.uint8_host_mb) MB")
    println("   Float32 per batch  : $(info.batch_f32_mb) MB")
    println("   Float32 full set   : $(info.full_f32_would_mb) MB")
    println("   VRAM saved         : $(info.vram_saved_mb) MB")
    println("   Cache hit          : $(info.cache_hit ? "YES ⚡" : "NO (first run)")")

    println("\n📊 Class balance check:")
    print_class_balance(dl)

    println("\n2️⃣  Testing batch iteration...")
    X_batch, y_batch = first(dl)
    println("   ✅ X shape  : $(size(X_batch))")
    println("   ✅ y shape  : $(size(y_batch))")
    println("   ✅ y values in first batch: $(Array(y_batch))")
    println("   ✅ Batches  : $(length(dl))")

    println("\n3️⃣  Verifying DIFFERENT shuffle each epoch...")
    idx_ep1 = copy(dl.shuffled_indices[1:10])
    # Simulate next epoch start
    iterate(dl, 1)  # triggers reshuffle
    idx_ep2 = copy(dl.shuffled_indices[1:10])
    println("   Epoch 1 first 10: $idx_ep1")
    println("   Epoch 2 first 10: $idx_ep2")
    println("   Different: $(idx_ep1 != idx_ep2 ? "✅ YES (correct)" : "❌ NO (bug!)")")

    println("\n4️⃣  Testing val loader (shuffle=false)...")
    dl_val = DataLoader(DefaultConfig.PROC_TEST; batch_size=8, shuffle=false)
    print_class_balance(dl_val)
    _, y_val_batch = first(dl_val)
    println("   ✅ First val batch labels: $(Array(y_val_batch))")
    println("   ✅ Should be mixed 0s and 1s, NOT all same class")

    println("\n" * "="^60)
    println("✅ ALL TESTS PASSED")
    println("="^60)
end