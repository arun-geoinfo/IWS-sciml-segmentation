# ================================
# train_gpu.jl  –  SAR CNN v2.2 GPU Training (Maximum Speed)
#
#  TARGET: ~70 min (down from 127 min) with same ~96% accuracy
#
#  SPEED OPTIMIZATIONS OVER v2.1:
#    1. FULL batch forward+backward (no micro-batching) — VRAM allows it
#    2. BN state updated every 5 batches (not every batch)
#       Reuses BN update forward pass for inline train accuracy (free)
#    3. GC + reclaim only at epoch boundaries
#    4. Val every 3 epochs for first 60%, every epoch after
#    5. Val loader uses full BATCH_SIZE (fewer kernel launches)
#    6. Patience 15 (was 20)
#    7. Reduced progress bar overhead
#
#  ALL ACCURACY FEATURES PRESERVED:
#    - Input normalization, label smoothing α=0.1, OneCycleLR
#    - Full augmentation (flips + rotations + random erasing)
#    - AdamW weight decay, gradient clipping
#    - Best model checkpoint to CPU, TTA 4×
#    - Early stopping with patience display
#    - Full VRAM monitoring, epoch banners, config logging
#
#  SciML v3.0 ADDITIONS:
#    - Physics-informed loss terms: gradient structure + symmetry
#    - Controlled by λ_GRADIENT and λ_SYMMETRY (set to 0.0 = disabled)
#    - All original behaviour preserved when λ values are 0.0
# ================================

using Lux, LuxCore, NNlib
using Random, Statistics, Printf, Dates, JLD2
using ProgressMeter
using Optimisers
using Zygote
using CUDA, LuxCUDA

CUDA.set_runtime_version!(; local_toolkit=true)
ENV["JULIA_CUDA_VERBOSE"]                = "0"
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"
ENV["JULIA_CUDA_MEMORY_POOL"]           = "none"

include("../config/default_config.jl")
using .DefaultConfig
using .DefaultConfig.DeviceConfig
include("../dataloaders/lux_dataloader.jl")
include("../models/sar_cnn_model.jl")

# =============================================================================
# Hyperparameters
# =============================================================================
const BATCH_SIZE        = 128
const EPOCHS            = 100
const BASE_LR           = Float32(1e-4)
const LEARNING_RATE     = BASE_LR
const MAX_LR            = Float32(5e-4)
const WEIGHT_DECAY      = Float32(1e-2)
const GRAD_CLIP_NORM    = 1.0f0
const LABEL_SMOOTH_α    = 0.1f0
const BN_UPDATE_FREQ    = 5         # Update BN running stats every N batches
const VAL_SKIP_FRAC     = 0.6f0     # Skip val for first 60% of epochs

const PATIENCE          = 15
const MIN_DELTA         = Float32(1e-4)

const DEVICE  = gpu_device()
const BACKEND = :cuda

# =============================================================================
# SciML — Physics-Informed Loss Weights
#
#  Set any λ to 0.0f0 to completely disable that physics loss term.
#  When all are 0.0, training is IDENTICAL to the original v2.2.
#
#  Suggested ablation schedule:
#    Run 0: λ_GRADIENT=0.0,  λ_SYMMETRY=0.0   (baseline, must match v2.2)
#    Run 1: λ_GRADIENT=0.01, λ_SYMMETRY=0.0   (gradient only)
#    Run 2: λ_GRADIENT=0.0,  λ_SYMMETRY=0.005 (symmetry only)
#    Run 3: λ_GRADIENT=0.01, λ_SYMMETRY=0.005 (both)
# =============================================================================
const λ_GRADIENT  = 0.0f0      # Gradient structure loss weight
const λ_SYMMETRY  = 0.0f0      # Symmetry loss weight

# =============================================================================
# Memory utilities
# =============================================================================
function gpu_memory_info()
    free  = CUDA.available_memory() / (1024^3)
    total = CUDA.total_memory() / (1024^3)
    used  = total - free
    return (free=round(free, digits=2),
            used=round(used, digits=2),
            total=round(total, digits=2))
end

function reclaim_memory!()
    GC.gc(true)
    CUDA.reclaim()
end

function save_best_to_cpu(ps, st)
    return Lux.cpu_device()(ps), Lux.cpu_device()(st)
end

# =============================================================================
# Input normalization
# =============================================================================
struct DatasetStats
    mean::Float32
    std::Float32
end

function compute_dataset_stats(dl::DataLoader)
    X = dl.X_preloaded
    μ = Float32(mean(X)) / 255.0f0
    σ = Float32(std(Float32.(X) ./ 255.0f0))
    return DatasetStats(μ, max(σ, 1.0f-6))
end

function normalize_batch(X::AbstractArray{Float32}, s::DatasetStats)
    return (X .- s.mean) ./ s.std
end

# =============================================================================
# Loss
# =============================================================================
function binary_cross_entropy(y_pred, y_true; ϵ=1.0f-8)
    if size(y_true) != size(y_pred)
        y_true = reshape(y_true, size(y_pred))
    end
    y_safe = clamp.(Float32.(y_pred), ϵ, 1.0f0 - ϵ)
    yt     = Float32.(y_true)
    return -mean(yt .* log.(y_safe) .+ (1.0f0 .- yt) .* log.(1.0f0 .- y_safe))
end

function binary_cross_entropy_smooth(y_pred, y_true; ϵ=1.0f-8, α=LABEL_SMOOTH_α)
    if size(y_true) != size(y_pred)
        y_true = reshape(y_true, size(y_pred))
    end
    y_smooth = label_smooth(Float32.(y_true); α=Float32(α))
    y_safe   = clamp.(Float32.(y_pred), ϵ, 1.0f0 - ϵ)
    return -mean(y_smooth .* log.(y_safe) .+
                 (1.0f0 .- y_smooth) .* log.(1.0f0 .- y_safe))
end

function compute_accuracy(y_pred, y_true)
    if size(y_true) != size(y_pred)
        y_true = reshape(y_true, size(y_pred))
    end
    return mean((Float32.(y_pred) .> 0.5f0) .== (Float32.(y_true) .> 0.5f0))
end

# =============================================================================
# SciML — Physics-Informed Loss Functions
#
#  These are pure array operations — fully Zygote-differentiable on GPU.
#  They operate on intermediate feature maps returned by the model.
# =============================================================================

"""
    gradient_structure_loss(feat)

Internal wave SAR signatures show strong gradients perpendicular to wave crests
and weak gradients parallel to crests. This loss penalises isotropic (blobby)
gradient fields in feature maps — encouraging anisotropic (stripe-like) structure.

Returns a value near 1.0 (bad: isotropic) or near 0.0 (good: anisotropic).
Operates on 4D feature maps (H×W×C×B).
"""
function gradient_structure_loss(feat::AbstractArray{Float32})
    ∂x = feat[2:end, :, :, :] .- feat[1:end-1, :, :, :]
    ∂y = feat[:, 2:end, :, :] .- feat[:, 1:end-1, :, :]
    E_x = mean(∂x .^ 2)
    E_y = mean(∂y .^ 2)
    return min(E_x, E_y) / (max(E_x, E_y) + 1.0f-7)
end

"""
    symmetry_loss(feat)

Wave-train SAR intensity is approximately invariant to 180° rotation.
This loss penalises asymmetry in deep feature maps.

Operates on 4D feature maps (H×W×C×B).
"""
function symmetry_loss(feat::AbstractArray{Float32})
    rotated = feat[end:-1:1, end:-1:1, :, :]
    return mean((feat .- rotated) .^ 2)
end

# =============================================================================
# Data augmentation
# =============================================================================
function augment_batch(X::AbstractArray{Float32, 4})
    do_hflip = rand() > 0.5f0
    do_vflip = rand() > 0.5f0
    rot      = rand(0:3)
    do_erase = rand() < 0.3f0

    if do_hflip
        X = X[end:-1:1, :, :, :]
    end
    if do_vflip
        X = X[:, end:-1:1, :, :]
    end
    if rot == 1
        X = permutedims(X, (2, 1, 3, 4))[end:-1:1, :, :, :]
    elseif rot == 2
        X = X[end:-1:1, end:-1:1, :, :]
    elseif rot == 3
        X = permutedims(X, (2, 1, 3, 4))[:, end:-1:1, :, :]
    end
    if do_erase
        h, w = size(X, 1), size(X, 2)
        frac = 0.05f0 + rand(Float32) * 0.15f0
        eh   = max(1, round(Int, h * sqrt(frac)))
        ew   = max(1, round(Int, w * sqrt(frac)))
        y0   = rand(1:max(1, h - eh))
        x0   = rand(1:max(1, w - ew))
        X[y0:min(y0+eh-1, h), x0:min(x0+ew-1, w), :, :] .= 0.0f0
    end
    return X
end

# =============================================================================
# OneCycleLR schedule
# =============================================================================
function onecycle_lr(epoch::Int, total::Int, base::Float32, peak::Float32;
                     warmup_frac::Float32 = 0.3f0)
    min_lr = base * 0.01f0
    wu     = round(Int, total * warmup_frac)
    if epoch <= wu
        p = Float32(epoch) / Float32(wu)
        return base + (peak - base) * p
    else
        p = Float32(epoch - wu) / Float32(total - wu)
        return min_lr + (peak - min_lr) * 0.5f0 * (1.0f0 + cos(Float32(π) * p))
    end
end

function apply_lr_schedule!(opt_state, epoch, base, peak, total)
    lr = onecycle_lr(epoch, total, base, peak)
    Optimisers.adjust!(opt_state, lr)
    return lr
end

# =============================================================================
# Gradient clipping
# =============================================================================
function clip_gradients(gs::NamedTuple, max_norm::Float32)
    total = grad_norm(gs)
    s = min(1.0f0, max_norm / (total + 1.0f-8))
    return s < 1.0f0 ? scale_tree(gs, s) : gs
end

function grad_norm(gs::NamedTuple)
    norms = Float32[]
    _collect_norms!(norms, gs)
    return sqrt(sum(norms))
end
function _collect_norms!(n, gs::NamedTuple)
    for v in values(gs); _collect_norms!(n, v); end
end
function _collect_norms!(n, x::AbstractArray)
    push!(n, Float32(sum(x .^ 2)))
end
function _collect_norms!(n, x) end

function scale_tree(nt::NamedTuple, s::Float32)
    return map(x -> scale_tree(x, s), nt)
end
function scale_tree(x::AbstractArray, s::Float32)
    return x .* s
end
function scale_tree(x, s::Float32)
    return x
end

# =============================================================================
# TTA 4× — original + hflip + vflip + 180°
# =============================================================================
function tta_predict(model, X, ps, st)
    y0, _, _ = model(X, ps, st)
    y1, _, _ = model(X[end:-1:1, :, :, :], ps, st)
    y2, _, _ = model(X[:, end:-1:1, :, :], ps, st)
    y3, _, _ = model(X[end:-1:1, end:-1:1, :, :], ps, st)
    return (Float32.(y0) .+ Float32.(y1) .+ Float32.(y2) .+ Float32.(y3)) ./ 4.0f0
end

# =============================================================================
# Main
# =============================================================================
function train_with_lr_schedule()

    DefaultConfig.print_config()
    println("="^60)
    println("SAR CNN v2.2 TRAINING — MAXIMUM SPEED + SciML")
    println("="^60)

    mem = gpu_memory_info()
    println("🖥️  GPU Memory: $(mem.used) / $(mem.total) GiB used ($(mem.free) GiB free)")

    Random.seed!(DefaultConfig.RNG_SEED)
    println("✅ Random seed : ", DefaultConfig.RNG_SEED)
    println("   LR           : $LEARNING_RATE → peak $MAX_LR (OneCycle)")
    println("   Weight decay  : $WEIGHT_DECAY")
    println("   Batch size    : $BATCH_SIZE (full — no micro-batching)")
    println("   Epochs        : $EPOCHS")
    println("   Patience      : $PATIENCE")
    println("   Grad clip     : $GRAD_CLIP_NORM")
    println("   Label smooth  : α=$LABEL_SMOOTH_α")
    println("   LR schedule   : OneCycleLR (30% warmup)")
    println("   BN update     : every $BN_UPDATE_FREQ batches")
    println("   Val skip      : first $(round(Int, VAL_SKIP_FRAC*100))% epochs (every 3rd)")
    println("   Augmentation  : flips + 90° rot + random erasing")
    println("   TTA           : 4× at final eval")
    println("   GC            : epoch boundaries only")
    println("   CUDA pool     : disabled")
    println("   ── SciML Physics Losses ──")
    println("   λ_GRADIENT    : $λ_GRADIENT", λ_GRADIENT > 0 ? " ✅ ACTIVE" : " (disabled)")
    println("   λ_SYMMETRY    : $λ_SYMMETRY", λ_SYMMETRY > 0 ? " ✅ ACTIVE" : " (disabled)")

    # ── Data ──────────────────────────────────────────────────────────────────
    println("\n📊 Loading data...")
    train_loader = DataLoader(DefaultConfig.PROC_TRAIN;
                              batch_size = BATCH_SIZE,
                              shuffle    = true)
    val_loader   = DataLoader(DefaultConfig.PROC_TEST;
                              batch_size = BATCH_SIZE,
                              shuffle    = false)

    println("   Train samples : $(dataset_size(train_loader))")
    println("   Val samples   : $(dataset_size(val_loader))")

    # ── Class balance diagnostics ─────────────────────────────────────────────
    println("\n📊 Class balance check:")
    println("   TRAIN:")
    print_class_balance(train_loader)
    println("   VAL:")
    print_class_balance(val_loader)

    # ── Compute normalization stats ───────────────────────────────────────────
    println("\n📐 Computing dataset normalization statistics...")
    stats = compute_dataset_stats(train_loader)
    println("   Mean: $(round(stats.mean, digits=4))  Std: $(round(stats.std, digits=4))")

    # ── Model ─────────────────────────────────────────────────────────────────
    println("\n🧠 Initializing model...")
    model  = SAR_CNN()
    rng    = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ps     = ps |> DEVICE
    st     = st |> DEVICE
    println("   Parameters : $(LuxCore.parameterlength(ps))")
    println("   Device     : $BACKEND")

    mem = gpu_memory_info()
    println("   GPU after model: $(mem.used) / $(mem.total) GiB")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt       = Optimisers.AdamW(LEARNING_RATE, (0.9f0, 0.999f0), WEIGHT_DECAY)
    opt_state = Optimisers.setup(opt, ps)

    # ── Bookkeeping — best model on CPU ───────────────────────────────────────
    best_train_loss  = Inf32
    best_val_loss    = Inf32
    best_ps_cpu      = Lux.cpu_device()(deepcopy(ps))
    best_st_cpu      = Lux.cpu_device()(deepcopy(st))
    best_epoch       = 0
    patience_counter = 0

    reclaim_memory!()

    history = (
        train_loss = Float32[],
        val_loss   = Float32[],
        train_acc  = Float32[],
        val_acc    = Float32[],
        lr         = Float32[]
    )

    println("\n🚀 Starting training...")
    training_start = time()

    # =========================================================================
    # Epoch loop
    # =========================================================================
    for epoch in 1:EPOCHS

        epoch_start = time()
        current_lr  = apply_lr_schedule!(opt_state, epoch, LEARNING_RATE, MAX_LR, EPOCHS)

        println("\n" * "="^60)
        mem = gpu_memory_info()
        println("Epoch $epoch/$EPOCHS  (LR: $(round(current_lr, sigdigits=3))  |  VRAM: $(mem.used)/$(mem.total) GiB)")
        println("="^60)

        # =====================================================================
        # TRAIN phase — single fwd+bwd per batch, no micro-batching
        # =====================================================================
        st            = set_dropout_mode(st, true)
        epoch_loss    = 0.0f0
        epoch_correct = 0
        epoch_samples = 0
        batch_count   = 0
        n_batches     = length(train_loader)

        prog_train = Progress(n_batches;
                              desc   = "  🏋  Train ",
                              barlen = 30,
                              color  = :cyan)

        for (X_batch, y_batch) in train_loader

            X_batch = normalize_batch(X_batch |> DEVICE, stats)
            y_batch = y_batch |> DEVICE
            X_batch = augment_batch(X_batch)

            batch_count += 1

            # ── Single forward+backward — full batch ──────────────────────────
            #    SciML CHANGE: model now returns 3-tuple (pred, state, feats).
            #    Physics losses are computed on intermediate feature maps inside
            #    the pullback so Zygote can differentiate through them.
            #    When λ values are 0.0, this reduces to the original BCE-only loss.
            # ──────────────────────────────────────────────────────────────────
            st_ref = st
            loss_val, back = Zygote.pullback(ps) do p
                yp, _, feats = model(X_batch, p, st_ref)

                L_bce = binary_cross_entropy_smooth(yp, y_batch)

                L_grad = (λ_GRADIENT > 0) ?
                    λ_GRADIENT * gradient_structure_loss(feats.h3) : 0.0f0
                L_sym  = (λ_SYMMETRY > 0) ?
                    λ_SYMMETRY * symmetry_loss(feats.h4)          : 0.0f0

                L_bce + L_grad + L_sym
            end

            gs   = back(1.0f0)[1]
            back = nothing   # Free Zygote tape

            gs = clip_gradients(gs, GRAD_CLIP_NORM)
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
            gs = nothing

            # ── BN state update every N batches ───────────────────────────────
            #    Reuse this forward pass for inline train accuracy (FREE)
            if batch_count % BN_UPDATE_FREQ == 0 || batch_count == n_batches
                st_for_bn          = set_dropout_mode(st, false)
                yp_eval, st_upd, _ = model(X_batch, ps, st_for_bn)
                st                  = set_dropout_mode(st_upd, true)

                # ── Inline accuracy from BN update pass (no extra cost) ───────
                n_b           = size(X_batch, 4)
                correct       = sum((Float32.(yp_eval) .> 0.5f0) .== (Float32.(y_batch) .> 0.5f0))
                epoch_correct += Int(correct)
                epoch_samples += n_b
            end

            epoch_loss += Float32(loss_val)
            next!(prog_train)
        end

        avg_train_loss = epoch_loss / batch_count
        avg_train_acc  = epoch_samples > 0 ? Float32(epoch_correct) / Float32(epoch_samples) : 0.0f0

        # =====================================================================
        # VALIDATION — skip in early epochs to save time
        #   First 60%: run every 3rd epoch + first 2 epochs
        #   Last 40%: run every epoch
        # =====================================================================
        skip_threshold = round(Int, EPOCHS * VAL_SKIP_FRAC)
        run_val = (epoch > skip_threshold) || (epoch % 3 == 0) || (epoch <= 2)

        local avg_val_loss, avg_val_acc

        if run_val
            st_eval     = set_dropout_mode(st, false)
            val_loss    = 0.0f0
            val_acc_sum = 0.0f0
            val_batches = 0

            prog_val = Progress(length(val_loader);
                                desc   = "  🔍  Val   ",
                                barlen = 30,
                                color  = :green)

            for (X_batch, y_batch) in val_loader
                X_batch = normalize_batch(X_batch |> DEVICE, stats)
                y_batch = y_batch |> DEVICE
                y_pred, _, _ = model(X_batch, ps, st_eval)
                val_loss    += binary_cross_entropy(y_pred, y_batch)
                val_acc_sum += compute_accuracy(y_pred, y_batch)
                val_batches += 1
                next!(prog_val)
            end

            avg_val_loss = val_loss    / val_batches
            avg_val_acc  = val_acc_sum / val_batches
        else
            avg_val_loss = isempty(history.val_loss) ? Inf32 : history.val_loss[end]
            avg_val_acc  = isempty(history.val_acc)  ? 0.0f0 : history.val_acc[end]
            println("  ⏩ Val skipped (early training — using last val metrics)")
        end

        epoch_time = time() - epoch_start
        println()
        @printf("  Train  →  loss: %.4f  |  acc: %.2f%%\n",
                avg_train_loss, avg_train_acc * 100)
        @printf("  Val    →  loss: %.4f  |  acc: %.2f%%  %s\n",
                avg_val_loss, avg_val_acc * 100, run_val ? "" : "(cached)")
        @printf("  ⏱  Epoch time: %.1f s\n", epoch_time)

        push!(history.train_loss, avg_train_loss)
        push!(history.val_loss,   avg_val_loss)
        push!(history.train_acc,  Float32(avg_train_acc))
        push!(history.val_acc,    Float32(avg_val_acc))
        push!(history.lr,         Float32(current_lr))

        if avg_train_loss < best_train_loss
            best_train_loss = avg_train_loss
        end

        # ── Best model tracking + early stopping (only when val ran) ──────────
        if run_val
            if avg_val_loss < (best_val_loss - MIN_DELTA)
                best_val_loss    = avg_val_loss
                best_ps_cpu, best_st_cpu = save_best_to_cpu(ps, st)
                best_epoch       = epoch
                patience_counter = 0
                println("  ⭐ New best val loss: $(round(best_val_loss, digits=4))  (epoch $epoch)")
            else
                patience_counter += 1
                println("  ⏳ No improvement ($patience_counter/$PATIENCE)")
                if patience_counter >= PATIENCE
                    println("\n⛔ EARLY STOPPING at epoch $epoch (no improvement for $PATIENCE epochs)")
                    break
                end
            end
        end

        # ── GC only at epoch boundary ─────────────────────────────────────────
        reclaim_memory!()
        mem = gpu_memory_info()
        @printf("  🧹 VRAM after cleanup: %.2f / %.2f GiB\n", mem.used, mem.total)
    end

    # =========================================================================
    # Final TTA evaluation using best model
    # =========================================================================
    println("\n" * "="^60)
    println("🎯 Final TTA evaluation (4×) using best model (epoch $best_epoch)")
    println("="^60)

    best_ps_gpu = best_ps_cpu |> DEVICE
    best_st_gpu = set_dropout_mode(best_st_cpu |> DEVICE, false)

    tta_acc_sum = 0.0f0
    tta_batches = 0

    prog_tta = Progress(length(val_loader);
                        desc   = "  🎯  TTA   ",
                        barlen = 30,
                        color  = :magenta)

    for (X_batch, y_batch) in val_loader
        X_batch = normalize_batch(X_batch |> DEVICE, stats)
        y_batch = y_batch |> DEVICE
        y_pred  = tta_predict(model, X_batch, best_ps_gpu, best_st_gpu)
        tta_acc_sum += compute_accuracy(y_pred, y_batch)
        tta_batches += 1
        next!(prog_tta)
    end

    tta_val_acc = tta_acc_sum / tta_batches

    best_ps_gpu = nothing
    best_st_gpu = nothing
    reclaim_memory!()

    total_time = time() - training_start
    epochs_run = length(history.train_loss)
    println("\n" * "="^60)
    println("⏱️  TRAINING COMPLETE")
    println("="^60)
    @printf("  Total time      : %.1f s  (%.1f min)\n", total_time, total_time/60)
    @printf("  Epochs run      : %d / %d\n",            epochs_run, EPOCHS)
    @printf("  Avg / epoch     : %.1f s\n",             total_time / epochs_run)
    @printf("  Best val loss   : %.4f  (epoch %d)\n",   best_val_loss, best_epoch)
    @printf("  Final train acc : %.2f%%\n",             history.train_acc[end] * 100)
    @printf("  Final val acc   : %.2f%%\n",             history.val_acc[end]   * 100)
    @printf("  TTA val acc     : %.2f%%  (4× augmentation)\n", tta_val_acc * 100)
    println("="^60)

    return model, best_ps_cpu, best_st_cpu, best_train_loss, best_val_loss, history,
           stats, tta_val_acc
end

# =============================================================================
# Run
# =============================================================================
println("Starting v2.2 training (maximum speed + SciML)...")
println("🖥️  CUDA pool: $(get(ENV, "JULIA_CUDA_MEMORY_POOL", "default"))")

model, best_ps_cpu, best_st_cpu, best_train_loss, best_val_loss, history,
    norm_stats, tta_val_acc = train_with_lr_schedule()

ps_cpu = best_ps_cpu
st_cpu = set_dropout_mode(best_st_cpu, false)

println("\n" * "="^60)
@printf("✅ Training completed!  Best train: %.4f  Best val: %.4f  TTA val acc: %.2f%%\n",
        best_train_loss, best_val_loss, tta_val_acc * 100)

mkpath(DefaultConfig.MODEL_SAVE_DIR)
timestamp  = Dates.format(now(), "yyyymmdd_HHMMSS")
model_path = joinpath(DefaultConfig.MODEL_SAVE_DIR, "model_v2_$(timestamp).jld2")

jldsave(model_path;
    model_version   = "SAR_CNN_v2_SciML",
    ps              = ps_cpu,
    st              = st_cpu,
    best_train_loss = best_train_loss,
    best_val_loss   = best_val_loss,
    history         = history,
    norm_mean       = norm_stats.mean,
    norm_std        = norm_stats.std,
    tta_val_acc     = tta_val_acc,
    backend         = BACKEND,
    batch_size      = BATCH_SIZE,
    learning_rate   = LEARNING_RATE,
    max_lr          = MAX_LR,
    weight_decay    = WEIGHT_DECAY,
    label_smooth    = LABEL_SMOOTH_α,
    bn_update_freq  = BN_UPDATE_FREQ,
    λ_gradient      = λ_GRADIENT,
    λ_symmetry      = λ_SYMMETRY,
    epochs          = length(history.train_loss),
    timestamp       = timestamp
)
println("✅ Model saved to: $model_path")
println("   (includes norm_mean=$(round(norm_stats.mean, digits=4)), norm_std=$(round(norm_stats.std, digits=4)) for inference)")

reclaim_memory!()
mem = gpu_memory_info()
println("🧹 Final VRAM: $(mem.used) / $(mem.total) GiB")
println("="^60)