# ============================================================================
# seg_train.jl — Training Loop for KdV-Regularized U-Net Segmentation
#
# Spec: specs/training.yaml (TR-001 through TR-006)
# ============================================================================

module SegTraining

using Random, Statistics, Printf, Dates
using Lux, LuxCore, Optimisers, Zygote, JLD2

# Import our modules
include(joinpath(@__DIR__, "..", "models", "seg_unet_model.jl"))
include(joinpath(@__DIR__, "physics_losses.jl"))
using .SegUNet
using .PhysicsLosses

export TrainConfig, TrainingLog, train!, train_step!, validate, EpochMetrics

# ============================================================================
# Configuration
# ============================================================================
struct TrainConfig
    epochs::Int
    lr::Float32
    weight_decay::Float32
    grad_clip_norm::Float32
    patience::Int
    checkpoint_path::String
    loss_weights::LossWeights
end

function TrainConfig(;
    epochs::Int = 50,
    lr::Float32 = 1.0f-3,
    weight_decay::Float32 = 1.0f-2,
    grad_clip_norm::Float32 = 1.0f0,
    patience::Int = 10,
    checkpoint_path::String = "best_seg_model.jld2",
    loss_weights::LossWeights = LossWeights()
)
    return TrainConfig(epochs, lr, weight_decay, grad_clip_norm, patience,
                       checkpoint_path, loss_weights)
end

# ============================================================================
# Training log — stores per-epoch metrics (TR-004)
# ============================================================================
struct EpochMetrics
    train_loss::Float32
    val_loss::Float32
    val_dice::Float32
    val_iou::Float32
end

mutable struct TrainingLog
    epochs::Vector{EpochMetrics}
    best_val_dice::Float32
    best_epoch::Int
    patience_counter::Int
    stopped_early::Bool
end

TrainingLog() = TrainingLog(EpochMetrics[], -Inf32, 0, 0, false)

# ============================================================================
# Segmentation metrics (Dice, IoU) for validation
# ============================================================================
function compute_dice(pred::AbstractArray, target::AbstractArray; threshold=0.5f0)
    p = Float32.(pred .>= threshold)
    t = Float32.(target .>= threshold)
    intersection = sum(p .* t)
    union_sum = sum(p) + sum(t)
    union_sum == 0 && return 1.0f0  # Both empty = perfect
    return Float32(2.0f0 * intersection / union_sum)
end

function compute_iou(pred::AbstractArray, target::AbstractArray; threshold=0.5f0)
    p = Float32.(pred .>= threshold)
    t = Float32.(target .>= threshold)
    intersection = sum(p .* t)
    union_val = sum(p) + sum(t) - intersection
    union_val == 0 && return 1.0f0  # Both empty = perfect
    return Float32(intersection / union_val)
end

# ============================================================================
# Gradient clipping
# ============================================================================
function clip_gradients(grads, max_norm::Float32)
    total_norm = _grad_norm(grads)
    scale = min(1.0f0, max_norm / (total_norm + 1.0f-7))
    scale ≈ 1.0f0 && return grads
    return _scale_grads(grads, scale)
end

function _grad_norm(g::NamedTuple)
    s = 0.0f0
    for k in keys(g)
        s += _grad_norm(g[k])
    end
    return s
end
_grad_norm(g::AbstractArray) = Float32(sum(abs2, g))
_grad_norm(::Nothing) = 0.0f0
_grad_norm(::Any) = 0.0f0

function _scale_grads(g::NamedTuple, s::Float32)
    vals = map(k -> _scale_grads(g[k], s), keys(g))
    return NamedTuple{keys(g)}(vals)
end
_scale_grads(g::AbstractArray, s::Float32) = g .* s
_scale_grads(g::Nothing, ::Float32) = nothing
_scale_grads(g::Any, ::Float32) = g

# ============================================================================
# TR-005: Single training step
# ============================================================================
"""
    train_step!(model, ps, st, opt_state, X, Y, config) -> (loss, ps, st, opt_state, components)

Execute one training step: forward pass, loss computation, backward pass,
parameter update. Returns updated params, state, optimizer state.
"""
function train_step!(model, ps, st, opt_state, X, Y, config::TrainConfig)
    # Ensure training mode for forward pass inside gradient computation
    st_train = Lux.trainmode(st)

    # Compute loss and gradients
    (loss_val, components), back = Zygote.pullback(ps) do p
        pred, st_new, intermediates = model(X, p, st_train)
        total, comps = combined_loss(pred, Y, intermediates, config.loss_weights)
        return total, comps
    end

    gs = back((one(Float32), nothing))[1]

    # Clip gradients
    gs = clip_gradients(gs, config.grad_clip_norm)

    # Update parameters
    opt_state, ps = Optimisers.update(opt_state, ps, gs)

    # Update BN running statistics with a non-gradient forward pass
    _, st_updated, _ = Lux.apply(model, X, ps, Lux.trainmode(st))

    return loss_val, ps, st_updated, opt_state, components
end

# ============================================================================
# Validation pass (no gradients)
# ============================================================================
"""
    validate(model, ps, st, val_loader, loss_weights) -> (val_loss, val_dice, val_iou)
"""
function validate(model, ps, st, val_loader, loss_weights::LossWeights)
    st_eval = Lux.testmode(st)
    total_loss = 0.0f0
    total_dice = 0.0f0
    total_iou = 0.0f0
    n_batches = 0

    for (X, Y) in val_loader
        pred, _, intermediates = model(X, ps, st_eval)
        loss, _ = combined_loss(pred, Y, intermediates, loss_weights)
        total_loss += Float32(loss)
        total_dice += compute_dice(pred, Y)
        total_iou += compute_iou(pred, Y)
        n_batches += 1
    end

    n_batches == 0 && return (0.0f0, 0.0f0, 0.0f0)
    return (total_loss / n_batches,
            total_dice / n_batches,
            total_iou / n_batches)
end

# ============================================================================
# TR-001, TR-002, TR-003: Main training loop
# ============================================================================
"""
    train!(model, ps, st, train_loader, val_loader, config; verbose=true)
        -> (ps, st, log)

Full training loop with early stopping and checkpointing.
"""
function train!(model, ps, st, train_loader, val_loader, config::TrainConfig;
                verbose::Bool=true)
    log = TrainingLog()

    # Setup optimizer (AdamW)
    opt = Optimisers.AdamW(; eta=config.lr, lambda=config.weight_decay)
    opt_state = Optimisers.setup(opt, ps)

    for epoch in 1:config.epochs
        epoch_start = time()

        # === Training phase ===
        epoch_loss = 0.0f0
        n_batches = 0

        for (X, Y) in train_loader
            loss, ps, st, opt_state, _ = train_step!(model, ps, st, opt_state,
                                                      X, Y, config)
            epoch_loss += Float32(loss)
            n_batches += 1
        end

        avg_train_loss = n_batches > 0 ? epoch_loss / n_batches : 0.0f0

        # === Validation phase (no gradients) ===
        val_loss, val_dice, val_iou = validate(model, ps, st, val_loader,
                                                config.loss_weights)

        # === Log metrics (TR-004) ===
        metrics = EpochMetrics(avg_train_loss, val_loss, val_dice, val_iou)
        push!(log.epochs, metrics)

        epoch_time = time() - epoch_start
        if verbose
            @printf("Epoch %3d/%d | train_loss: %.4f | val_loss: %.4f | val_dice: %.4f | val_iou: %.4f | %.1fs\n",
                    epoch, config.epochs, avg_train_loss, val_loss, val_dice, val_iou, epoch_time)
        end

        # === Checkpoint on improvement (TR-002) ===
        if val_dice > log.best_val_dice
            log.best_val_dice = val_dice
            log.best_epoch = epoch
            log.patience_counter = 0

            # Save checkpoint
            ps_cpu = Lux.cpu_device()(ps)
            st_cpu = Lux.cpu_device()(Lux.testmode(st))
            jldsave(config.checkpoint_path;
                    params=ps_cpu, state=st_cpu,
                    epoch=epoch, val_dice=val_dice, val_iou=val_iou,
                    config=config)

            if verbose
                @printf("  -> New best! Dice=%.4f saved to %s\n",
                        val_dice, config.checkpoint_path)
            end
        else
            log.patience_counter += 1
        end

        # === Early stopping (TR-003) ===
        if log.patience_counter >= config.patience
            log.stopped_early = true
            if verbose
                println("Early stopping at epoch $epoch (patience=$(config.patience))")
            end
            break
        end

        # Reshuffle training data for next epoch
        if hasproperty(train_loader, :epoch)
            train_loader.epoch = epoch
            if train_loader.shuffle
                rng = Random.MersenneTwister(train_loader.seed + epoch)
                Random.shuffle!(rng, train_loader.indices)
            end
        end
    end

    return ps, st, log
end

# ============================================================================
# Checkpoint loading
# ============================================================================
"""
    load_checkpoint(path) -> (params, state, metadata)
"""
function load_checkpoint(path::String)
    data = JLD2.load(path)
    return data["params"], data["state"],
           (epoch=data["epoch"], val_dice=data["val_dice"], val_iou=data["val_iou"])
end

end # module SegTraining
