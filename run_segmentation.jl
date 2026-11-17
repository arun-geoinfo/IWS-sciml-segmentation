# ============================================================================
# run_segmentation.jl — Full KdV-Regularized U-Net Training Pipeline
#
# Usage: julia --project=. run_segmentation.jl
# ============================================================================

using Random, Printf, Dates, Statistics
using Lux, LuxCore, LuxCUDA, Optimisers, Zygote, JLD2
using CUDA

# --- Include pipeline modules ---
include("src/dataloaders/seg_dataloader.jl")
include("src/training/seg_train.jl")
include("src/evaluation/seg_evaluate.jl")

using .SegDataLoader
using .SegTraining
using .SegTraining.SegUNet: build_unet, count_parameters
using .SegTraining.PhysicsLosses: LossWeights
using .SegEvaluation

# ============================================================================
# Configuration
# ============================================================================
const DATA_DIR = joinpath(@__DIR__, "data", "zenodo_oceanic_phenomena")
const OUTPUT_DIR = joinpath(@__DIR__, "outputs")
mkpath(OUTPUT_DIR)

const RNG = Random.MersenneTwister(42)

# Device setup
const USE_GPU = CUDA.functional()
const DEV = USE_GPU ? Lux.gpu_device() : Lux.cpu_device()

println("=" ^ 60)
println("  KdV-Regularized U-Net — IW Segmentation Training")
println("=" ^ 60)
println("  Date:   $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("  Device: $(USE_GPU ? "GPU ($(CUDA.name(CUDA.device())))" : "CPU")")
if USE_GPU
    mem = CUDA.MemoryInfo()
    println("  VRAM:   $(round(mem.total_bytes / 1024^3, digits=1)) GB total, $(round(mem.free_bytes / 1024^3, digits=1)) GB free")
end
println("=" ^ 60)

# ============================================================================
# Step 1: Load Data
# ============================================================================
println("\n[1/5] Loading dataset...")
gpu_transfer = USE_GPU ? (x -> DEV(x)) : nothing

train_loader, val_loader, test_loader = create_loaders(
    DATA_DIR;
    target_class = 12,       # IW class
    target_size  = (256, 256),
    batch_size   = 4,        # conservative for 8GB VRAM
    split_ratios = (0.7, 0.15, 0.15),
    seed         = 42,
    gpu_transfer = gpu_transfer
)

sizes = get_split_sizes(train_loader, val_loader, test_loader)
println("  Train: $(sizes.train), Val: $(sizes.val), Test: $(sizes.test)")

# ============================================================================
# Step 2: Build Model
# ============================================================================
println("\n[2/5] Building U-Net model...")
model, ps, st = build_unet(RNG; in_channels=1, out_channels=1)
n_params = count_parameters(ps)
println("  Parameters: $(n_params) ($(round(n_params * 4 / 1024^2, digits=1)) MB)")

# Move to GPU
ps = DEV(ps)
st = DEV(st)
println("  Model moved to $(USE_GPU ? "GPU" : "CPU")")

# ============================================================================
# Step 3: Configure Training
# ============================================================================
println("\n[3/5] Configuring training...")

loss_weights = LossWeights(
    lambda_dice = 1.0f0,
    lambda_bce  = 1.0f0,
    lambda_kdv  = 0.01f0,   # gentle KdV regularization
    lambda_grad = 0.005f0   # gentle gradient anisotropy
)

config = TrainConfig(;
    epochs         = 50,
    lr             = 1.0f-3,
    weight_decay   = 1.0f-2,
    grad_clip_norm = 1.0f0,
    patience       = 15,
    checkpoint_path = joinpath(OUTPUT_DIR, "best_seg_unet.jld2"),
    loss_weights   = loss_weights
)

println("  Epochs:       $(config.epochs)")
println("  LR:           $(config.lr)")
println("  Weight decay: $(config.weight_decay)")
println("  Patience:     $(config.patience)")
println("  Loss weights: dice=$(loss_weights.lambda_dice), bce=$(loss_weights.lambda_bce), kdv=$(loss_weights.lambda_kdv), grad=$(loss_weights.lambda_grad)")
println("  Checkpoint:   $(config.checkpoint_path)")

# ============================================================================
# Step 4: Train
# ============================================================================
println("\n[4/5] Training...")
println("-" ^ 100)

train_start = time()
ps, st, log = train!(model, ps, st, train_loader, val_loader, config; verbose=true)
train_elapsed = time() - train_start

println("-" ^ 100)
println("Training complete in $(round(train_elapsed / 60, digits=1)) minutes")
println("  Best val Dice: $(round(log.best_val_dice, digits=4)) at epoch $(log.best_epoch)")
println("  Early stopped: $(log.stopped_early)")

# ============================================================================
# Step 5: Evaluate on Test Set
# ============================================================================
println("\n[5/5] Evaluating on test set...")

# Load best checkpoint
best_ps, best_st, meta = SegTraining.load_checkpoint(config.checkpoint_path)
best_ps = DEV(best_ps)
best_st = DEV(best_st)
println("  Loaded best checkpoint (epoch $(meta.epoch), val_dice=$(round(meta.val_dice, digits=4)))")

# Run evaluation
metrics, all_preds, all_targets = evaluate_segmentation(model, best_ps, best_st,
                                                         test_loader; threshold=0.5f0)

println("\n  Test Metrics:")
println("    Dice:      $(round(metrics.dice, digits=4))")
println("    IoU:       $(round(metrics.iou, digits=4))")
println("    Precision: $(round(metrics.precision, digits=4))")
println("    Recall:    $(round(metrics.recall, digits=4))")
println("    F1:        $(round(metrics.f1, digits=4))")
println("    Accuracy:  $(round(metrics.accuracy, digits=4))")

# Save evaluation report
report_path = joinpath(OUTPUT_DIR,
    "evaluation_$(Dates.format(now(), "yyyymmdd_HHMMSS")).json")

config_info = Dict{String,Any}(
    "model"         => "KdV-Regularized U-Net",
    "epochs_trained" => length(log.epochs),
    "best_epoch"    => log.best_epoch,
    "batch_size"    => 4,
    "lr"            => Float64(config.lr),
    "lambda_dice"   => Float64(loss_weights.lambda_dice),
    "lambda_bce"    => Float64(loss_weights.lambda_bce),
    "lambda_kdv"    => Float64(loss_weights.lambda_kdv),
    "lambda_grad"   => Float64(loss_weights.lambda_grad),
    "train_samples" => sizes.train,
    "val_samples"   => sizes.val,
    "test_samples"  => sizes.test,
    "device"        => USE_GPU ? "GPU" : "CPU",
    "train_time_min" => round(train_elapsed / 60, digits=1),
    "early_stopped" => log.stopped_early
)

report = save_report(report_path, metrics; threshold=0.5f0, config_info=config_info)
println("\n  Report saved to: $report_path")

# Save training log
log_path = joinpath(OUTPUT_DIR, "training_log_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2")
jldsave(log_path; training_log=log.epochs, best_epoch=log.best_epoch,
        best_val_dice=log.best_val_dice, early_stopped=log.stopped_early)
println("  Training log saved to: $log_path")

println("\n" * "=" ^ 60)
println("  Pipeline Complete!")
println("=" ^ 60)
