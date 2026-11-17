# ============================================================================
# plot_training_curves.jl — Visualize training history
#
# Usage: julia --project=. plot_training_curves.jl
# ============================================================================

using JLD2, CairoMakie

# Load training log
log_path = let
    files = filter(f -> startswith(f, "training_log_") && endswith(f, ".jld2"),
                   readdir("outputs"))
    joinpath("outputs", last(sort(files)))  # most recent
end

println("Loading: $log_path")
data = JLD2.load(log_path)
epochs_data = data["training_log"]
best_epoch = data["best_epoch"]
best_dice = data["best_val_dice"]

n = length(epochs_data)
ep = 1:n
train_loss = [e.train_loss for e in epochs_data]
val_loss   = [e.val_loss for e in epochs_data]
val_dice   = [e.val_dice for e in epochs_data]
val_iou    = [e.val_iou for e in epochs_data]

# ============================================================================
# Create figure with 3 subplots
# ============================================================================
fig = Figure(size=(1200, 900), fontsize=14)

# --- Panel 1: Loss curves ---
ax1 = Axis(fig[1, 1:2],
    title = "Training & Validation Loss",
    xlabel = "Epoch", ylabel = "Loss",
    xlabelsize = 13, ylabelsize = 13)

lines!(ax1, ep, train_loss, color=:dodgerblue, linewidth=2, label="Train Loss")
lines!(ax1, ep, val_loss, color=:orangered, linewidth=2, label="Val Loss")
vlines!(ax1, [best_epoch], color=:gray60, linestyle=:dash, linewidth=1,
        label="Best (epoch $best_epoch)")
axislegend(ax1, position=:rt)

# --- Panel 2: Dice & IoU ---
ax2 = Axis(fig[2, 1:2],
    title = "Validation Dice & IoU",
    xlabel = "Epoch", ylabel = "Score",
    xlabelsize = 13, ylabelsize = 13)

lines!(ax2, ep, val_dice, color=:seagreen, linewidth=2, label="Val Dice")
lines!(ax2, ep, val_iou, color=:mediumpurple, linewidth=2, label="Val IoU")
scatter!(ax2, [best_epoch], [best_dice], color=:red, markersize=12,
         marker=:star5, label="Best Dice = $(round(best_dice, digits=4))")
vlines!(ax2, [best_epoch], color=:gray60, linestyle=:dash, linewidth=1)
axislegend(ax2, position=:rb)

# --- Panel 3: Loss component breakdown (train) ---
ax3 = Axis(fig[3, 1:2],
    title = "Training Loss Trend (smoothed)",
    xlabel = "Epoch", ylabel = "Loss",
    xlabelsize = 13, ylabelsize = 13)

# Exponential moving average for smoother visualization
function ema(x; alpha=0.3)
    s = similar(x)
    s[1] = x[1]
    for i in 2:length(x)
        s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
    end
    return s
end

lines!(ax3, ep, train_loss, color=(:dodgerblue, 0.3), linewidth=1, label="Raw")
lines!(ax3, ep, ema(train_loss), color=:dodgerblue, linewidth=2.5, label="Smoothed (EMA)")
lines!(ax3, ep, val_loss, color=(:orangered, 0.3), linewidth=1)
lines!(ax3, ep, ema(val_loss), color=:orangered, linewidth=2.5, label="Val (smoothed)")
axislegend(ax3, position=:rt)

# Supertitle
Label(fig[0, :], "KdV-Regularized U-Net — IW Segmentation Training",
      fontsize=18, font=:bold)

out_path = "outputs/training_curves.png"
save(out_path, fig, px_per_unit=2)
println("Saved: $out_path")
