using JSON3, CairoMakie, PrettyTables
using Statistics: mean, std
output_dir = "/home/arun/Desktop/IWS_Julia/COMPARISON JULIA/RESULTS"
mkpath(output_dir)
# ──────────────────────────────────────────────────────────────────────────────
# CONFIG – adjust only these paths if needed
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# CONFIG – absolute paths (safest)
# ──────────────────────────────────────────────────────────────────────────────

julia_dir = "/home/arun/Desktop/IWS_Julia/COMPARISON JULIA/OUTPUT_JSONS"
python_dir = "/home/arun/Desktop/IWS_python/COMPARISON_PYTHON/OUTPUT_JSONS"
output_dir = "/home/arun/Desktop/IWS_Julia/COMPARISON JULIA/RESULTS"

mkpath(output_dir)

# Load JSONs
julia_eval = JSON3.read(joinpath(julia_dir, "evaluation_metrics_julia_20260306_115854.json"))
julia_power = JSON3.read(joinpath(julia_dir, "gpu_power_profile_julia_2026-03-06_11-30-17.json"))
julia_inf   = JSON3.read(joinpath(julia_dir, "inference_metrics_julia_2026-03-06_11-30-16.json"))

python_eval = JSON3.read(joinpath(python_dir, "evaluation_metrics_20260305_132730.json"))
python_complete = JSON3.read(joinpath(python_dir, "complete_metrics_20260306_151947.json"))

# ──────────────────────────────────────────────────────────────────────────────
# Extract key numbers
# ──────────────────────────────────────────────────────────────────────────────

# Power efficiency (net & gross mJ/image)
julia_net_mj   = julia_power["energy_metrics"]["net_energy_per_image_mj"]
julia_gross_mj = julia_power["energy_metrics"]["energy_per_image_mj"]
python_net_mj   = python_complete["results"]["combined_power_profile"]["publishable_summary"]["energy_per_image_mj_net"]
python_gross_mj = python_complete["results"]["combined_power_profile"]["publishable_summary"]["energy_per_image_mj_gross"]

# Throughput (images/sec)
julia_throughput = julia_inf["images_per_second"]
python_throughput = python_complete["results"]["combined"]["throughput_images_per_second"]

# Latency per image (ms)
julia_latency_ms = julia_inf["single_image_latency"] * 1000
python_latency_ms = python_complete["results"]["combined"]["avg_latency_per_image_ms"]

# Peak memory
julia_peak_gpu = julia_inf["peak_gpu_mb"]
julia_peak_cpu = julia_inf["peak_cpu_mb"]
python_peak_gpu = python_complete["results"]["combined"]["peak_gpu_memory_inference_mb"]
python_peak_cpu = python_complete["results"]["combined"]["peak_cpu_memory_mb"]

# AUC-ROC
julia_auc = julia_eval["model_performance"]["auc_roc"]
python_auc = python_eval["model_performance"]["auc_roc"]

println("Key numbers extracted:")
println("Julia net mJ/img:   $julia_net_mj")
println("Python net mJ/img:  $python_net_mj")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Power Efficiency – Bar chart (log scale)
# ──────────────────────────────────────────────────────────────────────────────

fig_power = Figure(size=(800, 500))
ax = Axis(fig_power[1,1],
    title = "Energy per Image – Julia vs Python (net & gross)",
    ylabel = "mJ per image (log scale)",
    xticks = ([1,2], ["Julia", "Python"]),
    yscale = log10,
    yticks = [1, 10, 100, 1000, 10000],
    yminorticksvisible = true)

barplot!(ax, [1,2], [julia_net_mj, python_net_mj], color=:skyblue, label="Net")
barplot!(ax, [1,2], [julia_gross_mj, python_gross_mj], color=:lightcoral, label="Gross")

axislegend(ax, position=:lt)
save(joinpath(output_dir, "power_efficiency_log.png"), fig_power)
println("Saved: power_efficiency_log.png")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Throughput bar chart
# ──────────────────────────────────────────────────────────────────────────────

fig_throughput = Figure(size=(600, 400))
ax_t = Axis(fig_throughput[1,1],
    title = "Throughput (images/sec)",
    xticks = ([1,2], ["Julia", "Python"]),
    ylabel = "images/sec")

barplot!(ax_t, [1,2], [julia_throughput, python_throughput], color=:seagreen)
text!(ax_t, 1, julia_throughput*1.1, text="$(round(julia_throughput,digits=0))", align=(:center,:bottom))
text!(ax_t, 2, python_throughput*1.1, text="$(round(python_throughput,digits=1))", align=(:center,:bottom))

save(joinpath(output_dir, "throughput_comparison.png"), fig_throughput)
println("Saved: throughput_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Memory utilization – side-by-side bars
# ──────────────────────────────────────────────────────────────────────────────

fig_mem = Figure(size=(800, 450))
ax_mem = Axis(fig_mem[1,1],
    title = "Peak Memory Usage",
    xticks = ([1,2,3,4], ["Julia GPU", "Julia CPU", "Python GPU", "Python CPU"]),
    ylabel = "Peak memory (MB)")

barplot!(ax_mem, [1,2,3,4], [julia_peak_gpu, julia_peak_cpu, python_peak_gpu, python_peak_cpu],
         color=[:cornflowerblue, :lightblue, :coral, :lightsalmon])

save(joinpath(output_dir, "peak_memory_comparison.png"), fig_mem)
println("Saved: peak_memory_comparison.png")

# 4. AUC-ROC comparison (simple bar + text)
fig_auc = Figure(size=(500, 400))
ax_auc = Axis(fig_auc[1,1],
    title = "AUC-ROC Score",
    xticks = ([1,2], ["Julia", "Python"]),
    ylabel = "AUC-ROC")

barplot!(ax_auc, [1,2], [julia_auc, python_auc], color=:gray)
text!(ax_auc, 1, julia_auc + 0.002, text="$(round(julia_auc,digits=4))", align=(:center,:bottom))
text!(ax_auc, 2, python_auc + 0.002, text="$(round(python_auc,digits=4))", align=(:center,:bottom))

ylims!(ax_auc, 0.95, 1.00)   # ← this fixes it (or use limits as above)

save(joinpath(output_dir, "auc_roc_comparison.png"), fig_auc)
println("Saved: auc_roc_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Threshold sweep – Accuracy & F1 vs threshold (multi-line)
# ──────────────────────────────────────────────────────────────────────────────

# Julia sweep
julia_sweep = julia_eval["threshold_sweep"]
julia_thresh = [x["threshold"] for x in julia_sweep]
julia_acc   = [x["accuracy"] for x in julia_sweep]
julia_f1    = [x["f1_score"] for x in julia_sweep]

# Python sweep
python_sweep = python_eval["threshold_sweep"]
python_thresh = [x["threshold"] for x in python_sweep]
python_acc   = [x["accuracy"] for x in python_sweep]
python_f1    = [x["f1_score"] for x in python_sweep]

fig_thresh = Figure(size=(900, 600))
ax_th = Axis(fig_thresh[1,1],
    title = "Threshold vs Accuracy & F1-score",
    xlabel = "Decision Threshold",
    ylabel = "Score",
    limits = (0.25, 0.75, 0.85, 0.97))

lines!(ax_th, julia_thresh, julia_acc, color=:blue, linewidth=2, label="Julia Accuracy")
lines!(ax_th, julia_thresh, julia_f1,  color=:dodgerblue, linestyle=:dash, linewidth=2, label="Julia F1")

lines!(ax_th, python_thresh, python_acc, color=:red, linewidth=2, label="Python Accuracy")
lines!(ax_th, python_thresh, python_f1,  color=:orangered, linestyle=:dash, linewidth=2, label="Python F1")

axislegend(ax_th, position=:lb)
save(joinpath(output_dir, "threshold_sensitivity_comparison.png"), fig_thresh)
println("Saved: threshold_sensitivity_comparison.png")

println("\nAll plots saved in: $output_dir")
println("Done.")
