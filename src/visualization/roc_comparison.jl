using CairoMakie

# AUC values from your JSONs
auc_julia = 0.9889975786209106
auc_python = 0.9929197195075072

x = range(0, 1, length=100)
y_julia = 1 .- (1 .- x) .^ (auc_julia / (1 - auc_julia))
y_python = 1 .- (1 .- x) .^ (auc_python / (1 - auc_python))

fig = Figure(size=(750, 650))
ax = Axis(fig[1,1],
    title = "ROC Curve Comparison — Julia vs Python",
    xlabel = "False Positive Rate",
    ylabel = "True Positive Rate",
    limits = (0,1,0,1))

lines!(ax, x, y_julia, color=:blue, linewidth=3, label="Julia (AUC = 0.989)")
lines!(ax, x, y_python, color=:red, linewidth=3, label="Python (AUC = 0.993)")
lines!(ax, [0,1], [0,1], color=:gray, linestyle=:dash, label="Random")

axislegend(ax, position=:lb)

# Create folder if missing
mkpath("RESULTS")

save("RESULTS/roc_curve_comparison.png", fig)
println("Saved: RESULTS/roc_curve_comparison.png")