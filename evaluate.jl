# evaluate.jl - GPU-ENABLED EVALUATION WITH AUC-ROC AND THRESHOLD OPTIMIZATION
using Lux, LuxCore, NNlib
using Random, Statistics, Printf, Dates, JLD2
using ProgressBars
using Optimisers
using Zygote
using CairoMakie
using LuxCUDA  # GPU support
import CairoMakie: Axis, Figure, lines!, axislegend, xlims!, ylims!, save, vlines!, heatmap!, text!, Colorbar

# Include project modules
include("src/config/default_config.jl")
include("src/dataloaders/lux_dataloader.jl")
include("src/models/sar_cnn_model.jl")
using .DefaultConfig
using .DefaultConfig.DeviceConfig

# Function to calculate metrics at a given threshold
function calculate_metrics_at_threshold(y_true::Vector{Float32}, y_scores::Vector{Float32}, threshold::Float32)
    predictions = y_scores .>= threshold
    labels = y_true .>= 0.5f0
    
    # Calculate confusion matrix components
    tp = sum(predictions .& labels)
    tn = sum(.!predictions .& .!labels)
    fp = sum(predictions .& .!labels)
    fn = sum(.!predictions .& labels)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp > 0 ? tp / (tp + fp) : 0.0f0
    recall = tp > 0 ? tp / (tp + fn) : 0.0f0
    f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0f0
    
    return Dict(
        "threshold" => threshold,
        "accuracy" => accuracy,
        "precision" => precision,
        "recall" => recall,
        "f1_score" => f1_score,
        "tp" => tp,
        "tn" => tn,
        "fp" => fp,
        "fn" => fn
    )
end

# Function to plot confusion matrix
function plot_confusion_matrix(metrics::Dict, save_path::String="confusion_matrix.png")
    tp = Int(metrics["tp"])
    tn = Int(metrics["tn"])
    fp = Int(metrics["fp"])
    fn = Int(metrics["fn"])
    
    # Create confusion matrix
    conf_matrix = Float32[tn fp; fn tp]
    
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], 
              xlabel="Predicted Label", 
              ylabel="True Label",
              title="Confusion Matrix (Threshold = $(round(metrics["threshold"], digits=3)))",
              xticks=([1, 2], ["Non-IW", "IW"]),
              yticks=([1, 2], ["Non-IW", "IW"]))
    
    # Plot heatmap
    hm = heatmap!(ax, conf_matrix, colormap=:Blues, colorrange=(0, maximum(conf_matrix)))
    
    # Add text annotations
    text!(ax, 1, 1, text="TN\n$tn", align=(:center, :center), fontsize=24, color=:black)
    text!(ax, 2, 1, text="FP\n$fp", align=(:center, :center), fontsize=24, color=:black)
    text!(ax, 1, 2, text="FN\n$fn", align=(:center, :center), fontsize=24, color=:black)
    text!(ax, 2, 2, text="TP\n$tp", align=(:center, :center), fontsize=24, color=:black)
    
    # Add colorbar
    Colorbar(fig[1, 2], hm, label="Count")
    
    save(save_path, fig)
    println("   Confusion matrix saved to: $save_path")
    
    return fig
end

# Function to analyze FP vs FN with scientific interpretation
function analyze_fp_vs_fn(metrics::Dict, y_true::Vector{Float32}, y_scores::Vector{Float32})
    println("\n" * "="^60)
    println("🔬 FALSE POSITIVE vs FALSE NEGATIVE ANALYSIS")
    println("="^60)
    
    tp = Int(metrics["tp"])
    tn = Int(metrics["tn"])
    fp = Int(metrics["fp"])
    fn = Int(metrics["fn"])
    total = tp + tn + fp + fn
    
    # Calculate error rates
    fp_rate = fp / total
    fn_rate = fn / total
    
    println("\n📊 CONFUSION MATRIX BREAKDOWN:")
    println("   ┌─────────────────┬─────────────────┐")
    println("   │                 │   PREDICTED     │")
    println("   │                 ├────────┬────────┤")
    println("   │                 │ Non-IW │   IW   │")
    println("   ├─────────────────┼────────┼────────┤")
    println("   │ ACTUAL: Non-IW  │   $tn   │   $fp   │")
    println("   │ ACTUAL: IW      │   $fn   │   $tp   │")
    println("   └─────────────────┴────────┴────────┘")
    
    println("\n🎯 ERROR TYPE BREAKDOWN:")
    println("   True Positives (TP):  $tp (correctly detected IW)")
    println("   True Negatives (TN):  $tn (correctly detected Non-IW)")
    println("   False Positives (FP): $fp (Non-IW misclassified as IW)")
    println("   False Negatives (FN): $fn (IW misclassified as Non-IW)")
    
    println("\n📈 ERROR RATES:")
    println("   FP Rate: $(round(fp_rate*100, digits=2))% of total samples")
    println("   FN Rate: $(round(fn_rate*100, digits=2))% of total samples")
    
    println("\n⚖️  SCIENTIFIC INTERPRETATION:")
    println("   FALSE POSITIVES (FP = $fp):")
    println("   → Non-IW pixels classified as IW")
    println("   → Impact: NOISY SCIENCE - adds spurious detections")
    println("   → Consequence: Overestimation of IW events")
    println("   → Resource cost: Wasted analysis time on false detections")
    
    println("\n   FALSE NEGATIVES (FN = $fn):")
    println("   → IW pixels classified as Non-IW")
    println("   → Impact: MISSED EVENTS - lose real phenomena")
    println("   → Consequence: Underestimation of IW occurrence")
    println("   → Scientific cost: Lost scientific data and insights")
    
    # Determine which error is more prevalent
    if fp > fn
        error_bias = "FALSE POSITIVES"
        bias_amount = fp - fn
        println("\n⚠️  CURRENT ERROR BIAS: $error_bias dominate by $bias_amount samples")
        println("   → Model is too LIBERAL (predicting IW too often)")
        println("   → Suggestion: INCREASE threshold to reduce FP")
    elseif fn > fp
        error_bias = "FALSE NEGATIVES"
        bias_amount = fn - fp
        println("\n⚠️  CURRENT ERROR BIAS: $error_bias dominate by $bias_amount samples")
        println("   → Model is too CONSERVATIVE (missing IW events)")
        println("   → Suggestion: DECREASE threshold to reduce FN")
    else
        println("\n✅ ERROR BALANCE: FP and FN are equal")
        println("   → Model has balanced error types")
    end
    
    println("\n💡 THRESHOLD BIAS RECOMMENDATIONS:")
    println("   If FN is more costly (missing IW is worse):")
    println("   → Use LOWER threshold (e.g., 0.4) - favor recall")
    println("   → Accepts more FP to minimize FN")
    
    println("\n   If FP is more costly (false alarms are worse):")
    println("   → Use HIGHER threshold (e.g., 0.6) - favor precision")
    println("   → Accepts more FN to minimize FP")
    
    println("="^60)
    
    return fp, fn, fp_rate, fn_rate
end

# Function to optimize threshold with cost-based selection
function optimize_threshold_with_cost(y_true::Vector{Float32}, y_scores::Vector{Float32}; 
                                     threshold_range=0.3f0:0.01f0:0.7f0,
                                     optimization_metric="accuracy",
                                     fp_cost::Float32=1.0f0,
                                     fn_cost::Float32=1.0f0)
    println("\n🔍 THRESHOLD OPTIMIZATION WITH COST ANALYSIS")
    println("="^60)
    println("⚠️  NOTE: We do NOT assume threshold = 0.5 is optimal!")
    println("   Sweeping thresholds from $(first(threshold_range)) to $(last(threshold_range))")
    println("   Optimization metric: $optimization_metric")
    println("   FP Cost: $fp_cost | FN Cost: $fn_cost")
    println("="^60)
    
    best_metric_value = Inf
    best_threshold = 0.5f0
    best_metrics = nothing
    all_results = []
    
    for threshold in threshold_range
        metrics = calculate_metrics_at_threshold(y_true, y_scores, threshold)
        
        # Calculate cost-weighted error
        total_cost = metrics["fp"] * fp_cost + metrics["fn"] * fn_cost
        metrics["total_cost"] = total_cost
        
        push!(all_results, metrics)
        
        # Select based on optimization metric
        metric_value = if optimization_metric == "accuracy"
            -metrics["accuracy"]  # negative because we minimize
        elseif optimization_metric == "f1_score"
            -metrics["f1_score"]
        elseif optimization_metric == "balanced"
            abs(metrics["precision"] - metrics["recall"])
        elseif optimization_metric == "cost"
            total_cost  # minimize cost
        else
            -metrics["accuracy"]  # default
        end
        
        if metric_value < best_metric_value
            best_metric_value = metric_value
            best_threshold = threshold
            best_metrics = metrics
        end
    end
    
    return best_threshold, best_metrics, all_results
end

# Function to plot threshold analysis
function plot_threshold_analysis(all_results::Vector, best_threshold::Float32, save_path::String="threshold_analysis.png")
    thresholds = [r["threshold"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]
    precisions = [r["precision"] for r in all_results]
    recalls = [r["recall"] for r in all_results]
    f1_scores = [r["f1_score"] for r in all_results]
    
    fig = Figure(size=(1200, 800))
    
    # Plot metrics vs threshold
    ax = Axis(fig[1, 1], 
              xlabel="Decision Threshold", 
              ylabel="Metric Value",
              title="Metrics vs Decision Threshold")
    
    lines!(ax, thresholds, accuracies, color=:blue, linewidth=2, label="Accuracy")
    lines!(ax, thresholds, precisions, color=:green, linewidth=2, label="Precision")
    lines!(ax, thresholds, recalls, color=:red, linewidth=2, label="Recall")
    lines!(ax, thresholds, f1_scores, color=:purple, linewidth=2, label="F1-Score")
    
    # Mark optimal threshold
    vlines!(ax, [best_threshold], color=:black, linestyle=:dash, linewidth=2, label="Optimal Threshold")
    
    axislegend(ax, position=:rb)
    xlims!(ax, first(thresholds), last(thresholds))
    ylims!(ax, 0, 1)
    
    save(save_path, fig)
    println("   Threshold analysis plot saved to: $save_path")
    
    return fig
end

# Function to calculate AUC-ROC
function calculate_auc_roc(y_true::Vector{Float32}, y_scores::Vector{Float32})
    # Sort by scores in descending order
    sorted_indices = sortperm(y_scores, rev=true)
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate TPR and FPR at different thresholds
    n_positive = sum(y_true_sorted)
    n_negative = length(y_true_sorted) - n_positive
    
    tpr = Float32[]
    fpr = Float32[]
    
    push!(tpr, 0.0f0)
    push!(fpr, 0.0f0)
    
    tp = 0.0f0
    fp = 0.0f0
    
    for i in 1:length(y_true_sorted)
        if y_true_sorted[i] > 0.5f0
            tp += 1.0f0
        else
            fp += 1.0f0
        end
        push!(tpr, tp / n_positive)
        push!(fpr, fp / n_negative)
    end
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0f0
    for i in 2:length(fpr)
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0f0
    end
    
    return auc, fpr, tpr
end

# Function to plot ROC curve
function plot_roc_curve(fpr::Vector{Float32}, tpr::Vector{Float32}, auc::Float32, save_path::String="roc_curve.png")
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], 
              xlabel="False Positive Rate (FPR)", 
              ylabel="True Positive Rate (TPR)",
              title="ROC Curve (AUC = $(round(auc, digits=4)))")
    
    # Plot ROC curve
    lines!(ax, fpr, tpr, color=:blue, linewidth=2, label="ROC Curve")
    
    # Plot diagonal reference line (random classifier)
    lines!(ax, [0.0f0, 1.0f0], [0.0f0, 1.0f0], color=:red, linestyle=:dash, linewidth=2, label="Random Classifier")
    
    # Add legend
    axislegend(ax, position=:rb)
    
    # Set limits
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)
    
    # Save figure
    save(save_path, fig)
    println("   ROC curve saved to: $save_path")
    
    return fig
end

function evaluate_model(model_path::String="model_trained.jld2")
    # ADDED: Start timing
    evaluation_start_time = time()
    
    # Print configuration
    DefaultConfig.print_config()
    
    println("="^60)
    println("SAR CNN MODEL EVALUATION WITH THRESHOLD OPTIMIZATION")
    println("="^60)
    
    # LOAD TRAINED MODEL
    println("\n📂 Loading trained model from: $model_path")
    loaded_data = load(model_path)
    model = loaded_data["model"]
    ps = loaded_data["ps"]
    st = loaded_data["st"]
    
    # Transfer to device
    ps = ps |> DefaultConfig.DEVICE
    st = st |> DefaultConfig.DEVICE
    
    println("✅ Model loaded successfully on $(DefaultConfig.BACKEND)")
    
    # LOAD VALIDATION DATA
    println("\n📊 Loading validation data...")
    # CHANGED: Use DefaultConfig.BATCH_SIZE instead of fixed 32
    val_loader = DataLoader(DefaultConfig.PROC_TEST; 
                          batch_size=DefaultConfig.BATCH_SIZE, 
                          shuffle=false)
    println("   Validation samples: ", length(val_loader.image_paths))
    println("   Batch size: $(DefaultConfig.BATCH_SIZE)")  # ADDED: Show batch size
    
    # EVALUATION - Collect predictions and labels
    println("\n🔍 Evaluating model...")
    st_val = LuxCore.testmode(st)
    
    # Store all predictions and true labels
    all_predictions = Float32[]
    all_labels = Float32[]
    
    for (X_val, y_val) in val_loader
        # Hybrid mode: transfer to GPU if needed
        if DefaultConfig.COMPUTE_MODE == :hybrid
            X_val = DeviceConfig.safe_gpu_transfer(X_val, DefaultConfig.DEVICE, DefaultConfig.BACKEND)
            y_val = DeviceConfig.safe_gpu_transfer(y_val, DefaultConfig.DEVICE, DefaultConfig.BACKEND)
        end
        
        y_pred, _ = model(X_val, ps, st_val)
        
        # Transfer to CPU for metrics calculation
        y_pred_cpu = y_pred |> cpu_device() |> vec
        y_val_cpu = y_val |> cpu_device() |> vec
        
        # Store raw predictions (probabilities) and labels
        append!(all_predictions, y_pred_cpu)
        append!(all_labels, y_val_cpu)
    end
    
    # BASELINE: Calculate accuracy with default threshold = 0.5
    println("\n📊 BASELINE (Threshold = 0.5)")
    println("="^60)
    baseline_metrics = calculate_metrics_at_threshold(all_labels, all_predictions, 0.5f0)
    println("   Accuracy:  $(round(baseline_metrics["accuracy"]*100, digits=2))%")
    println("   Precision: $(round(baseline_metrics["precision"]*100, digits=2))%")
    println("   Recall:    $(round(baseline_metrics["recall"]*100, digits=2))%")
    println("   F1-Score:  $(round(baseline_metrics["f1_score"]*100, digits=2))%")
    println("="^60)
    
    # THRESHOLD OPTIMIZATION
    best_threshold, best_metrics, all_results = optimize_threshold_with_cost(
        all_labels, 
        all_predictions; 
        threshold_range=0.3f0:0.01f0:0.7f0,
        optimization_metric="accuracy",
        fp_cost=1.0f0,
        fn_cost=1.0f0
    )
    
    println("\n✨ OPTIMAL THRESHOLD RESULTS")
    println("="^60)
    println("   Optimal Threshold: $(round(best_threshold, digits=3))")
    println("   Accuracy:  $(round(best_metrics["accuracy"]*100, digits=2))% (Δ = $(round((best_metrics["accuracy"] - baseline_metrics["accuracy"])*100, digits=2))%)")
    println("   Precision: $(round(best_metrics["precision"]*100, digits=2))%")
    println("   Recall:    $(round(best_metrics["recall"]*100, digits=2))%")
    println("   F1-Score:  $(round(best_metrics["f1_score"]*100, digits=2))%")
    println("="^60)
    
    # FALSE POSITIVE vs FALSE NEGATIVE ANALYSIS
    fp, fn, fp_rate, fn_rate = analyze_fp_vs_fn(best_metrics, all_labels, all_predictions)
    
    # PLOT CONFUSION MATRIX
    println("\n📊 Plotting confusion matrix...")
    plot_confusion_matrix(best_metrics, "confusion_matrix_optimal.png")
    
    println("📊 Plotting baseline confusion matrix...")
    plot_confusion_matrix(baseline_metrics, "confusion_matrix_baseline.png")
    
    # CALCULATE AUC-ROC
    println("\n📈 Calculating AUC-ROC...")
    auc, fpr, tpr = calculate_auc_roc(all_labels, all_predictions)
    println("   AUC-ROC: $(round(auc, digits=4))")
    
    # PLOT THRESHOLD ANALYSIS
    println("\n📊 Plotting threshold analysis...")
    plot_threshold_analysis(all_results, best_threshold, "threshold_analysis.png")
    
    # PLOT ROC CURVE
    println("\n📊 Plotting ROC curve...")
    plot_roc_curve(fpr, tpr, auc, "roc_curve.png")
    
    # ADDED: Calculate total evaluation time
    total_evaluation_time = time() - evaluation_start_time
    
    # FINAL SUMMARY
    println("\n" * "="^60)
    println("FINAL EVALUATION SUMMARY")
    println("="^60)
    println("Baseline Accuracy (threshold=0.5): $(round(baseline_metrics["accuracy"]*100, digits=2))%")
    println("Optimized Accuracy (threshold=$(round(best_threshold, digits=3))): $(round(best_metrics["accuracy"]*100, digits=2))%")
    println("Accuracy Gain: $(round((best_metrics["accuracy"] - baseline_metrics["accuracy"])*100, digits=2))%")
    println("")
    println("Error Analysis at Optimal Threshold:")
    
    
    println("   FP Rate: $(round(fp_rate*100, digits=2))% of total samples")
    println("   FN Rate: $(round(fn_rate*100, digits=2))% of total samples")
    
    
    
    println("")
    println("AUC-ROC Score: $(round(auc, digits=4))")
    println("="^60)
    
    # ADDED: Evaluation time summary
    println("\n" * "="^60)
    println("⏱️  EVALUATION TIME SUMMARY")
    println("="^60)
    
    println("   Total evaluation time: $(round(total_evaluation_time, digits=1)) seconds")
    println("   Time per sample: $(round(total_evaluation_time/length(val_loader.image_paths)*1000, digits=1)) ms")
    println("="^60)
    
    # Transfer to CPU for saving
    ps_cpu = ps |> cpu_device()
    st_cpu = st |> cpu_device()
    
    # CHECK PERFORMANCE LEVEL
    if best_metrics["accuracy"] >= 0.88f0
        println("\n🎯 EXCELLENT PERFORMANCE: Validation accuracy >= 88%!")
        println("   Saving best model with optimal threshold...")
        jldsave("model_best.jld2"; 
                model=model, ps=ps_cpu, st=st_cpu, 
                val_accuracy=best_metrics["accuracy"], 
                auc_roc=auc,
                optimal_threshold=best_threshold,
                baseline_accuracy=baseline_metrics["accuracy"],
                false_positives=fp,
                false_negatives=fn,
                fp_rate=fp_rate,
                fn_rate=fn_rate,
                backend=DefaultConfig.BACKEND,
                compute_mode=DefaultConfig.COMPUTE_MODE)
    elseif best_metrics["accuracy"] >= 0.85f0
        println("\n✅ SUCCESS! Validation accuracy >= 85%")
        println("Saving best model to model_best.jld2")
        jldsave("model_best.jld2"; 
                model=model, ps=ps_cpu, st=st_cpu, 
                val_accuracy=best_metrics["accuracy"], 
                auc_roc=auc,
                optimal_threshold=best_threshold,
                baseline_accuracy=baseline_metrics["accuracy"],
                false_positives=fp,
                false_negatives=fn,
                fp_rate=fp_rate,
                fn_rate=fn_rate,
                backend=DefaultConfig.BACKEND,
                compute_mode=DefaultConfig.COMPUTE_MODE)
    else
        println("\n⚠️  Validation accuracy: $(round(best_metrics["accuracy"]*100, digits=2))%")
        println("Model saved, but consider further tuning.")
        jldsave("model_best.jld2"; 
                model=model, ps=ps_cpu, st=st_cpu, 
                val_accuracy=best_metrics["accuracy"], 
                auc_roc=auc,
                optimal_threshold=best_threshold,
                baseline_accuracy=baseline_metrics["accuracy"],
                false_positives=fp,
                false_negatives=fn,
                fp_rate=fp_rate,
                fn_rate=fn_rate,
                backend=DefaultConfig.BACKEND,
                compute_mode=DefaultConfig.COMPUTE_MODE)
    end
    
    return baseline_metrics["accuracy"], best_metrics["accuracy"], best_threshold, auc, fp, fn
end

# RUN EVALUATION
println("Starting comprehensive model evaluation...")
baseline_acc, optimized_acc, optimal_threshold, auc, fp, fn = evaluate_model()

println("\n" * "="^60)
println("🎯 FINAL RESULTS")
println("="^60)
println("Baseline Accuracy (0.5):     $(round(baseline_acc*100, digits=2))%")
println("Optimized Accuracy ($(round(optimal_threshold, digits=3))): $(round(optimized_acc*100, digits=2))%")
println("Improvement:                  $(round((optimized_acc - baseline_acc)*100, digits=2))%")
println("")
println("Error Breakdown:")
println("  False Positives: $fp")
println("  False Negatives: $fn")
println("")
println("AUC-ROC Score:                $(round(auc, digits=4))")
println("="^60)