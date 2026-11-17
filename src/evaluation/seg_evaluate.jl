# ============================================================================
# seg_evaluate.jl — Evaluation for Segmentation Model
#
# Computes Dice, IoU, Precision, Recall, F1, Accuracy on the test set.
# Saves a JSON report with all metrics, config, and timestamp.
#
# Spec: specs/evaluation.yaml (EV-001 through EV-006)
# ============================================================================

module SegEvaluation

using Statistics, Dates, JSON3

export SegMetrics, evaluate_segmentation, compute_metrics, binarize,
       save_report, dice_iou_consistent

# ============================================================================
# EV-006: Binarize predictions at threshold
# ============================================================================
"""
    binarize(pred, threshold) -> Array{Float32}

Values >= threshold become 1.0, values < threshold become 0.0.
"""
function binarize(pred::AbstractArray{<:Real}, threshold::Float32=0.5f0)
    return Float32.(pred .>= threshold)
end

# ============================================================================
# Metric computation
# ============================================================================
struct SegMetrics
    dice::Float32
    iou::Float32
    precision::Float32
    recall::Float32
    f1::Float32
    accuracy::Float32
end

"""
    compute_metrics(pred_binary, target) -> SegMetrics

Compute all segmentation metrics from binarized predictions and targets.
Both inputs should be binary (0.0 or 1.0).
"""
function compute_metrics(pred::AbstractArray{Float32}, target::AbstractArray{Float32})
    tp = sum(pred .* target)
    fp = sum(pred .* (1.0f0 .- target))
    fn = sum((1.0f0 .- pred) .* target)
    tn = sum((1.0f0 .- pred) .* (1.0f0 .- target))

    # Dice (EV-001)
    dice_denom = 2.0f0 * tp + fp + fn
    dice = dice_denom == 0.0f0 ? 1.0f0 : 2.0f0 * tp / dice_denom

    # IoU / Jaccard (EV-002)
    iou_denom = tp + fp + fn
    iou = iou_denom == 0.0f0 ? 1.0f0 : tp / iou_denom

    # Precision
    prec_denom = tp + fp
    prec = prec_denom == 0.0f0 ? 1.0f0 : tp / prec_denom

    # Recall
    recall_denom = tp + fn
    recall = recall_denom == 0.0f0 ? 1.0f0 : tp / recall_denom

    # F1 (harmonic mean of precision and recall)
    f1_denom = prec + recall
    f1 = f1_denom == 0.0f0 ? 0.0f0 : 2.0f0 * prec * recall / f1_denom

    # Accuracy
    total = tp + fp + fn + tn
    accuracy = total == 0.0f0 ? 1.0f0 : (tp + tn) / total

    return SegMetrics(
        Float32(dice), Float32(iou), Float32(prec),
        Float32(recall), Float32(f1), Float32(accuracy)
    )
end

# ============================================================================
# EV-001, EV-002: Full evaluation over a data loader
# ============================================================================
"""
    evaluate_segmentation(model, ps, st, test_loader; threshold=0.5f0)
        -> (metrics::SegMetrics, all_preds, all_targets)

Run the model on all test batches, aggregate metrics.
"""
function evaluate_segmentation(model, ps, st, test_loader;
                                threshold::Float32=0.5f0)
    st_eval = if hasproperty(st, :training)
        # Simple training flag
        merge(st, (training=Val(false),))
    else
        # Lux model — use testmode if available
        try
            # Dynamic dispatch to handle Lux.testmode
            Base.invokelatest(_testmode, st)
        catch
            st
        end
    end

    all_preds = Float32[]
    all_targets = Float32[]

    for (X, Y) in test_loader
        pred, _, _ = model(X, ps, st_eval)

        # Flatten and collect
        append!(all_preds, vec(Array(pred)))
        append!(all_targets, vec(Array(Y)))
    end

    # Binarize predictions (EV-006)
    pred_binary = binarize(all_preds, threshold)
    target_binary = Float32.(all_targets .>= 0.5f0)

    # Compute metrics
    metrics = compute_metrics(pred_binary, target_binary)

    return metrics, all_preds, all_targets
end

# Helper to avoid Lux dependency at module level
function _testmode(st)
    # Will be resolved at runtime if Lux is loaded
    return Lux.testmode(st)
end

# ============================================================================
# Dice-IoU consistency check (invariant)
# ============================================================================
"""
    dice_iou_consistent(dice, iou; atol=1e-4) -> Bool

Verify Dice = 2*IoU / (1+IoU).
"""
function dice_iou_consistent(dice::Float32, iou::Float32; atol=1.0f-4)
    expected_dice = 2.0f0 * iou / (1.0f0 + iou)
    return abs(dice - expected_dice) < atol
end

# ============================================================================
# EV-005: Save JSON report
# ============================================================================
"""
    save_report(path, metrics, threshold, config_info)

Save evaluation metrics as a JSON file with timestamp.
"""
function save_report(path::String, metrics::SegMetrics;
                     threshold::Float32=0.5f0,
                     config_info::Dict{String,Any}=Dict{String,Any}())
    report = Dict{String,Any}(
        "metrics" => Dict{String,Any}(
            "dice" => Float64(metrics.dice),
            "iou" => Float64(metrics.iou),
            "precision" => Float64(metrics.precision),
            "recall" => Float64(metrics.recall),
            "f1" => Float64(metrics.f1),
            "accuracy" => Float64(metrics.accuracy)
        ),
        "threshold" => Float64(threshold),
        "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "config" => config_info
    )

    open(path, "w") do f
        JSON3.pretty(f, report)
    end

    return report
end

end # module SegEvaluation
