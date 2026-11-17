# ============================================================================
# physics_losses.jl — Physics-Informed Loss Functions for IW Segmentation
#
# Combines standard segmentation losses (Dice + BCE) with physics-informed
# regularization based on ocean internal wave structural priors:
#
#   1. Dice loss + BCE loss (base segmentation)
#   2. KdV soliton regularization (quasi-periodic wave crest structure)
#   3. Gradient anisotropy loss (directional wave features)
#
# All losses are Zygote-differentiable and operate on Float32 arrays.
#
# Spec: specs/physics_losses.yaml (PL-001 through PL-007)
# ============================================================================

module PhysicsLosses

using Statistics

export dice_loss, bce_loss, kdv_regularization_loss, gradient_structure_loss,
       combined_loss, LossWeights

# ============================================================================
# Loss weight configuration
# ============================================================================
struct LossWeights
    lambda_dice::Float32
    lambda_bce::Float32
    lambda_kdv::Float32
    lambda_grad::Float32
end

LossWeights(; lambda_dice=1.0f0, lambda_bce=1.0f0,
              lambda_kdv=0.0f0, lambda_grad=0.0f0) =
    LossWeights(lambda_dice, lambda_bce, lambda_kdv, lambda_grad)

# ============================================================================
# PL-001: Base segmentation losses — Dice + BCE
# ============================================================================

"""
    dice_loss(pred, target; smooth=1.0f0)

Soft Dice loss for binary segmentation.
Returns 1 - Dice coefficient. Range [0, 1].

When both pred and target are all zeros, returns 0.0 (PL-007).
"""
function dice_loss(pred::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:Real};
                   smooth::Float32=1.0f0)
    # Flatten spatial dims, keep batch
    p = reshape(pred, :, size(pred, 4))
    t = reshape(target, :, size(target, 4))

    intersection = sum(p .* t; dims=1)
    union = sum(p; dims=1) .+ sum(t; dims=1)

    dice = mean((2.0f0 .* intersection .+ smooth) ./ (union .+ smooth))
    return 1.0f0 - dice
end

"""
    bce_loss(pred, target; eps=1.0f-7)

Binary cross-entropy loss with numerical stability.
"""
function bce_loss(pred::AbstractArray{<:AbstractFloat}, target::AbstractArray{<:Real};
                  eps::Float32=1.0f-7)
    p = clamp.(pred, eps, 1.0f0 - eps)
    return -mean(target .* log.(p) .+ (1.0f0 .- target) .* log.(1.0f0 .- p))
end

# ============================================================================
# PL-002: KdV Soliton Regularization Loss
#
# Ocean internal waves are described by the Korteweg-de Vries (KdV) equation:
#   ∂η/∂t + c₀∂η/∂x + α·η·∂η/∂x + β·∂³η/∂x³ = 0
#
# The soliton solution: η(x) = A · sech²((x - ct) / L)
#
# Key structural properties of IW in SAR imagery:
#   1. Wave crests are quasi-linear (elongated in one direction)
#   2. Crests have consistent spacing (quasi-periodic)
#   3. Intensity profile across crests resembles sech² (smooth, peaked)
#
# Rather than solving KdV, we enforce these structural properties on the
# predicted segmentation mask as soft constraints:
#
#   L_kdv = L_periodicity + L_crest_smoothness
#
#   - L_periodicity: penalizes irregular spacing in the autocorrelation
#   - L_crest_smoothness: penalizes sharp/jagged edges (sech² is smooth)
#
# This is differentiable and doesn't require bathymetry or stratification data.
# ============================================================================

"""
    kdv_regularization_loss(pred_mask)

KdV-inspired regularization for soliton-like spatial structure.
Penalizes masks that lack smooth, quasi-periodic wave crest structure.

Operates on predicted mask (H,W,1,B) or (H,W,C,B).
Returns scalar >= 0. Near zero for ideal soliton-like patterns.
"""
function kdv_regularization_loss(pred::AbstractArray{Float32})
    l_smooth = crest_smoothness_loss(pred)
    l_period = periodicity_loss(pred)
    return l_smooth + l_period
end

"""
    crest_smoothness_loss(pred)

Penalizes high-frequency spatial variations in the predicted mask.
A sech² profile is smooth — jagged or noisy masks get penalized.

Uses the Laplacian (second-order spatial derivative) as a smoothness measure.
For a perfect sech² profile, the Laplacian is smooth and bounded.
Random noise has a large Laplacian.
"""
function crest_smoothness_loss(pred::AbstractArray{Float32})
    # Spatial second derivatives (discrete Laplacian)
    # ∂²/∂x²: pred[i+1] - 2*pred[i] + pred[i-1]
    laplacian_x = pred[3:end, :, :, :] .- 2.0f0 .* pred[2:end-1, :, :, :] .+ pred[1:end-2, :, :, :]
    laplacian_y = pred[:, 3:end, :, :] .- 2.0f0 .* pred[:, 2:end-1, :, :] .+ pred[:, 1:end-2, :, :]

    # Mean squared Laplacian — lower is smoother
    return mean(laplacian_x .^ 2) + mean(laplacian_y .^ 2)
end

"""
    periodicity_loss(pred)

Encourages quasi-periodic structure by penalizing variance in row/column
intensity profiles. For wave crests with consistent spacing, the row-wise
mean intensity profile should have clear periodic peaks.

This avoids FFT (not Zygote-friendly) and uses spatial statistics instead.
"""
function periodicity_loss(pred::AbstractArray{Float32})
    # Average across channel and batch dims -> (H, W) profile
    # Then compute row-wise and column-wise mean intensity
    p = mean(pred; dims=(3, 4))  # (H, W, 1, 1)
    p = dropdims(p; dims=(3, 4))  # (H, W)

    row_profile = mean(p; dims=2)  # (H, 1) — mean intensity per row
    col_profile = mean(p; dims=1)  # (1, W) — mean intensity per column

    # For a periodic signal, the first derivative should oscillate regularly.
    # Irregular spacing → high variance in the magnitude of first derivatives.
    # We penalize the variance of the absolute gradient of profiles.
    d_row = row_profile[2:end, :] .- row_profile[1:end-1, :]
    d_col = col_profile[:, 2:end] .- col_profile[:, 1:end-1]

    # Variance of absolute gradients — lower means more regular spacing
    var_row = var(abs.(d_row))
    var_col = var(abs.(d_col))

    # Return the minimum — we care about periodicity in at least one direction
    # (waves have one dominant orientation)
    return min(var_row, var_col)
end

# ============================================================================
# PL-003: Gradient Structure Loss (Anisotropy)
#
# Internal wave SAR signatures show strong gradients perpendicular to wave
# crests and weak gradients parallel to crests. This loss penalizes
# isotropic gradient fields — encouraging directional (stripe-like) structure.
# ============================================================================

"""
    gradient_structure_loss(feat)

Penalizes isotropic gradients in feature maps. Returns ratio of min/max
directional gradient energy. Near 0.0 = anisotropic (good for waves),
near 1.0 = isotropic (bad, blobby).

Operates on 4D arrays (H,W,C,B).
"""
function gradient_structure_loss(feat::AbstractArray{Float32})
    dx = feat[2:end, :, :, :] .- feat[1:end-1, :, :, :]
    dy = feat[:, 2:end, :, :] .- feat[:, 1:end-1, :, :]
    E_x = mean(dx .^ 2)
    E_y = mean(dy .^ 2)
    return min(E_x, E_y) / (max(E_x, E_y) + 1.0f-7)
end

# ============================================================================
# PL-005: Combined loss with configurable weights
# ============================================================================

"""
    combined_loss(pred, target, features, weights) -> (total, components)

Compute weighted sum of all loss components.
Returns total scalar loss and a NamedTuple of individual components.
"""
function combined_loss(pred::AbstractArray{<:AbstractFloat},
                       target::AbstractArray{<:Real},
                       features::Union{NamedTuple, Nothing},
                       weights::LossWeights)
    # Base losses (always computed)
    l_dice = dice_loss(pred, target)
    l_bce  = bce_loss(pred, target)

    # Physics losses (computed only if weight > 0, but always differentiable)
    l_kdv  = weights.lambda_kdv > 0 ? kdv_regularization_loss(pred) : 0.0f0
    l_grad = if weights.lambda_grad > 0 && features !== nothing
        # Use the deepest encoder features for gradient structure
        feat = _get_deepest_feature(features)
        gradient_structure_loss(feat)
    else
        0.0f0
    end

    total = weights.lambda_dice * l_dice +
            weights.lambda_bce  * l_bce +
            weights.lambda_kdv  * l_kdv +
            weights.lambda_grad * l_grad

    components = (dice=l_dice, bce=l_bce, kdv=l_kdv, grad=l_grad, total=total)

    return total, components
end

"""Extract the deepest feature map from intermediates for gradient loss."""
function _get_deepest_feature(features::NamedTuple)
    # Prefer enc4 (deepest encoder), fall back through the hierarchy
    if hasproperty(features, :enc4)
        return features.enc4
    elseif hasproperty(features, :enc3)
        return features.enc3
    elseif hasproperty(features, :bottleneck)
        return features.bottleneck
    else
        # Use whatever is available
        return features[first(keys(features))]
    end
end

end # module PhysicsLosses
