# src/models/sar_cnn_model.jl
using Lux, LuxCore, NNlib, CUDA, Random
using Statistics, LinearAlgebra  # ✨ Added these imports
include("../config/default_config.jl")
using .DefaultConfig

""" Great to use on the feature extraction logic that we'll build later 
# -------------------------
# 0️⃣ Gabor Kernel Initialization Helper
# -------------------------

Create a single Gabor-like kernel for orientation bias.
Parameters:
- size: kernel size (e.g., 7)
- σ: standard deviation (spatial envelope)
- θ: orientation in radians (0 = horizontal, π/2 = vertical)
- λ: wavelength (fixed, not swept)
- γ: aspect ratio (spatial frequency)

function gabor_kernel_2d(size::Int, σ::Float32, θ::Float32, λ::Float32, γ::Float32=1.0f0)
    kernel = zeros(Float32, size, size)
    center = (size + 1) / 2
    
    for x in 1:size
        for y in 1:size
            # Center coordinates
            x_c = x - center
            y_c = y - center
            
            # Rotation
            x_θ = x_c * cos(θ) + y_c * sin(θ)
            y_θ = -x_c * sin(θ) + y_c * cos(θ)
            
            # Gabor function
            gaussian = exp(-(x_θ^2 + γ^2 * y_θ^2) / (2 * σ^2))
            sinusoid = cos(2π * x_θ / λ)
            
            kernel[x, y] = gaussian * sinusoid
        end
    end
    
    # Normalize to zero mean
    kernel = kernel .- mean(kernel)
    # Normalize to unit norm
    kernel = kernel ./ (norm(kernel) + 1f-8)
    
    return kernel
end
"""

"""
Initialize a subset of conv filters with Gabor orientation bias.
- Only affects `n_gabor` filters out of total `out_channels`
- Uses multiple orientations (horizontal, diagonal, vertical)
- Remaining filters: standard random initialization

function initialize_gabor_biased_conv(rng, kernel_size::Tuple{Int,Int}, in_ch::Int, out_ch::Int, n_gabor::Int)
    # Standard random initialization for all filters
    weights = randn(rng, Float32, kernel_size[1], kernel_size[2], in_ch, out_ch) .* sqrt(2.0f0 / (kernel_size[1] * kernel_size[2] * in_ch))
    
    # Gabor parameters (fixed, single scale)
    σ = Float32(kernel_size[1]) / 6.0f0  # Adaptive to kernel size
    λ = Float32(kernel_size[1]) / 2.0f0  # Single wavelength
    
    # Orientations: horizontal, 45°, vertical, 135°
    orientations = Float32[0, π/4, π/2, 3π/4]
    
    # Replace first `n_gabor` filters with Gabor-biased versions
    for i in 1:min(n_gabor, out_ch)
        θ = orientations[((i-1) % length(orientations)) + 1]  # Cycle through orientations
        gabor = gabor_kernel_2d(kernel_size[1], σ, θ, λ)
        
        # Assign to first channel (grayscale input has 1 channel)
        weights[:, :, 1, i] = gabor
    end
    
    return weights
end
"""
# -------------------------
# 1️⃣ Block Definitions
# -------------------------
# Conv Block with optional pooling
struct ConvBlock <: LuxCore.AbstractLuxContainerLayer{(:conv, :bn, :pool)}
    conv::Conv
    bn::BatchNorm
    pool::Union{MaxPool, Nothing}
end

function ConvBlock(kernel, in_ch, out_ch; stride=(1,1), pad=0, pool_size=nothing)
    conv = Conv(kernel, in_ch=>out_ch; stride=stride, pad=pad)
    bn = BatchNorm(out_ch)
    pool = isnothing(pool_size) ? nothing : MaxPool(pool_size)
    return ConvBlock(conv, bn, pool)
end

# Apply function for ConvBlock
function (cb::ConvBlock)(x, ps, st)
    y, st_conv = cb.conv(x, ps.conv, st.conv)
    y, st_bn   = cb.bn(y, ps.bn, st.bn)
    y = relu.(y)
    if cb.pool !== nothing
        y, st_pool = cb.pool(y, ps.pool, st.pool)
    else
        st_pool = NamedTuple()
    end
    return y, (conv=st_conv, bn=st_bn, pool=st_pool)
end

# -------------------------
# 2️⃣ Aggregation Block
# -------------------------
struct AggBlock <: LuxCore.AbstractLuxLayer
end

function (ab::AggBlock)(x, ps, st)
    # Global average pooling: (H,W,C,B) -> (1,1,C,B) -> (C,B)
    y = sum(x, dims=(1,2)) ./ (size(x,1) * size(x,2))
    y = reshape(y, :, size(y, 4))
    return y, NamedTuple()
end

# -------------------------
# 3️⃣ Classifier Block
# -------------------------
struct ClassifierBlock <: LuxCore.AbstractLuxContainerLayer{(:dense1, :dense2)}
    dense1::Dense
    dropout_rate::Float32
    dense2::Dense
end

function ClassifierBlock(in_features::Int; dropout_rate=0.3)
    dense1 = Dense(in_features, 32)
    dense2 = Dense(32, 1)
    return ClassifierBlock(dense1, dropout_rate, dense2)
end

function (cb::ClassifierBlock)(x, ps, st)
    y, st1 = cb.dense1(x, ps.dense1, st.dense1)
    # Simple dropout at train time
    y = y .* (1 .- cb.dropout_rate)
    y, st2 = cb.dense2(y, ps.dense2, st.dense2)
    y = 1 ./ (1 .+ exp.(-y))  # sigmoid for binary classification
    return y, (dense1=st1, dense2=st2)
end

# -------------------------
# 4️⃣ Full SAR CNN Model (WITH GABOR BIAS IN BLOCK1 ONLY)
# -------------------------
struct SAR_CNN <: LuxCore.AbstractLuxContainerLayer{(:block1, :block2, :block3, :agg, :classifier)}
    block1::ConvBlock
    block2::ConvBlock
    block3::ConvBlock
    agg::AggBlock
    classifier::ClassifierBlock
end

function SAR_CNN()
    block1 = ConvBlock((7,7), 1, 16; stride=(2,2), pad=3)
    block2 = ConvBlock((5,5), 16, 32; stride=(1,1), pad=2, pool_size=(2,2))
    block3 = ConvBlock((3,3), 32, 64; stride=(1,1), pad=1, pool_size=(2,2))
    agg    = AggBlock()
    classifier = ClassifierBlock(64)
    return SAR_CNN(block1, block2, block3, agg, classifier)
end

# ✨ CUSTOM SETUP FUNCTION WITH GABOR INITIALIZATION
function setup_with_gabor_bias(rng, model::SAR_CNN; n_gabor_filters=4)
    # Standard Lux setup
    ps, st = LuxCore.setup(rng, model)
    
    # ✨ REPLACE Block1 conv weights with Gabor-biased initialization
    # Only affects first `n_gabor_filters` out of 16 total filters
    gabor_weights = initialize_gabor_biased_conv(rng, (7, 7), 1, 16, n_gabor_filters)
    
    # Update parameters (Block1 conv weights only)
    ps = (
        block1 = (
            conv = (weight = gabor_weights, bias = ps.block1.conv.bias),
            bn = ps.block1.bn,
            pool = ps.block1.pool
        ),
        block2 = ps.block2,  # Unchanged
        block3 = ps.block3,  # Unchanged
        agg = ps.agg,
        classifier = ps.classifier
    )
    
   # println("✨ Gabor bias applied to Block1: $n_gabor_filters out of 16 filters")
    println("   Orientations: 0°, 45°, 90°, 135° (horizontal, diagonal, vertical)")
    println("   Remaining $(16 - n_gabor_filters) filters: standard random initialization")
    println("   ALL filters remain TRAINABLE (not frozen)")
    
    return ps, st
end

# Apply function for full SAR_CNN
function (m::SAR_CNN)(x, ps, st)
    y, st1 = m.block1(x, ps.block1, st.block1)
    y, st2 = m.block2(y, ps.block2, st.block2)
    y, st3 = m.block3(y, ps.block3, st.block3)
    y, st4 = m.agg(y, ps.agg, st.agg)
    y, st5 = m.classifier(y, ps.classifier, st.classifier)
    return y, (block1=st1, block2=st2, block3=st3, agg=st4, classifier=st5)
end

# -------------------------
# Optional: Instantiate for inspection
# -------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    rng = Random.default_rng()
    Random.seed!(rng, DefaultConfig.RNG_SEED)
    model = SAR_CNN()
    ps, st = setup_with_gabor_bias(rng, model; n_gabor_filters=4)
    println("SAR CNN model ready ✅")
    println("Model parameters: ", LuxCore.parameterlength(ps))
    println(model)
end
