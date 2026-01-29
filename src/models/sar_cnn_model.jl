# src/models/sar_cnn_model.jl
# EXACT REPLICA of iws_model_without_gabor.jl (GPU-compatible, no architectural changes)
using Lux, LuxCore, NNlib, Random
using Statistics, LinearAlgebra
include("../config/default_config.jl")
using .DefaultConfig

# -------------------------
# 1️⃣ Block Definitions
# -------------------------
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
    y = y .* (1 .- cb.dropout_rate)
    y, st2 = cb.dense2(y, ps.dense2, st.dense2)
    y = 1 ./ (1 .+ exp.(-y))
    return y, (dense1=st1, dense2=st2)
end

# -------------------------
# 4️⃣ Full SAR CNN Model
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
    ps, st = LuxCore.setup(rng, model)  # simple setup without Gabor
    println("SAR CNN model ready ✅")
    println("Model parameters: ", LuxCore.parameterlength(ps))
    println(model)
end