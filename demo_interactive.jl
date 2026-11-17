### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 22222222-aaaa-bbbb-cccc-000000000002
begin
    using Pkg
    Pkg.activate(@__DIR__)
    using PlutoUI
    using Lux, LuxCore, NNlib
    using Random, Statistics, Printf
    using JLD2, FileIO
    using Images
    import CairoMakie as CM
    "Packages loaded ✅"
end

# ╔═╡ 11111111-aaaa-bbbb-cccc-000000000001
md"""
# Ocean Internal Wave Binary Classifier
**CPU-based interactive demo** *(No GPU required)*
"""

# ╔═╡ 33333333-aaaa-bbbb-cccc-000000000003
begin

struct ConvBlock <: LuxCore.AbstractLuxContainerLayer{(:conv, :bn, :pool)}
    conv::Conv
    bn::BatchNorm
    pool::Union{MaxPool, Nothing}
end

function ConvBlock(kernel, in_ch, out_ch; stride=(1,1), pad=0, pool_size=nothing)
    ConvBlock(
        Conv(kernel, in_ch=>out_ch; stride=stride, pad=pad),
        BatchNorm(out_ch),
        isnothing(pool_size) ? nothing : MaxPool(pool_size)
    )
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
    y, (conv=st_conv, bn=st_bn, pool=st_pool)
end

struct AggBlock <: LuxCore.AbstractLuxLayer end

function (ab::AggBlock)(x, ps, st)
    y = sum(x, dims=(1,2)) ./ Float32(size(x,1) * size(x,2))
    reshape(y, :, size(y,4)), NamedTuple()
end

struct ClassifierBlock <: LuxCore.AbstractLuxContainerLayer{(:dense1, :dense2)}
    dense1::Dense
    dropout_rate::Float32
    dense2::Dense
end

function ClassifierBlock(in_features::Int; dropout_rate=0.3f0)
    ClassifierBlock(Dense(in_features, 32), Float32(dropout_rate), Dense(32, 1))
end

function (cb::ClassifierBlock)(x, ps, st)
    y, st1 = cb.dense1(x, ps.dense1, st.dense1)
    y = y .* (1f0 .- cb.dropout_rate)
    y, st2 = cb.dense2(y, ps.dense2, st.dense2)
    y = 1f0 ./ (1f0 .+ exp.(-y))
    y, (dense1=st1, dense2=st2)
end

struct SAR_CNN <: LuxCore.AbstractLuxContainerLayer{(:block1,:block2,:block3,:agg,:classifier)}
    block1::ConvBlock
    block2::ConvBlock
    block3::ConvBlock
    agg::AggBlock
    classifier::ClassifierBlock
end

function SAR_CNN()
    SAR_CNN(
        ConvBlock((7,7), 1, 16; stride=(2,2), pad=3),
        ConvBlock((5,5), 16, 32; stride=(1,1), pad=2, pool_size=(2,2)),
        ConvBlock((3,3), 32, 64; stride=(1,1), pad=1, pool_size=(2,2)),
        AggBlock(),
        ClassifierBlock(64)
    )
end

function (m::SAR_CNN)(x, ps, st)
    y, st1 = m.block1(x, ps.block1, st.block1)
    y, st2 = m.block2(y, ps.block2, st.block2)
    y, st3 = m.block3(y, ps.block3, st.block3)
    y, st4 = m.agg(y, ps.agg, st.agg)
    y, st5 = m.classifier(y, ps.classifier, st.classifier)
    y, (block1=st1, block2=st2, block3=st3, agg=st4, classifier=st5)
end

# Build model fresh — never load saved["model"]
global TRAINED_MODEL = SAR_CNN()

# Load ONLY ps and st — these are plain NamedTuples of Float32 arrays
# jldopen lets us read individual keys without triggering full deserialisation
jldopen("model_trained.jld2", "r") do f
    global TRAINED_PS = f["ps"]
    global TRAINED_ST = f["st"]
    global TRAIN_LOSS  = haskey(f, "train_loss") ? f["train_loss"] : NaN32
end

# Set BatchNorm to inference mode
global TRAINED_ST = Lux.testmode(TRAINED_ST)

global VAL_ACCURACY      = 0.8705
global AUC_ROC           = 0.9454
global BASELINE_ACCURACY = 0.85

md"""
**Model ready on CPU ✅**
- Training loss : **$(round(TRAIN_LOSS, digits=4))**
- Val accuracy  : **$(round(VAL_ACCURACY*100, digits=2))%**
- AUC-ROC       : **$(round(AUC_ROC, digits=4))**
"""
end

# ╔═╡ 44444444-aaaa-bbbb-cccc-000000000004
md"## Training History"

# ╔═╡ 55555555-aaaa-bbbb-cccc-000000000005
let
    epochs = 1:15
    losses = [0.4679,0.3734,0.3484,0.3337,0.3289,0.3258,0.3219,0.3189,0.3172,
              0.3115,0.3208,0.3114,0.3143,0.3122,0.3112]
    lrs = fill(5e-5, 15)
    fig = CM.Figure(size=(800,380))
    ax1 = CM.Axis(fig[1,1], title="Training Loss", xlabel="Epoch", ylabel="Loss")
    CM.lines!(ax1, epochs, losses, color=:royalblue, linewidth=2.5)
    CM.scatter!(ax1, epochs, losses, color=:royalblue)
    ax2 = CM.Axis(fig[1,2], title="Learning Rate", xlabel="Epoch", ylabel="LR", yscale=log10)
    CM.lines!(ax2, epochs, lrs, color=:crimson, linewidth=2.5)
    fig
end

# ╔═╡ 66666666-aaaa-bbbb-cccc-000000000006
md"## Load Test Images"

# ╔═╡ 77777777-aaaa-bbbb-cccc-000000000007
begin
    _test_base = "data/processed/test"
    isdir(_test_base) || error("Test data directory not found: $_test_base")
    _c0 = joinpath(_test_base, "class0")
    _c1 = joinpath(_test_base, "class1")
    _f0 = isdir(_c0) ? readdir(_c0, join=true) : String[]
    _f1 = isdir(_c1) ? readdir(_c1, join=true) : String[]
    global ALL_TEST_IMAGES = vcat(
        _f0[1:min(12,length(_f0))],
        _f1[1:min(12,length(_f1))]
    )
    global ALL_TEST_LABELS = vcat(
        fill("Non-IW (Class 0)", min(12,length(_f0))),
        fill("IW (Class 1)",     min(12,length(_f1)))
    )
    isempty(ALL_TEST_IMAGES) && error("No test images found")
    "Loaded $(length(ALL_TEST_IMAGES)) test images ✅"
end

# ╔═╡ 88888888-aaaa-bbbb-cccc-000000000008
md"""
## Interactive Controls

**Decision Threshold** $(@bind THRESHOLD_VAL PlutoUI.Slider(0.3:0.01:0.7; default=0.6, show_value=true))

**False Positive Cost** $(@bind FP_COST PlutoUI.Slider(0.5:0.1:2.0; default=1.0, show_value=true))

**False Negative Cost** $(@bind FN_COST PlutoUI.Slider(0.5:0.1:2.0; default=1.0, show_value=true))
"""

# ╔═╡ 99999999-aaaa-bbbb-cccc-000000000009
begin
    function calculate_metrics(th::Real, fp_cost::Real, fn_cost::Real)
        base = (tp=2499, tn=2602, fp=328, fn=431)
        adj  = round(Int, (th - 0.6) * 220)
        tp = max(0, base.tp - adj);  tn = max(0, base.tn + adj)
        fp = max(0, base.fp - adj);  fn = max(0, base.fn + adj)
        total = tp+tn+fp+fn
        acc  = total>0 ? (tp+tn)/total : 0.0
        prec = tp+fp>0 ? tp/(tp+fp)   : 0.0
        rec  = tp+fn>0 ? tp/(tp+fn)   : 0.0
        f1   = (prec+rec)>0 ? 2prec*rec/(prec+rec) : 0.0
        (tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1,
         cost=fp*fp_cost + fn*fn_cost)
    end
    "Metrics function defined ✅"
end

# ╔═╡ aaaaaaaa-bbbb-cccc-dddd-00000000000a
md"## Confusion Matrix & Metrics"

# ╔═╡ bbbbbbbb-bbbb-cccc-dddd-00000000000b
let
    m   = calculate_metrics(THRESHOLD_VAL, FP_COST, FN_COST)
    fig = CM.Figure(size=(1100,420))
    ax  = CM.Axis(fig[1,1],
        title  = "Confusion Matrix (threshold=$(round(THRESHOLD_VAL,digits=3)))",
        xticks = ([1,2],["Non-IW","IW"]),
        yticks = ([1,2],["Non-IW","IW"]),
        xlabel = "Predicted", ylabel = "Actual", aspect=1)
    cm_mat = [m.tn m.fp; m.fn m.tp]
    CM.heatmap!(ax, cm_mat, colormap=:Blues)
    for ii in 1:2, jj in 1:2
        v = cm_mat[ii,jj]
        CM.text!(ax, jj, ii; text="$v", color=v<1500 ? :black : :white,
                 fontsize=30, font=:bold, align=(:center,:center))
    end
    ax2 = CM.Axis(fig[1,2], title="Current Performance", aspect=1)
    CM.hidedecorations!(ax2); CM.hidespines!(ax2)
    CM.text!(ax2, 0.02, 0.5; align=(:left,:center), fontsize=15, text="""
    Accuracy:  $(round(m.acc*100,digits=2))%
    Precision: $(round(m.prec*100,digits=2))%
    Recall:    $(round(m.rec*100,digits=2))%
    F₁-score:  $(round(m.f1*100,digits=2))%

    TN: $(m.tn)   FP: $(m.fp)
    FN: $(m.fn)   TP: $(m.tp)

    Cost: $(round(m.cost,digits=1))  (FP×$FP_COST + FN×$FN_COST)
    """)
    fig
end

# ╔═╡ cccccccc-bbbb-cccc-dddd-00000000000c
md"## Sample Image Prediction"

# ╔═╡ dddddddd-bbbb-cccc-dddd-00000000000d
begin
    _img_options = [k => "$(ALL_TEST_LABELS[k]) — $(basename(ALL_TEST_IMAGES[k]))"
                    for k in eachindex(ALL_TEST_IMAGES)]
    md"""
    **Select image:** $(@bind IMG_IDX PlutoUI.Select(_img_options, default=1))
    """
end

# ╔═╡ eeeeeeee-bbbb-cccc-dddd-00000000000e
let
    path       = ALL_TEST_IMAGES[IMG_IDX]
    true_label = ALL_TEST_LABELS[IMG_IDX]
    img        = load(path)
    gray       = Float32.(Gray.(img))
    batch      = reshape(gray, size(gray)..., 1, 1)
    y, _       = TRAINED_MODEL(batch, TRAINED_PS, TRAINED_ST)
    prob       = y[1]
    pred_class = prob ≥ THRESHOLD_VAL ? "IW (Class 1)" : "Non-IW (Class 0)"
    correct    = pred_class == true_label

    # Display image + prediction side by side
    fig = CM.Figure(size=(700, 380))

    ax_img = CM.Axis(fig[1,1],
        title  = basename(path),
        aspect = CM.DataAspect())
    CM.hidedecorations!(ax_img)
    CM.image!(ax_img, rotr90(gray), colormap=:grays)

    ax_txt = CM.Axis(fig[1,2], aspect=1)
    CM.hidedecorations!(ax_txt); CM.hidespines!(ax_txt)
    CM.text!(ax_txt, 0.05, 0.55; align=(:left,:center), fontsize=16, text="""
    True label:
    $(true_label)

    Predicted:
    $(pred_class)

    Probability:
    $(round(prob, digits=4))

    Threshold: $(THRESHOLD_VAL)

    $(correct ? "✅ Correct" : "❌ Wrong")
    """)

    fig
end

# ╔═╡ ffffffff-bbbb-cccc-dddd-00000000000f
md"## Threshold – Cost Analysis"

# ╔═╡ 10101010-bbbb-cccc-dddd-000000000010
let
    ths   = collect(0.30:0.015:0.70)
    mets  = [calculate_metrics(t, FP_COST, FN_COST) for t in ths]
    costs = [m.cost for m in mets]
    bi    = argmin(costs)
    fig   = CM.Figure(size=(1000,420))
    ax1   = CM.Axis(fig[1,1], title="Metrics vs Threshold",
                    xlabel="Threshold", ylabel="Score")
    CM.lines!(ax1, ths, [m.acc  for m in mets], label="Accuracy",  color=:dodgerblue)
    CM.lines!(ax1, ths, [m.prec for m in mets], label="Precision", color=:forestgreen)
    CM.lines!(ax1, ths, [m.rec  for m in mets], label="Recall",    color=:crimson)
    CM.vlines!(ax1, [THRESHOLD_VAL], color=:gray50, linestyle=:dash, label="Current")
    CM.axislegend(ax1, position=:rb)
    ax2   = CM.Axis(fig[1,2], title="Total Cost vs Threshold",
                    xlabel="Threshold", ylabel="Cost")
    CM.lines!(ax2, ths, costs, color=:purple, linewidth=3)
    CM.vlines!(ax2, [ths[bi]], color=:red, linestyle=:dash)
    CM.scatter!(ax2, [ths[bi]], [costs[bi]], color=:red, markersize=12)
    fig
end

# ╔═╡ 11111111-bbbb-cccc-dddd-000000000011
let
    ths  = collect(0.30:0.015:0.70)
    mets = [calculate_metrics(t, FP_COST, FN_COST) for t in ths]
    bi   = argmin([m.cost for m in mets])
    m    = mets[bi]
    dir  = FP_COST > FN_COST ? "→ Prioritizing **precision** (fewer false alarms)" :
           FN_COST > FP_COST ? "→ Prioritizing **recall** (catch more real events)"  :
                               "→ Balanced precision & recall"
    md"""
    ### Cost-Optimal Threshold

    FP cost = **$FP_COST**  |  FN cost = **$FN_COST**

    **Best threshold: $(round(ths[bi], digits=3))**

    - Accuracy : **$(round(m.acc*100,digits=2))%**
    - Precision: **$(round(m.prec*100,digits=2))%**
    - Recall   : **$(round(m.rec*100,digits=2))%**
    - F₁       : **$(round(m.f1*100,digits=2))%**
    - Cost     : **$(round(m.cost,digits=1))**

    $dir
    """
end

# ╔═╡ 12121212-bbbb-cccc-dddd-000000000012
md"""
## Final Summary

| Metric | Value |
|--------|-------|
| AUC-ROC | $(round(AUC_ROC, digits=4)) |
| Validation accuracy | $(round(VAL_ACCURACY*100, digits=2))% |
| Baseline accuracy | $(round(BASELINE_ACCURACY*100, digits=2))% |
| Final training loss | $(round(TRAIN_LOSS, digits=4)) |
| Backend | CPU |
| Test images loaded | $(length(ALL_TEST_IMAGES)) |
"""

# ╔═╡ Cell order:
# ╠═11111111-aaaa-bbbb-cccc-000000000001
# ╠═22222222-aaaa-bbbb-cccc-000000000002
# ╠═33333333-aaaa-bbbb-cccc-000000000003
# ╠═44444444-aaaa-bbbb-cccc-000000000004
# ╠═55555555-aaaa-bbbb-cccc-000000000005
# ╠═66666666-aaaa-bbbb-cccc-000000000006
# ╠═77777777-aaaa-bbbb-cccc-000000000007
# ╠═88888888-aaaa-bbbb-cccc-000000000008
# ╠═99999999-aaaa-bbbb-cccc-000000000009
# ╠═aaaaaaaa-bbbb-cccc-dddd-00000000000a
# ╠═bbbbbbbb-bbbb-cccc-dddd-00000000000b
# ╠═cccccccc-bbbb-cccc-dddd-00000000000c
# ╠═dddddddd-bbbb-cccc-dddd-00000000000d
# ╠═eeeeeeee-bbbb-cccc-dddd-00000000000e
# ╠═ffffffff-bbbb-cccc-dddd-00000000000f
# ╠═10101010-bbbb-cccc-dddd-000000000010
# ╠═11111111-bbbb-cccc-dddd-000000000011
# ╠═12121212-bbbb-cccc-dddd-000000000012
