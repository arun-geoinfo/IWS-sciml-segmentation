using Test

if !isdefined(Main, :SegEvaluation)
    include(joinpath(@__DIR__, "..", "src", "evaluation", "seg_evaluate.jl"))
end
using .SegEvaluation
using Dates

@testset "Evaluation" begin

    @testset "EV-001: Dice coefficient in [0,1]" begin
        # Random predictions vs random targets
        pred = Float32.(rand(Float32, 1000) .> 0.5f0)
        target = Float32.(rand(Float32, 1000) .> 0.5f0)
        m = compute_metrics(pred, target)
        @test 0.0f0 <= m.dice <= 1.0f0

        # All positive predictions, mixed targets
        pred_all1 = ones(Float32, 1000)
        target_mixed = Float32.(rand(Float32, 1000) .> 0.5f0)
        m2 = compute_metrics(pred_all1, target_mixed)
        @test 0.0f0 <= m2.dice <= 1.0f0

        # All negative predictions
        pred_all0 = zeros(Float32, 1000)
        m3 = compute_metrics(pred_all0, target_mixed)
        @test 0.0f0 <= m3.dice <= 1.0f0
    end

    @testset "EV-002: IoU in [0,1]" begin
        pred = Float32.(rand(Float32, 1000) .> 0.5f0)
        target = Float32.(rand(Float32, 1000) .> 0.5f0)
        m = compute_metrics(pred, target)
        @test 0.0f0 <= m.iou <= 1.0f0

        # Verify Dice-IoU consistency: Dice = 2*IoU / (1+IoU)
        @test dice_iou_consistent(m.dice, m.iou)
    end

    @testset "EV-003: Perfect predictions give Dice=1, IoU=1" begin
        # Identical non-trivial predictions and targets
        target = Float32.([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
        pred = copy(target)
        m = compute_metrics(pred, target)
        @test m.dice ≈ 1.0f0
        @test m.iou ≈ 1.0f0
        @test m.precision ≈ 1.0f0
        @test m.recall ≈ 1.0f0

        # All zeros — both empty, still perfect
        z = zeros(Float32, 100)
        mz = compute_metrics(z, z)
        @test mz.dice ≈ 1.0f0
        @test mz.iou ≈ 1.0f0

        # All ones — also perfect
        o = ones(Float32, 100)
        mo = compute_metrics(o, o)
        @test mo.dice ≈ 1.0f0
        @test mo.iou ≈ 1.0f0
    end

    @testset "EV-004: All-zero predictions give Recall=0, Dice=0" begin
        pred = zeros(Float32, 100)
        target = ones(Float32, 100)  # all positive targets
        m = compute_metrics(pred, target)
        @test m.recall ≈ 0.0f0
        @test m.dice ≈ 0.0f0

        # With mixed targets (some positives exist)
        target_mixed = Float32.([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        pred_zero = zeros(Float32, 10)
        m2 = compute_metrics(pred_zero, target_mixed)
        @test m2.recall ≈ 0.0f0
        @test m2.dice ≈ 0.0f0
    end

    @testset "EV-005: Report JSON contains required fields" begin
        metrics = SegMetrics(0.75f0, 0.6f0, 0.8f0, 0.7f0, 0.74f0, 0.9f0)
        report_path = tempname() * ".json"

        config = Dict{String,Any}("model" => "UNet", "batch_size" => 8)
        report = save_report(report_path, metrics;
                             threshold=0.5f0, config_info=config)

        @test isfile(report_path)

        # Check required fields
        @test haskey(report, "metrics")
        @test haskey(report, "threshold")
        @test haskey(report, "timestamp")
        @test haskey(report, "config")

        # Check metrics subfields
        m = report["metrics"]
        @test haskey(m, "dice")
        @test haskey(m, "iou")
        @test haskey(m, "precision")
        @test haskey(m, "recall")
        @test haskey(m, "f1")
        @test haskey(m, "accuracy")

        # Verify values round-trip correctly
        @test m["dice"] ≈ 0.75 atol=1e-4
        @test report["threshold"] ≈ 0.5

        # Timestamp should be parseable
        ts = report["timestamp"]
        @test length(ts) > 0
        @test occursin("T", ts)  # ISO format

        # Config should be preserved
        @test report["config"]["model"] == "UNet"

        rm(report_path)
    end

    @testset "EV-006: Threshold binarization" begin
        pred = Float32.([0.1, 0.4, 0.5, 0.6, 0.9, 0.0, 1.0])

        # Default threshold 0.5
        b = binarize(pred, 0.5f0)
        @test b == Float32.([0, 0, 1, 1, 1, 0, 1])

        # Higher threshold
        b_high = binarize(pred, 0.7f0)
        @test b_high == Float32.([0, 0, 0, 0, 1, 0, 1])

        # Lower threshold
        b_low = binarize(pred, 0.2f0)
        @test b_low == Float32.([0, 1, 1, 1, 1, 0, 1])

        # Threshold 0.0 — everything becomes 1 (except exact 0.0 which is < 0.0? No, >= 0.0)
        b_zero = binarize(pred, 0.0f0)
        @test all(b_zero .== 1.0f0)

        # Threshold 1.0 — only exact 1.0 survives
        b_one = binarize(pred, 1.0f0)
        @test b_one == Float32.([0, 0, 0, 0, 0, 0, 1])

        # Verify output type
        @test eltype(b) == Float32
    end

end
