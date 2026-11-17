using Test

@testset "Evaluation" begin

    @testset "EV-001: Dice coefficient in [0,1]" begin
        # TODO: Evaluate model, verify Dice is in valid range
        @test_skip "not yet implemented"
    end

    @testset "EV-002: IoU in [0,1]" begin
        # TODO: Evaluate model, verify IoU is in valid range
        @test_skip "not yet implemented"
    end

    @testset "EV-003: Perfect predictions give Dice=1, IoU=1" begin
        # TODO: Identical pred and target -> all metrics = 1.0
        @test_skip "not yet implemented"
    end

    @testset "EV-004: All-zero predictions give Recall=0, Dice=0" begin
        # TODO: Zero predictions vs positive targets -> Recall=0, Dice=0
        @test_skip "not yet implemented"
    end

    @testset "EV-005: Report JSON contains required fields" begin
        # TODO: Verify JSON has metrics, threshold, config, timestamp
        @test_skip "not yet implemented"
    end

    @testset "EV-006: Threshold binarization" begin
        # TODO: Values >= threshold -> 1, below -> 0
        @test_skip "not yet implemented"
    end

end
