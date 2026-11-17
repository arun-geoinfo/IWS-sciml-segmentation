using Test

@testset "Segmentation Model" begin

    @testset "SEG-001: Output shape matches input spatial dims" begin
        # TODO: Create model, pass (256,256,1,2) input, verify output is (256,256,1,2)
        @test_skip "not yet implemented"
    end

    @testset "SEG-002: Encoder has 4 conv blocks (1->32->64->128->128)" begin
        # TODO: Inspect model layers, verify channel progression
        @test_skip "not yet implemented"
    end

    @testset "SEG-003: Decoder has 4 upsampling blocks with skip connections" begin
        # TODO: Inspect decoder structure
        @test_skip "not yet implemented"
    end

    @testset "SEG-004: Output values in [0,1] (sigmoid)" begin
        # TODO: Forward pass with random input, verify all values in [0,1]
        @test_skip "not yet implemented"
    end

    @testset "SEG-005: Load encoder from classifier checkpoint" begin
        # TODO: Load .jld2 classifier weights into encoder, verify shapes match
        @test_skip "not yet implemented"
    end

    @testset "SEG-006: Fits in 8GB VRAM at batch_size=16" begin
        # TODO: Estimate param count + activation memory, verify < 8GB
        @test_skip "not yet implemented"
    end

    @testset "SEG-007: Intermediates dict from forward pass" begin
        # TODO: Forward pass, verify intermediates has encoder feature maps
        @test_skip "not yet implemented"
    end

end
