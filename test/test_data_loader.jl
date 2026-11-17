using Test

@testset "Data Loader" begin

    @testset "DL-001: Discover matching image-mask pairs" begin
        # TODO: Initialize loader on zenodo dir, verify it finds paired files
        @test_skip "not yet implemented"
    end

    @testset "DL-002: Images are Float32 normalized WHCN" begin
        # TODO: Load a 16-bit TIFF, verify output is Float32 in [0,1], shape (256,256,1,B)
        @test_skip "not yet implemented"
    end

    @testset "DL-003: Binary mask from class index" begin
        # TODO: Load a uint8 PNG mask, verify class 12 -> 1.0, others -> 0.0
        @test_skip "not yet implemented"
    end

    @testset "DL-004: Train/val/test split ratios" begin
        # TODO: Split N images at [0.7, 0.15, 0.15], verify counts and no overlap
        @test_skip "not yet implemented"
    end

    @testset "DL-005: GPU tensors when CUDA available" begin
        # TODO: Load batch, verify CuArray type if CUDA is available
        @test_skip "not yet implemented"
    end

    @testset "DL-006: Filter for IW-containing images" begin
        # TODO: Verify only images with at least one IW pixel are included
        @test_skip "not yet implemented"
    end

end
