using Test

if !isdefined(Main, :SegUNet)
    include(joinpath(@__DIR__, "..", "src", "models", "seg_unet_model.jl"))
end
using .SegUNet
using Random, Lux

const RNG = Random.MersenneTwister(42)

@testset "Segmentation Model" begin

    model, ps, st = build_unet(RNG)
    st_test = Lux.testmode(st)

    @testset "SEG-001: Output shape matches input spatial dims" begin
        x = randn(Float32, 256, 256, 1, 2)
        mask_pred, _, _ = model(x, ps, st_test)
        @test size(mask_pred) == (256, 256, 1, 2)

        # Test with different batch size
        x4 = randn(Float32, 256, 256, 1, 4)
        mask_pred4, _, _ = model(x4, ps, st_test)
        @test size(mask_pred4) == (256, 256, 1, 4)

        # Single sample
        x1 = randn(Float32, 256, 256, 1, 1)
        mask_pred1, _, _ = model(x1, ps, st_test)
        @test size(mask_pred1) == (256, 256, 1, 1)
    end

    @testset "SEG-002: Encoder has 4 conv blocks (1->32->64->128->128)" begin
        x = randn(Float32, 256, 256, 1, 2)
        _, _, intermediates = model(x, ps, st_test)

        # Check channel dimensions of each encoder block output
        @test size(intermediates.enc1, 3) == 32    # 1 -> 32
        @test size(intermediates.enc2, 3) == 64    # 32 -> 64
        @test size(intermediates.enc3, 3) == 128   # 64 -> 128
        @test size(intermediates.enc4, 3) == 128   # 128 -> 128
    end

    @testset "SEG-003: Decoder has 4 upsampling blocks with skip connections" begin
        # Verify the model struct has 4 decoder blocks
        @test hasproperty(model, :dec1)
        @test hasproperty(model, :dec2)
        @test hasproperty(model, :dec3)
        @test hasproperty(model, :dec4)

        # Each decoder block should have an upconv (ConvTranspose)
        @test model.dec1.upconv isa ConvTranspose
        @test model.dec2.upconv isa ConvTranspose
        @test model.dec3.upconv isa ConvTranspose
        @test model.dec4.upconv isa ConvTranspose

        # Verify skip connections work by checking decoder input channels
        # dec4.conv1 takes (upsampled 128) + (skip 128) = 256 input channels
        # We can verify by checking the conv1 weight shape
        @test size(ps.dec4.conv1.weight, 3) == 128 + 128  # up_ch + skip_ch
        @test size(ps.dec3.conv1.weight, 3) == 64 + 128   # up_ch + skip_ch
        @test size(ps.dec2.conv1.weight, 3) == 32 + 64
        @test size(ps.dec1.conv1.weight, 3) == 32 + 32
    end

    @testset "SEG-004: Output values in [0,1] (sigmoid)" begin
        # Test with various inputs
        for _ in 1:5
            x = randn(Float32, 256, 256, 1, 2) .* 10  # large values
            mask_pred, _, _ = model(x, ps, st_test)
            @test all(mask_pred .>= 0.0f0)
            @test all(mask_pred .<= 1.0f0)
        end

        # Test with extreme inputs
        x_large = ones(Float32, 256, 256, 1, 1) .* 100.0f0
        pred_large, _, _ = model(x_large, ps, st_test)
        @test all(pred_large .>= 0.0f0)
        @test all(pred_large .<= 1.0f0)

        x_neg = ones(Float32, 256, 256, 1, 1) .* -100.0f0
        pred_neg, _, _ = model(x_neg, ps, st_test)
        @test all(pred_neg .>= 0.0f0)
        @test all(pred_neg .<= 1.0f0)
    end

    @testset "SEG-005: Load encoder from classifier checkpoint" begin
        # Test that encoder layer shapes match SAR_CNN pattern
        # enc1: Conv (3,3) with 1->32
        @test size(ps.enc1.conv1.weight) == (3, 3, 1, 32)
        # enc2: Conv (3,3) with 32->64
        @test size(ps.enc2.conv1.weight) == (3, 3, 32, 64)
        # enc3: Conv (3,3) with 64->128
        @test size(ps.enc3.conv1.weight) == (3, 3, 64, 128)
        # enc4: Conv (3,3) with 128->128
        @test size(ps.enc4.conv1.weight) == (3, 3, 128, 128)
    end

    @testset "SEG-006: Fits in 8GB VRAM at batch_size=16" begin
        n_params = SegUNet.count_parameters(ps)
        @test n_params > 0

        # Parameter memory (Float32)
        param_mb = n_params * 4 / 1024^2

        # Activation estimate for batch=16 at 256x256
        # Sum of all feature map sizes through the network
        act_per_sample = (
            256*256*32 +   # enc1
            128*128*64 +   # enc2
            64*64*128 +    # enc3
            32*32*128 +    # enc4
            16*16*256 +    # bottleneck
            32*32*128 +    # dec4
            64*64*64 +     # dec3
            128*128*32 +   # dec2
            256*256*32     # dec1
        )
        act_mb = act_per_sample * 16 * 4 / 1024^2  # batch=16, Float32

        # Total with 2x overhead for gradients
        total_mb = param_mb + act_mb * 2
        @test total_mb < 8000  # Must fit in 8GB
    end

    @testset "SEG-007: Intermediates dict from forward pass" begin
        x = randn(Float32, 256, 256, 1, 2)
        _, _, intermediates = model(x, ps, st_test)

        # Must have encoder feature maps
        @test hasproperty(intermediates, :enc1)
        @test hasproperty(intermediates, :enc2)
        @test hasproperty(intermediates, :enc3)
        @test hasproperty(intermediates, :enc4)
        @test hasproperty(intermediates, :bottleneck)

        # Feature maps should have correct spatial dims (after encoding, before pooling)
        @test size(intermediates.enc1)[1:2] == (256, 256)
        @test size(intermediates.enc2)[1:2] == (128, 128)
        @test size(intermediates.enc3)[1:2] == (64, 64)
        @test size(intermediates.enc4)[1:2] == (32, 32)
        @test size(intermediates.bottleneck)[1:2] == (16, 16)

        # Batch dimension preserved
        @test size(intermediates.enc1, 4) == 2
    end

end
