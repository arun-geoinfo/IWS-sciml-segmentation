using Test

include(joinpath(@__DIR__, "..", "src", "training", "physics_losses.jl"))
using .PhysicsLosses
using Statistics

@testset "Physics Losses" begin

    @testset "PL-001: Base loss is Dice + BCE combination" begin
        pred = rand(Float32, 64, 64, 1, 4)
        target = Float32.(rand(Float32, 64, 64, 1, 4) .> 0.5f0)

        d = dice_loss(pred, target)
        b = bce_loss(pred, target)

        # Both should be positive scalars
        @test d isa Float32 || d isa Float64
        @test b isa Float32 || b isa Float64
        @test d >= 0
        @test b >= 0

        # Combined with only Dice+BCE weights should equal their weighted sum
        w = LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0, lambda_kdv=0.0f0, lambda_grad=0.0f0)
        total, comp = combined_loss(pred, target, nothing, w)
        @test comp.dice ≈ d
        @test comp.bce ≈ b
        @test total ≈ d + b atol=1e-5
    end

    @testset "PL-002: KdV regularization penalizes non-soliton structure" begin
        # Random noise mask — no wave structure
        noise = rand(Float32, 256, 256, 1, 2)
        kdv_noise = kdv_regularization_loss(noise)

        # Soliton-like mask: sech² profile repeated as wave crests
        x = collect(Float32, 1:256)
        soliton_1d = sech.(Float32.(0.1f0 .* (x .- 128.0f0))) .^ 2
        soliton_2d = repeat(soliton_1d, 1, 256)
        soliton_mask = reshape(soliton_2d, 256, 256, 1, 1)
        kdv_soliton = kdv_regularization_loss(soliton_mask)

        # Soliton should have MUCH lower loss than noise
        @test kdv_soliton < kdv_noise
        @test kdv_soliton < kdv_noise * 0.01  # At least 100x lower

        # Both should be non-negative
        @test kdv_noise >= 0
        @test kdv_soliton >= 0
    end

    @testset "PL-003: Gradient structure loss encourages anisotropy" begin
        # Isotropic features (random)
        feat_iso = rand(Float32, 32, 32, 64, 2)
        grad_iso = gradient_structure_loss(feat_iso)

        # Anisotropic features (horizontal stripes)
        feat_stripes = zeros(Float32, 32, 32, 64, 2)
        for i in 1:32
            feat_stripes[i, :, :, :] .= Float32(sin(i * 0.5))
        end
        grad_stripes = gradient_structure_loss(feat_stripes)

        # Stripes should have much lower loss (more anisotropic)
        @test grad_stripes < grad_iso

        # Isotropic should be near 1.0 (equal energy in both directions)
        @test grad_iso > 0.8f0

        # Pure stripes should be near 0.0
        @test grad_stripes < 0.1f0
    end

    @testset "PL-004: KdV loss near zero for soliton profile" begin
        # Single soliton: η(x) = sech²((x-x0)/L)
        x = collect(Float32, 1:256)
        soliton_1d = sech.(Float32.(0.1f0 .* (x .- 128.0f0))) .^ 2
        soliton_2d = repeat(soliton_1d, 1, 256)  # uniform along y (wave crest)
        soliton_mask = reshape(soliton_2d, 256, 256, 1, 1)

        kdv = kdv_regularization_loss(soliton_mask)

        # Should be very small (not exactly zero due to discretization)
        @test kdv < 0.001f0

        # Multiple solitons (periodic wave train)
        multi_soliton_1d = zeros(Float32, 256)
        for center in [64, 128, 192]
            multi_soliton_1d .+= sech.(Float32.(0.15f0 .* (x .- Float32(center)))) .^ 2
        end
        multi_2d = repeat(multi_soliton_1d, 1, 256)
        multi_mask = reshape(multi_2d, 256, 256, 1, 1)
        kdv_multi = kdv_regularization_loss(multi_mask)

        @test kdv_multi < 0.1f0  # Still structured, low loss
    end

    @testset "PL-005: Total loss is weighted sum of components" begin
        pred = rand(Float32, 64, 64, 1, 2)
        target = Float32.(rand(Float32, 64, 64, 1, 2) .> 0.5f0)
        feat = rand(Float32, 16, 16, 32, 2)

        w = LossWeights(lambda_dice=0.5f0, lambda_bce=0.3f0,
                        lambda_kdv=0.1f0, lambda_grad=0.05f0)

        total, comp = combined_loss(pred, target, (enc4=feat,), w)

        expected = 0.5f0 * comp.dice + 0.3f0 * comp.bce +
                   0.1f0 * comp.kdv + 0.05f0 * comp.grad
        @test total ≈ expected atol=1e-5

        # With all lambdas = 0, total should be 0
        w_zero = LossWeights(lambda_dice=0.0f0, lambda_bce=0.0f0,
                             lambda_kdv=0.0f0, lambda_grad=0.0f0)
        total_zero, _ = combined_loss(pred, target, nothing, w_zero)
        @test total_zero ≈ 0.0f0 atol=1e-7
    end

    @testset "PL-006: All losses are finite" begin
        # Normal inputs
        pred = rand(Float32, 64, 64, 1, 2)
        target = Float32.(rand(Float32, 64, 64, 1, 2) .> 0.5f0)
        feat = rand(Float32, 16, 16, 32, 2)

        @test isfinite(dice_loss(pred, target))
        @test isfinite(bce_loss(pred, target))
        @test isfinite(kdv_regularization_loss(pred))
        @test isfinite(gradient_structure_loss(feat))

        w = LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0,
                        lambda_kdv=0.1f0, lambda_grad=0.01f0)
        total, comp = combined_loss(pred, target, (enc4=feat,), w)
        @test isfinite(total)
        @test isfinite(comp.dice)
        @test isfinite(comp.bce)
        @test isfinite(comp.kdv)
        @test isfinite(comp.grad)

        # Edge case: all zeros
        z = zeros(Float32, 64, 64, 1, 2)
        @test isfinite(dice_loss(z, z))
        @test isfinite(bce_loss(z, z))
        @test isfinite(kdv_regularization_loss(z))

        # Edge case: all ones
        o = ones(Float32, 64, 64, 1, 2)
        @test isfinite(dice_loss(o, o))
        @test isfinite(bce_loss(o, o))
        @test isfinite(kdv_regularization_loss(o))
    end

    @testset "PL-007: Dice loss is 0 for matching all-zero pred and target" begin
        z = zeros(Float32, 64, 64, 1, 4)
        d = dice_loss(z, z)
        @test d ≈ 0.0f0 atol=1e-5

        # Perfect match (both ones) should also be 0
        o = ones(Float32, 64, 64, 1, 4)
        d_ones = dice_loss(o, o)
        @test d_ones ≈ 0.0f0 atol=1e-5

        # Mismatch should be > 0
        d_mismatch = dice_loss(o, z)
        @test d_mismatch > 0.0f0
    end

end
