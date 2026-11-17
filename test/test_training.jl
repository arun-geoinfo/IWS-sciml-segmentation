using Test

if !isdefined(Main, :SegTraining)
    include(joinpath(@__DIR__, "..", "src", "training", "seg_train.jl"))
end
using .SegTraining
using Random, Lux, Optimisers, JLD2

# Qualified references to avoid name conflicts when run from runtests.jl
const _build_unet = SegTraining.SegUNet.build_unet
const _LossWeights = SegTraining.PhysicsLosses.LossWeights

# ============================================================================
# Helper: create tiny synthetic data for fast testing
# ============================================================================
function make_tiny_data(; n_samples=8, img_size=32, batch_size=4)
    # Simple pattern: target is a horizontal stripe in the middle
    batches = []
    for i in 1:batch_size:n_samples
        b_end = min(i + batch_size - 1, n_samples)
        B = b_end - i + 1
        X = rand(Float32, img_size, img_size, 1, B)
        Y = zeros(Float32, img_size, img_size, 1, B)
        # Put a stripe in rows 12-20
        lo = max(1, img_size ÷ 3)
        hi = min(img_size, 2 * img_size ÷ 3)
        Y[lo:hi, :, :, :] .= 1.0f0
        push!(batches, (X, Y))
    end
    return batches
end

# Simple iterator wrapper for test batches
struct TestLoader
    batches::Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}
end
Base.iterate(tl::TestLoader, i=1) = i > length(tl.batches) ? nothing : (tl.batches[i], i+1)
Base.length(tl::TestLoader) = length(tl.batches)

@testset "Training" begin

    # Shared setup: tiny model on CPU
    rng = Random.MersenneTwister(42)
    model, ps, st = _build_unet(rng; in_channels=1, out_channels=1)

    train_data = make_tiny_data(n_samples=8, img_size=32, batch_size=4)
    val_data = make_tiny_data(n_samples=4, img_size=32, batch_size=4)
    train_loader = TestLoader(train_data)
    val_loader = TestLoader(val_data)

    @testset "TR-001: Trains for specified epochs with logging" begin
        config = TrainConfig(
            epochs=3, lr=1.0f-3, patience=100,
            checkpoint_path=tempname() * ".jld2",
            loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0)
        )

        ps_t, st_t, log = train!(model, deepcopy(ps), deepcopy(st),
                                  train_loader, val_loader, config; verbose=false)

        # Should have exactly 3 epoch entries
        @test length(log.epochs) == 3
        @test !log.stopped_early

        # Clean up
        isfile(config.checkpoint_path) && rm(config.checkpoint_path)
    end

    @testset "TR-002: Checkpoint saved on val Dice improvement" begin
        ckpt_path = tempname() * ".jld2"
        config = TrainConfig(
            epochs=3, lr=1.0f-3, patience=100,
            checkpoint_path=ckpt_path,
            loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0)
        )

        _, _, log = train!(model, deepcopy(ps), deepcopy(st),
                           train_loader, val_loader, config; verbose=false)

        # Checkpoint should exist
        @test isfile(ckpt_path)

        # Should be loadable
        loaded_ps, loaded_st, meta = SegTraining.load_checkpoint(ckpt_path)
        @test meta.epoch == log.best_epoch
        @test meta.val_dice ≈ log.best_val_dice

        # Loaded params should have same structure
        @test keys(loaded_ps) == keys(ps)

        rm(ckpt_path)
    end

    @testset "TR-003: Early stopping on patience exhaustion" begin
        # Use very small patience — training should stop before max epochs
        config = TrainConfig(
            epochs=100, lr=1.0f-6, patience=2,  # tiny LR = no improvement
            checkpoint_path=tempname() * ".jld2",
            loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0)
        )

        _, _, log = train!(model, deepcopy(ps), deepcopy(st),
                           train_loader, val_loader, config; verbose=false)

        # Should have stopped early (well before 100 epochs)
        @test log.stopped_early
        @test length(log.epochs) < 100
        # Should have stopped after best_epoch + patience
        @test length(log.epochs) <= log.best_epoch + config.patience

        isfile(config.checkpoint_path) && rm(config.checkpoint_path)
    end

    @testset "TR-004: Training log contains required fields" begin
        config = TrainConfig(
            epochs=2, lr=1.0f-3, patience=100,
            checkpoint_path=tempname() * ".jld2",
            loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0)
        )

        _, _, log = train!(model, deepcopy(ps), deepcopy(st),
                           train_loader, val_loader, config; verbose=false)

        for metrics in log.epochs
            @test hasfield(typeof(metrics), :train_loss)
            @test hasfield(typeof(metrics), :val_loss)
            @test hasfield(typeof(metrics), :val_dice)
            @test hasfield(typeof(metrics), :val_iou)
            @test isfinite(metrics.train_loss)
            @test isfinite(metrics.val_loss)
            @test isfinite(metrics.val_dice)
            @test isfinite(metrics.val_iou)
            @test 0 <= metrics.val_dice <= 1
            @test 0 <= metrics.val_iou <= 1
        end

        isfile(config.checkpoint_path) && rm(config.checkpoint_path)
    end

    @testset "TR-005: Parameters update after one step" begin
        ps_before = deepcopy(ps)
        st_copy = deepcopy(st)
        opt = Optimisers.AdamW(; eta=1.0f-3, lambda=1.0f-2)
        opt_state = Optimisers.setup(opt, ps_before)

        X, Y = train_data[1]
        config = TrainConfig(epochs=1, lr=1.0f-3,
                             checkpoint_path=tempname() * ".jld2",
                             loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0))

        loss, ps_after, _, _, _ = train_step!(model, ps_before, st_copy,
                                               opt_state, X, Y, config)

        # Loss should be finite
        @test isfinite(loss)
        @test loss > 0

        # Parameters should have changed
        w_before = ps.enc1.conv1.weight
        w_after = ps_after.enc1.conv1.weight
        @test w_before != w_after  # At least some weights changed
    end

    @testset "TR-006: Physics lambdas=0 gives standard segmentation" begin
        # Train with physics losses disabled
        config_no_physics = TrainConfig(
            epochs=2, lr=1.0f-3, patience=100,
            checkpoint_path=tempname() * ".jld2",
            loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0,
                                     lambda_kdv=0.0f0, lambda_grad=0.0f0)
        )

        # Train with physics losses enabled
        config_with_physics = TrainConfig(
            epochs=2, lr=1.0f-3, patience=100,
            checkpoint_path=tempname() * ".jld2",
            loss_weights=_LossWeights(lambda_dice=1.0f0, lambda_bce=1.0f0,
                                     lambda_kdv=0.1f0, lambda_grad=0.01f0)
        )

        # Both should work without errors
        _, _, log_no = train!(model, deepcopy(ps), deepcopy(st),
                              train_loader, val_loader, config_no_physics; verbose=false)
        _, _, log_with = train!(model, deepcopy(ps), deepcopy(st),
                                train_loader, val_loader, config_with_physics; verbose=false)

        @test length(log_no.epochs) == 2
        @test length(log_with.epochs) == 2

        # Both should produce valid metrics
        @test all(isfinite(e.train_loss) for e in log_no.epochs)
        @test all(isfinite(e.train_loss) for e in log_with.epochs)

        # With physics losses, training loss should be different
        # (physics terms add to the loss)
        @test log_no.epochs[1].train_loss != log_with.epochs[1].train_loss

        for p in [config_no_physics.checkpoint_path, config_with_physics.checkpoint_path]
            isfile(p) && rm(p)
        end
    end

end
