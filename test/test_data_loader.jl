using Test

if !isdefined(Main, :SegDataLoader)
    include(joinpath(@__DIR__, "..", "src", "dataloaders", "seg_dataloader.jl"))
end
using .SegDataLoader

const DATA_DIR = joinpath(@__DIR__, "..", "data", "zenodo_oceanic_phenomena")
const DATA_AVAILABLE = isdir(joinpath(DATA_DIR, "Image_Geo")) && isdir(joinpath(DATA_DIR, "label"))

@testset "Data Loader" begin

    @testset "DL-001: Discover matching image-mask pairs" begin
        if !DATA_AVAILABLE
            @test_skip "Dataset not available at $DATA_DIR"
        else
            imgs, masks = SegDataLoader.discover_pairs(DATA_DIR)
            @test length(imgs) > 0
            @test length(imgs) == length(masks)
            # Verify stems match
            for i in 1:min(5, length(imgs))
                img_stem = splitext(basename(imgs[i]))[1]
                msk_stem = splitext(basename(masks[i]))[1]
                @test img_stem == msk_stem
            end
            # Images are .tiff, masks are .png
            @test endswith(imgs[1], ".tiff") || endswith(imgs[1], ".tif")
            @test endswith(masks[1], ".png")
        end
    end

    @testset "DL-002: Images are Float32 normalized WHCN" begin
        if !DATA_AVAILABLE
            @test_skip "Dataset not available"
        else
            imgs, masks = SegDataLoader.discover_pairs(DATA_DIR)
            img = SegDataLoader.load_image(imgs[1], (256, 256))
            @test size(img) == (256, 256, 1)
            @test eltype(img) == Float32
            @test minimum(img) >= 0.0f0
            @test maximum(img) <= 1.0f0
        end
    end

    @testset "DL-003: Binary mask from class index" begin
        if !DATA_AVAILABLE
            @test_skip "Dataset not available"
        else
            imgs, masks = SegDataLoader.discover_pairs(DATA_DIR)
            # Find a mask that contains IW (class 12)
            iw_mask_path = nothing
            for mp in masks[1:min(100, length(masks))]
                raw = SegDataLoader.load_mask_raw(mp)
                if UInt8(12) in raw
                    iw_mask_path = mp
                    break
                end
            end
            if iw_mask_path === nothing
                @test_skip "No IW mask found in first 100 samples"
            else
                raw = SegDataLoader.load_mask_raw(iw_mask_path)
                bmask = SegDataLoader.load_mask_binary(iw_mask_path, 12, (256, 256))
                @test size(bmask) == (256, 256, 1)
                @test eltype(bmask) == Float32
                # Only 0.0 and 1.0 values
                unique_vals = sort(unique(bmask))
                @test all(v -> v == 0.0f0 || v == 1.0f0, unique_vals)
                # Pixels that were class 12 should be 1.0
                iw_count_raw = count(==(UInt8(12)), raw)
                iw_count_bin = count(==(1.0f0), bmask)
                @test iw_count_bin == iw_count_raw
                # Non-IW pixels should be 0.0
                non_iw_count = count(!=(UInt8(12)), raw)
                zero_count = count(==(0.0f0), bmask)
                @test zero_count == non_iw_count
            end
        end
    end

    @testset "DL-004: Train/val/test split ratios" begin
        # Test with synthetic paths (no data needed)
        N = 100
        fake_imgs = ["img_$i.tiff" for i in 1:N]
        fake_masks = ["mask_$i.png" for i in 1:N]
        ratios = (0.7, 0.15, 0.15)

        train, val, test = SegDataLoader.split_data(fake_imgs, fake_masks, ratios; seed=42)

        n_train = length(train[1])
        n_val = length(val[1])
        n_test = length(test[1])

        # Sizes should be approximately correct
        @test n_train == 70
        @test n_val == 15
        @test n_test == 15
        # Total should equal N
        @test n_train + n_val + n_test == N

        # No overlap between splits
        all_train = Set(train[1])
        all_val = Set(val[1])
        all_test = Set(test[1])
        @test isempty(intersect(all_train, all_val))
        @test isempty(intersect(all_train, all_test))
        @test isempty(intersect(all_val, all_test))

        # Image and mask paths are paired correctly within each split
        for (im_paths, mk_paths) in [train, val, test]
            @test length(im_paths) == length(mk_paths)
        end
    end

    @testset "DL-005: GPU tensors when CUDA available" begin
        # Test GPU transfer function parameter
        # We test the mechanism without requiring CUDA
        if !DATA_AVAILABLE
            @test_skip "Dataset not available"
        else
            imgs, masks = SegDataLoader.discover_pairs(DATA_DIR)
            # Find a few IW images
            iw_imgs = String[]
            iw_masks = String[]
            for i in 1:min(50, length(imgs))
                raw = SegDataLoader.load_mask_raw(masks[i])
                if UInt8(12) in raw
                    push!(iw_imgs, imgs[i])
                    push!(iw_masks, masks[i])
                    length(iw_imgs) >= 4 && break
                end
            end

            if length(iw_imgs) < 2
                @test_skip "Not enough IW images in first 50"
            else
                # Test with a mock GPU transfer (identity + type tag)
                transferred = Ref(false)
                mock_gpu = x -> begin
                    transferred[] = true
                    x  # return as-is
                end

                iter = SegDataLoader.SegBatchIterator(iw_imgs, iw_masks;
                    target_size=(256,256), target_class=12, batch_size=2,
                    shuffle=false, seed=42, gpu_transfer=mock_gpu)

                (X, Y), _ = iterate(iter)
                @test transferred[]
                @test size(X)[end] <= 2  # batch dimension
            end
        end
    end

    @testset "DL-006: Filter for IW-containing images" begin
        if !DATA_AVAILABLE
            @test_skip "Dataset not available"
        else
            imgs, masks = SegDataLoader.discover_pairs(DATA_DIR)
            # Take a small sample and filter
            sample_n = min(50, length(imgs))
            filtered_imgs, filtered_masks = SegDataLoader.filter_by_class(
                imgs[1:sample_n], masks[1:sample_n], 12)

            # All filtered images should have IW pixels in their masks
            for mp in filtered_masks
                raw = SegDataLoader.load_mask_raw(mp)
                @test UInt8(12) in raw
            end

            # Filtered count should be <= original
            @test length(filtered_imgs) <= sample_n
            @test length(filtered_imgs) == length(filtered_masks)
        end
    end

end
