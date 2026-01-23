# src/preprocessing/preprocess_data.jl
using CSV, DataFrames, Random, FilePathsBase, FileIO

include("../config/default_config.jl")
using .DefaultConfig

# -------------------------
# Helper: Process one split
# -------------------------
function preprocess_split(image_dir::String, csv_file::String, proc_dir::String)
    # 1. Read CSV
    df = CSV.read(csv_file, DataFrame)
    # Assume CSV has columns: "id", "label"
image_to_label = Dict(split(basename(row.id), ".")[1] => row.ground_truth for row in eachrow(df))

    # 2. Get all image files
    img_files = readdir(image_dir; join=true)

    # 3. Map filenames to labels
    labeled_files = []
    for img_path in img_files
        fname = split(basename(img_path), ".")[1]  # remove extension
        if haskey(image_to_label, fname)
            push!(labeled_files, (img_path, image_to_label[fname]))
        else
            @warn "No label found for $fname"
        end
    end

    # 4. Shuffle reproducibly
    Random.seed!(DefaultConfig.RNG_SEED)
    labeled_files = shuffle(labeled_files)

    # 5. Copy images to processed folder by class
    mkpath(joinpath(proc_dir, "class0"))
    mkpath(joinpath(proc_dir, "class1"))

    for (src, label) in labeled_files
        dest_dir = joinpath(proc_dir, "class$(label)")
        dest_path = joinpath(dest_dir, basename(src))
        cp(src, dest_path; force=true)
    end

    println("Processed $(length(labeled_files)) images into $proc_dir")
end

# -------------------------
# Run for TRAIN and TEST
# -------------------------
preprocess_split(DefaultConfig.RAW_TRAIN_IMAGES, DefaultConfig.TRAIN_CSV, DefaultConfig.PROC_TRAIN)
preprocess_split(DefaultConfig.RAW_TEST_IMAGES, DefaultConfig.TEST_CSV, DefaultConfig.PROC_TEST)
