# ================================
# preprocess_gpu.jl
# Data preprocessing with monitoring & alerts
# ================================

using LuxCUDA
using CSV, DataFrames, Random, FileIO, ProgressMeter, SHA
using Dates
using Logging

# GPU initialization
using CUDA
CUDA.set_runtime_version!(; local_toolkit=true)
ENV["JULIA_CUDA_VERBOSE"] = "0"
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# Load configuration
include("../config/default_config.jl")
using .DefaultConfig

# ========================================
# EMAIL CONFIGURATION
# ========================================
const ENABLE_EMAIL_ALERTS = false  # Set true when SMTP configured
const ADMIN_EMAIL = "admin@example.com"
const SMTP_SERVER = "smtp.company.com"
const SMTP_PORT = 587
const SMTP_USER = "your-email@company.com"
const SMTP_PASSWORD = "your-password"

function send_email_alert(subject::String, body::String; level::Symbol=:info)
    !ENABLE_EMAIL_ALERTS && return

    try
        println("📧 Email alert: $subject")
        @info "Email alert queued: $subject"
    catch e
        @warn "Failed to send email alert" exception=e
    end
end

# ========================================
# Checksum generator (data integrity)
# ========================================
function file_checksum(path)
    open(path) do io
        return bytes2hex(sha256(read(io)))
    end
end

# ========================================
# Dataset validation helper
# ========================================
function validate_dataset(labeled_files)
    missing_files = filter(x -> !isfile(x[1]), labeled_files)

    if !isempty(missing_files)
        error_msg = """
        ❌ CRITICAL: Dataset validation failed!

        Missing $(length(missing_files)) files

        First 5 missing files:
        $(join([basename(f[1]) for f in missing_files[1:min(5, length(missing_files))]], "\n"))
        """
        @error error_msg
        send_email_alert("🚨 CRITICAL: Missing Image Files", error_msg; level=:critical)
        error(error_msg)
    end

    println("✅ Dataset validation passed: $(length(labeled_files)) files")
end

# ========================================
# Main preprocessing with monitoring
# ========================================
function preprocess_split(image_dir::String, csv_file::String, proc_dir::String;
                          enable_checksum=false)

    start_time = now()

    println("\n" * "="^60)
    println("📂 Processing split: ", basename(proc_dir))
    println("="^60)

    # ❌ CRITICAL: Image directory not found
    if !isdir(image_dir)
        error_msg = """
        ❌ CRITICAL: Image directory not found!

        Expected path: $image_dir

        Please verify:
        1. Data has been downloaded
        2. Path is correct in config
        3. Directory exists
        """
        @error error_msg
        send_email_alert("🚨 CRITICAL: Image Directory Missing", error_msg; level=:critical)
        error(error_msg)
    end

    # ❌ CRITICAL: CSV file not found
    if !isfile(csv_file)
        error_msg = """
        ❌ CRITICAL: CSV file not found!

        Expected path: $csv_file

        Please verify:
        1. CSV file exists
        2. Path is correct in config
        3. File has not been moved/deleted
        """
        @error error_msg
        send_email_alert("🚨 CRITICAL: CSV File Missing", error_msg; level=:critical)
        error(error_msg)
    end

    # Read CSV
    println("📋 Reading CSV labels...")
    local image_to_label

    try
        df = CSV.read(csv_file, DataFrame)

        # Validate CSV structure
        if !("id" in names(df) && "ground_truth" in names(df))
            error("CSV missing required columns: 'id' and 'ground_truth'")
        end

        image_to_label = Dict(
            split(basename(row.id), ".")[1] => row.ground_truth
            for row in eachrow(df)
        )

        println("   ✅ Loaded $(length(image_to_label)) labels")
    catch e
        error_msg = """
        ❌ CRITICAL: Failed to read CSV file!

        File: $csv_file
        Error: $e

        Please verify CSV format and integrity
        """
        @error error_msg
        send_email_alert("🚨 CRITICAL: CSV Read Failed", error_msg; level=:critical)
        rethrow(e)
    end

    # Scan directory
    println("🔍 Scanning image directory...")
    img_files = readdir(image_dir; join=true)
    labeled_files = Tuple{String,Int}[]

    for img_path in img_files
        fname = split(basename(img_path), ".")[1]
        if haskey(image_to_label, fname)
            push!(labeled_files, (img_path, image_to_label[fname]))
        else
            @warn "No label found for $fname"
        end
    end

    println("   ✅ Found $(length(labeled_files)) labeled images")

    # Validation
    validate_dataset(labeled_files)

    # Shuffle
    Random.seed!(DefaultConfig.RNG_SEED)
    labeled_files = shuffle(labeled_files)

    # Create output directories
    mkpath(joinpath(proc_dir, "class0"))
    mkpath(joinpath(proc_dir, "class1"))

    # Track failures
    failed_copies      = String[]
    checksum_failures  = String[]

    println("🚀 Copying $(length(labeled_files)) files using $(Threads.nthreads()) threads")
    prog      = Progress(length(labeled_files))
    copy_lock = ReentrantLock()

    # ── Multithreaded copy with integrity checks ──────────────────────────────
    Threads.@threads for i in eachindex(labeled_files)
        src, label = labeled_files[i]
        dest_dir   = joinpath(proc_dir, "class$(label)")
        dest_path  = joinpath(dest_dir, basename(src))

        try
            cp(src, dest_path; force=true)

            # Checksum verification
            if enable_checksum
                if file_checksum(src) != file_checksum(dest_path)
                    lock(copy_lock) do
                        push!(checksum_failures, basename(src))
                    end
                    @warn "Checksum mismatch: $(basename(src))"
                end
            end

        catch e
            lock(copy_lock) do
                push!(failed_copies, basename(src))
            end
            @error "Failed to copy $(basename(src))" exception=e
        end

        next!(prog)   # ── progress bar tick
    end
    # ─────────────────────────────────────────────────────────────────────────

    elapsed  = Dates.value(now() - start_time) / 1000  # seconds
    n_success = length(labeled_files) - length(failed_copies)
    n_class0  = count(x -> x[2] == 0, labeled_files)
    n_class1  = count(x -> x[2] == 1, labeled_files)

    println("\n" * "="^60)
    println("📊 PREPROCESSING SUMMARY")
    println("="^60)
    println("   Split:     $(basename(proc_dir))")
    println("   Total:     $(length(labeled_files))")
    println("   Successful: $n_success")
    println("   Failed:    $(length(failed_copies))")
    println("   Class 0:   $n_class0")
    println("   Class 1:   $n_class1")
    println("   Time:      $(round(elapsed, digits=1))s")

    if enable_checksum
        println("   Checksum failures: $(length(checksum_failures))")
    end
    println("="^60)

    # ⚠️ WARNING: copy failures
    if !isempty(failed_copies)
        warn_msg = """
        ⚠️ WARNING: Some images failed to copy

        Split: $(basename(proc_dir))
        Failed: $(length(failed_copies)) / $(length(labeled_files))

        Failed files (first 10):
        $(join(failed_copies[1:min(10, length(failed_copies))], "\n"))

        Processing continued with remaining images.
        """
        @warn warn_msg
        send_email_alert("⚠️ Image Copy Failures", warn_msg; level=:warning)
    end

    # ⚠️ WARNING: checksum mismatches
    if enable_checksum && !isempty(checksum_failures)
        warn_msg = """
        ⚠️ WARNING: Checksum verification failed

        Split: $(basename(proc_dir))
        Mismatches: $(length(checksum_failures)) / $(length(labeled_files))

        Failed files (first 10):
        $(join(checksum_failures[1:min(10, length(checksum_failures))], "\n"))

        Files may be corrupted. Manual verification recommended.
        """
        @warn warn_msg
        send_email_alert("⚠️ Checksum Failures", warn_msg; level=:warning)
    end

    # 📧 completion summary
    completion_msg = """
    ✅ Preprocessing completed successfully

    Split: $(basename(proc_dir))

    Statistics:
    - Total images:        $(length(labeled_files))
    - Successfully copied: $n_success
    - Failed:              $(length(failed_copies))
    - Class 0:             $n_class0 ($(round(n_class0/length(labeled_files)*100, digits=1))%)
    - Class 1:             $n_class1 ($(round(n_class1/length(labeled_files)*100, digits=1))%)
    - Time taken:          $(round(elapsed, digits=1))s
    - Checksum enabled:    $enable_checksum
    - Checksum failures:   $(enable_checksum ? length(checksum_failures) : "N/A")

    Output directory: $proc_dir
    """
    send_email_alert("✅ Preprocessing Complete: $(basename(proc_dir))", completion_msg; level=:info)

    println("✅ Completed split: $(basename(proc_dir))")

    return (
        total             = length(labeled_files),
        success           = n_success,
        failed            = length(failed_copies),
        checksum_failures = length(checksum_failures),
        time              = elapsed
    )
end

# ========================================
# Run preprocessing
# ========================================

println("="^60)
println("🚀 DATA PREPROCESSING PIPELINE")
println("="^60)
println("Started at: $(now())")
println("="^60)

overall_start = now()

try
    train_stats = preprocess_split(
        DefaultConfig.RAW_TRAIN_IMAGES,
        DefaultConfig.TRAIN_CSV,
        DefaultConfig.PROC_TRAIN
    )

    test_stats = preprocess_split(
        DefaultConfig.RAW_TEST_IMAGES,
        DefaultConfig.TEST_CSV,
        DefaultConfig.PROC_TEST
    )

    overall_elapsed = Dates.value(now() - overall_start) / 1000

    println("\n" * "="^60)
    println("🎉 PREPROCESSING PIPELINE COMPLETE")
    println("="^60)
    println("Total time: $(round(overall_elapsed, digits=1))s")
    println("\nTrain split:")
    println("   Success: $(train_stats.success) / $(train_stats.total)")
    println("   Failed:  $(train_stats.failed)")
    println("\nTest split:")
    println("   Success: $(test_stats.success) / $(test_stats.total)")
    println("   Failed:  $(test_stats.failed)")
    println("="^60)

    total_success = train_stats.success + test_stats.success
    total_images  = train_stats.total   + test_stats.total
    total_failed  = train_stats.failed  + test_stats.failed

    final_msg = """
    ✅ Full preprocessing pipeline completed

    Overall Statistics:
    - Total images processed: $total_images
    - Successful: $total_success ($(round(total_success/total_images*100, digits=1))%)
    - Failed:     $total_failed
    - Total time: $(round(overall_elapsed, digits=1))s

    Train split: $(train_stats.success) / $(train_stats.total)
    Test split:  $(test_stats.success)  / $(test_stats.total)

    Output directories:
    - Train: $(DefaultConfig.PROC_TRAIN)
    - Test:  $(DefaultConfig.PROC_TEST)
    """
    send_email_alert("✅ Full Preprocessing Complete", final_msg; level=:info)

catch e
    error_msg = """
    ❌ CRITICAL: Preprocessing pipeline failed!

    Error: $e

    Stacktrace:
    $(sprint(showerror, e, catch_backtrace()))

    Pipeline halted. Manual intervention required.
    """
    @error error_msg
    send_email_alert("🚨 CRITICAL: Preprocessing Failed", error_msg; level=:critical)
    rethrow(e)
end
