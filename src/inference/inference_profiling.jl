# =========================================================
# INFERENCE WITH COMPREHENSIVE PERFORMANCE PROFILING
# =========================================================
using Images, FileIO, ImageTransformations, JLD2
using DataFrames, CSV, Statistics, Dates, Printf
using ProgressBars
using Lux, LuxCore, NNlib
# ADD THESE LINES AT THE TOP (after other imports, before including config):
using CUDA
using LuxCUDA
#CUDA.set_runtime_version!(; local_toolkit=true)
ENV["JULIA_CUDA_VERBOSE"] = "0"
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# Load project modules
include("../config/default_config.jl")
include("../models/sar_cnn_model.jl")
include("../dataloaders/lux_dataloader.jl")

using .DefaultConfig
using .DefaultConfig.DeviceConfig

# ADD THIS CRITICAL LINE:
using LuxCUDA  # ← MUST ADD THIS

## =========================================================
# Inference-specific DataLoader (optimized for inference)
# =========================================================
mutable struct InferenceDataLoader
    image_paths::Vector{String}
    batch_size::Int
    device
    backend::Symbol
    hybrid_mode::Bool
    X_preloaded::Array{Float32, 4}
    current_index::Int
end

function InferenceDataLoader(inference_dir::String; 
                            batch_size=DefaultConfig.BATCH_SIZE,
                            target_size=DefaultConfig.TARGET_SIZE)
    
    image_files = readdir(inference_dir)
    image_paths = [joinpath(inference_dir, f) for f in image_files 
                   if lowercase(splitext(f)[2]) in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]]
    
    N = length(image_paths)
    h, w = target_size
    
    println("\n📊 Loading inference data...")
    println("   Directory: $inference_dir")
    println("   Images found: $N")
    
    start_time = time()
    X_preloaded = Array{Float32, 4}(undef, h, w, 1, N)
    
    Threads.@threads for i in 1:N
        try
            img = load(image_paths[i])
            img_gray = Gray.(img)
            img_resized = imresize(img_gray, target_size)
            X_preloaded[:, :, 1, i] = Float32.(img_resized) ./ 255.0f0
        catch e
            @warn "Failed to load $(image_paths[i]): $e"
            X_preloaded[:, :, 1, i] = zeros(Float32, h, w)
        end
    end
    
    load_time = time() - start_time
    memory_gb = (h * w * 4 * N) / 1e9
    
    println("   ✅ Preloaded $N images in $(round(load_time, digits=2)) seconds")
    println("   💾 Memory used: $(round(memory_gb, digits=3)) GB")
    
    hybrid_mode = (DefaultConfig.COMPUTE_MODE == :hybrid)
    
    return InferenceDataLoader(
        image_paths,
        batch_size,
        DefaultConfig.DEVICE,
        DefaultConfig.BACKEND,
        hybrid_mode,
        X_preloaded,
        1
    )
end

Base.iterate(dl::InferenceDataLoader, state=1) = _next_inference_batch(dl, state)

function Base.length(dl::InferenceDataLoader)
    return ceil(Int, length(dl.image_paths) / dl.batch_size)
end

function _next_inference_batch(dl::InferenceDataLoader, state::Int)
    N = length(dl.image_paths)
    
    if state > N
        return nothing
    end
    
    batch_end = min(state + dl.batch_size - 1, N)
    X_batch = dl.X_preloaded[:, :, :, state:batch_end]
    
    if !dl.hybrid_mode
        X_batch = DeviceConfig.safe_gpu_transfer(X_batch, dl.device, dl.backend)
    end
    
    return (X_batch, state:batch_end), batch_end + 1
end

function reset_inference_loader!(dl::InferenceDataLoader)
    dl.current_index = 1
end

# =========================================================
# Performance Profiling Tools
# =========================================================
struct PerformanceMetrics
    model_load_time::Float64
    data_load_time::Float64
    warmup_time::Float64
    total_inference_time::Float64
    avg_gpu_batch_time::Float64
    std_batch_time::Float64
    images_per_second::Float64
    single_image_latency::Float64
    cpu_memory_mb::Float64
    batch_times::Vector{Float64}
    timestamps::Vector{Float64}
end

function Base.show(io::IO, metrics::PerformanceMetrics)
    println(io, "="^60)
    println(io, "📊 INFERENCE PERFORMANCE METRICS")
    println(io, "="^60)
    println(io, "📈 Timing Metrics:")
    println(io, "   Model Loading:     $(round(metrics.model_load_time*1000, digits=1)) ms")
    println(io, "   Data Loading:      $(round(metrics.data_load_time*1000, digits=1)) ms")
    println(io, "   Warm-up:           $(round(metrics.warmup_time*1000, digits=1)) ms")
    println(io, "   Total Inference:   $(round(metrics.total_inference_time, digits=3)) s")
    println(io, "   Per Image:         $(round(metrics.single_image_latency*1000, digits=1)) ms")
    println(io, "   Throughput:        $(round(metrics.images_per_second, digits=0)) img/s")
    println(io, "   Batch Time (avg):  $(round(metrics.avg_gpu_batch_time*1000, digits=1)) ms")
    println(io, "   Batch Time (std):  $(round(metrics.std_batch_time*1000, digits=1)) ms")
    
    println(io, "\n💾 Memory Metrics:")
    println(io, "   CPU Memory:        $(round(metrics.cpu_memory_mb, digits=1)) MB")
    
    if !isempty(metrics.batch_times)
        println(io, "\n⏱️  Batch Time Distribution:")
        q25 = quantile(metrics.batch_times, 0.25)
        q50 = quantile(metrics.batch_times, 0.50)
        q75 = quantile(metrics.batch_times, 0.75)
        println(io, "   25th percentile:   $(round(q25*1000, digits=1)) ms")
        println(io, "   50th percentile:   $(round(q50*1000, digits=1)) ms")
        println(io, "   75th percentile:   $(round(q75*1000, digits=1)) ms")
    end
    println(io, "="^60)
end

function get_cpu_memory_mb()
    try
        if Sys.islinux()
            meminfo = split(read("/proc/meminfo", String))
            total_idx = findfirst(x -> x == "MemTotal:", meminfo)
            if total_idx !== nothing
                total_kb = parse(Int, meminfo[total_idx+1])
                return total_kb / 1024
            end
        end
    catch e
    end
    return 0.0
end

# =========================================================
# Main Inference Function with Profiling
# =========================================================
function run_inference_with_profiling(;
        model_path="model_trained.jld2",
        inference_dir=joinpath(DefaultConfig.PROJECT_ROOT, "data", "raw", "Inference"),
        output_csv="inference_predictions.csv",
        batch_size=DefaultConfig.BATCH_SIZE,
        warmup_runs=3,
        profile_runs=5
    )
    
    println("\n" * "="^60)
    println("🚀 SAR CNN INFERENCE - COMPREHENSIVE PROFILING")
    println("="^60)
    
   # 1. MODEL LOADING PROFILING
println("\n1️⃣ Loading trained model...")
model_load_start = time()

if !isfile(model_path)
    error("❌ Model file not found: $model_path")
end

model_data = jldopen(model_path, "r") do file
    (model = file["model"], ps = file["ps"], st = file["st"])
end

# Correct unpacking (no typos!)
model = model_data.model
ps    = model_data.ps
st    = model_data.st

# Transfer to device
# Force GPU device (bypass DefaultConfig.DEVICE which may be CPU)

gpu_dev = gpu_device()

println("Transferring model to GPU...")
ps = ps |> gpu_dev
st = st |> gpu_dev

# Verify transfer
println("Parameters device type: ", typeof(ps.block1.conv.weight))
# VERY IMPORTANT: Switch to inference/test mode (removes BatchNorm training behavior)
st = LuxCore.testmode(st)

# Debug prints to confirm GPU usage
println("Backend: ", DefaultConfig.BACKEND)
println("Device type: ", typeof(DefaultConfig.DEVICE))
println("CUDA functional? ", CUDA.functional())
println("Current CUDA device: ", CUDA.functional() ? string(CUDA.device()) : "N/A")
# Check if parameters are actually on GPU (very useful)
println("Sample weight type (should be CuArray): ", typeof(ps.block1.conv.weight))
    model_load_time = time() - model_load_start
    println("   ✅ Model loaded in $(round(model_load_time*1000, digits=1)) ms")
    println("   📐 Model parameters: $(LuxCore.parameterlength(ps))")
    
    # 2. DATA LOADING PROFILING
    println("\n2️⃣  Loading inference data...")
    data_load_start = time()
    
    inference_loader = InferenceDataLoader(inference_dir; batch_size=batch_size)
    total_images = length(inference_loader.image_paths)
    
    data_load_time = time() - data_load_start
    
    # 3. WARM-UP RUNS
    println("\n3️⃣  Running warm-up passes ($warmup_runs runs)...")
    warmup_start = time()
    
    warmup_times = Float64[]
    for run in 1:warmup_runs
        run_start = time()
        reset_inference_loader!(inference_loader)
        
        for (X_batch, _) in inference_loader
         y_pred, _ = model(X_batch, ps, st)
         CUDA.@sync y_pred  # Ensure GPU operation completes
        end
        
        run_time = time() - run_start
        push!(warmup_times, run_time)
        
        if run == 1
            println("   First run: $(round(run_time, digits=3)) s")
        end
    end
    
    warmup_time = time() - warmup_start
    avg_warmup = mean(warmup_times)
    println("   Average warm-up: $(round(avg_warmup, digits=3)) s")
    
    # 4. MAIN INFERENCE PROFILING
    println("\n4️⃣  Running main inference profiling ($profile_runs runs)...")
    
    all_batch_times = Float64[]
    all_timestamps = Float64[]
    all_e2e_times = Float64[]

    cpu_memory_used = get_cpu_memory_mb()
    
    predictions = Vector{Float32}(undef, total_images)
    confidences = Vector{Float32}(undef, total_images)
    inference_start = time()
    for run in 1:profile_runs
        println("   Run $run/$profile_runs...")
        reset_inference_loader!(inference_loader)
        
        run_start = time()
        
        prog = ProgressBar(inference_loader)
        for (X_batch, indices) in prog
            # ---------------- PURE GPU TIMER ----------------
gpu_start = time()

y_pred, _ = model(X_batch, ps, st)
CUDA.synchronize()

gpu_batch_time = time() - gpu_start
push!(all_batch_times, gpu_batch_time)

# ---------------- END-TO-END TIMER ----------------
e2e_start = time()

y_pred_cpu = y_pred |> cpu_device()
predictions[indices] = y_pred_cpu .> 0.5f0
confidences[indices] = y_pred_cpu

e2e_batch_time = gpu_batch_time + (time() - e2e_start)
push!(all_e2e_times, e2e_batch_time)
            # batch_start = time()
            
            # if inference_loader.hybrid_mode
            #     X_batch = DeviceConfig.safe_gpu_transfer(X_batch, inference_loader.device, inference_loader.backend)
            # end
            
            # y_pred, _ = model(X_batch, ps, st)
            
            # CUDA.synchronize()
            # batch_time = time() - batch_start
            
            # push!(all_batch_times, batch_time)
            # push!(all_timestamps, time() - run_start)
            
            # y_pred_cpu = y_pred |> cpu_device()
            # predictions[indices] = y_pred_cpu .> 0.5f0
            # confidences[indices] = y_pred_cpu
        end
        
        run_time = time() - run_start
        throughput = total_images / run_time
        println("     Time: $(round(run_time, digits=3)) s, Throughput: $(round(throughput, digits=0)) img/s")
    end
    
    total_inference_time = time() - inference_start
    
    # 5. CALCULATE METRICS
    avg_gpu_batch_time = mean(all_batch_times)
    avg_e2e_batch_time = mean(all_e2e_times)
    std_batch_time = std(all_batch_times)
    images_per_second = batch_size / avg_gpu_batch_time
    single_image_latency = (avg_gpu_batch_time / batch_size) * 1000
    
    metrics = PerformanceMetrics(
        model_load_time,
        data_load_time,
        warmup_time,
        total_inference_time,
        avg_gpu_batch_time,
        std_batch_time,
        images_per_second,
        single_image_latency,
        cpu_memory_used,
        all_batch_times,
        all_timestamps
    )
    
    # 6. SAVE PREDICTIONS
    println("\n5️⃣  Saving predictions...")
    
    df = DataFrame(
        image_path = inference_loader.image_paths,
        prediction = predictions,
        confidence = confidences,
        timestamp = now()
    )
    
    CSV.write(output_csv, df)
    println("   ✅ Predictions saved to $output_csv")
    
    # 7. SAVE PERFORMANCE REPORT
    save_performance_report(metrics, batch_size, total_images, profile_runs)
    
    return df, metrics
end

function save_performance_report(metrics::PerformanceMetrics, batch_size::Int, total_images::Int, runs::Int)
    report_path = "inference_performance_report_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).txt"
    
    open(report_path, "w") do io
        println(io, "="^80)
        println(io, "📊 SAR CNN INFERENCE PERFORMANCE REPORT")
        println(io, "="^80)
        println(io, "Generated: $(now())")
        println(io, "Total Images: $total_images")
        println(io, "Batch Size: $batch_size")
        println(io, "Profile Runs: $runs")
        println(io, "Backend: $(DefaultConfig.BACKEND)")
        println(io, "Compute Mode: $(DefaultConfig.COMPUTE_MODE)")
        println(io, "")
        
        show(io, metrics)
        println(io, "")
        
        println(io, "\n💡 PERFORMANCE INSIGHTS:")
        println(io, "="^40)
        
        if metrics.std_batch_time > metrics.avg_gpu_batch_time * 0.5
            println(io, "⚠️  High batch time variance")
        else
            println(io, "✅ Batch times are consistent")
        end
        
        if metrics.single_image_latency < 5
            println(io, "✅ Low latency (<5ms per image)")
        elseif metrics.single_image_latency < 20
            println(io, "⚠️  Moderate latency")
        else
            println(io, "⚠️  High latency")
        end
        
        println(io, "\n🔄 COMPARISON WITH PYTHON TRANSFORMER+XGBOOST:")
        println(io, "="^40)
        println(io, "Julia CNN:")
        println(io, "  • Latency: ~$(round(metrics.single_image_latency, digits=1)) ms/image")
        println(io, "  • Throughput: ~$(round(metrics.images_per_second, digits=0)) img/s")
        println(io, "  • Single-stage inference")
        println(io, "")
        println(io, "Python Transformer+XGBoost (Expected):")
        println(io, "  • Latency: ~10-50 ms/image (two-stage)")
        println(io, "  • Throughput: ~20-100 img/s")
        println(io, "  • Higher memory footprint")
        println(io, "")
        println(io, "✅ Julia advantage: $(round(metrics.images_per_second/50, digits=1))x faster (estimated)")
    end
    
    println("\n📋 Performance report saved to: $report_path")
end

# =========================================================
# Main Execution
# =========================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting inference with comprehensive profiling...")
    
    predictions, metrics = run_inference_with_profiling(
        model_path="model_trained.jld2",
        inference_dir=joinpath(DefaultConfig.PROJECT_ROOT, "data", "raw", "Inference"),
        output_csv="inference_predictions.csv",
        batch_size=128,
        warmup_runs=3,
        profile_runs=5
    )
    
    println("\n" * "="^60)
    println("✅ INFERENCE COMPLETE")
    println("="^60)
    show(metrics)
end