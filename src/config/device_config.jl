module DeviceConfig

using Lux, LuxCore

# ========================================
# GPU Backend Detection with Top-Level Imports
# ========================================

# Check CUDA availability at module load time
const CUDA_AVAILABLE = try
    @eval Main using CUDA
    Main.CUDA.functional()
catch
    false
end

# Check AMDGPU availability at module load time
const AMDGPU_AVAILABLE = try
    @eval Main using AMDGPU
    Main.AMDGPU.functional()
catch
    false
end

# Check Metal availability at module load time
const METAL_AVAILABLE = try
    @eval Main using Metal
    Main.Metal.functional()
catch
    false
end

# ========================================
# Device Detection Function
# ========================================
function detect_gpu_backend()
    if CUDA_AVAILABLE
        return :cuda
    elseif AMDGPU_AVAILABLE
        return :amdgpu
    elseif METAL_AVAILABLE
        return :metal
    else
        return :cpu
    end
end

# ========================================
# Device Selection Logic
# ========================================
"""
    get_device(mode::Symbol)

Get compute device based on user selection.

# Modes
- `:auto` - Auto-detect GPU, fallback to CPU
- `:cpu` - Force CPU
- `:gpu` - Force GPU (error if unavailable)
- `:hybrid` - Data on CPU, model on GPU

# Returns
- `device` - Lux device object
- `backend` - Symbol (:cuda, :amdgpu, :metal, :cpu)
"""
function get_device(mode::Symbol=:auto)
    backend = detect_gpu_backend()
    
    if mode == :cpu
        println("🖥️  COMPUTE MODE: CPU (forced)")
        return cpu_device(), :cpu
        
    elseif mode == :gpu
        if backend == :cpu
            error("❌ GPU mode requested but no GPU available!")
        end
        println("🚀 COMPUTE MODE: GPU-Only ($backend)")
        return gpu_device(), backend
        
    elseif mode == :auto
        if backend == :cpu
            println("🖥️  COMPUTE MODE: CPU (no GPU detected)")
            return cpu_device(), :cpu
        else
            println("🚀 COMPUTE MODE: GPU with CPU fallback ($backend)")
            return gpu_device(), backend
        end
        
    elseif mode == :hybrid
        if backend == :cpu
            @warn "Hybrid mode requested but no GPU available, falling back to CPU"
            return cpu_device(), :cpu
        end
        println("⚡ COMPUTE MODE: Hybrid (data=CPU, model=GPU)")
        return gpu_device(), backend
        
    else
        error("Invalid mode: $mode. Use :auto, :cpu, :gpu, or :hybrid")
    end
end

# ========================================
# Safe GPU Transfer with Fallback
# ========================================
"""
    safe_gpu_transfer(data, device, backend)

Transfer data to GPU with automatic fallback on OOM errors.
"""
function safe_gpu_transfer(data, device, backend)
    if backend == :cpu
        return data
    end
    
    try
        return data |> device
    catch e
        if isa(e, OutOfMemoryError) || contains(string(e), "out of memory")
            @warn "⚠️  GPU out of memory! Falling back to CPU for this batch"
            return data
        else
            rethrow(e)
        end
    end
end

end