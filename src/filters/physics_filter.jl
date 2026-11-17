module PhysicsFilter

using FFTW
using ImageFiltering
using ImageEdgeDetection
using Radon
using ImageMorphology: label_components
using Statistics

export apply_physics_filter, PhysicsFilterParams

# ============================================================================
# Parameters struct – holds all thresholds
# ============================================================================
struct PhysicsFilterParams
    min_wavelength_pixels::Int      # e.g., 20
    max_wavelength_pixels::Int      # e.g., 100
    radon_anisotropy_threshold::Float32  # e.g., 3.0
    min_edge_component_area::Int    # e.g., 500
    min_passes::Int                 # e.g., 2
end

# Default parameters (you will replace these after calibration)
function default_params()
    return PhysicsFilterParams(
        min_wavelength_pixels = 20,
        max_wavelength_pixels = 100,
        radon_anisotropy_threshold = 3.0f0,
        min_edge_component_area = 500,
        min_passes = 2
    )
end

# ============================================================================
# Filter 1: FFT Peak Wavelength Test
# ============================================================================
function fft_peak_wavelength_test(img::AbstractMatrix{Float32}, params::PhysicsFilterParams)
    h, w = size(img)
    
    # 2D FFT
    fft_img = abs.(fftshift(fft(img)))
    
    # Radial average
    center_h, center_w = h÷2 + 1, w÷2 + 1
    max_radius = min(h, w) ÷ 2
    
    radial_profile = zeros(Float32, max_radius)
    radius_counts = zeros(Int, max_radius)
    
    for i in 1:h, j in 1:w
        r = round(Int, sqrt((i - center_h)^2 + (j - center_w)^2))
        if r >= 1 && r <= max_radius
            radial_profile[r] += fft_img[i, j]
            radius_counts[r] += 1
        end
    end
    
    for r in 1:max_radius
        if radius_counts[r] > 0
            radial_profile[r] /= radius_counts[r]
        end
    end
    
    # Find peak (skip DC at r=1)
    peak_idx = argmax(radial_profile[2:end]) + 1
    wavelength_pixels = size(img, 1) / peak_idx
    
    # Check if within internal wave band
    passes = (wavelength_pixels >= params.min_wavelength_pixels) &&
             (wavelength_pixels <= params.max_wavelength_pixels)
    
    return passes, wavelength_pixels, peak_idx
end

# ============================================================================
# Filter 2: Radon Anisotropy Test
# ============================================================================
function radon_anisotropy_test(img::AbstractMatrix{Float32}, params::PhysicsFilterParams)
    theta = 0:1:179
    R = radon(img, theta)
    var_per_angle = var(R, dims=1)[:]
    anisotropy = maximum(var_per_angle) / (mean(var_per_angle) + 1e-7f0)
    passes = anisotropy >= params.radon_anisotropy_threshold
    return passes, anisotropy
end

# ============================================================================
# Filter 3: Edge Continuity Test
# ============================================================================
function edge_continuity_test(img::AbstractMatrix{Float32}, params::PhysicsFilterParams)
    edges = canny(img, 1.5, 0.1, 0.2)
    labels = label_components(edges)
    
    if maximum(labels) == 0
        return false, 0
    end
    
    component_areas = [count(labels .== i) for i in 1:maximum(labels)]
    max_area = maximum(component_areas)
    passes = max_area >= params.min_edge_component_area
    return passes, max_area
end

# ============================================================================
# Main Filter Function
# ============================================================================
function apply_physics_filter(
    img::AbstractMatrix{Float32},
    params::PhysicsFilterParams = default_params()
)
    # Filter 1: FFT wavelength
    fft_pass, wavelength, peak_idx = fft_peak_wavelength_test(img, params)
    
    # Filter 2: Radon anisotropy
    radon_pass, anisotropy = radon_anisotropy_test(img, params)
    
    # Filter 3: Edge continuity
    edge_pass, max_edge_area = edge_continuity_test(img, params)
    
    # Voting
    passes = [fft_pass, radon_pass, edge_pass]
    total_passes = sum(passes)
    final_decision = total_passes >= params.min_passes
    
    details = Dict(
        "fft_pass" => fft_pass,
        "fft_wavelength_pixels" => wavelength,
        "fft_peak_idx" => peak_idx,
        "radon_pass" => radon_pass,
        "radon_anisotropy" => anisotropy,
        "edge_pass" => edge_pass,
        "edge_max_area" => max_edge_area,
        "total_passes" => total_passes,
        "min_passes_required" => params.min_passes
    )
    
    return final_decision, details
end

# ============================================================================
# Batch Processing Helper
# ============================================================================
function batch_filter(
    images::Array{Float32, 4},
    params::PhysicsFilterParams = default_params()
)
    B = size(images, 4)
    decisions = Vector{Bool}(undef, B)
    all_details = Vector{Dict}(undef, B)
    
    for b in 1:B
        img = dropdims(images[:, :, :, b], dims=3)
        decisions[b], all_details[b] = apply_physics_filter(img, params)
    end
    
    return decisions, all_details
end

end  # module
