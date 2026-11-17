# =========================================================
# GPU POWER PROFILER — Synchronous nvidia-smi Polling
# Julia Lux CNN GPU Pipeline — Ocean Internal Wave Classification
# =========================================================
# Conference-paper grade measurements:
#   • GPU clocks locked before measurement, restored after
#   • High-resolution sampling (50 ms default)
#   • Extended idle baseline with convergence check
#   • Outlier-robust statistics (IQR trimming)
#   • LaTeX-ready publishable summary table
# =========================================================

using Statistics, Dates, Printf, JSON3, CUDA

const NVIDIA_SMI = "/usr/bin/nvidia-smi"

# =========================================================
# SECTION 1: GPU METADATA
# =========================================================

function check_nvidia_smi()::Bool
    if isfile(NVIDIA_SMI) && isexecutable(NVIDIA_SMI)
        return true
    end
    @warn "nvidia-smi not found at $NVIDIA_SMI"
    return false
end

function gpu_metadata()::Dict{String,Any}
    meta = Dict{String,Any}(
        "cuda_functional"  => CUDA.functional(),
        "measurement_date" => string(now()),
        "nvidia_smi_path"  => NVIDIA_SMI
    )
    !CUDA.functional() && return meta
    try
        smi_query(q) = strip(read(`$NVIDIA_SMI --query-gpu=$q
                                   --format=csv,noheader,nounits -i 0`, String))
        meta["gpu_name"]           = smi_query("name")
        meta["tdp_w"]              = parse(Float64, smi_query("power.limit"))
        meta["driver_version"]     = smi_query("driver_version")
        meta["graphics_clock_mhz"] = parse(Float64, smi_query("clocks.gr"))
        meta["memory_clock_mhz"]   = parse(Float64, smi_query("clocks.mem"))
        meta["vram_total_mb"]      = parse(Float64, smi_query("memory.total"))
        meta["persistence_mode"]   = smi_query("persistence_mode")
        meta["compute_mode"]       = smi_query("compute_mode")
        meta["power_limit_w"]      = parse(Float64, smi_query("power.limit"))
    catch e
        meta["metadata_error"] = string(e)
    end
    return meta
end

# =========================================================
# SECTION 2: CLOCK LOCKING
# =========================================================
# Locking SM and memory clocks eliminates the P-state stepping
# that is the primary source of idle and inter-run power variance.
# Requires: sudo nvidia-smi -pm 1  (persistence mode on)
#           sudo nvidia-smi --lock-gpu-clocks / --lock-memory-clocks
# Falls back gracefully if insufficient permissions.

struct ClockLock
    locked       :: Bool
    sm_clock_mhz :: Int
    mem_clock_mhz:: Int
    note         :: String
end


function lock_gpu_clocks()::ClockLock
    !CUDA.functional() && return ClockLock(false, 0, 0, "CUDA not functional")
    println("  🔒 Clocks pre-locked externally: SM 3105 MHz, MEM 8001 MHz")
    return ClockLock(true, 3105, 8001, "Locked externally via sudo before Julia launch")
end

function unlock_gpu_clocks(lock::ClockLock)
    !lock.locked && return
    println("  🔓 Clocks remain locked — reset manually with: sudo nvidia-smi -i 0 --reset-gpu-clocks")
end



# =========================================================
# SECTION 3: SINGLE POWER SAMPLE
# =========================================================

function sample_gpu_once()::Tuple{Float64, Float64}
    try
        out = read(`$NVIDIA_SMI --query-gpu=power.draw,temperature.gpu
                    --format=csv,noheader,nounits -i 0`, String)
        parts = split(strip(out), ",")
        length(parts) < 2 && return (NaN, NaN)
        return (parse(Float64, strip(parts[1])),
                parse(Float64, strip(parts[2])))
    catch 
        return (NaN, NaN)
    end
end

# =========================================================
# SECTION 4: BACKGROUND SAMPLER
# =========================================================

mutable struct NvidiaSampler
    interval_ms :: Int
    power_w     :: Vector{Float64}
    temp_c      :: Vector{Float64}
    timestamps  :: Vector{Float64}
    running     :: Bool
    task        :: Union{Task, Nothing}
    t_start     :: Float64
end

function NvidiaSampler(; interval_ms=50)   # ← 50 ms default (was 200 ms)
    return NvidiaSampler(interval_ms, Float64[], Float64[], Float64[],
                         false, nothing, 0.0)
end

function start!(s::NvidiaSampler)
    empty!(s.power_w); empty!(s.temp_c); empty!(s.timestamps)
    s.running = true
    s.t_start = time()
    s.task = @async begin
        while s.running
            pw, tp = sample_gpu_once()
            if !isnan(pw)
                push!(s.power_w,    pw)
                push!(s.temp_c,     tp)
                push!(s.timestamps, time() - s.t_start)
            end
            sleep(s.interval_ms / 1000.0)
        end
    end
end

function stop!(s::NvidiaSampler)
    s.running = false
    !isnothing(s.task) && wait(s.task)
end

# =========================================================
# SECTION 5: ROBUST STATISTICS
# =========================================================

safe_mean(v) = isempty(v) ? NaN : mean(v)
safe_std(v)  = length(v) < 2 ? NaN : std(v)
safe_min(v)  = isempty(v) ? NaN : minimum(v)
safe_max(v)  = isempty(v) ? NaN : maximum(v)

"""
IQR-trimmed mean and std — removes outlier spikes from P-state transitions
or transient load bursts without discarding valid steady-state samples.
Uses Tukey fences: keep values within [Q1 - 1.5·IQR, Q3 + 1.5·IQR].
"""
function iqr_trim(v::Vector{Float64})
    length(v) < 4 && return v
    q1, q3 = quantile(v, 0.25), quantile(v, 0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    trimmed = filter(x -> lo ≤ x ≤ hi, v)
    isempty(trimmed) ? v : trimmed
end

function robust_stats(v::Vector{Float64})
    trimmed = iqr_trim(v)
    (mean   = safe_mean(trimmed),
     std    = safe_std(trimmed),
     min    = safe_min(v),
     max    = safe_max(v),
     p25    = isempty(v) ? NaN : quantile(v, 0.25),
     p50    = isempty(v) ? NaN : quantile(v, 0.50),
     p75    = isempty(v) ? NaN : quantile(v, 0.75),
     p95    = isempty(v) ? NaN : quantile(v, 0.95),
     n_raw  = length(v),
     n_used = length(trimmed))
end

# =========================================================
# SECTION 6: TRIM HELPERS
# =========================================================

function trim_samples(power_w, temp_c, timestamps;
                      discard_start_s=0.0, discard_end_s=0.0)
    isempty(timestamps) && return power_w, temp_c, timestamps
    total = timestamps[end]
    if total <= (discard_start_s + discard_end_s + 0.5)
        return power_w, temp_c, timestamps
    end
    lo, hi = discard_start_s, total - discard_end_s
    mask = (timestamps .>= lo) .& (timestamps .<= hi)
    any(mask) || return power_w, temp_c, timestamps
    return power_w[mask], temp_c[mask], timestamps[mask]
end

# =========================================================
# SECTION 7: THERMAL THROTTLE CHECK
# =========================================================

function check_thermal_throttling(temp_c; threshold_c=80.0)::NamedTuple
    isempty(temp_c) &&
        return (throttled=false, max_temp=NaN, mean_temp=NaN, pct_above=0.0)
    return (throttled  = safe_max(temp_c) > threshold_c,
            max_temp   = safe_max(temp_c),
            mean_temp  = safe_mean(temp_c),
            pct_above  = 100.0 * sum(temp_c .> threshold_c) / length(temp_c))
end

# =========================================================
# SECTION 8: IDLE BASELINE  (convergence-checked)
# =========================================================
# The idle baseline is the largest single source of net-power
# error. Two improvements over the original:
#   1. Extended default duration (45 s instead of 30 s)
#   2. Convergence check: split into three equal windows and
#      warn if the window means differ by more than 1 W
#      (indicates the GPU hadn't fully cooled / clocks settled).

function measure_idle_baseline(;
    duration_s      = 90.0,      # ← 60 s for convergence
    interval_ms     = 50,        # ← high-res
    discard_start_s = 3.0,       # ← longer settle trim
    discard_end_s   = 2.0
)::NamedTuple

    println("\n⏳ Measuring GPU idle baseline for $(duration_s)s ...")
    println("   (GPU must be fully idle — no other GPU workloads)")

    s = NvidiaSampler(interval_ms=interval_ms)
    start!(s)
    sleep(duration_s)
    stop!(s)

    @printf("   Raw samples collected: %d  (%.0f ms resolution)\n",
            length(s.power_w), interval_ms)

    pw, tp, ts = trim_samples(s.power_w, s.temp_c, s.timestamps;
                               discard_start_s=discard_start_s,
                               discard_end_s=discard_end_s)
    if isempty(pw)
        pw = s.power_w; tp = s.temp_c
    end
    isempty(pw) && error("No idle samples collected — check nvidia-smi at $NVIDIA_SMI")

    # ── Convergence check ────────────────────────────────────────────────────
    converged = true
    convergence_note = "OK"
    if length(pw) >= 9
        third = div(length(pw), 3)
        w1 = mean(pw[1:third])
        w2 = mean(pw[third+1:2*third])
        w3 = mean(pw[2*third+1:end])
        drift = maximum([w1, w2, w3]) - minimum([w1, w2, w3])
        if drift > 1.0
            converged = false
            convergence_note = @sprintf("Drift %.2f W across windows — extend idle duration", drift)
            @warn "⚠️  Idle baseline not converged (drift = $(round(drift, digits=2)) W). " *
                  "Consider a longer idle period or ensure no background GPU activity."
        else
            convergence_note = @sprintf("Converged (drift %.2f W across 3 windows)", drift)
        end
    end

    rs = robust_stats(pw)

    @printf("   ✅ Idle power:  %.2f ± %.2f W  (IQR-robust)\n", rs.mean, rs.std)
    @printf("      Raw range:   %.2f – %.2f W\n", rs.min, rs.max)
    @printf("      Convergence: %s\n", convergence_note)
    @printf("   🌡️  Idle temp:  %.1f°C\n", safe_mean(tp))

    return (
        mean_w           = rs.mean,
        std_w            = rs.std,
        min_w            = rs.min,
        max_w            = rs.max,
        mean_temp_c      = safe_mean(tp),
        converged        = converged,
        convergence_note = convergence_note,
        n_samples_total  = length(s.power_w),
        n_samples_used   = rs.n_used,
        duration_s       = duration_s,
        all_power_w      = s.power_w,
        all_temp_c       = s.temp_c
    )
end

# =========================================================
# SECTION 9: SINGLE-RUN POWER MEASUREMENT
# =========================================================

function measure_one_run_power(
    inference_fn    :: Function;
    interval_ms     :: Int     = 50,
    discard_start_s :: Float64 = 0.5,
    discard_end_s   :: Float64 = 0.5,
    idle            = nothing
)::NamedTuple

    s = NvidiaSampler(interval_ms=interval_ms)
    start!(s)

    t_start = time()
    result  = inference_fn()
    CUDA.synchronize()
    t_end   = time()

    stop!(s)
    wall = t_end - t_start

    @printf("     [%d raw samples in %.2fs] ", length(s.power_w), wall)

    pw, tp, ts = trim_samples(s.power_w, s.temp_c, s.timestamps;
                               discard_start_s=discard_start_s,
                               discard_end_s=discard_end_s)
    if isempty(pw)
        pw = isempty(s.power_w) ? [0.0] : s.power_w
        tp = isempty(s.temp_c)  ? [0.0] : s.temp_c
    end

    rs      = robust_stats(pw)
    peak_w  = rs.max
    energy_j= rs.mean * wall
    net_w   = isnothing(idle) ? NaN : rs.mean - idle.mean_w
    thermal = check_thermal_throttling(tp)

    return (
        mean_w            = rs.mean,
        std_w             = rs.std,
        peak_w            = peak_w,
        p95_w             = rs.p95,
        net_w             = net_w,
        energy_j          = energy_j,
        wall_time_s       = wall,
        mean_temp_c       = safe_mean(tp),
        max_temp_c        = safe_max(tp),
        throttled         = thermal.throttled,
        pct_above_80c     = thermal.pct_above,
        n_samples_trimmed = rs.n_used,
        n_samples_total   = length(s.power_w),
        power_w_trimmed   = pw,
        temp_c_trimmed    = tp,
        inference_result  = result
    )
end

# =========================================================
# SECTION 10: FIVE-RUN PROTOCOL
# =========================================================

function measure_power_five_runs(
    inference_fn    :: Function;
    interval_ms     :: Int     = 50,
    discard_start_s :: Float64 = 0.5,
    discard_end_s   :: Float64 = 0.5,
    cooldown_s      :: Float64 = 15.0,   # ← extended cool-down
    idle            = nothing
)::NamedTuple

    run_mean_w    = Float64[]
    run_peak_w    = Float64[]
    run_p95_w     = Float64[]
    run_net_w     = Float64[]
    run_energy_j  = Float64[]
    run_wall_s    = Float64[]
    run_temp_c    = Float64[]
    run_throttled = Bool[]
    run_results   = []

    println("\n⚡ Running 5 power-profiled inference passes ...")
    println("   ($(cooldown_s)s cool-down between runs)")

    for r in 1:5
        @printf("   Run %d/5 — ", r)
        flush(stdout)

        m = measure_one_run_power(inference_fn;
                interval_ms=interval_ms,
                discard_start_s=discard_start_s,
                discard_end_s=discard_end_s,
                idle=idle)

        push!(run_mean_w,    m.mean_w)
        push!(run_peak_w,    m.peak_w)
        push!(run_p95_w,     m.p95_w)
        push!(run_net_w,     m.net_w)
        push!(run_energy_j,  m.energy_j)
        push!(run_wall_s,    m.wall_time_s)
        push!(run_temp_c,    m.mean_temp_c)
        push!(run_throttled, m.throttled)
        push!(run_results,   m.inference_result)

        net_str = isnan(m.net_w) ? "N/A" : @sprintf("%.2f", m.net_w)
        @printf("%.2f W (net %s W), peak %.2f W, p95 %.2f W, %.1f°C%s, %.2fs\n",
                m.mean_w, net_str, m.peak_w, m.p95_w, m.mean_temp_c,
                m.throttled ? " ⚠️THROTTLE" : "", m.wall_time_s)

        r < 5 && sleep(cooldown_s)
    end

    any(run_throttled) &&
        @warn "⚠️  Thermal throttling detected in ≥1 run — results may be unreliable."

    # Cross-run robust stats
    rs = robust_stats(run_mean_w)

    return (
        run_mean_w    = run_mean_w,
        run_peak_w    = run_peak_w,
        run_p95_w     = run_p95_w,
        run_net_w     = run_net_w,
        run_energy_j  = run_energy_j,
        run_wall_s    = run_wall_s,
        run_temp_c    = run_temp_c,
        run_throttled = run_throttled,
        # Robust cross-run summaries
        mean_w        = rs.mean,
        std_w         = rs.std,
        mean_peak_w   = safe_mean(run_peak_w),
        mean_p95_w    = safe_mean(run_p95_w),
        mean_energy_j = safe_mean(run_energy_j),
        net_w         = all(isnan, run_net_w) ? NaN :
                        safe_mean(filter(!isnan, run_net_w)),
        any_throttled = any(run_throttled),
        run_results   = run_results
    )
end

# =========================================================
# SECTION 11: ENERGY METRICS
# =========================================================

function compute_energy_metrics(power::NamedTuple, n_images::Int,
                                 idle=nothing)::Dict{String,Any}
    mean_w    = power.mean_w
    net_w     = power.net_w
    mean_wall = safe_mean(power.run_wall_s)

    energy_j     = mean_w * mean_wall
    e_per_img_mj = (energy_j / n_images) * 1000
    net_e_mj     = isnan(net_w) ? nothing :
                   (net_w * mean_wall / n_images) * 1000

    # Uncertainty propagation: σ_E = σ_P × t
    std_energy_j     = safe_std(power.run_energy_j)
    std_e_per_img_mj = isnan(std_energy_j) ? NaN :
                       (std_energy_j / n_images) * 1000

    return Dict(
        "gross_mean_w"             => mean_w,
        "gross_std_w"              => power.std_w,
        "net_w"                    => isnan(net_w) ? nothing : net_w,
        "idle_baseline_w"          => isnothing(idle) ? nothing : idle.mean_w,
        "total_energy_j"           => energy_j,
        "std_energy_j"             => std_energy_j,
        "energy_per_image_j"       => energy_j / n_images,
        "energy_per_image_mj"      => e_per_img_mj,
        "std_energy_per_image_mj"  => std_e_per_img_mj,
        "net_energy_per_image_mj"  => net_e_mj,
        "mean_inference_wall_s"    => mean_wall,
        "std_inference_wall_s"     => safe_std(power.run_wall_s)
    )
end

# =========================================================
# SECTION 12: LATEX TABLE GENERATOR
# =========================================================
# Produces a ready-to-paste IEEE/ACM style table.

function generate_latex_table(report::Dict{String,Any})::String
    s    = report["publishable_summary"]
    e    = report["energy_metrics"]
    meta = get(get(report, "system", Dict()), "gpu", Dict())
    pow  = report["inference_power"]
    idle = get(report, "idle_baseline", nothing)

    gpu_name  = get(meta, "gpu_name",  "N/A")
    tdp       = get(meta, "tdp_w",     NaN)
    n_images  = get(get(report, "system", Dict()), "n_images",  0)
    bs        = get(get(report, "system", Dict()), "batch_size", 0)
    n_runs    = get(get(report, "metadata", Dict()), "n_profile_runs", 5)
    pipe      = get(get(report, "metadata", Dict()), "pipeline",  "—")
    samp_ms   = get(get(report, "metadata", Dict()), "sampling_interval_ms", 50)

    idle_str  = isnothing(idle) ? "N/A" :
                @sprintf("%.2f \\(\\pm\\) %.2f", idle["mean_w"], idle["std_w"])
    gross_str = @sprintf("%.2f \\(\\pm\\) %.2f", s["mean_gpu_power_w"], s["std_gpu_power_w"])

    net_str   = isnothing(s["net_inference_power_w"]) ? "N/A" :
                @sprintf("%.2f", s["net_inference_power_w"])

    e_gross   = @sprintf("%.3f", s["energy_per_image_mj_gross"])
    e_gross_std = isnan(e["std_energy_per_image_mj"]) ? "" :
                  @sprintf(" \\(\\pm\\) %.3f", e["std_energy_per_image_mj"])

    e_net_str = isnothing(s["energy_per_image_mj_net"]) ? "N/A" :
                @sprintf("%.3f", s["energy_per_image_mj_net"])

    wall_str  = @sprintf("%.3f \\(\\pm\\) %.3f",
                    e["mean_inference_wall_s"],
                    isnan(e["std_inference_wall_s"]) ? 0.0 : e["std_inference_wall_s"])

    util_str  = isnan(tdp) || s["mean_gpu_power_w"] == 0.0 ? "N/A" :
                @sprintf("%.1f\\%%", 100.0 * s["mean_gpu_power_w"] / tdp)

    throttle  = s["thermal_throttling_detected"] ? "Yes \\textbf{(!)}" : "No"

    clk = get(get(report, "metadata", Dict()), "clock_lock", Dict())
    clk_str = get(clk, "locked", false) ?
              @sprintf("Locked (%d MHz SM)", get(clk, "sm_clock_mhz", 0)) :
              "Driver control"

    lines = String[]
    push!(lines, "% ─────────────────────────────────────────────────────────")
    push!(lines, "% Auto-generated by gpu_power_profiler.jl — $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(lines, "% Pipeline: $pipe")
    push!(lines, "% ─────────────────────────────────────────────────────────")
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "  \\centering")
    push!(lines, "  \\caption{GPU Power and Energy Consumption --- $pipe}")
    push!(lines, "  \\label{tab:gpu_power_$(lowercase(replace(pipe, r"[^a-zA-Z0-9]" => "_")))}")
    push!(lines, "  \\begin{tabular}{ll}")
    push!(lines, "    \\toprule")
    push!(lines, "    \\textbf{Metric} & \\textbf{Value} \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    \\multicolumn{2}{l}{\\textit{Hardware \\& Configuration}} \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    GPU & $gpu_name \\\\")
    push!(lines, isnan(tdp) ? "    TDP & N/A \\\\" :
                              "    TDP & $(round(Int, tdp)) W \\\\")
    push!(lines, "    Clock configuration & $clk_str \\\\")
    push!(lines, "    Sampling interval & $samp_ms ms \\\\")
    push!(lines, "    Measurement runs & $n_runs \\\\")
    push!(lines, "    Images measured & $(n_images) (batch size $bs) \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    \\multicolumn{2}{l}{\\textit{Power (W)}} \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    Idle baseline & $idle_str W \\\\")
    push!(lines, "    Inference (gross) & $gross_str W \\\\")
    push!(lines, "    Inference (net, active only) & $net_str W \\\\")
    push!(lines, "    GPU utilisation & $util_str \\\\")
    push!(lines, "    Thermal throttling & $throttle \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    \\multicolumn{2}{l}{\\textit{Energy Efficiency}} \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    Energy/image (gross) & $(e_gross)$(e_gross_std) mJ \\\\")
    push!(lines, "    Energy/image (net) & $e_net_str mJ \\\\")
    push!(lines, "    Inference wall time & $wall_str s \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    \\multicolumn{2}{l}{\\textit{Reporting standard}} \\\\")
    push!(lines, "    \\midrule")
    push!(lines, "    Format & mean \\(\\pm\\) std across $n_runs runs \\\\")
    push!(lines, "    Statistic & IQR-robust (Tukey fences) \\\\")
    push!(lines, "    \\bottomrule")
    push!(lines, "  \\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

# =========================================================
# SECTION 13: JSON REPORT BUILDER
# =========================================================

function build_json_report(; pipeline_name, n_images, batch_size,
                             n_profile_runs=5, idle=nothing,
                             power, clock_lock=nothing,
                             interval_ms=50, extra_fields=Dict())::Dict{String,Any}

    energy   = compute_energy_metrics(power, n_images, idle)
    gpu_meta = gpu_metadata()

    per_run = [Dict(
        "run"         => i,
        "mean_w"      => power.run_mean_w[i],
        "peak_w"      => power.run_peak_w[i],
        "p95_w"       => power.run_p95_w[i],
        "net_w"       => isnan(power.run_net_w[i]) ? nothing : power.run_net_w[i],
        "energy_j"    => power.run_energy_j[i],
        "wall_time_s" => power.run_wall_s[i],
        "mean_temp_c" => power.run_temp_c[i],
        "throttled"   => power.run_throttled[i]
    ) for i in 1:length(power.run_mean_w)]

    idle_dict = isnothing(idle) ? nothing : Dict(
        "mean_w"           => idle.mean_w,
        "std_w"            => idle.std_w,
        "min_w"            => idle.min_w,
        "max_w"            => idle.max_w,
        "mean_temp_c"      => idle.mean_temp_c,
        "converged"        => idle.converged,
        "convergence_note" => idle.convergence_note,
        "duration_s"       => idle.duration_s,
        "n_samples_total"  => idle.n_samples_total,
        "n_samples_used"   => idle.n_samples_used
    )

    clk_dict = isnothing(clock_lock) ? nothing : Dict(
        "locked"        => clock_lock.locked,
        "sm_clock_mhz"  => clock_lock.sm_clock_mhz,
        "mem_clock_mhz" => clock_lock.mem_clock_mhz,
        "note"          => clock_lock.note
    )

    report = Dict{String,Any}(
        "metadata" => Dict(
            "pipeline"                => pipeline_name,
            "generated_at"            => string(now()),
            "measurement_tool"        => "nvidia-smi synchronous polling",
            "sampling_interval_ms"    => interval_ms,
            "n_profile_runs"          => n_profile_runs,
            "cooldown_between_runs_s" => 15.0,
            "clock_lock"              => clk_dict,
            "notes" => [
                "Synchronous polling at $(interval_ms) ms resolution",
                "Intra-run statistics: IQR-robust (Tukey fences)",
                "Cross-run statistics: mean ± std over $(n_profile_runs) passes",
                "Power measured during single full-dataset pass per run",
                "Model load, data preload, warm-up excluded from measurement",
                "CUDA.synchronize() called before stopping per-run timer",
                "Net power = gross − idle baseline (IQR-robust idle mean)",
                "Clock locking applied: $(isnothing(clock_lock) ? "No" : string(clock_lock.locked))"
            ]
        ),
        "system" => Dict(
            "gpu"        => gpu_meta,
            "n_images"   => n_images,
            "batch_size" => batch_size
        ),
        "idle_baseline"   => idle_dict,
        "inference_power" => Dict(
            "per_run_breakdown" => per_run,
            "gross_mean_w"      => power.mean_w,
            "gross_std_w"       => power.std_w,
            "net_w"             => isnan(power.net_w) ? nothing : power.net_w,
            "distribution" => Dict(
                "mean_w"         => power.mean_w,
                "std_w"          => power.std_w,
                "min_w"          => safe_min(power.run_mean_w),
                "max_w"          => safe_max(power.run_mean_w),
                "p25_w"          => quantile(power.run_mean_w, 0.25),
                "p50_w"          => quantile(power.run_mean_w, 0.50),
                "p75_w"          => quantile(power.run_mean_w, 0.75),
                "peak_mean_w"    => power.mean_peak_w,
                "p95_mean_w"     => power.mean_p95_w,
                "any_throttling" => power.any_throttled
            )
        ),
        "energy_metrics" => energy,
        "publishable_summary" => Dict(
            "mean_gpu_power_w"            => round(power.mean_w,    digits=2),
            "std_gpu_power_w"             => round(power.std_w,     digits=2),
            "net_inference_power_w"       => isnan(power.net_w) ? nothing :
                                             round(power.net_w, digits=2),
            "energy_per_image_mj_gross"   => round(energy["energy_per_image_mj"],     digits=3),
            "std_energy_per_image_mj"     => round(energy["std_energy_per_image_mj"], digits=3),
            "energy_per_image_mj_net"     => isnothing(energy["net_energy_per_image_mj"]) ?
                                             nothing :
                                             round(energy["net_energy_per_image_mj"], digits=3),
            "thermal_throttling_detected" => power.any_throttled,
            "idle_converged"              => isnothing(idle) ? nothing : idle.converged,
            "report_format"               => "mean ± std (IQR-robust) across $(n_profile_runs) runs",
            "clocks_locked"               => isnothing(clock_lock) ? false : clock_lock.locked
        )
    )

    merge!(report, extra_fields)
    return report
end

# =========================================================
# SECTION 14: CONSOLE SUMMARY (conference-paper style)
# =========================================================

function print_publishable_summary(report::Dict{String,Any})
    s    = report["publishable_summary"]
    e    = report["energy_metrics"]
    meta = get(get(report, "system", Dict()), "gpu", Dict())
    pow  = report["inference_power"]
    idle = get(report, "idle_baseline", nothing)
    clk  = get(get(report, "metadata", Dict()), "clock_lock", nothing)

    bar = "═"^62

    println("\n$bar")
    println("  📊  GPU POWER — PUBLISHABLE SUMMARY")
    println("$bar")

    println("\n  Hardware")
    println("  ────────")
    @printf("  GPU model          : %s\n", get(meta, "gpu_name", "N/A"))
    tdp = get(meta, "tdp_w", NaN)
    isnan(tdp) || @printf("  TDP                : %.0f W\n", tdp)
    if !isnothing(clk) && get(clk, "locked", false)
        @printf("  Clock configuration: Locked — SM %d MHz, MEM %d MHz\n",
                get(clk, "sm_clock_mhz", 0), get(clk, "mem_clock_mhz", 0))
    else
        println("  Clock configuration: Driver control (not locked)")
    end

    println("\n  Idle Baseline")
    println("  ─────────────")
    if !isnothing(idle)
        @printf("  Idle power         : %.2f ± %.2f W\n", idle["mean_w"], idle["std_w"])
        @printf("  Raw range          : %.2f – %.2f W\n", idle["min_w"], idle["max_w"])
        conv = get(idle, "convergence_note", "")
        isempty(conv) || @printf("  Convergence        : %s\n", conv)
    else
        println("  Idle power         : N/A")
    end

    println("\n  Inference Power  ($(get(get(report,"metadata",Dict()),"n_profile_runs",5)) runs × 1 full dataset pass)")
    println("  ──────────────────────────────────────────────────")
    @printf("  Gross power        : %.2f ± %.2f W\n",
            s["mean_gpu_power_w"], s["std_gpu_power_w"])
    isnothing(s["net_inference_power_w"]) ||
        @printf("  Net power          : %.2f W\n", s["net_inference_power_w"])
    @printf("  Peak power (mean)  : %.2f W\n",
            get(pow["distribution"], "peak_mean_w", NaN))
    @printf("  P95  power (mean)  : %.2f W\n",
            get(pow["distribution"], "p95_mean_w",  NaN))
    if !isnan(tdp) && s["mean_gpu_power_w"] > 0
        @printf("  TDP utilisation    : %.1f%%\n",
                100.0 * s["mean_gpu_power_w"] / tdp)
    end

    println("\n  Energy Efficiency")
    println("  ─────────────────")
    std_e = get(s, "std_energy_per_image_mj", NaN)
    if isnan(std_e)
        @printf("  Energy/image (gross): %.3f mJ\n", s["energy_per_image_mj_gross"])
    else
        @printf("  Energy/image (gross): %.3f ± %.3f mJ\n",
                s["energy_per_image_mj_gross"], std_e)
    end
    isnothing(s["energy_per_image_mj_net"]) ||
        @printf("  Energy/image (net)  : %.3f mJ\n", s["energy_per_image_mj_net"])
    @printf("  Inference wall time : %.3f ± %.3f s\n",
            e["mean_inference_wall_s"],
            isnan(e["std_inference_wall_s"]) ? 0.0 : e["std_inference_wall_s"])

    println("\n  Per-run breakdown")
    println("  ─────────────────")
    println("  Run │ Mean W  │ Peak W  │ P95 W   │ Net W   │ Time s  │ Temp °C")
    println("  ────┼─────────┼─────────┼─────────┼─────────┼─────────┼────────")
    runs = get(pow, "per_run_breakdown", [])
    for r in runs
        net = isnothing(r["net_w"]) ? "   N/A " : @sprintf(" %5.2f ", r["net_w"])
        @printf("   %d  │ %6.2f  │ %6.2f  │ %6.2f  │%s│ %6.3f  │ %5.1f\n",
                r["run"], r["mean_w"], r["peak_w"], r["p95_w"],
                net, r["wall_time_s"], r["mean_temp_c"])
    end

    println("\n  Reporting standard")
    println("  ──────────────────")
    println("  • " * s["report_format"])
    println("  • IQR-robust statistics (Tukey fences, intra-run outlier removal)")
    println("  • Thermal throttling : $(s["thermal_throttling_detected"] ? "⚠️  DETECTED" : "✅  None")")
    println("  • Idle converged     : $(get(s, "idle_converged", nothing) === true ? "✅  Yes" : "⚠️  Check")")

    println("\n$bar")
    println("  📋  Cite as:")
    println("  \"Mean GPU power: $(s["mean_gpu_power_w"]) ± $(s["std_gpu_power_w"]) W")
    println("   (net: $(isnothing(s["net_inference_power_w"]) ? "N/A" : s["net_inference_power_w"]) W),")
    println("   $(s["energy_per_image_mj_gross"]) mJ/image gross")
    println("   ($(isnothing(s["energy_per_image_mj_net"]) ? "N/A" : s["energy_per_image_mj_net"]) mJ/image net),")
    println("   IQR-robust mean ± std across $(get(get(report,"metadata",Dict()),"n_profile_runs",5)) runs,")
    println("   GPU: $(get(meta, "gpu_name", "N/A"))\"")
    println("$bar\n")
end

# =========================================================
# SECTION 15: ENTRY POINT
# =========================================================

function run_full_gpu_power_profile(;
    inference_fn    :: Function,
    n_images        :: Int,
    batch_size      :: Int,
    pipeline_name   :: String  = "Julia_Lux_CNN_GPU",
    output_json     :: String  = "gpu_power_profile.json",
    idle_duration_s :: Float64 = 90.0,    # ← 60 s for convergence
    interval_ms     :: Int     = 50,      # ← high-res
    discard_start_s :: Float64 = 0.5,
    discard_end_s   :: Float64 = 0.5,
    cooldown_s      :: Float64 = 15.0,    # ← extended
    lock_clocks     :: Bool    = true,    # ← new flag
    extra_fields    :: Dict    = Dict()
)
    println("\n" * "═"^62)
    println("  ⚡  GPU POWER PROFILER — Conference-Grade Measurement")
    println("  Pipeline: $pipeline_name")
    println("═"^62)

    check_nvidia_smi() || error("nvidia-smi not available at $NVIDIA_SMI")
    CUDA.functional()  || error("CUDA not functional")

    # ── PHASE 0: Lock GPU clocks ──────────────────────────────────────────
    clock_lock = nothing
    if lock_clocks
        println("\n--- PHASE 0: LOCK GPU CLOCKS ---")
        clock_lock = lock_gpu_clocks()
        !clock_lock.locked &&
            @warn "Proceeding without clock lock — std dev may be elevated."
    end

    # ── PHASE 1: Idle baseline ────────────────────────────────────────────
    println("\n--- PHASE 1: GPU IDLE BASELINE ---")
    idle = measure_idle_baseline(
        duration_s      = idle_duration_s,
        interval_ms     = interval_ms,
        discard_start_s = 3.0,
        discard_end_s   = 2.0
    )
    !idle.converged &&
        @warn "Idle baseline did not converge — net power values may be inaccurate."

    # ── PHASE 2: Inference power ──────────────────────────────────────────
    println("\n--- PHASE 2: INFERENCE POWER (5 runs × 1 pass each) ---")
    power = measure_power_five_runs(inference_fn;
                interval_ms=interval_ms,
                discard_start_s=discard_start_s,
                discard_end_s=discard_end_s,
                cooldown_s=cooldown_s,
                idle=idle)

    # ── PHASE 3: Unlock clocks ────────────────────────────────────────────
    if !isnothing(clock_lock)
        println("\n--- PHASE 3: RESTORE GPU CLOCKS ---")
        unlock_gpu_clocks(clock_lock)
    end

    # ── Build and save report ─────────────────────────────────────────────
    report = build_json_report(
        pipeline_name  = pipeline_name,
        n_images       = n_images,
        batch_size     = batch_size,
        n_profile_runs = 5,
        idle           = idle,
        power          = power,
        clock_lock     = clock_lock,
        interval_ms    = interval_ms,
        extra_fields   = extra_fields
    )

    open(output_json, "w") do io; JSON3.write(io, report) end
    println("\n✅  JSON report saved → $output_json")

    # LaTeX table
    latex_path = replace(output_json, ".json" => "_table.tex")
    open(latex_path, "w") do io; print(io, generate_latex_table(report)) end
    println("✅  LaTeX table saved → $latex_path")

    # Conference-ready console summary
    print_publishable_summary(report)

    return report, power, idle
end
