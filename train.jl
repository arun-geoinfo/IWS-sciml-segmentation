# train.jl - GPU-ENABLED TRAINING (exact replica of train_cpu.jl with GPU support)
using Lux, LuxCore, NNlib
using Random, Statistics, Printf, Dates, JLD2
using ProgressBars
using Optimisers
using Zygote
using LuxCUDA  # GPU support

# Include project modules
include("src/config/default_config.jl")
include("src/dataloaders/lux_dataloader.jl")
include("src/models/sar_cnn_model.jl")
using .DefaultConfig
using .DefaultConfig.DeviceConfig

function train_with_lr_schedule()
    # Print configuration
    DefaultConfig.print_config()
    
    println("="^60)
    println("SAR CNN TRAINING - WITH LR SCHEDULE")
    println("="^60)
    
    # FIX RANDOM SEEDS
    Random.seed!(DefaultConfig.RNG_SEED)
    println("✅ Random seed fixed: ", DefaultConfig.RNG_SEED)
    
    # LOAD DATA
    println("\n📊 Loading data...")
    train_loader = DataLoader(DefaultConfig.PROC_TRAIN; batch_size=DefaultConfig.BATCH_SIZE, shuffle=true)
    
    println("   Training samples: ", length(train_loader.image_paths))
    
    # INITIALIZE MODEL
    println("\n🧠 Initializing model...")
    model = SAR_CNN()
    rng = Random.default_rng()
    ps, st = LuxCore.setup(rng, model)
    
    # Transfer to device
    ps = ps |> DefaultConfig.DEVICE
    st = st |> DefaultConfig.DEVICE
    
    println("   Model parameters: ", LuxCore.parameterlength(ps))
    println("   Device: $(DefaultConfig.BACKEND)")
    
    # ✨ REFINED OPTIMIZER SETTINGS
    println("\n⚙️  Setting up optimizer with learning rate schedule...")
    
    initial_lr = DefaultConfig.LEARNING_RATE
    println("   Initial learning rate: $initial_lr")
    println("   Schedule: Step decay (×0.5 every 3 epochs)")
    
    opt = Optimisers.Adam(initial_lr)
    opt_state = Optimisers.setup(opt, ps)
    current_lr = initial_lr  
    
    # ✨ LEARNING RATE SCHEDULER
    function update_learning_rate(opt_state, epoch::Int, initial_lr::Float64)
        # Step decay: reduce LR by 50% every 3 epochs
        decay_factor = 0.5 ^ (floor(Int, (epoch - 1) / 3))
        new_lr = initial_lr * decay_factor
        
        # Recreate optimizer with new learning rate
        new_opt = Optimisers.Adam(new_lr)
        new_opt_state = Optimisers.setup(new_opt, ps)
        
        return new_opt_state, new_lr
    end
    
    # TRAINING LOOP
    println("\n🚀 Starting training...")

    training_start_time = time()
    
    num_epochs = DefaultConfig.EPOCHS
    
    best_train_loss = Inf
    best_ps = ps
    best_st = st
    best_epoch = 0
    
    for epoch in 1:num_epochs
        # ✨ UPDATE LEARNING RATE
        opt_state, current_lr = update_learning_rate(opt_state, epoch, initial_lr)
        
        println("\n" * "="^60)
        println("Epoch $epoch/$num_epochs (LR: $(round(current_lr, sigdigits=3)))")
        println("="^60)
        
        epoch_loss = 0.0f0
        batch_count = 0
        
        # Training batches
        prog = ProgressBar(train_loader)
        for (X_batch, y_batch) in prog
            # Hybrid mode: transfer to GPU if needed
            if DefaultConfig.COMPUTE_MODE == :hybrid
                X_batch = DeviceConfig.safe_gpu_transfer(X_batch, DefaultConfig.DEVICE, DefaultConfig.BACKEND)
                y_batch = DeviceConfig.safe_gpu_transfer(y_batch, DefaultConfig.DEVICE, DefaultConfig.BACKEND)
            end
            
            # Loss function closure
            function loss_function(ps_inner)
                y_pred, st_ = model(X_batch, ps_inner, st)
                loss_val = binary_cross_entropy(y_pred, y_batch)
                return loss_val, st_
            end
            
            # Gradient computation
            (loss, st_new), back = Zygote.pullback(loss_function, ps)
            gs = back((1.0f0, nothing))[1]
            
            # Parameter update
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
            st = st_new
            
            epoch_loss += loss
            batch_count += 1
        end
        
        avg_loss = epoch_loss / batch_count
        
        println("   Train Loss: $(round(avg_loss, digits=4))")
        
        # ✨ TRACK BEST MODEL (based on lowest training loss)
        if avg_loss < best_train_loss
            best_train_loss = avg_loss
            best_ps = ps
            best_st = st
            best_epoch = epoch
            println("   ⭐ New best training loss!")
        end
    end
    
    total_training_time = time() - training_start_time
    
    println("\n" * "="^60)
    println("⏱️  TRAINING TIME SUMMARY")
    println("="^60)
    println("   Total training time: $(round(total_training_time, digits=1)) seconds")
    println("   Training time: $(round(total_training_time/60, digits=1)) minutes")
    println("   Average time per epoch: $(round(total_training_time/num_epochs, digits=1)) seconds")
    println("="^60) 
    
    return model, best_ps, best_st, best_train_loss
end  # ⬅️ THIS WAS MISSING - ADD THIS LINE!

# Binary cross-entropy helper (GPU-safe version)
function binary_cross_entropy(y_pred, y_true; ϵ=1.0f-8)
    # Ensure y_true matches y_pred shape
    if size(y_true) != size(y_pred)
        y_true = reshape(y_true, size(y_pred))
    end
    
    # Clamp predictions to avoid log(0)
    y_pred_safe = clamp.(y_pred, ϵ, 1.0f0 - ϵ)
    
    loss = -mean(y_true .* log.(y_pred_safe) .+ 
                 (1.0f0 .- y_true) .* log.(1.0f0 .- y_pred_safe))
    return loss
end

# RUN TRAINING
println("Starting training with GPU support...")
model, ps, st, train_loss = train_with_lr_schedule()

# Transfer to CPU for saving
ps_cpu = ps |> cpu_device()
st_cpu = st |> cpu_device()

println("\n" * "="^60)
println("✅ Training completed! Final training loss: $(round(train_loss, digits=4))")
println("Saving trained model to model_trained.jld2")
jldsave("model_trained.jld2"; 
    model=model, 
    ps=ps_cpu, 
    st=st_cpu, 
    train_loss=train_loss,
    backend=DefaultConfig.BACKEND,
    compute_mode=DefaultConfig.COMPUTE_MODE
)
println("="^60)