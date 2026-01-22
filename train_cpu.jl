# train_cpu_only.jl - TRAINING ONLY WITH LEARNING RATE REFINEMENT
using Lux, LuxCore, NNlib
using Random, Statistics, Printf, Dates, JLD2
using ProgressBars
using Optimisers
using Zygote

# Include project modules
include("src/config/default_config.jl")
include("src/dataloaders/lux_dataloader.jl")
include("src/models/sar_cnn_model.jl")
using .DefaultConfig

function train_cpu_with_lr_schedule()
    println("="^60)
    println("SAR CNN CPU TRAINING - WITH LR SCHEDULE")
    println("="^60)
    
    # FIX RANDOM SEEDS
    Random.seed!(DefaultConfig.RNG_SEED)
    println("✅ Random seed fixed: ", DefaultConfig.RNG_SEED)
    
    # LOAD DATA
    println("\n📊 Loading data...")
    train_loader = DataLoader(DefaultConfig.PROC_TRAIN; batch_size=32, shuffle=true)
    
    println("   Training samples: ", length(train_loader.image_paths))
    
    # INITIALIZE MODEL
    println("\n🧠 Initializing model...")
    model = SAR_CNN()
    rng = Random.default_rng()
    ps, st = LuxCore.setup(rng, model)
    println("   Model parameters: ", LuxCore.parameterlength(ps))
    
    # ✨ REFINED OPTIMIZER SETTINGS
    println("\n⚙️  Setting up optimizer with learning rate schedule...")
    initial_lr = 5.0f-5  # ✨ REDUCED from 1e-4 to 5e-5 (50% reduction)
    println("   Initial learning rate: $initial_lr")
    println("   Schedule: Step decay (×0.5 every 3 epochs)")
    
    opt = Optimisers.Adam(initial_lr)
    opt_state = Optimisers.setup(opt, ps)
    
    # ✨ LEARNING RATE SCHEDULER
    function update_learning_rate(opt_state, epoch::Int, initial_lr::Float32)
        # Step decay: reduce LR by 50% every 3 epochs
        decay_factor = 0.5f0 ^ (floor(Int, (epoch - 1) / 3))
        new_lr = initial_lr * decay_factor
        
        # Recreate optimizer with new learning rate
        new_opt = Optimisers.Adam(new_lr)
        new_opt_state = Optimisers.setup(new_opt, ps)
        
        return new_opt_state, new_lr
    end
    
    # TRAINING LOOP
    println("\n🚀 Starting training...")
    num_epochs = 15  # ✨ Increased from 5 to 15 to benefit from LR decay
    
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
        for (X_batch, y_batch) in train_loader
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
    
    println("\n" * "="^60)
    println("Training completed!")
    println("Best training loss: $(round(best_train_loss, digits=4)) at epoch $best_epoch")
    println("="^60)
    
    return model, best_ps, best_st, best_train_loss
end

# Binary cross-entropy helper
function binary_cross_entropy(y_pred, y_true; ϵ=1.0f-8)
    y_true_2d = reshape(y_true, 1, :)
    loss = -mean(y_true_2d .* log.(y_pred .+ ϵ) .+ 
                 (1 .- y_true_2d) .* log.(1 .- y_pred .+ ϵ))
    return loss
end

# RUN TRAINING
println("Starting CPU training with learning rate schedule...")
model, ps, st, train_loss = train_cpu_with_lr_schedule()

println("\n" * "="^60)
println("✅ Training completed! Final training loss: $(round(train_loss, digits=4))")
println("Saving trained model to cpu_model_trained.jld2")
jldsave("cpu_model_trained.jld2"; model=model, ps=ps, st=st, train_loss=train_loss)
println("="^60)
