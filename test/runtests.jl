using Test

# Run all test files
@testset "IWS_sciml Tests" begin
    include("test_data_loader.jl")
    include("test_segmentation_model.jl")
    include("test_physics_losses.jl")
    include("test_training.jl")
    include("test_evaluation.jl")
end
