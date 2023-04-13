@testset "Unit test for inner loss function when the surrogate does not initialize successfuly " begin
    surrogate_solution =
        PowerSimulationNODE.PhysicalModel_solution([0.0, 0.1, 0.2], [], [1.0], true)
    real_current_subset = [0.0, 0.1, 0.2]
    imag_current_subset = [0.3, 0.4, 0.5]
    params = TrainParams()
    Random.seed!(params.rng_seed) #Seed call usually happens at start of train()
    opt_ix = 1
    loss_init, loss_dynamic = PowerSimulationNODE._inner_loss_function(
        surrogate_solution,
        real_current_subset,
        imag_current_subset,
        params,
        opt_ix,
    )
    @test loss_init == 0.0
    @test loss_dynamic == 1.0e9
end
