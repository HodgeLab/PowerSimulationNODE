path = (joinpath(pwd(), "build-param-list-test-dir"))
!isdir(path) && mkdir(path)

try
    params_data = TrainParams[]
    no_change_params = Dict{Symbol, Any}()
    change_params = Dict{Symbol, Any}()

    #INDICATE CONSTANT, NON-DEFAULT PARAMETERS
    no_change_params[:maxiters] = 10
    no_change_params[:node_layers] = 2
    no_change_params[:node_unobserved_states] = 19
    no_change_params[:node_width] = 25

    #INDICATE PARAMETES TO ITERATE OVER COMBINATORIALLY 
    change_params[:optimizer_η] = [0.001, 0.0005]

    #SPECIAL HANDLING TO BUILD ITERATOR FOR TRAINING GROUPS 
    no_change_fields = Dict{Symbol, Any}()
    change_fields = Dict{Symbol, Any}()
    no_change_fields[:tspan] = (0.0, 1.0)
    no_change_fields[:training_groups] = 1
    no_change_fields[:shoot_times] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    change_fields[:multiple_shoot_continuity_term] =
        [(0.0, 100.0), (0.0, 10.0), (0.0, 1.0), (1000.0, 100.0), (100.0, 10.0), (10.0, 1.0)]
    change_fields[:batching_sample_factor] = [0.2, 0.4, 0.6, 0.8, 1.0]
    change_params[:training_groups] =
        build_training_groups_list(no_change_fields, change_fields)

    build_params_list!(params_data, no_change_params, change_params)
    @test length(params_data) == 60

finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
