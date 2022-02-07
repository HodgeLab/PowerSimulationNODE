path = (joinpath(pwd(), "test-train-dir"))
try
    #Test with pure NODE and 0 feedback states passes 
    #TODO - need additional tests for ode_model = "vsm" and node_unobserved_states != 0 once these are enabled. 
    p = NODETrainParams(
        base_path = path,
        ode_model = "none",
        node_unobserved_states = 1, #1
        learn_initial_condition_unobserved_states = false,
        groupsize_faults = 2,
        verify_psid_node_off = false,
        maxiters = 12,
        optimizer_η = 0.001,
        node_input_scale = 1.0,
        training_groups = [
            (
                tspan = (0.0, 1.2),
                shoot_times = [0.2],
                multiple_shoot_continuity_term = 100,
                batching_sample_factor = 1.0,
            ),
            (
                tspan = (0.0, 2.6),
                shoot_times = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                multiple_shoot_continuity_term = 100,
                batching_sample_factor = 1.0,
            ),
        ],
        node_state_inputs = [],
        #= node_state_inputs = [
            ("gen1", :ir_filter),
            ("gen1", :ii_filter),
            ("gen1", :θ_pll),
            ("gen1", :ϕq_ic),
            ("gen1", :ϕd_ic),
        ], =#
    )
    status = train(p)
    @test status

finally
    @info("removing test files")
    #rm(path, force = true, recursive = true)
end
