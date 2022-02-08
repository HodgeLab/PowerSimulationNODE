"""
Returns a total loss after trying a 'Direct multiple shooting' on ODE data
and an array of predictions from each of the groups (smaller intervals).
In Direct Multiple Shooting, the Neural Network divides the interval into smaller intervals
and solves for them separately.
The default continuity term is 100, implying any losses arising from the non-continuity
of 2 different groups will be scaled by 100.
```julia
multiple_shoot(p, ode_data, tsteps, prob, loss_function, solver, group_size;
               continuity_term=100, kwargs...)
multiple_shoot(p, ode_data, tsteps, prob, loss_function, continuity_loss, solver, group_size;
               continuity_term=100, kwargs...)
```
Arguments:
  - `p`: The parameters of the Neural Network to be trained.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `prob`: ODE problem that the Neural Network attempts to solve.
  - `loss_function`: Any arbitrary function to calculate loss.
  - `continuity_loss`: Function that takes states ``\\hat{u}_{end}`` of group ``k`` and
  ``u_{0}`` of group ``k+1`` as input and calculates prediction continuity loss
  between them.
  If no custom `continuity_loss` is specified, `sum(abs, û_end - u_0)` is used.
  - `solver`: ODE Solver algorithm.
  - `group_size`: The group size achieved after splitting the ode_data into equal sizes.
  - `continuity_term`: Weight term to ensure continuity of predictions throughout
    different groups.
  - `kwargs`: Additional arguments splatted to the ODE solver. Refer to the
  [Local Sensitivity Analysis](https://diffeq.sciml.ai/dev/analysis/sensitivity/) and
  [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
Note:
The parameter 'continuity_term' should be a relatively big number to enforce a large penalty
whenever the last point of any group doesn't coincide with the first point of next group.
"""
#TODO - lots of the logic is going to be in this function, will be worthwile to make it as clean/well-named as possible 
function batch_multiple_shoot(
    θ_node,
    θ_u0,
    θ_observation,
    ode_data::AbstractArray,
    tsteps::AbstractArray,
    fault_data,
    loss_function,
    continuity_term::Real,
    solver::DiffEqBase.AbstractODEAlgorithm,
    shooting_ranges::AbstractArray,
    batching_factor::Float64,
    params::NODETrainParams,
    observation_function;
    kwargs...,
)
    prob = fault_data[:surr_problem]

    P = fault_data[:P]
    P.nn = θ_node
    p = vectorize(P)
    ranges_batch = batch_ranges(batching_factor, shooting_ranges)
    u0s = generate_initial_conditions(ode_data, params, θ_u0, ranges_batch, prob)
    @assert length(ranges_batch) == length(u0s)

    sols = [
        OrdinaryDiffEq.solve(
            OrdinaryDiffEq.remake(
                prob;
                p = p,
                tspan = (tsteps[first(rg)], tsteps[last(rg)]),
                u0 = u0s[i],
            ),
            solver;
            saveat = tsteps[rg],
            kwargs...,
        ) for (i, rg) in enumerate(ranges_batch)
    ]
    group_predictions = Array.(sols)
    group_observations = [
        observation_function(group_prediction, θ_observation) for
        group_prediction in group_predictions
    ]

    # Abort and return infinite loss if one of the integrations failed
    retcodes = [sol.retcode for sol in sols]
    if any(retcodes .!= :Success)
        return Inf, group_predictions
    end

    # Calculate multiple shooting loss
    loss = 0
    for (i, rg) in enumerate(ranges_batch)
        u = ode_data[:, rg]
        û = group_observations[i]
        loss += loss_function(u, û)
        if i > 1
            loss +=
                continuity_term * _default_continuity_loss(
                    group_predictions[i - 1][:, end],
                    group_predictions[i][:, 1],
                )  # Continuity loss for all states 
        end
    end
    t_predictions = [tsteps[r] for r in ranges_batch]
    return loss, group_predictions, group_observations, t_predictions
end

function batch_ranges(batching_factor::Float64, ranges)
    batch_ranges = []
    for rg in ranges
        n_points = length(rg) - 2
        n_choose = Int(floor(n_points * batching_factor))
        if n_choose < 1
            @error "Shooting nodes are too close togeter for data provided and batching factor"
        end
        batch_range = sort!(
            vcat(
                rg[1],
                StatsBase.sample(rg[2:(end - 1)], n_choose, replace = false),
                rg[end],
            ),
        )
        push!(batch_ranges, batch_range)
    end
    return batch_ranges
end

function generate_initial_conditions(ode_data, params, θ_u0, ranges_batch, prob)
    if params.learn_initial_condition_unobserved_states
        @assert size(θ_u0)[1] == (length(ranges_batch) * params.node_unobserved_states)
        u0s = [
            vcat(
                ode_data[:, first(rg)],
                θ_u0[((i - 1) * params.node_unobserved_states + 1):(i * params.node_unobserved_states)],
            ) for (i, rg) in enumerate(ranges_batch)
        ]
    else
        @assert size(θ_u0)[1] ==
                ((length(ranges_batch) - 1) * params.node_unobserved_states)
        u0s = [
            if i == 1
                vcat(
                    ode_data[:, first(rg)],
                    prob.u0[(end - (params.node_unobserved_states - 1)):end],
                )
            else
                vcat(
                    ode_data[:, first(rg)],
                    θ_u0[((i - 2) * params.node_unobserved_states + 1):((i - 1) * params.node_unobserved_states)],
                )
            end

            for (i, rg) in enumerate(ranges_batch)
        ]
    end
    return u0s
end

function shooting_ranges(tsteps::AbstractArray, shoot_times::AbstractArray)
    shoot_times = vcat(tsteps[1], shoot_times, tsteps[end])
    @assert all(in(tsteps).(shoot_times))
    shooting_indices = [indexin(x, tsteps)[1][2] for x in shoot_times]
    return [
        (shooting_indices[i]):(shooting_indices[i + 1]) for
        i in 1:(length(shooting_indices) - 1)
    ]
end

function _default_continuity_loss(û_end::AbstractArray, u_0::AbstractArray)
    return sum(abs, û_end - u_0)
end
