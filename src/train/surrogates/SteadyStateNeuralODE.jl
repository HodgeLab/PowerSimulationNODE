using Flux
abstract type SteadyStateNeuralODELayer <: Function end
Flux.trainable(m::SteadyStateNeuralODELayer) = (p = m.p,)

#TODO - update docstring for the layer
#TODO - split SteadyStateNeuralODE and SteadyStateNeuralODEObs into two different layers? 
"""
Constructs a custom surrogate layer designed for dynamic power systems simulation.
The layer contains three models with distinct parameters.
- An initializer (`h`) parameterized by `α` (predicts initial conditions of hidden states from power flow solution).
- A NODE (`f`) parameterized by `θ` (describes the dynamics in the hidden space).
- An observation (`g`) parameterized by `ϕ` (predicts output quantity of interest from hidden space).
```julia
PowerSystemSurr(model1, model2, model3, exogenous, pf, tspan, alg=nothing,args...;kwargs...)
```
Arguments:
- `model1`: A Chain neural network that defines the initializer. Takes 1 input 
- `model2`: A Chain neural network that defines the neural ODE. Takes 2 input
- `model3`: A Chain neural network that defines the observation. Takes 1 input
- `exogenous` : An exogenous input `x(y,t)` where y is the model output.
- `pf` : The power flow solution (V*, I*)
- `tspan`: The timespan to be solved on
- `alg`: The algorithm used to solve the ODE. Defaults to `nothing`, i.e. the
  default algorithm from DifferentialEquations.jl.
- `sensealg`: The choice of differentiation algorthm used in the backpropogation.
  Defaults to an adjoint method.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
  [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
"""
struct SteadyStateNeuralODE{PT, PF, M, RE, M2, RE2, M3, RE3, IN, TN, TNI, RF, RFI, PM} <:
       SteadyStateNeuralODELayer
    p_train::PT
    p_fixed::PF
    len::Int        #length of p1 
    len2::Int       #length of p2 
    model1::M       #Initializer model 
    re1::RE
    model2::M2      #Dynamics model
    re2::RE2
    model3::M3      #Observation model 
    re3::RE3
    input_normalization::IN
    target_normalization::TN
    target_normalization_inverse::TNI
    ref_frame::RF
    ref_frame_inverse::RFI
    p_map::PM

    function SteadyStateNeuralODE(
        model1,
        model2,
        model3,
        input_normalization,
        target_normalization,
        target_normalization_inverse,
        ref_frame,
        ref_frame_inverse;
        p = nothing,
    )           #This is an inner constructor 
        p1, re1 = Flux.destructure(model1)
        p2, re2 = Flux.destructure(model2)
        p3, re3 = Flux.destructure(model3)
        if p === nothing
            p = [p1; p2; p3]
        end
        new{
            typeof(p),
            typeof(p),
            typeof(model1),
            typeof(re1),
            typeof(model2),
            typeof(re2),
            typeof(model3),
            typeof(re3),      #The type of len and len2 (Int) is automatically derived: https://docs.julialang.org/en/v1/manual/constructors/
            typeof(input_normalization),
            typeof(target_normalization),
            typeof(target_normalization_inverse),
            typeof(ref_frame),
            typeof(ref_frame_inverse),
            Vector{Int64},
        }(
            p,
            [],
            length(p1),
            length(p2),
            model1,
            re1,
            model2,
            re2,
            model3,
            re3,
            input_normalization,
            target_normalization,
            target_normalization_inverse,
            ref_frame,
            ref_frame_inverse,
            1:length(p),
        )
    end
end

Flux.@functor SteadyStateNeuralODE
Flux.trainable(m::SteadyStateNeuralODE) = (p = m.p_train,)

function (s::SteadyStateNeuralODE)(
    V,
    v0,
    i0,
    tsteps,
    tstops,
    ss_solver,
    ss_solver_params,
    dyn_solver,
    dyn_solver_params,
    dyn_sensealg;
    p_fixed = s.p_fixed,
    p_train = s.p_train,
    p_map = s.p_map,
)
    p = vcat(p_fixed, p_train)
    Vd0, Vq0 = s.ref_frame([v0[1]; v0[2]], [v0[1]; v0[2]])    #first argument defines the angle for the transformation. Second argument is the input.
    Id0, Iq0 = s.ref_frame([v0[1]; v0[2]], [i0[1]; i0[2]])

    #u[1:(end-2)] = surrogate states 
    #u[end-1:end] = refs 
    function dudt_ss(u, p, t)
        Vd, Vq = s.ref_frame([v0[1]; v0[2]], V(0.0))
        dyn_input = vcat(
            u[1:(end - SURROGATE_N_REFS)],
            s.input_normalization([Vd, Vq]),
            u[(end - (SURROGATE_N_REFS - 1)):end],
        )
        return vcat(
            s.re2(p[p_map[(s.len + 1):(s.len + s.len2)]])(dyn_input),
            s.target_normalization_inverse(
                s.re3(p[p_map[(s.len + s.len2 + 1):end]])(u[1:(end - SURROGATE_N_REFS)]),
            )[1] .- Id0,
            s.target_normalization_inverse(
                s.re3(p[p_map[(s.len + s.len2 + 1):end]])(u[1:(end - SURROGATE_N_REFS)]),
            )[2] .- Iq0,
            #  zeros(SURROGATE_N_REFS - 2),
        )
    end

    #u[1:end] = surrogate states 
    function dudt_dyn(u, p, t)
        Vd, Vq = s.ref_frame([v0[1]; v0[2]], V(t))
        dyn_input = vcat(u[1:end], s.input_normalization([Vd, Vq]), refs)
        return vcat(s.re2(p[p_map[(s.len + 1):(s.len + s.len2)]])(dyn_input))
    end
    #PREDICTOR 
    ss_input =
        vcat([s.input_normalization([Vd0, Vq0])[2]], s.target_normalization([Id0, Iq0])) #how to deal with this in a more general way? When there is a ref frame transformation, one of the inputs is arbitrary shouldn't be included... 
    u0_pred = s.re1(p[p_map[1:(s.len)]])(ss_input)

    #SOLVE PROBLEM TO STEADY STATE 
    ff_ss = OrdinaryDiffEq.ODEFunction{false}(dudt_ss)
    ss_solution = _solve_steadystate_problem(ff_ss, u0_pred, p, ss_solver, ss_solver_params)
    deq_iterations = _calculate_deq_iterations(ss_solution)
    #TODO - extra call (dummy) to propogate gradients needed after ss_solution is reached? Not sure if needed. 
    #https://github.com/SciML/DeepEquilibriumNetworks.jl/blob/9c2626d6080bbda3c06b81d2463744f5e395003f/src/layers/deq.jl#L41
    if ss_solution.retcode == SciMLBase.ReturnCode.Success
        #SOLVE DYNAMICS
        refs = ss_solution.u[(end - (SURROGATE_N_REFS - 1)):end] #NOTE: refs is used in dudt_dyn
        ff = OrdinaryDiffEq.ODEFunction{false}(dudt_dyn)
        prob_dyn = OrdinaryDiffEq.ODEProblem{false}(
            ff,
            ss_solution.u[1:(end - SURROGATE_N_REFS)],
            (tsteps[1], tsteps[end]),
            p;
            tstops = tstops,
            saveat = tsteps,
        )
        sol = OrdinaryDiffEq.solve(
            prob_dyn,
            dyn_solver;
            sensealg = dyn_sensealg,
            reltol = dyn_solver_params.reltol,
            abstol = dyn_solver_params.abstol,
            maxiters = dyn_solver_params.maxiters,
        )

        return SteadyStateNeuralODE_solution(
            u0_pred,
            Array(ss_solution.u),
            deq_iterations,
            ss_solution.stats,
            tsteps,
            Array(sol[1:end, :]),
            s.ref_frame_inverse(
                [v0[1]; v0[2]],
                s.target_normalization_inverse(
                    s.re3(p[p_map[(s.len + s.len2 + 1):end]])(sol[1:end, :]),
                ),
            ),
            ss_solution.resid,
            true,
            sol.destats,
        )
    else
        return SteadyStateNeuralODE_solution(
            u0_pred,
            Array(ss_solution.u),
            deq_iterations,
            ss_solution.stats,
            tsteps,
            Array(ss_solution.u),
            s.ref_frame_inverse(
                [v0[1]; v0[2]],
                s.target_normalization_inverse(
                    s.re3(p[p_map[(s.len + s.len2 + 1):end]])(ss_solution.u[1:(end - 2)]),
                ),
            ),
            ss_solution.resid,
            false,
            nothing,
        )
    end
end

function _calculate_deq_iterations(ss_solution)
    if ss_solution.original !== nothing
        return ss_solution.original.iterations
    else
        return 0
    end
end

function _solve_steadystate_problem(
    ff_ss,
    u0_pred,
    p,
    ss_solver::SS,
    ss_solver_params,
) where {SS <: SteadyStateDiffEq.SteadyStateDiffEqAlgorithm}
    prob_ss = SteadyStateDiffEq.SteadyStateProblem(
        OrdinaryDiffEq.ODEProblem{false}(
            ff_ss,
            u0_pred,
            (zero(u0_pred[1]), one(u0_pred[1]) * 100),
            p,
        ),
    )
    ss_solution = SteadyStateDiffEq.solve(prob_ss, ss_solver)
    return ss_solution
end

function _solve_steadystate_problem(
    ff_ss,
    u0_pred,
    p,
    ss_solver::SS,
    ss_solver_params,
) where {SS <: SciMLBase.AbstractNonlinearAlgorithm}
    prob_nl = NonlinearSolve.NonlinearProblem(
        OrdinaryDiffEq.ODEProblem{false}(
            ff_ss,
            u0_pred,
            (zero(u0_pred[1]), one(u0_pred[1]) * 100),
            p,
        ),
    )
    ss_solution = SteadyStateDiffEq.solve(
        prob_nl,
        ss_solver;
        abstol = ss_solver_params.abstol,
        reltol = ss_solver_params.reltol,
    )
    return ss_solution
end

#This was for comparing the initializer network with learning the initial conditions directly.
struct OutputParams{P <: AbstractArray}
    p::P
    function OutputParams(p::P) where {P <: AbstractArray}
        new{P}(p)
    end
end

function OutputParams(n::I) where {I <: Integer}
    OutputParams(rand(n))
end

function (m::OutputParams)(x)
    return m.p
end
Flux.@functor OutputParams
Flux.trainable(a::OutputParams) = (a.p,)
