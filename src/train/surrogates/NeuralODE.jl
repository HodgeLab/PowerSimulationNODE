using Flux
abstract type NeuralODELayer <: Function end
Flux.trainable(m::NeuralODELayer) = (p = m.p,)

#TODO - update docstring for the layer

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
struct NeuralODE{PT, PF, M, RE, M2, RE2, IN, TN, TNI, RF, RFI, PM} <: NeuralODELayer
    p_train::PT
    p_fixed::PF
    len::Int        #length of p1 
    model1::M       #Initializer model 
    re1::RE
    model2::M2      #Dynamics model
    re2::RE2
    input_normalization::IN
    target_normalization::TN
    target_normalization_inverse::TNI
    ref_frame::RF
    ref_frame_inverse::RFI
    p_map::PM

    function NeuralODE(
        model1,
        model2,
        input_normalization,
        target_normalization,
        target_normalization_inverse,
        ref_frame,
        ref_frame_inverse;
        p = nothing,
    )           #This is an inner constructor 
        p1, re1 = Flux.destructure(model1)
        p2, re2 = Flux.destructure(model2)
        if p === nothing
            p = [p1; p2]
        end
        new{
            typeof(p),
            typeof(p),
            typeof(model1),
            typeof(re1),
            typeof(model2),
            typeof(re2),
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
            model1,
            re1,
            model2,
            re2,
            input_normalization,
            target_normalization,
            target_normalization_inverse,
            ref_frame,
            ref_frame_inverse,
            1:length(p),
        )
    end
end

Flux.@functor NeuralODE
Flux.trainable(m::NeuralODE) = (p = m.p_train,)

function (s::NeuralODE)(
    V,
    v0,
    i0,
    tsteps,
    tstops,
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

    #u[1:end] = surrogate states 
    function dudt_dyn(u, p_and_refs, t)
        p = p_and_refs[1:(end - 2)]
        refs = p_and_refs[(end - 1):end]
        Vd, Vq = s.ref_frame([v0[1]; v0[2]], V(t))
        dyn_input = vcat(u[1:end], s.input_normalization([Vd, Vq]), refs)
        return vcat(s.re2(p[p_map[(s.len + 1):end]])(dyn_input))
    end
    #PREDICTOR 
    ss_input =
        vcat([s.input_normalization([Vd0, Vq0])[2]], s.target_normalization([Id0, Iq0])) #how to deal with this in a more general way? When there is a ref frame transformation, one of the inputs is arbitrary shouldn't be included... 
    u0_pred = s.re1(p[p_map[1:(s.len)]])(ss_input)

    #SOLVE DYNAMICS
    refs = u0_pred[(end - (SURROGATE_N_REFS - 1)):end] #NOTE: refs is used in dudt_dyn - possible performance issue here? 

    ff = OrdinaryDiffEq.ODEFunction{false}(dudt_dyn)
    prob_dyn = OrdinaryDiffEq.ODEProblem{false}(
        ff,
        u0_pred[1:(end - SURROGATE_N_REFS)],
        (tsteps[1], tsteps[end]),
        vcat(p, refs);
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
        Array(u0_pred),
        0,
        nothing,
        tsteps,
        Array(sol[1:end, :]),
        s.ref_frame_inverse([v0[1]; v0[2]], s.target_normalization_inverse(sol[1:2, :])),
        s.ref_frame_inverse([v0[1]; v0[2]], s.target_normalization_inverse(sol[1:2, :])),
        true,
        sol.stats,
    )
end
