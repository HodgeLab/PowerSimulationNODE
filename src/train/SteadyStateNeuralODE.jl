using Flux
abstract type SteadyStateNeuralODELayer <: Function end
basic_tgrad(u, p, t) = zero(u) #? 
Flux.trainable(m::SteadyStateNeuralODELayer) = (p = m.p,)

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
struct SteadyStateNeuralODE{P, M, RE, M2, RE2, M3, RE3, SS, DS, A, K} <:
       SteadyStateNeuralODELayer
    p::P
    len::Int        #length of p1 
    len2::Int       #length of p2 
    model1::M       #Initializer model 
    re1::RE
    model2::M2      #Dynamics model
    re2::RE2
    model3::M3      #Observation model 
    re3::RE3
    ss_solver::SS
    dyn_solver::DS
    args::A
    kwargs::K

    function SteadyStateNeuralODE(
        model1,
        model2,
        model3,
        ss_solver,
        dyn_solver,
        args...;
        p = nothing,
        kwargs...,
    )           #This is an inner constructor 
        p1, re1 = Flux.destructure(model1)
        p2, re2 = Flux.destructure(model2)
        p3, re3 = Flux.destructure(model3)
        if p === nothing
            p = [p1; p2; p3]
        end
        new{
            typeof(p),
            typeof(model1),
            typeof(re1),
            typeof(model2),
            typeof(re2),
            typeof(model3),
            typeof(re3),      #The type of len and len2 (Int) is automatically derived: https://docs.julialang.org/en/v1/manual/constructors/
            typeof(ss_solver),
            typeof(dyn_solver),
            typeof(args),
            typeof(kwargs),
        }(
            p,
            length(p1),
            length(p2),
            model1,
            re1,
            model2,
            re2,
            model3,
            re3,
            ss_solver,
            dyn_solver,
            args,
            kwargs,
        )
    end
end

Flux.@functor SteadyStateNeuralODE
Flux.trainable(m::SteadyStateNeuralODE) = (p = m.p,)

function (s::SteadyStateNeuralODE)(ex, x, tsteps, p = s.p)   #r, ex, refs (order of inputs) 
    dudt_ss(u, p, t) = vcat(
        s.re2(p[(s.len + 1):(s.len + s.len2)])((
            u[1:(end - 2)],
            ex(0.0, s.re3(p[(s.len + s.len2 + 1):end])(u[1:(end - 2)])),
            u[(end - 1):end],
        )),
        s.re3(p[(s.len + s.len2 + 1):end])(u[1:(end - 2)]) .- x[1:2], #_PQVθ_to_IrIi(x),
    )

    dudt_dyn(u, p, t) = s.re2(p[(s.len + 1):(s.len + s.len2)])((
        u,
        ex(t, s.re3(p[(s.len + s.len2 + 1):end])(u)),
        refs,
    ))

    #PREDICTOR 
    u0_pred = s.re1(p[1:(s.len)])(x)

    #SOLVE PROBLEM TO STEADY STATE 
    ff_ss = OrdinaryDiffEq.ODEFunction{false}(dudt_ss)
    prob_ss = SteadyStateDiffEq.SteadyStateProblem(
        OrdinaryDiffEq.ODEProblem{false}(
            ff_ss,
            u0_pred,
            (zero(u0_pred[1]), one(1.0) * 100),
            #(zero(u0_pred[1]), zero(u0_pred[1])),  #Possible this is the issue with SS solve? Restricts time to 0?? 
            p,
        ),
    )
    ss_solution = SteadyStateDiffEq.solve(
        prob_ss,
        s.ss_solver;
        abstol = s.args[2],
        maxiters = s.args[1],
    )
    #=     display(s.args[1])
        display(ss_solution.original)
        display(dudt_ss(u0_pred, p, 0.0))
        display(ss_solution.u) =#

    res = dudt_ss(ss_solution.u, p, 0.0)

    #TODO - extra call (dummy) to propogate gradients needed after ss_solution is reached? 
    #https://github.com/SciML/DeepEquilibriumNetworks.jl/blob/9c2626d6080bbda3c06b81d2463744f5e395003f/src/layers/deq.jl#L41

    if NLsolve.converged(ss_solution.original)
        #SOLVE DYNAMICS
        refs = ss_solution.u[(end - 1):end] #NOTE: refs is used in dudt_dyn
        ff = OrdinaryDiffEq.ODEFunction{false}(dudt_dyn) #,tgrad=basic_tgrad)    
        prob_dyn = OrdinaryDiffEq.ODEProblem{false}(
            ff,
            ss_solution.u[1:(end - 2)],
            (tsteps[1], tsteps[end]),
            p;
            saveat = tsteps,
        )
        sol = OrdinaryDiffEq.solve(prob_dyn, s.dyn_solver; s.kwargs...)

        return SteadyStateNeuralODE_solution(
            u0_pred,
            Array(ss_solution.u),
            tsteps,
            Array(sol),
            s.re3(p[(s.len + s.len2 + 1):end])(Array(sol)),
            res,
        )
    else
        return SteadyStateNeuralODE_solution(
            u0_pred,
            Array(ss_solution.u),
            tsteps,
            Array(ss_solution.u),
            s.re3(p[(s.len + s.len2 + 1):end])(ss_solution.u[1:(end - 2)]),
            res,
        )
    end
end

function _PQVθ_to_IrIi(powerflow)
    P = powerflow[1]
    Q = powerflow[2]
    Vm = powerflow[3]
    θ = powerflow[4]
    S = P + Q * 1im
    Vr = Vm * cos(θ)
    Vi = Vm * sin(θ)
    V = Vr + Vi * 1im
    I = conj(S / V)
    return [real(I), imag(I)]
end

struct SteadyStateNeuralODE_solution{T}        #TODO - look into making this compatible with SciML ecosystem of solution types 
    r0_pred::AbstractArray{T}
    r0::AbstractArray{T}
    t_series::AbstractArray{T}
    r_series::AbstractArray{T}
    i_series::AbstractArray{T}
    res::AbstractArray{T}
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
