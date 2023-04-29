@doc raw"""
    FrankWolfeState <: AbstractManoptSolverState

A struct to store the current state of the [`Frank_Wolfe_method`](@ref)

It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

* `p` – the current iterate, i.e. a point on the manifold
* `X` – the current gradient ``\operatorname{grad} F(p)``, i.e. a tangent vector to `p`.
* `inverse_retraction_method` – (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction method to use within Frank Wolfe.
* `sub_problem` – an [`AbstractManoptProblem`](@ref) problem for the subsolver
* `sub_state` – an [`AbstractManoptSolverState`](@ref) for the subsolver
* `stop` – ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`) a [`StoppingCriterion`](@ref)
* `stepsize` - ([`DecreasingStepsize`](@ref)`(; length=2.0, shift=2)`) ``s_k`` which by default is set to ``s_k = \frac{2}{k+2}``.
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a retraction to use within Frank-Wolfe

For the subtask, we need a method to solve

```math
    \operatorname*{argmin}_{q∈\mathcal M} ⟨X, \log_p q⟩,\qquad \text{ where }X=\operatorname{grad} f(p)
```

# Constructor

    FrankWolfeState(M, p, X, sub_problem, sub_task)

where the remaining fields from above are keyword arguments with their defaults already given in brackets.
"""
mutable struct FrankWolfeState{
    P,
    T,
    Pr,
    St,
    TStep<:Stepsize,
    TStop<:StoppingCriterion,
    TM<:AbstractRetractionMethod,
    ITM<:AbstractInverseRetractionMethod,
} <: AbstractGradientSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    stop::TStop
    stepsize::TStep
    retraction_method::TM
    inverse_retraction_method::ITM
    function FrankWolfeState(
        M::AbstractManifold,
        p::P,
        sub_problem::Pr,
        sub_state::Op;
        initial_vector::T=zero_vector(M, p),
        stopping_criterion::TStop=StopAfterIteration(200) |
                                  StopWhenGradientNormLess(1.0e-6),
        stepsize::TStep=default_stepsize(M, FrankWolfeState),
        retraction_method::TM=default_retraction_method(M, typeof(p)),
        inverse_retraction_method::ITM=default_inverse_retraction_method(M, typeof(p)),
    ) where {
        P,
        Pr,
        Op,
        T,
        TStop<:StoppingCriterion,
        TStep<:Stepsize,
        TM<:AbstractRetractionMethod,
        ITM<:AbstractInverseRetractionMethod,
    }
        return new{P,T,Pr,Op,TStep,TStop,TM,ITM}(
            p,
            initial_vector,
            sub_problem,
            sub_state,
            stopping_criterion,
            stepsize,
            retraction_method,
            inverse_retraction_method,
        )
    end
end
function default_stepsize(::AbstractManifold, ::Type{FrankWolfeState})
    return DecreasingStepsize(; length=2.0, shift=2)
end
get_gradient(fws::FrankWolfeState) = fws.X
get_iterate(fws::FrankWolfeState) = fws.p
function get_message(fws::FrankWolfeState)
    # for now only the sub solver might have messages
    return get_message(fws.sub_state)
end
function get_message(::FrankWolfeState{P,T,F,<:InplaceEvaluation}) where {P,T,F}
    return ""
end

function set_iterate!(fws::FrankWolfeState, p)
    fws.p = p
    return fws
end
function show(io::IO, fws::FrankWolfeState)
    i = get_count(fws, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(fws.stop) ? "Yes" : "No"
    sub = repr(fws.sub_state)
    sub = replace(sub, "\n" => "\n    | ")
    s = """
    # Solver state for `Manopt.jl`s Frank Wolfe Method
    $Iter
    ## Parameters
    * inverse retraction method: $(fws.inverse_retraction_method)
    * retraction method: $(fws.retraction_method)
    * sub solver state:
        | $(sub)

    ## Stepsize
    $(fws.stepsize)

    ## Stopping Criterion
    $(status_summary(fws.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    Frank_Wolfe_method(M, f, grad_f, p)
    Frank_Wolfe_method(M, gradient_objective, p; kwargs...)

Perform the Frank-Wolfe algorithm to compute for ``\mathcal C \subset \mathcal M``

```math
    \operatorname*{arg\,min}_{p∈\mathcal C} f(p)
```

Where the main step is a constrained optimisation is within the algorithm,
that is the sub problem (Oracle)

```math
    q_k = \operatorname{arg\,min}_{q \in C} ⟨\operatorname{grad} F(p_k), \log_{p_k}q⟩.
```

for every iterate ``p_k`` together with a stepsize ``s_k≤1``, by default ``s_k = \frac{2}{k+2}``.

The next iterate is then given by ``p_{k+1} = γ_{p_k,q_k}(s_k)``,
where by default ``γ`` is the shortest geodesic between the two points but can also be changed to
use a retraction and its inverse.

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f: \mathcal M→ℝ`` to find a minimizer ``p^*`` for
* `grad_f` – the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of f
  - as a function `(M, p) -> X` or a function `(M, X, p) -> X`
* `p` – an initial value ``p ∈ \mathcal C``, note that it really has to be a feasible point

Alternatively to `f` and `grad_f` you can prodive
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.

## Keyword Arguments

* `evaluation` ([`AllocatingEvaluation`](@ref)) whether `grad_F` is an inplace or allocating (default) funtion
* `initial_vector` – (`zero_vectoir(M,p)`) how to initialize the inner gradient tangent vector
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(500) | `[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`) a stopping criterion
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a type of retraction
* `stepsize` ([`DecreasingStepsize`](@ref)`(; length=2.0, shift=2)`
  a [`Stepsize`](@ref) to use; but it has to be always less than 1. The default is the one proposed by Frank & Wolfe:
  ``s_k = \frac{2}{k+2}``.

All other keyword arguments are passed to [`decorate_state!`](@ref) for decorators or
[`decorate_objective!`](@ref), respectively.
If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
Frank_Wolfe_method(M::AbstractManifold, args...; kwargs...)
function Frank_Wolfe_method(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return Frank_Wolfe_method(M, mgo, p; evaluation=evaluation, kwargs...)
end
function Frank_Wolfe_method(
    M::AbstractManifold,
    f,
    grad_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    # redefine our initial point
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    rs = Frank_Wolfe_method(M, f_, grad_f_, q; evaluation=evaluation, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function Frank_Wolfe_method(
    M::AbstractManifold, mgo::ManifoldGradientObjective, p; kwargs...
)
    q = copy(M, p)
    return Frank_Wolfe_method!(M, mgo, q; kwargs...)
end

@doc raw"""
    Frank_Wolfe_method!(M, f, grad_f, p; kwargs...)
    Frank_Wolfe_method!(M, gradient_objective, p; kwargs...)

Peform the Frank Wolfe method in place of `p`.

For all options and keyword arguments, see [`Franke_Wolfe_method`](@ref).
"""
Frank_Wolfe_method!(M::AbstractManifold, args...; kwargs...)
function Frank_Wolfe_method!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return Frank_Wolfe_method!(M, mgo, p; evaluation=evaluation, kwargs...)
end
function Frank_Wolfe_method!(
    M::AbstractManifold,
    mgo::AbstractManifoldGradientObjective,
    p;
    initial_vector=zero_vector(M, p),
    evaluation=AllocatingEvaluation(),
    retraction_method=default_retraction_method(M, typeof(p)),
    stepsize::TStep=default_stepsize(M, FrankWolfeState),
    stopping_criterion::TStop=StopAfterIteration(200) |
                              StopWhenGradientNormLess(1.0e-8) |
                              StopWhenChangeLess(1.0e-8),
    sub_cost=FrankWolfeCost(p, initial_vector),
    sub_grad=FrankWolfeGradient(p, initial_vector),
    sub_objective=ManifoldGradientObjective(sub_cost, sub_grad),
    sub_problem=DefaultManoptProblem(M, sub_objective),
    sub_kwargs=[],
    sub_stopping_criterion=StopAfterIteration(300) | StopWhenStepsizeLess(1e-8),
    sub_state::Union{AbstractManoptSolverState,AbstractEvaluationType}=if sub_problem isa
        Function
        evaluation
    else
        decorate_state!(
            GradientDescentState(
                M,
                copy(M, p);
                stopping_criterion=sub_stopping_criterion,
                stepsize=default_stepsize(
                    M, GradientDescentState; retraction_method=retraction_method
                ),
            ),
            sub_kwargs...,
        )
    end,
    kwargs..., #collect rest
) where {TStop<:StoppingCriterion,TStep<:Stepsize}
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    fws = FrankWolfeState(
        M,
        p,
        sub_problem,
        sub_state;
        initial_vector=initial_vector,
        retraction_method=retraction_method,
        stepsize=stepsize,
        stopping_criterion=stopping_criterion,
    )
    fws = decorate_state!(fws; kwargs...)
    return get_solver_return(solve!(dmp, fws))
end
function initialize_solver!(amp::AbstractManoptProblem, fws::FrankWolfeState)
    get_gradient!(amp, fws.X, fws.p)
    return fws
end
function step_solver!(
    amp::AbstractManoptProblem,
    fws::FrankWolfeState{P,T,<:AbstractManoptProblem,<:AbstractManoptSolverState},
    i,
) where {P,T}
    M = get_manifold(amp)
    # update gradient
    get_gradient!(amp, fws.X, fws.p) # evaluate grad F(p), store the result in O.X
    # solve subtask
    solve!(fws.sub_problem, fws.sub_state) # call the subsolver
    q = get_solver_result(fws.sub_state)
    s = fws.stepsize(amp, fws, i)
    # step along the geodesic
    retract!(
        M,
        fws.p,
        fws.p,
        s .* inverse_retract(M, fws.p, q, fws.inverse_retraction_method),
        fws.retraction_method,
    )
    return fws
end
#
# Variant II: subtask is a mutating function providing a closed form soltuion
#
function step_solver!(
    amp::AbstractManoptProblem, fws::FrankWolfeState{P,T,F,InplaceEvaluation}, i
) where {P,T,F}
    M = get_manifold(amp)
    get_gradient!(amp, fws.X, fws.p) # evaluate grad F in place for O.X
    q = copy(M, fws.p)
    fws.sub_problem(M, q, fws.p, fws.X) # evaluate the closed form solution and store the result in q
    s = fws.stepsize(amp, fws, i)
    # step along the geodesic
    retract!(
        M,
        fws.p,
        fws.p,
        s .* inverse_retract(M, fws.p, q, fws.inverse_retraction_method),
        fws.retraction_method,
    )
    return fws
end
#
# Variant II: subtask is an allocating function providing a closed form soltuion
#
function step_solver!(
    amp::AbstractManoptProblem, fws::FrankWolfeState{P,T,F,AllocatingEvaluation}, i
) where {P,T,F}
    M = get_manifold(amp)
    get_gradient!(amp, fws.X, fws.p) # evaluate grad F in place for O.X
    q = fws.sub_problem(M, fws.p, fws.X) # evaluate the closed form solution and store the result in O.p
    # step along the geodesic
    s = fws.stepsize(amp, fws, i)
    # step along the geodesic
    retract!(
        M,
        fws.p,
        fws.p,
        s .* inverse_retract(M, fws.p, q, fws.inverse_retraction_method),
        fws.retraction_method,
    )
    return fws
end
