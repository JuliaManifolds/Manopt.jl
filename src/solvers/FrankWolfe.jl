
_doc_FW_sub = raw"""
```math
   \operatorname*{arg\,min}_{q ∈ C} ⟨\operatorname{grad} f(p_k), \log_{p_k}q⟩.
```
"""

@doc """
    FrankWolfeState <: AbstractManoptSolverState

A struct to store the current state of the [`Frank_Wolfe_method`](@ref)

It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

* $_field_iterate
* $_field_gradient
* $_field_inv_retr
* $_field_sub_problem
* $_field_sub_state
* $_field_stop
* $_field_step
* $_field_retr

The sub task requires a method to solve

$_doc_FW_sub

# Constructor

    FrankWolfeState(M, p, sub_problem, sub_state; kwargs...)

Initialise the Frank Wolfe method state with.

## Input

* $_arg_M
* $_arg_p
* $_arg_X
* $_arg_sub_problem
* $_arg_sub_state

## Keyword arguments

* `stopping_criterion=`[`StopAfterIteration`](@ref)`(200)`$_sc_any[`StopWhenGradientNormLess`](@ref)`(1e-6)` $_kw_stop_note
* `stepsize=`[`default_stepsize`](@ref)`(M, FrankWolfeState)`
* $_kw_retraction_method_default
* $_kw_inverse_retraction_method_default
* $_kw_X_default

where the remaining fields from before are keyword arguments.
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
        initial_vector::T=zero_vector(M, p), #deprecated
        X::T=initial_vector,
        stopping_criterion::TStop=StopAfterIteration(200) | StopWhenGradientNormLess(1e-6),
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

    ## Stopping criterion

    $(status_summary(fws.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

_doc_FW_problem = raw"""
```math
    \operatorname*{arg\,min}_{p∈\mathcal C} f(p),
```
"""
_doc_FW_sk_default = raw"``s_k = \frac{2}{k+2}``"
_doc_Frank_Wolfe_method = """
    Frank_Wolfe_method(M, f, grad_f, p)
    Frank_Wolfe_method(M, gradient_objective, p; kwargs...)
    Frank_Wolfe_method!(M, f, grad_f, p; kwargs...)
    Frank_Wolfe_method!(M, gradient_objective, p; kwargs...)

Perform the Frank-Wolfe algorithm to compute for ``$_l_C_subset_M``
the constrained problem

$_doc_FW_problem

where the main step is a constrained optimisation is within the algorithm,
that is the sub problem (Oracle)

$_doc_FW_sub

for every iterate ``p_k`` together with a stepsize ``s_k≤1``.
The algorhtm can be performed in-place of `p`.

This algorithm is inspired by but slightly more general than [WeberSra:2022](@cite).

The next iterate is then given by ``p_{k+1} = γ_{p_k,q_k}(s_k)``,
where by default ``γ`` is the shortest geodesic between the two points but can also be changed to
use a retraction and its inverse.

# Input

$_arg_M
$_arg_f
$_arg_grad_f
$_arg_p

$_arg_alt_mgo

# Keyword arguments

* $_kw_evaluation_default:
  $_kw_evaluation $_kw_evaluation_example

* $_kw_retraction_method_default:
  $_kw_retraction_method

* `stepsize=`[`DecreasingStepsize`](@ref)`(; length=2.0, shift=2)`:
  $_kw_stepsize, where the default is the step size $_doc_FW_sk_default

* `stopping_criterion=`[`StopAfterIteration`](@ref)`(500)`$_sc_any[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`)
  $_kw_stopping_criterion

* $_kw_X_default:
  $_kw_X, the evaluated gradient ``$_l_grad f`` evaluated at ``p^{(k)}``.

* `sub_cost=`[`FrankWolfeCost`](@ref)`(p, X)`:
  the cost of the Frank-Wolfe sub problem. $(_kw_used_in("sub_objective"))

* `sub_grad=`[`FrankWolfeGradient`](@ref)`(p, X)`:
  the gradient of the Frank-Wolfe sub problem. $(_kw_used_in("sub_objective"))

* `sub_kwargs=(;)`:
  specify keyword arguments passed to decorators of both the `sub_problem=` and `sub_state` as well as to the default state of the `substate`.

* `sub_objective=`[`ManifoldGradientObjective`](@ref)`(sub_cost, sub_gradient)`:
  the objective for the Frank-Wolfe sub problem. $(_kw_used_in("sub_problem"))

* `sub_problem=`[`DefaultManoptProblem`](@ref)`(M, sub_objective)`): the sub problem to solve.
  This can be given in three forms
   1. as an [`AbstractManoptProblem`](@ref), then the `sub_state=` specifies the solver to use
   2. as a closed form solution, as a function evaluating with new allocations `(M, p, X) -> q` that solves the sub problem on `M` given the current iterate `p` and (sub)gradient `X`.
   3. as a closed form solution, as a function `(M, q, p, X) -> q` working in place of `q`.
  For points 2 and 3 the `sub_state` has to be set to the corresponding [`AbstractEvaluationType`](@ref), [`AllocatingEvaluation`](@ref) and [`InplaceEvaluation`](@ref), respectively
  This keyword takes further into account `sub_kwargs` to evejtually decorate the problem

* `sub_state= if sub_problem isa Function evaluation else GradientDescentState(M, copy(M,p); kwargs...)`:

  specify either the solver for a `sub_problem` or the kind of evaluation if the sub problem is given by a closed form solution
  this keyword takes into account the `sub_stopping_criterion`, and the `sub_kwargs`, that are also used to potentially decorate the state.

* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(300)`$_sc_any[`StopWhenStepsizeLess`](@ref)`(1e-8)`:
  $_kw_stopping_criterion for the sub solver. $(_kw_used_in("sub_state"))

$_kw_others

If you provide the [`ManifoldGradientObjective`](@ref) directly, the `evaluation=` keyword is ignored.
The decorations are still applied to the objective.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""

@doc "$_doc_Frank_Wolfe_method"
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
    # redefine initial point
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    rs = Frank_Wolfe_method(M, f_, grad_f_, q; evaluation=evaluation, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function Frank_Wolfe_method(
    M::AbstractManifold, mgo::O, p; kwargs...
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return Frank_Wolfe_method!(M, mgo, q; kwargs...)
end

@doc "$_doc_Frank_Wolfe_method"
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
    mgo::O,
    p;
    initial_vector=zero_vector(M, p), #deprecated
    X=initial_vector,
    evaluation=AllocatingEvaluation(),
    objective_type=:Riemannian,
    retraction_method=default_retraction_method(M, typeof(p)),
    stepsize::TStep=default_stepsize(M, FrankWolfeState),
    stopping_criterion::TStop=StopAfterIteration(200) |
                              StopWhenGradientNormLess(1.0e-8) |
                              StopWhenChangeLess(1.0e-8),
    sub_cost=FrankWolfeCost(p, X),
    sub_grad=FrankWolfeGradient(p, X),
    sub_kwargs=(;),
    sub_objective=ManifoldGradientObjective(sub_cost, sub_grad),
    sub_problem=DefaultManoptProblem(
        M,
        decorate_objective!(M, sub_objective; objective_type=objective_type, sub_kwargs...),
    ),
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
                sub_kwargs...,
            );
            objective_type=objective_type,
            sub_kwargs...,
        )
    end,
    kwargs...,
) where {
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective},
}
    dmgo = decorate_objective!(M, mgo; objective_type=objective_type, kwargs...)
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
    dfws = decorate_state!(fws; kwargs...)
    solve!(dmp, dfws)
    return get_solver_return(get_objective(dmp), dfws)
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
    # solve sub task
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
# Variant 2: sub task is a mutating function providing a closed form solution
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
# Variant 3: sub task is an allocating function providing a closed form solution
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
