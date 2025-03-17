
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

$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :X; add=[:as_Gradient]))
$(_var(:Field, :inverse_retraction_method))
$(_var(:Field, :vector_transport_method))
$(_var(:Field, :sub_problem))
$(_var(:Field, :sub_state))
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :stepsize))
$(_var(:Field, :retraction_method))

The sub task requires a method to solve

$_doc_FW_sub

# Constructor

    FrankWolfeState(M, sub_problem, sub_state; kwargs...)

Initialise the Frank Wolfe method state.

FrankWolfeState(M, sub_problem; evaluation=AllocatingEvaluation(), kwargs...)

Initialise the Frank Wolfe method state, where `sub_problem` is a closed form solution with `evaluation` as type of evaluation.

## Input

$(_var(:Argument, :M; type=true))
$(_var(:Argument, :sub_problem))
$(_var(:Argument, :sub_state))

## Keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
$(_var(:Keyword, :inverse_retraction_method))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-6)`"))
$(_var(:Keyword, :stepsize; default="[`default_stepsize`](@ref)`(M, FrankWolfeState)`"))
$(_var(:Keyword, :X; add=:as_Memory))

where the remaining fields from before are keyword arguments.
"""
mutable struct FrankWolfeState{
    P,
    T,
    Pr,
    St<:AbstractManoptSolverState,
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
        sub_problem::Pr,
        sub_state::St;
        p::P=rand(M),
        X::T=zero_vector(M, p),
        stopping_criterion::TStop=StopAfterIteration(200) | StopWhenGradientNormLess(1e-6),
        stepsize::TStep=default_stepsize(M, FrankWolfeState),
        retraction_method::TM=default_retraction_method(M, typeof(p)),
        inverse_retraction_method::ITM=default_inverse_retraction_method(M, typeof(p)),
    ) where {
        P,
        T,
        Pr<:Union{AbstractManoptProblem,F} where {F},
        St<:AbstractManoptSolverState,
        TStop<:StoppingCriterion,
        TStep<:Stepsize,
        TM<:AbstractRetractionMethod,
        ITM<:AbstractInverseRetractionMethod,
    }
        return new{P,T,Pr,St,TStep,TStop,TM,ITM}(
            p,
            X,
            sub_problem,
            sub_state,
            stopping_criterion,
            stepsize,
            retraction_method,
            inverse_retraction_method,
        )
    end
end
function FrankWolfeState(
    M::AbstractManifold, sub_problem; evaluation::E=AllocatingEvaluation(), kwargs...
) where {E<:AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation=evaluation)
    return FrankWolfeState(M, sub_problem, cfs; kwargs...)
end

function default_stepsize(M::AbstractManifold, ::Type{FrankWolfeState})
    return DecreasingStepsize(M; length=2.0, shift=2.0)
end
get_gradient(fws::FrankWolfeState) = fws.X
get_iterate(fws::FrankWolfeState) = fws.p
function get_message(fws::FrankWolfeState)
    # for now only the sub solver might have messages
    return get_message(fws.sub_state)
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
    Frank_Wolfe_method(M, f, grad_f, p=rand(M))
    Frank_Wolfe_method(M, gradient_objective, p=rand(M); kwargs...)
    Frank_Wolfe_method!(M, f, grad_f, p; kwargs...)
    Frank_Wolfe_method!(M, gradient_objective, p; kwargs...)

Perform the Frank-Wolfe algorithm to compute for ``$(_tex(:Cal, "C")) ⊂ $(_tex(:Cal, "M"))``
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

$(_var(:Argument, :M; type=true))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))
$(_var(:Argument, :p))

$(_note(:GradientObjective))

# Keyword arguments

$(_var(:Keyword, :evaluation))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stepsize; default="[`DecreasingStepsize`](@ref)`(; length=2.0, shift=2)`"))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`)"))
* `sub_cost=`[`FrankWolfeCost`](@ref)`(p, X)`:
  the cost of the Frank-Wolfe sub problem. $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_grad=`[`FrankWolfeGradient`](@ref)`(p, X)`:
  the gradient of the Frank-Wolfe sub problem. $(_note(:KeywordUsedIn, "sub_objective"))
$(_var(:Keyword, :sub_kwargs))

* `sub_objective=`[`ManifoldGradientObjective`](@ref)`(sub_cost, sub_gradient)`:
  the objective for the Frank-Wolfe sub problem. $(_note(:KeywordUsedIn, "sub_problem"))

$(_var(:Keyword, :sub_problem; default="[`DefaultManoptProblem`](@ref)`(M, sub_objective)`"))
$(_var(:Keyword, :sub_state; default="[`GradientDescentState`](@ref)`(M, copy(M,p))`"))

$(_var(:Keyword, :X; add=:as_Gradient))
$(_var(:Keyword, :stopping_criterion, "sub_stopping_criterion"; default="`[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-8)`"))
  $(_note(:KeywordUsedIn, "sub_state"))
$(_var(:Keyword, :X; add=:as_Gradient))

$(_note(:OtherKeywords))

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
    p=rand(M);
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    mgo = ManifoldGradientObjective(f_, grad_f_; evaluation=evaluation)
    rs = Frank_Wolfe_method(M, mgo, p_; evaluation=evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function Frank_Wolfe_method(
    M::AbstractManifold, mgo::O, p=rand(M); kwargs...
) where {O<:Union{AbstractManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
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
    X=zero_vector(M, p),
    evaluation=AllocatingEvaluation(),
    objective_type=:Riemannian,
    retraction_method=default_retraction_method(M, typeof(p)),
    stepsize::Union{Stepsize,ManifoldDefaultsFactory}=default_stepsize(M, FrankWolfeState),
    stopping_criterion::TStop=StopAfterIteration(200) |
                              StopWhenGradientNormLess(1.0e-8) |
                              StopWhenChangeLess(M, 1.0e-8),
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
                M;
                p=copy(M, p),
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
    O<:Union{AbstractManifoldGradientObjective,AbstractDecoratedManifoldObjective},
}
    dmgo = decorate_objective!(M, mgo; objective_type=objective_type, kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    fws = FrankWolfeState(
        M,
        sub_problem,
        sub_state_storage;
        p=p,
        X=X,
        retraction_method=retraction_method,
        stepsize=_produce_type(stepsize, M),
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
function step_solver!(amp::AbstractManoptProblem, fws::FrankWolfeState, k)
    M = get_manifold(amp)
    # update gradient
    get_gradient!(amp, fws.X, fws.p) # evaluate grad F(p), store the result in O.X
    # solve sub task
    solve!(fws.sub_problem, fws.sub_state) # call the subsolver
    q = get_solver_result(fws.sub_state)
    s = fws.stepsize(amp, fws, k)
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
    amp::AbstractManoptProblem,
    fws::FrankWolfeState{P,T,F,ClosedFormSubSolverState{InplaceEvaluation}},
    k,
) where {P,T,F}
    M = get_manifold(amp)
    get_gradient!(amp, fws.X, fws.p) # evaluate grad F in place for O.X
    q = copy(M, fws.p)
    fws.sub_problem(M, q, fws.p, fws.X) # evaluate the closed form solution and store the result in q
    s = fws.stepsize(amp, fws, k)
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
    amp::AbstractManoptProblem,
    fws::FrankWolfeState{P,T,F,ClosedFormSubSolverState{AllocatingEvaluation}},
    k,
) where {P,T,F}
    M = get_manifold(amp)
    get_gradient!(amp, fws.X, fws.p) # evaluate grad F in place for O.X
    q = fws.sub_problem(M, fws.p, fws.X) # evaluate the closed form solution and store the result in O.p
    # step along the geodesic
    s = fws.stepsize(amp, fws, k)
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
