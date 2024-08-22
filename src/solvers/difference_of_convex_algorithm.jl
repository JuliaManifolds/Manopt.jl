
@doc """
    DifferenceOfConvexState{Pr,St,P,T,SC<:StoppingCriterion} <:
               AbstractManoptSolverState

A struct to store the current state of the [`difference_of_convex_algorithm`])(@ref).
It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :X; add=[:as_Subgradient]))
$(_var(:Field, :sub_problem))
$(_var(:Field, :sub_state))
$(_var(:Field, :stopping_criterion, "stop"))

The sub task consists of a method to solve

```math
    $(_tex(:argmin))_{q∈$(_math(:M))}\\ g(p) - ⟨X, $(_tex(:log))_p q⟩
```

is needed. Besides a problem and a state, one can also provide a function and
an [`AbstractEvaluationType`](@ref), respectively, to indicate
a closed form solution for the sub task.

# Constructors

    DifferenceOfConvexState(M, sub_problem, sub_state; kwargs...)
    DifferenceOfConvexState(M, sub_solver; evaluation=InplaceEvaluation(), kwargs...)

Generate the state either using a solver from Manopt, given by
an [`AbstractManoptProblem`](@ref) `sub_problem` and an [`AbstractManoptSolverState`](@ref) `sub_state`,
or a closed form solution `sub_solver` for the sub-problem the function expected to be of the form `(M, p, X) -> q` or `(M, q, p, X) -> q`,
where by default its [`AbstractEvaluationType`](@ref) `evaluation` is in-place of `q`.
Here the elements passed are the current iterate `p` and the subgradient `X` of `h` can be passed to that function.

## further keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(200)`"))
$(_var(:Keyword, :X; add=:as_Memory))
"""
mutable struct DifferenceOfConvexState{
    P,T,Pr,St<:AbstractManoptSolverState,SC<:StoppingCriterion
} <: AbstractSubProblemSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    stop::SC
    function DifferenceOfConvexState(
        M::AbstractManifold,
        sub_problem::Pr,
        sub_state::St;
        p::P=rand(M),
        X::T=zero_vector(M, p),
        stopping_criterion::SC=StopAfterIteration(300) | StopWhenChangeLess(M, 1e-9),
    ) where {
        P,
        T,
        Pr<:Union{AbstractManoptProblem,F} where {F},
        St<:AbstractManoptSolverState,
        SC<:StoppingCriterion,
    }
        return new{P,T,Pr,St,SC}(p, X, sub_problem, sub_state, stopping_criterion)
    end
end
function DifferenceOfConvexState(
    M::AbstractManifold, sub_problem; evaluation::E=AllocatingEvaluation(), kwargs...
) where {E<:AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation=evaluation)
    return DifferenceOfConvexState(M, sub_problem, cfs; kwargs...)
end

get_iterate(dcs::DifferenceOfConvexState) = dcs.p
function set_iterate!(dcs::DifferenceOfConvexState, M, p)
    copyto!(M, dcs.p, p)
    return dcs
end
get_gradient(dcs::DifferenceOfConvexState) = dcs.X
function set_gradient!(dcs::DifferenceOfConvexState, M, p, X)
    copyto!(M, dcs.X, p, X)
    return dcs
end
function get_message(dcs::DifferenceOfConvexState)
    # for now only the sub solver might have messages
    return get_message(dcs.sub_state)
end

function show(io::IO, dcs::DifferenceOfConvexState)
    i = get_count(dcs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(dcs.stop) ? "Yes" : "No"
    sub = repr(dcs.sub_state)
    sub = replace(sub, "\n" => "\n    | ")
    s = """
    # Solver state for `Manopt.jl`s Difference of Convex Algorithm
    $Iter
    ## Parameters
    * sub solver state:
        | $(sub)

    ## Stopping criterion

    $(status_summary(dcs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
_doc_DoC = """
    difference_of_convex_algorithm(M, f, g, ∂h, p=rand(M); kwargs...)
    difference_of_convex_algorithm(M, mdco, p; kwargs...)
    difference_of_convex_algorithm!(M, f, g, ∂h, p; kwargs...)
    difference_of_convex_algorithm!(M, mdco, p; kwargs...)

Compute the difference of convex algorithm [BergmannFerreiraSantosSouza:2023](@cite) to minimize

```math
    $(_tex(:argmin))_{p∈$(_math(:M))}\\ g(p) - h(p)
```

where you need to provide ``f(p) = g(p) - h(p)``, ``g`` and the subdifferential ``∂h`` of ``h``.

This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,…``

1. Take ``X^{(k)}  ∈ ∂h(p^{(k)})``
2. Set the next iterate to the solution of the subproblem
```math
  p^{(k+1)} ∈ $(_tex(:argmin))_{q ∈ $(_math(:M))} g(q) - ⟨X^{(k)}, $(_tex(:log))_{p^{(k)}}q⟩
```

until the stopping criterion (see the `stopping_criterion` keyword is fulfilled.

# Keyword arguments

$(_var(:Keyword, :evaluation))
* `gradient=nothing`:        specify ``$(_tex(:grad)) f``, for debug / analysis or enhancing the `stopping_criterion=`
* `grad_g=nothing`:          specify the gradient of `g`. If specified, a subsolver is automatically set up.
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-8)`"))
* `g=nothing`:               specify the function `g` If specified, a subsolver is automatically set up.
* `sub_cost=`[`LinearizedDCCost`](@ref)`(g, p, initial_vector)`: a cost to be used within the default `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_grad=`[`LinearizedDCGrad`](@ref)`(grad_g, p, initial_vector; evaluation=evaluation)`:
  gradient to be used within the default `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_hess`:              (a finite difference approximation using `sub_grad` by default):
   specify a Hessian of the `sub_cost`, which the default solver, see `sub_state=` needs.
  $(_note(:KeywordUsedIn, "sub_objective"))
$(_var(:Keyword, :sub_kwargs))
* `sub_objective`:         a gradient or Hessian objective based on `sub_cost=`, `sub_grad=`, and `sub_hess`if provided
   the objective used within `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_problem"))
$(_var(:Keyword, :sub_state; default="([`GradientDescentState`](@ref) or [`TrustRegionsState`](@ref) if `sub_hessian` is provided)"))
$(_var(:Keyword, :sub_problem; default="[`DefaultManoptProblem`](@ref)`(M, sub_objective)`"))
* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-9)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-9)`:
  a stopping criterion used withing the default `sub_state=`
  $(_note(:KeywordUsedIn, "sub_state"))
* `sub_stepsize=`[`ArmijoLinesearch`](@ref)`(M)`) specify a step size used within the `sub_state`.
  $(_note(:KeywordUsedIn, "sub_state"))
$(_var(:Keyword, :X; add=:as_Memory))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_DoC)"
difference_of_convex_algorithm(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_algorithm(
    M::AbstractManifold,
    f,
    g,
    ∂h,
    p=rand(M);
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    grad_g=nothing,
    gradient=nothing,
    kwargs...,
)
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    g_ = _ensure_mutating_cost(g, p)
    gradient_ = _ensure_mutating_gradient(gradient, p, evaluation)
    grad_g_ = _ensure_mutating_gradient(grad_g, p, evaluation)
    ∂h_ = _ensure_mutating_gradient(∂h, p, evaluation)
    mdco = ManifoldDifferenceOfConvexObjective(
        f_, ∂h_; gradient=gradient_, evaluation=evaluation
    )
    rs = difference_of_convex_algorithm(
        M,
        mdco,
        p_;
        g=g_,
        evaluation=evaluation,
        gradient=gradient_,
        grad_g=grad_g_,
        kwargs...,
    )
    return _ensure_matching_output(p, rs)
end
function difference_of_convex_algorithm(
    M::AbstractManifold, mdco::O, p; kwargs...
) where {O<:Union{ManifoldDifferenceOfConvexObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return difference_of_convex_algorithm!(M, mdco, q; kwargs...)
end

@doc "$(_doc_DoC)"
difference_of_convex_algorithm!(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_algorithm!(
    M::AbstractManifold,
    f,
    g,
    ∂h,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    gradient=nothing,
    kwargs...,
)
    mdco = ManifoldDifferenceOfConvexObjective(
        f, ∂h; gradient=gradient, evaluation=evaluation
    )
    return difference_of_convex_algorithm!(
        M, mdco, p; g=g, evaluation=evaluation, kwargs...
    )
end
function difference_of_convex_algorithm!(
    M::AbstractManifold,
    mdco::O,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    grad_g=nothing,
    gradient=nothing,
    X=zero_vector(M, p),
    objective_type=:Riemannian,
    stopping_criterion=if isnothing(gradient)
        StopAfterIteration(300) | StopWhenChangeLess(M, 1e-9)
    else
        StopAfterIteration(300) |
        StopWhenChangeLess(M, 1e-9) |
        StopWhenGradientNormLess(1e-9)
    end,
    # Subsolver Magic Cascade.
    sub_cost=if isnothing(g)
        nothing
    else
        LinearizedDCCost(g, copy(M, p), copy(M, p, X))
    end,
    sub_grad=if isnothing(grad_g)
        nothing
    else
        LinearizedDCGrad(grad_g, copy(M, p), copy(M, p, X); evaluation=evaluation)
    end,
    sub_hess=ApproxHessianFiniteDifference(M, copy(M, p), sub_grad; evaluation=evaluation),
    sub_kwargs=(;),
    sub_stopping_criterion=StopAfterIteration(300) | StopWhenGradientNormLess(1e-8),
    sub_objective=if isnothing(sub_cost) || isnothing(sub_grad)
        nothing
    else
        decorate_objective!(
            M,
            if isnothing(sub_hess)
                ManifoldGradientObjective(sub_cost, sub_grad; evaluation=evaluation)
            else
                ManifoldHessianObjective(
                    sub_cost, sub_grad, sub_hess; evaluation=evaluation
                )
            end;
            objective_type=objective_type,
            sub_kwargs...,
        )
    end,
    sub_problem::Union{AbstractManoptProblem,Function,Nothing}=if isnothing(sub_objective)
        nothing
    else
        DefaultManoptProblem(M, sub_objective)
    end,
    sub_state::Union{AbstractManoptSolverState,AbstractEvaluationType,Nothing}=if sub_problem isa
        Function
        evaluation
    elseif isnothing(sub_objective)
        nothing
    else
        decorate_state!(
            if isnothing(sub_hess)
                GradientDescentState(
                    M;
                    p=copy(M, p),
                    stopping_criterion=sub_stopping_criterion,
                    sub_kwargs...,
                )
            else
                TrustRegionsState(
                    M,
                    sub_objective;
                    p=copy(M, p),
                    stopping_criterion=sub_stopping_criterion,
                    sub_kwargs...,
                )
            end;
            sub_kwargs...,
        )
    end,
    kwargs..., #collect rest
) where {O<:Union{ManifoldDifferenceOfConvexObjective,AbstractDecoratedManifoldObjective}}
    dmdco = decorate_objective!(M, mdco; objective_type=objective_type, kwargs...)
    dmp = DefaultManoptProblem(M, dmdco)
    isnothing(sub_problem) && error(
        """
        Subproblem not correctly initialized. Please provide _either_
        * a `sub_problem=` to be solved
        * a `sub_objective` to automatically generate the sub problem,
        * `sub_grad=` (as well as the usually given `sub_cost=`) to automatically generate the sub objective _or_
        * `grad_g=` keywords to automatically generate the sub problems gradient.
        """,
    )
    dcs = DifferenceOfConvexState(
        M,
        sub_problem,
        maybe_wrap_evaluation_type(sub_state);
        p=p,
        stopping_criterion=stopping_criterion,
        X=X,
    )
    ddcs = decorate_state!(dcs; kwargs...)
    solve!(dmp, ddcs)
    return get_solver_return(get_objective(dmp), ddcs)
end
function initialize_solver!(::AbstractManoptProblem, dcs::DifferenceOfConvexState)
    return dcs
end
function step_solver!(amp::AbstractManoptProblem, dcs::DifferenceOfConvexState, kw)
    M = get_manifold(amp)
    get_subtrahend_gradient!(amp, dcs.X, dcs.p)
    set_parameter!(dcs.sub_problem, :Objective, :Cost, :p, dcs.p)
    set_parameter!(dcs.sub_problem, :Objective, :Cost, :X, dcs.X)
    set_parameter!(dcs.sub_problem, :Objective, :Gradient, :p, dcs.p)
    set_parameter!(dcs.sub_problem, :Objective, :Gradient, :X, dcs.X)
    set_iterate!(dcs.sub_state, M, copy(M, dcs.p))
    solve!(dcs.sub_problem, dcs.sub_state) # call the subsolver
    # copy result from subsolver to current iterate
    copyto!(M, dcs.p, get_solver_result(dcs.sub_state))
    # small hack: store `gradient_f` in X at end of iteration for the gradient norm stopping criterion
    if !isnothing(get_gradient_function(get_objective(amp)))
        get_gradient!(amp, dcs.X, dcs.p)
    end
    return dcs
end
#
# Variant II: sub task is a mutating function providing a closed form solution
#
function step_solver!(
    amp::AbstractManoptProblem,
    dcs::DifferenceOfConvexState{P,T,F,ClosedFormSubSolverState{InplaceEvaluation}},
    i,
) where {P,T,F}
    M = get_manifold(amp)
    get_subtrahend_gradient!(amp, dcs.X, dcs.p) # evaluate grad F in place for O.X
    dcs.sub_problem(M, dcs.p, dcs.p, dcs.X) # evaluate the closed form solution and store the result in p
    return dcs
end
#
# Variant II: sub task is an allocating function providing a closed form solution
#
function step_solver!(
    amp::AbstractManoptProblem,
    dcs::DifferenceOfConvexState{P,T,F,ClosedFormSubSolverState{AllocatingEvaluation}},
    i,
) where {P,T,F}
    M = get_manifold(amp)
    get_subtrahend_gradient!(amp, dcs.X, dcs.p) # evaluate grad F in place for O.X
    # run the subsolver in-place of a copy of the current iterate and copy it back to the current iterate
    copyto!(M, dcs.p, dcs.sub_problem(M, copy(M, dcs.p), dcs.X))
    return dcs
end
get_solver_result(dcs::DifferenceOfConvexState) = dcs.p
