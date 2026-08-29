@doc """
    ManifoldDifferenceOfConvexObjective <: AbstractManifoldCostObjective

Specify an objective for a [`difference_of_convex_algorithm`](@ref).

The objective ``f: $(_math(:Manifold)) → ℝ`` is given as

```math
    f(p) = g(p) - h(p)
```

where both ``g`` and ``h`` are convex, lower semicontinuous and proper.
Furthermore the subdifferential ``∂h`` of ``h`` is required.

# Fields

* `cost`: an implementation of ``f(p) = g(p)-h(p)`` as a function `f(M,p)`.
* `gradient!` a gradient of the smooth component `g`
* `∂h!`: a deterministic version of ``∂h: $(_math(:Manifold))→ T$(_math(:Manifold)))``,
  in the sense that calling `∂h(M, p)` returns a subgradient of ``h`` at `p` and
  if there is more than one, it returns a deterministic choice.

Note that the gradient and the subdifferential might be given in two possible signatures

* `(M,p) -> X` which does an [`AllocatingEvaluation`](@ref)
* `(M, X, p) -> X` which does an [`InplaceEvaluation`](@ref) in place of `X`.

# Constructor

    ManifoldDifferenceOfConvexObjective(cost, ∂h; gradient = missing, evaluation = AllocatingEvaluation(), p = missing)

Create the difference of convex objective given a `cost` function and the subdifferential `∂h` of the non-smooth part
The `gradient` of the smooth part and the `evaluation = ` type are keywords.

## Keyword Arguments

$(_kwargs(:evaluation))
* `gradient = missing` provide a gradient of the smooth part
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.
"""
struct ManifoldDifferenceOfConvexObjective{F, G, S} <:
    AbstractManifoldFirstOrderObjective{F, G}
    cost::F
    gradient!::G
    ∂h!::S
    function ManifoldDifferenceOfConvexObjective(
            cost::TC, ∂h::TSH;
            gradient::TG = missing, evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
        ) where {TC, TG, TSH}
        cost_ = maybe_wrap_function(cost, p; result = :Number)
        gradient_ = ismissing(gradient) ? gradient : maybe_wrap_function(gradient, p, evaluation; result = :TangentVector)
        ∂h_ = maybe_wrap_function(∂h, p, evaluation; result = :TangentVector)
        return new{typeof(cost_), typeof(gradient_), typeof(∂h_)}(cost_, gradient_, ∂h_)
    end
end
function get_gradient_function(doco::ManifoldDifferenceOfConvexObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    ismissing(doco.gradient!) && return missing
    if evaluation isa AllocatingEvaluation
        return (M, p) -> doco.gradient!(M, zero_vector(M, p), p)
    else
        return doco.gradient!
    end
end
function get_gradient(
        M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective, p
    )
    X = zero_vector(M, p)
    return doco.gradient!(M, X, p)
end
function get_gradient!(M::AbstractManifold, X, doco::ManifoldDifferenceOfConvexObjective, p)
    return doco.gradient!(M, X, p)
end

function get_subtrahend_gradient(M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective, p)
    X = zero_vector(M, p)
    return doco.∂h!(M, X, p)
end
function get_subtrahend_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_subtrahend_gradient(M, get_objective(admo, false), p)
end
function get_subtrahend_gradient!(
        M::AbstractManifold, X, doco::ManifoldDifferenceOfConvexObjective, p
    )
    return doco.∂h!(M, X, p)
end
function get_subtrahend_gradient!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
    )
    return get_subtrahend_gradient!(M, X, get_objective(admo, false), p)
end

function Base.show(io::IO, doco::ManifoldDifferenceOfConvexObjective)
    print(io, "ManifoldDifferenceOfConvexObjective("); print(io, doco.cost); print(io, ", ")
    print(io, doco.∂h!); print(io, "; ")
    if !ismissing(doco.gradient!)
        print(io, ", gradient = ")
        print(io, doco.gradient!)
    end
    return print(io, ")")
end
function status_summary(doco::ManifoldDifferenceOfConvexObjective; context::Symbol = :default)
    (context === :short) && (return repr(doco))
    gs = ismissing(doco.gradient!) ? "" : "including a gradient of the smooth component"
    (context === :inline) && (return "A difference of convex objective on a manifold $gs")
    gsd = ismissing(doco.gradient!) ? "" : "\n* gradient of `g`:  $(_MANOPT_INDENT)$(doco.gradient!)"
    return """
    A difference of convex objective on a manifold.

    ## Functions
    * cost `f = g + h`: $(_MANOPT_INDENT)$(doco.cost)$(gsd)
    * ∂h:               $(_MANOPT_INDENT)$(doco.∂h!)"""
end

@doc """
    DifferenceOfConvexState{Pr,St,P,T,SC<:StoppingCriterion} <:
               AbstractManoptSolverState

A struct to store the current state of the [`difference_of_convex_algorithm`])(@ref).
It comes in two forms, depending on the realization of the `subproblem`.

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:X; add_properties = [:as_Subgradient]))
$(_fields([:sub_problem, :sub_state]))
$(_fields(:stopping_criterion; name = "stop"))

The sub task consists of a method to solve

```math
    $(_tex(:argmin))_{q∈$(_math(:Manifold))}\\ g(p) - ⟨X, $(_tex(:log))_p q⟩
```

is needed. Besides a problem and a state, one can also provide a function and
an [`AbstractEvaluationType`](@ref), respectively, to indicate
a closed form solution for the sub task.

# Constructors

    DifferenceOfConvexState(M, sub_problem, sub_state; kwargs...)
    DifferenceOfConvexState(M, sub_solver; evaluation=InplaceEvaluation(), kwargs...)

Generate the state either using a solver from Manopt, given by
an [`AbstractManoptProblem`](@ref) `sub_problem` and an [`AbstractManoptSolverState`](@ref) `sub_state`,
or a closed form solution `sub_solver` for the sub-problem the function expected to be of the form `(M, q, p, X) -> q`.

## further keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)"))
$(_kwargs(:X; add_properties = [:as_Memory]))
"""
mutable struct DifferenceOfConvexState{
        P, T, Pr, St <: AbstractManoptSolverState, C <: AbstractDict{Symbol}, SC <: StoppingCriterion,
    } <: AbstractSubProblemSolverState
    callbacks::C
    p::P
    stop::SC
    sub_problem::Pr
    sub_state::St
    X::T
    function DifferenceOfConvexState(
            M::AbstractManifold, sub_problem::Pr, sub_state::St;
            callbacks::C = Dict{Symbol, Function}(),
            p::P = rand(M), X::T = zero_vector(M, p),
            stopping_criterion::SC = StopAfterIteration(300) | StopWhenChangeLess(M, 1.0e-9),
        ) where {
            P, T, C <: AbstractDict{Symbol}, Pr <: Union{AbstractManoptProblem, F} where {F},
            St <: AbstractManoptSolverState, SC <: StoppingCriterion,
        }
        return DifferenceOfConvexState(
            sub_problem, sub_state; callbacks = callbacks, p = p, X = X, stopping_criterion = stopping_criterion
        )
    end
    # resolve an ambiguity
    DifferenceOfConvexState(M::AbstractManifold, st::AbstractManoptSolverState; kwargs...) = error("Difference of Convex Method state can not be constructed based on $M and the sub state $st, a sub_problem is missing")
    function DifferenceOfConvexState(
            sub_problem::Pr, sub_state::St; callbacks::C, p::P, X::T, stopping_criterion::SC
        ) where {
            P, T, C <: AbstractDict{Symbol}, Pr <: Union{AbstractManoptProblem, F} where {F},
            St <: AbstractManoptSolverState, SC <: StoppingCriterion,
        }
        return new{P, T, Pr, St, C, SC}(callbacks, p, stopping_criterion, sub_problem, sub_state, X)
    end
end
provided_callbacks(::Type{DifferenceOfConvexState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:BeforeSubsolver, :Subsolver])
get_callbacks(dcs::DifferenceOfConvexState) = dcs.callbacks
function DifferenceOfConvexState(M::AbstractManifold, sub_problem, sub_state::AbstractEvaluationType; kwargs...)
    return DifferenceOfConvexState(M, sub_problem; evaluation = sub_state, kwargs...)
end
function DifferenceOfConvexState(M::AbstractManifold, sub_problem; evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...)
    sub_problem_ = maybe_wrap_function(sub_problem, evaluation; result = :Point)
    return DifferenceOfConvexState(M, sub_problem_, ClosedFormSubSolverState(); kwargs...)
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
function Base.show(io::IO, dcs::DifferenceOfConvexState)
    print(io, "DifferenceOfConvexState(", dcs.sub_problem, ", ", dcs.sub_state, "; ")
    print(io, "callbacks = "); print(io, dcs.callbacks); print(io, ", ")
    print(io, "p = "); print(io, dcs.p); print(io, ", ")
    print(io, "stopping_criterion = "); print(io, status_summary(dcs.stop; context = :short)); print(io, ", ")
    print(io, "X = "); print(io, dcs.X)
    return print(io, ")")
end
function status_summary(dcs::DifferenceOfConvexState; context::Symbol = :default)
    (context === :short) && return repr(dcs)
    i = get_count(dcs, :Iterations)
    conv_inl = (i > 0) ? (has_converged(dcs.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the differencce of convex algorithm$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(dcs.stop) ? "Yes" : "No"
    as = _callbacks_summary(dcs)
    sub = status_summary(dcs.sub_state; context = context)
    sub = replace(sub, "\n" => "\n    | ", "\n#" => "\n$(_MANOPT_INDENT)##")
    s = """
    # Solver state for `Manopt.jl`s Difference of Convex Algorithm
    $Iter
    ## Parameters$(as)
    * sub solver state:
        | $(sub)

    ## Stopping criterion
    $(_in_str(status_summary(dcs.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
    return s
end

_doc_DoC = """
    difference_of_convex_algorithm(M, f, g, ∂h, p=rand(M); kwargs...)
    difference_of_convex_algorithm(M, mdco, p; kwargs...)
    difference_of_convex_algorithm!(M, f, g, ∂h, p; kwargs...)
    difference_of_convex_algorithm!(M, mdco, p; kwargs...)

Compute the difference of convex algorithm [BergmannFerreiraSantosSouza:2024](@cite) to minimize

```math
    $(_tex(:argmin))_{p∈$(_math(:Manifold))\\ g(p) - h(p)
```

where you need to provide ``f(p) = g(p) - h(p)``, ``g`` and the subdifferential ``∂h`` of ``h``.

This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,…``

1. Take ``X^{(k)}  ∈ ∂h(p^{(k)})``
2. Set the next iterate to the solution of the subproblem
```math
  p^{(k+1)} ∈ $(_tex(:argmin))_{q ∈ $(_math(:Manifold))} g(q) - ⟨X^{(k)}, $(_tex(:log))_{p^{(k)}}q⟩
```

until the stopping criterion (see the `stopping_criterion` keyword is fulfilled.

# Input

$(_args([:M, :f]))
  total cost function ``f = g - h``
* `g`: the smooth part ``g`` of the cost function
$(_args(:subgrad_f; name = "∂h", f = "h"))
$(_args(:p))

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:evaluation))
* `gradient=missing`:        specify ``$(_tex(:grad)) f``, for debug / analysis or enhancing the `stopping_criterion=`
* `grad_g=missing`:          specify the gradient of `g`. If specified, a subsolver is automatically set up.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-8)"))
* `g=missing`:               specify the function `g` If specified, a subsolver is automatically set up.
* `sub_cost=`[`LinearizedDCCost`](@ref)`(g, p, initial_vector)`: a cost to be used within the default `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_grad=`[`LinearizedDCGrad`](@ref)`(grad_g, p, initial_vector; evaluation=evaluation)`:
  gradient to be used within the default `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_hess`:              (a finite difference approximation using `sub_grad` by default):
   specify a Hessian of the `sub_cost`, which the default solver, see `sub_state=` needs.
  $(_note(:KeywordUsedIn, "sub_objective"))
$(_kwargs(:sub_kwargs))
* `sub_objective`:         a gradient or Hessian objective based on `sub_cost=`, `sub_grad=`, and `sub_hess` if provided
   the objective used within `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_problem"))
$(_kwargs(:sub_state; default = "([`GradientDescentState`](@ref) or [`TrustRegionsState`](@ref) if `sub_hess` is provided)"))
$(_kwargs(:sub_problem; default = "`[`DefaultManoptProblem`](@ref)`(M, sub_objective)"))
* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-9)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-9)`:
  a stopping criterion used within the default `sub_state=`
  $(_note(:KeywordUsedIn, "sub_state"))
* `sub_stepsize=`[`ArmijoLinesearch`](@ref)`(M)`) specify a step size used within the `sub_state`.
  $(_note(:KeywordUsedIn, "sub_state"))
$(_kwargs(:X; add_properties = [:as_Memory]))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_DoC)"
difference_of_convex_algorithm(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_algorithm(
        M::AbstractManifold, f, g, ∂h, p = rand(M);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        grad_g = missing, gradient = missing, kwargs...,
    )
    p_ = maybe_wrap_variable(p)
    mdco = ManifoldDifferenceOfConvexObjective(
        f, ∂h; gradient = gradient, evaluation = evaluation, p = p
    )
    # to mutating
    g_ = ismissing(g) ? missing : maybe_wrap_function(g, p)
    grad_g_ = ismissing(grad_g) ? missing : maybe_wrap_function(grad_g, p)
    gradient_ = ismissing(gradient) ? missing : maybe_wrap_function(gradient, p)
    rs = difference_of_convex_algorithm(
        M, mdco, p_;
        g = g_, evaluation = evaluation, gradient = gradient_, grad_g = grad_g_, kwargs...,
    )
    return maybe_unwrap_variable(p, rs)
end
function difference_of_convex_algorithm(
        M::AbstractManifold, mdco::O, p; kwargs...
    ) where {O <: Union{ManifoldDifferenceOfConvexObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(difference_of_convex_algorithm; kwargs...)
    q = copy(M, p)
    return difference_of_convex_algorithm!(M, mdco, q; kwargs...)
end
calls_with_kwargs(::typeof(difference_of_convex_algorithm)) = (difference_of_convex_algorithm!,)

@doc "$(_doc_DoC)"
difference_of_convex_algorithm!(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_algorithm!(
        M::AbstractManifold, f, g, ∂h, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), gradient = missing, kwargs...,
    )
    mdco = ManifoldDifferenceOfConvexObjective(f, ∂h; gradient = gradient, evaluation = evaluation)
    return difference_of_convex_algorithm!(M, mdco, p; g = g, evaluation = evaluation, kwargs...)
end
function difference_of_convex_algorithm!(
        M::AbstractManifold, mdco::O, p;
        callbacks = Dict{Symbol, Function}(),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = missing, grad_g = missing, gradient = missing,
        X = zero_vector(M, p),
        objective_type = :Riemannian,
        stopping_criterion = if ismissing(gradient)
            StopAfterIteration(300) | StopWhenChangeLess(M, 1.0e-9)
        else
            StopAfterIteration(300) | StopWhenChangeLess(M, 1.0e-9) | StopWhenGradientNormLess(1.0e-9)
        end,
        # Subsolver Magic Cascade.
        sub_cost = ismissing(g) ? missing : LinearizedDCCost(g, copy(M, p), copy(M, p, X)),
        sub_grad = ismissing(grad_g) ? missing : LinearizedDCGrad(grad_g, copy(M, p), copy(M, p, X); evaluation = evaluation),
        sub_hess = ismissing(sub_grad) ? missing : ApproxHessianFiniteDifference(M, copy(M, p), sub_grad; evaluation = evaluation),
        sub_kwargs = (;),
        sub_stopping_criterion = StopAfterIteration(300) | StopWhenGradientNormLess(1.0e-8),
        sub_objective = if ismissing(sub_cost) || ismissing(sub_grad)
            missing
        else
            decorate_objective!(
                M,
                if ismissing(sub_hess)
                    ManifoldGradientObjective(sub_cost, sub_grad; evaluation = evaluation)
                else
                    ManifoldHessianObjective(sub_cost, sub_grad, sub_hess; evaluation = evaluation)
                end;
                objective_type = objective_type,
                sub_kwargs...,
            )
        end,
        sub_problem::Union{AbstractManoptProblem, Function, Missing} = if ismissing(sub_objective)
            missing
        else
            DefaultManoptProblem(M, sub_objective)
        end,
        sub_state::Union{AbstractManoptSolverState, AbstractEvaluationType, Missing} = if sub_problem isa
                Function
            evaluation
        elseif ismissing(sub_objective)
            missing
        else
            decorate_state!(
                if ismissing(sub_hess)
                    GradientDescentState(
                        M; p = copy(M, p), stopping_criterion = sub_stopping_criterion, sub_kwargs...
                    )
                else
                    TrustRegionsState(
                        M, sub_objective;
                        p = copy(M, p), stopping_criterion = sub_stopping_criterion, sub_kwargs...
                    )
                end;
                sub_kwargs...,
            )
        end,
        kwargs..., #collect rest
    ) where {O <: Union{ManifoldDifferenceOfConvexObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(difference_of_convex_algorithm!; kwargs...)
    dmdco = decorate_objective!(M, mdco; objective_type = objective_type, kwargs...)
    dmp = DefaultManoptProblem(M, dmdco)
    ismissing(sub_problem) && error(
        """
        Subproblem seems to be missing. Please provide _either_
        * a `sub_problem=` to be solved
        * a `sub_objective` to automatically generate the sub problem,
        * `sub_grad=` (as well as the usually given `sub_cost=`) to automatically generate the sub objective _or_
        * `grad_g=` keywords to automatically generate the sub problems gradient.
        """,
    )
    dcs = DifferenceOfConvexState(
        M, sub_problem, sub_state;
        callbacks = process_callbacks_arg(callbacks, DifferenceOfConvexState),
        p = p, stopping_criterion = stopping_criterion, X = X,
    )
    ddcs = decorate_state!(dcs; kwargs...)
    solve!(dmp, ddcs)
    return get_solver_return(get_objective(dmp), ddcs)
end
calls_with_kwargs(::typeof(difference_of_convex_algorithm!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(::AbstractManoptProblem, dcs::DifferenceOfConvexState)
    return dcs
end
function step_solver!(amp::AbstractManoptProblem, dcs::DifferenceOfConvexState, k)
    M = get_manifold(amp)
    get_subtrahend_gradient!(amp, dcs.X, dcs.p)
    set_parameter!(dcs.sub_problem, Val(:Objective), Val(:Cost), Val(:p), dcs.p)
    set_parameter!(dcs.sub_problem, Val(:Objective), Val(:Cost), Val(:X), dcs.X)
    set_parameter!(dcs.sub_problem, Val(:Objective), Val(:Gradient), Val(:p), dcs.p)
    set_parameter!(dcs.sub_problem, Val(:Objective), Val(:Gradient), Val(:X), dcs.X)
    set_iterate!(dcs.sub_state, M, copy(M, dcs.p))
    callback(:BeforeSubsolver, amp, dcs, k)
    solve!(dcs.sub_problem, dcs.sub_state) # call the subsolver
    callback(:Subsolver, amp, dcs, k)
    # copy result from subsolver to current iterate
    copyto!(M, dcs.p, get_solver_result(dcs.sub_state))
    # small hack: store `gradient_f` in X at end of iteration for the gradient norm stopping criterion
    !ismissing(get_gradient_function(get_objective(amp))) && get_gradient!(amp, dcs.X, dcs.p)
    return dcs
end
#
# Variant II: sub task is a mutating function providing a closed form solution
#
function step_solver!(
        amp::AbstractManoptProblem,
        dcs::DifferenceOfConvexState{P, T, F, ClosedFormSubSolverState},
        k,
    ) where {P, T, F}
    M = get_manifold(amp)
    get_subtrahend_gradient!(amp, dcs.X, dcs.p) # evaluate grad F in place for O.X
    callback(:BeforeSubsolver, amp, dcs, k)
    dcs.sub_problem(M, dcs.p, dcs.p, dcs.X) # evaluate the closed form solution and store the result in p
    callback(:Subsolver, amp, dcs, k)
    return dcs
end
get_solver_result(dcs::DifferenceOfConvexState) = dcs.p
