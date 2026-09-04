@doc """
    ManifoldDifferenceOfConvexProximalObjective <: AbstractManifoldFirstOrderObjective

Specify an objective [`difference_of_convex_proximal_point`](@ref) algorithm.
The problem is of the form

```math
    $(_tex(:argmin))_{p∈$(_math(:Manifold))} g(p) - h(p)
```

where both ``g`` and ``h`` are convex, lower semicontinuous and proper.

# Fields

* `cost`:     implementation of ``f(p) = g(p)-h(p)``
* `gradient!`: the gradient of the cost
* `grad_h!`: a function ``$(_tex(:grad))h: $(_math(:Manifold)) → T$(_math(:Manifold))``,

Note that both the gradients might be given in two possible signatures
as allocating or in-place.

 # Constructor

    ManifoldDifferenceOfConvexProximalObjective(
        grad_h;
        cost = missing, gradient = missing, evaluation = AllocatingEvaluation()
    )

and note that neither cost nor gradient are required for the algorithm,
just for eventual debug or recording functionality or for the stopping criterion.
"""
struct ManifoldDifferenceOfConvexProximalObjective{GH, F, G} <: AbstractManifoldFirstOrderObjective{F, G}
    cost::F
    gradient!::G
    grad_h!::GH
    function ManifoldDifferenceOfConvexProximalObjective(
            grad_h::THG; cost::TC = missing, gradient::TG = missing, evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing,
        ) where {TC, TG, THG}
        cost_ = ismissing(cost) ? missing : maybe_wrap_function(cost, p; result = :Number)
        grad_h_ = maybe_wrap_function(grad_h, p, evaluation; result = :TangentVector)
        grad_ = ismissing(gradient) ? missing : maybe_wrap_function(gradient, p, evaluation; result = :TangentVector)
        return new{typeof(grad_h_), typeof(cost_), typeof(grad_)}(cost_, grad_, grad_h_)
    end
end
function get_gradient!(M::AbstractManifold, X, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)
    return dcpo.gradient!(M, X, p)
end
function get_gradient_function(dcpo::ManifoldDifferenceOfConvexProximalObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    ismissing(dcpo.gradient!) && return missing
    if evaluation isa AllocatingEvaluation
        return (M, p) -> dcpo.gradient!(M, zero_vector(M, p), p)
    else
        return dcpo.gradient!
    end
end

@doc """
    X = get_subtrahend_gradient(M::AbstractManifold, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)
    get_subtrahend_gradient!(M::AbstractManifold, X, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)

Evaluate the gradient of the subtrahend ``h`` from within
a [`ManifoldDifferenceOfConvexProximalObjective`](@ref) `dcpo` at the point `p` (in place of `X`).
"""
get_subtrahend_gradient(M::AbstractManifold, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)

function get_subtrahend_gradient(M::AbstractManifold, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)
    X = zero_vector(M, p)
    dcpo.grad_h!(M, X, p)
    return X
end
function get_subtrahend_gradient!(M::AbstractManifold, X, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)
    dcpo.grad_h!(M, X, p)
    return X
end

function Base.show(io::IO, dcpo::ManifoldDifferenceOfConvexProximalObjective)
    print(io, "ManifoldDifferenceOfConvexProximalObjective(")
    print(io, dcpo.grad_h!); print(io, "; ")
    if !ismissing(dcpo.cost)
        print(io, "cost = ")
        print(io, dcpo.cost)
    end
    if !ismissing(dcpo.gradient!)
        !ismissing(dcpo.cost) && print(io, ", ")
        print(io, "gradient = ")
        print(io, dcpo.gradient!)
    end
    return print(io, ")")
end
function status_summary(dcpo::ManifoldDifferenceOfConvexProximalObjective; context::Symbol = :default)
    (context === :short) && (return repr(dcpo))
    cs = ismissing(dcpo.cost) ? "" : "an overall cost"
    gs = ismissing(dcpo.gradient!) ? "" : "an overall gradient"
    cgs = length(cs) * length(gs) > 0 ? "$cs and $gs" : "$cs$gs"
    s = length(cgs) == 0 ? "" : "including $cgs"
    (context === :inline) && (return "A difference of convex proximal objective on a manifold $s")
    csd = ismissing(dcpo.cost) ? "" : "\n* cost `f = g - h`:$(_MANOPT_INDENT)$(dcpo.cost)"
    gsd = ismissing(dcpo.gradient!) ? "" : "\n* gradient of `f` :$(_MANOPT_INDENT)$(dcpo.gradient!)"
    return """
    A difference of convex proximal objective on a manifold.

    ## Functions$(csd)$(gsd)
    * gradient of `h` :$(_MANOPT_INDENT)$(dcpo.grad_h!)"""
end

@doc """
    DifferenceOfConvexProximalState{P, T, Pr, St, S<:Stepsize, SC<:StoppingCriterion, RTR<:AbstractRetractionMethod, ITR<:AbstractInverseRetractionMethod}
        <: AbstractSubProblemSolverState

A struct to store the current state of the algorithm as well as the form.
It comes in two forms, depending on the realization of the `subproblem`.

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:inverse_retraction_method))
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:p; name = "q"))
 storing the gradient step
$(_fields(:p; name = "r"))
  storing the result of the proximal map
$(_fields(:retraction_method))
$(_fields(:stepsize))
$(_fields(:stopping_criterion; name = "stop"))
* `X`: the current gradient
  their common type is set by the keyword `X`
$(_fields([:sub_problem, :sub_state]))

# Constructor

    DifferenceOfConvexProximalState(M::AbstractManifold, sub_problem, sub_state; kwargs...)

construct an difference of convex proximal point state

    DifferenceOfConvexProximalState(M::AbstractManifold, sub_problem; evaluation=AllocatingEvaluation(), kwargs...)

construct an difference of convex proximal point state, where `sub_problem` is a closed form solution with `evaluation` as type of evaluation.

## Input

$(_args([:M, :sub_problem, :sub_state]))

# Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:inverse_retraction_method))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:retraction_method))

$(_kwargs(:stepsize; default = "`[`ConstantLength`](@ref)`()"))
$(_kwargs(:stopping_criterion; default = "`[`StopWhenChangeLess`](@ref)`(1e-8)"))
$(_kwargs(:X; add_properties = [:as_Memory]))
"""
mutable struct DifferenceOfConvexProximalState{
        P, T, Pr, St <: AbstractManoptSolverState, C <: AbstractDict{Symbol}, S <: Stepsize, SC <: StoppingCriterion,
        RTR <: AbstractRetractionMethod, ITR <: AbstractInverseRetractionMethod, Tλ,
    } <: AbstractSubProblemSolverState
    callbacks::C
    inverse_retraction_method::ITR
    λ::Tλ
    p::P
    q::P
    r::P
    retraction_method::RTR
    stepsize::S
    stop::SC
    sub_problem::Pr
    sub_state::St
    X::T
    function DifferenceOfConvexProximalState(
            M::AbstractManifold, sub_problem::Pr, sub_state::St;
            callbacks::C = Dict{Symbol, Function}(),
            p::P = rand(M), X::T = zero_vector(M, p),
            stepsize::S = ConstantStepsize(M),
            stopping_criterion::SC = StopWhenChangeLess(M, 1.0e-8),
            inverse_retraction_method::I = default_inverse_retraction_method(M, typeof(p)),
            retraction_method::R = default_retraction_method(M, typeof(p)),
            λ::Fλ = i -> 1,
        ) where {
            P, T, C <: AbstractDict{Symbol}, Pr <: Union{AbstractManoptProblem, F} where {F},
            S <: Stepsize, St <: AbstractManoptSolverState, SC <: StoppingCriterion,
            I <: AbstractInverseRetractionMethod, R <: AbstractRetractionMethod, Fλ,
        }
        return DifferenceOfConvexProximalState(
            sub_problem, sub_state;
            callbacks = callbacks, λ = λ, p = p, q = copy(M, p), r = copy(M, p), X = X,
            retraction_method = retraction_method, inverse_retraction_method = inverse_retraction_method,
            stepsize = stepsize, stopping_criterion = stopping_criterion,
        )
    end
    function DifferenceOfConvexProximalState(
            sub_problem::Pr, sub_state::St;
            callbacks::C, λ::Fλ, p::P, q::P, r::P, X::T,
            retraction_method::R, inverse_retraction_method::I, stepsize::S, stopping_criterion::SC
        ) where {
            P, T, C <: AbstractDict{Symbol}, Pr <: Union{AbstractManoptProblem, F} where {F},
            S <: Stepsize, St <: AbstractManoptSolverState, SC <: StoppingCriterion,
            I <: AbstractInverseRetractionMethod, R <: AbstractRetractionMethod, Fλ,
        }
        return new{P, T, Pr, St, C, S, SC, R, I, Fλ}(
            callbacks, inverse_retraction_method, λ, p, q, r,
            retraction_method, stepsize, stopping_criterion, sub_problem, sub_state, X,
        )
    end
end
additional_callbacks(::Type{<:DifferenceOfConvexProximalState}) = [:BeforeSubsolver, :Subsolver, :Stepsize]
get_callbacks(dcps::DifferenceOfConvexProximalState) = dcps.callbacks
# resolve an ambiguity
DifferenceOfConvexProximalState(M::AbstractManifold, st::AbstractManoptSolverState; kwargs...) = error("Difference of Convex Proximal Method state can not be constructed based on $M and the sub state $st, a sub_problem is missing")
function DifferenceOfConvexProximalState(M::AbstractManifold, sub_problem, e::AbstractEvaluationType; kwargs...)
    sub_problem_ = maybe_wrap_function(sub_problem, e)
    return DifferenceOfConvexProximalState(M, sub_problem_, ClosedFormSubSolverState(); kwargs...)
end
function DifferenceOfConvexProximalState(
        M::AbstractManifold, sub_problem;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    return DifferenceOfConvexProximalState(M, sub_problem, evaluation; kwargs...)
end
get_iterate(dcps::DifferenceOfConvexProximalState) = dcps.p
function set_iterate!(dcps::DifferenceOfConvexProximalState, M, p)
    copyto!(M, dcps.p, p)
    return dcps
end
get_gradient(dcs::DifferenceOfConvexProximalState) = dcs.X
function set_gradient!(dcps::DifferenceOfConvexProximalState, M, p, X)
    copyto!(M, dcps.X, p, X)
    return dcps
end
function get_message(dcs::DifferenceOfConvexProximalState)
    # for now only the sub solver might have messages
    return get_message(dcs.sub_state)
end
function Base.show(io::IO, dcps::DifferenceOfConvexProximalState)
    print(io, "DifferenceOfConvexProximalState(", dcps.sub_problem, ", ", dcps.sub_state, "; ")
    print(io, "callbacks = "); print(io, dcps.callbacks); print(io, ", ")
    print(io, "inverse_retraction_method = "); print(io, dcps.inverse_retraction_method); print(io, ", ")
    print(io, "λ = "); print(io, dcps.λ); print(io, ", ")
    print(io, "p = "); print(io, dcps.p); print(io, ", ")
    print(io, "q = "); print(io, dcps.q); print(io, ", ")
    print(io, "r = "); print(io, dcps.r); print(io, ", ")
    print(io, "retraction_method = "); print(io, dcps.retraction_method); print(io, ", ")
    print(io, "stepsize = "); print(io, dcps.stepsize); print(io, ", ")
    print(io, "stopping_criterion = "); print(io, status_summary(dcps.stop; context = :short)); print(io, ", ")
    print(io, "X = "); print(io, dcps.X)
    return print(io, ")")
end
function status_summary(dcps::DifferenceOfConvexProximalState; context::Symbol = :default)
    (context === :short) && return repr(dcps)
    i = get_count(dcps, :Iterations)
    (context === :inline) && return "A solver state for the difference of convex proximal point algorithm$(_iteration_suffix(dcps))"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(dcps.stop) ? "Yes" : "No"
    as = _callbacks_summary(dcps)
    sub = _in_str(repr(dcps.sub_state); indent = 1, indent_end = "| ")
    s = """
    # Solver state for `Manopt.jl`s Difference of Convex Proximal Point Algorithm
    $Iter
    ## Parameters$(as)
    * retraction method:         $(dcps.retraction_method)
    * inverse retraction method: $(dcps.inverse_retraction_method)
    * sub solver state:
    $(sub)

    ## Stepsize
    $(_in_str(status_summary(dcps.stepsize; context = context); indent = 0, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(dcps.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
    return s
end
#
# Prox approach
#
_doc_DCPPA = """
    difference_of_convex_proximal_point(M, grad_h, p=rand(M); kwargs...)
    difference_of_convex_proximal_point(M, mdcpo, p=rand(M); kwargs...)
    difference_of_convex_proximal_point!(M, grad_h, p; kwargs...)
    difference_of_convex_proximal_point!(M, mdcpo, p; kwargs...)

Compute the difference of convex proximal point algorithm [SouzaOliveira:2015](@cite) to minimize

```math
    $(_tex(:argmin))_{p∈$(_math(:Manifold))} g(p) - h(p)
```

where you have to provide the subgradient ``∂h`` of ``h`` and either
* the proximal map ``$(_tex(:prox))_{λg}`` of `g` as a function `prox_g(M, λ, p)` or  `prox_g(M, q, λ, p)`
* the functions `g` and `grad_g` to compute the proximal map using a sub solver
* your own sub-solver, specified by `sub_problem=`and `sub_state=`

This algorithm performs the following steps given a start point `p`= ``p^{(0)}``.
Then repeat for ``k=0,1,…``

1. ``X^{(k)}  ∈ $(_tex(:grad)) h(p^{(k)})``
2. ``q^{(k)} = $(_tex(:retr))_{p^{(k)}}(λ_kX^{(k)})``
3. ``r^{(k)} = $(_tex(:prox))_{λ_kg}(q^{(k)})``
4. ``X^{(k)} = $(_tex(:invretr))_{p^{(k)}}(r^{(k)})``
5. Compute a stepsize ``s_k`` and
6. set ``p^{(k+1)} = $(_tex(:retr))_{p^{(k)}}(s_kX^{(k)})``.

until the `stopping_criterion` is fulfilled.

See [AlmeidaNetoOliveiraSouza:2020](@cite) for more details on the modified variant,
where steps 4-6 are slightly changed, since here the classical proximal point method for
DC functions is obtained for ``s_k = 1`` and one can hence employ usual line search method.

# Input

$(_args([:M, :f]))
  total cost function ``f = g - h``
$(_args(:grad_f; name = "grad_h", f = "h"))
$(_args(:p))

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `λ`:                          ( `k -> 1/2` ) a function returning the sequence of prox parameters ``λ_k``
* `cost=missing`: provide the cost `f`, for debug reasons / analysis
$(_kwargs(:evaluation))
* `gradient=missing`: specify ``$(_tex(:grad)) f``, for debug / analysis
   or enhancing the `stopping_criterion`
* `prox_g=missing`: specify a proximal map for the sub problem _or_ both of the following
* `g=missing`: specify the function `g`.
* `grad_g=missing`: specify the gradient of `g`. If both `g`and `grad_g` are specified, a subsolver is automatically set up.
$(_kwargs([:inverse_retraction_method, :retraction_method]))
$(_kwargs(:stepsize; default = "`[`ConstantLength`](@ref)`()"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1.0e-9)`, plus `[`StopWhenGradientNormLess`](@ref)`(1.0e-9)` when a gradient is provided"))
* `sub_cost=`[`ProximalDCCost`](@ref)`(g, copy(M, p), λ(1))`):
  cost to be used within the default `sub_problem` that is initialized as soon as `g` is provided.
  $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_grad=`[`ProximalDCGrad`](@ref)`(grad_g, copy(M, p), λ(1); evaluation=evaluation)`:
  gradient to be used within the default `sub_problem`, that is initialized as soon as `grad_g` is provided.
  $(_note(:KeywordUsedIn, "sub_objective"))
* `sub_hess`:              (a finite difference approximation using `sub_grad` by default):
   specify a Hessian of the `sub_cost`, which the default solver, see `sub_state=` needs.
$(_kwargs(:sub_kwargs))
* `sub_objective`:         a gradient or Hessian objective based on `sub_cost=`, `sub_grad=`, and `sub_hess`if provided
   the objective used within `sub_problem`.
  $(_note(:KeywordUsedIn, "sub_problem"))
$(_kwargs(:sub_problem; default = "`[`DefaultManoptProblem`](@ref)`(M, sub_objective)"))
$(_kwargs(:sub_state; default = "(`[`GradientDescentState`](@ref)` or `[`TrustRegionsState`](@ref)` if `sub_hess` is provided)"))
$(_kwargs(:stopping_criterion; name = "sub_stopping_criterion", default = "(`[`StopAfterIteration`](@ref)`(300)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)"))
  $(_note(:KeywordUsedIn, "sub_state"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_DCPPA)"
difference_of_convex_proximal_point(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_proximal_point(
        M::AbstractManifold, grad_h, p = rand(M);
        cost = missing, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        gradient = missing, g = missing, grad_g = missing, prox_g = missing,
        kwargs...,
    )
    keywords_accepted(difference_of_convex_proximal_point; kwargs...)
    p_ = maybe_wrap_variable(p)
    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        grad_h; cost = cost, gradient = gradient, evaluation = evaluation, p = p
    )
    # to mutating
    cost_ = ismissing(cost) ? missing : maybe_wrap_function(cost, p)
    g_ = ismissing(g) ? missing : maybe_wrap_function(g, p)
    prox_g_ = ismissing(prox_g) ? missing : maybe_wrap_function(prox_g, p)
    grad_g_ = ismissing(grad_g) ? missing : maybe_wrap_function(grad_g, p)
    gradient_ = ismissing(gradient) ? missing : maybe_wrap_function(gradient, p)
    rs = difference_of_convex_proximal_point(
        M, mdcpo, p_;
        cost = cost_, evaluation = evaluation,
        gradient = gradient_, g = g_, grad_g = grad_g_, prox_g = prox_g_,
        kwargs...,
    )
    return maybe_unwrap_variable(p, rs)
end

function difference_of_convex_proximal_point(
        M::AbstractManifold, mdcpo::O, p; kwargs...
    ) where {
        O <: Union{ManifoldDifferenceOfConvexProximalObjective, AbstractDecoratedManifoldObjective},
    }
    keywords_accepted(difference_of_convex_proximal_point; kwargs...)
    q = copy(M, p)
    return difference_of_convex_proximal_point!(M, mdcpo, q; kwargs...)
end
calls_with_kwargs(::typeof(difference_of_convex_proximal_point)) = (difference_of_convex_proximal_point!,)

@doc "$(_doc_DCPPA)"
difference_of_convex_proximal_point!(M::AbstractManifold, args...; kwargs...)
function difference_of_convex_proximal_point!(
        M::AbstractManifold, grad_h, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        cost = missing, gradient = missing, kwargs...,
    )
    mdcpo = ManifoldDifferenceOfConvexProximalObjective(
        grad_h; cost = cost, gradient = gradient, evaluation = evaluation
    )
    return difference_of_convex_proximal_point!(
        M, mdcpo, p; evaluation = evaluation, kwargs...
    )
end
function difference_of_convex_proximal_point!(
        M::AbstractManifold, mdcpo::O, p;
        callbacks = Dict{Symbol, Function}(),
        g = missing, grad_g = missing, prox_g = missing,
        X = zero_vector(M, p),
        λ = i -> 1 / 2,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        inverse_retraction_method = default_inverse_retraction_method(M, typeof(p)),
        objective_type = :Riemannian,
        retraction_method = default_retraction_method(M, typeof(p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = ConstantLength(M),
        stopping_criterion = if ismissing(get_gradient_function(mdcpo))
            StopAfterIteration(300) | StopWhenChangeLess(M, 1.0e-9)
        else
            StopAfterIteration(300) | StopWhenChangeLess(M, 1.0e-9) | StopWhenGradientNormLess(1.0e-9)
        end,
        sub_cost = ismissing(g) ? missing : ProximalDCCost(g, copy(M, p), λ(1)),
        sub_grad = if ismissing(grad_g)
            missing
        else
            ProximalDCGrad(grad_g, copy(M, p), λ(1); evaluation = evaluation)
        end,
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
                    ManifoldHessianObjective(
                        sub_cost, sub_grad, sub_hess; evaluation = evaluation
                    )
                end;
                objective_type = objective_type, sub_kwargs...,
            )
        end,
        sub_problem::Union{AbstractManoptProblem, F, Missing} where {F} = if !ismissing(prox_g)
            prox_g # closed form solution
        else
            ismissing(sub_objective) ? missing : DefaultManoptProblem(M, sub_objective)
        end,
        sub_state::Union{AbstractEvaluationType, AbstractManoptSolverState, Missing} = if !ismissing(prox_g)
            evaluation
        elseif ismissing(sub_objective)
            missing
        else
            decorate_state!(
                if ismissing(sub_hess)
                    GradientDescentState(
                        M; p = copy(M, p),
                        stopping_criterion = sub_stopping_criterion, sub_kwargs...,
                    )
                else
                    TrustRegionsState(
                        M,
                        DefaultManoptProblem(
                            TangentSpace(M, copy(M, p)), TrustRegionModelObjective(sub_objective),
                        ),
                        TruncatedConjugateGradientState(TangentSpace(M, p); sub_kwargs...);
                        p = copy(M, p),
                        stopping_criterion = sub_stopping_criterion,
                    )
                end;
                sub_kwargs...,
            )
        end,
        kwargs...,
    ) where {
        O <: Union{ManifoldDifferenceOfConvexProximalObjective, AbstractDecoratedManifoldObjective},
    }
    keywords_accepted(difference_of_convex_proximal_point!; kwargs...)
    # Check whether either the right defaults were provided or a `sub_problem`.
    if ismissing(sub_problem)
        error(
            """
            The `sub_problem` is not correctly initialized. Provide _one of_ the following setups
            * `prox_g` as a closed form solution,
            * `g=` and `grad_g=` keywords to automatically generate the sub cost and gradient,
            * provide individual `sub_cost=` and `sub_grad=` to automatically generate the sub objective,
            * provide a `sub_objective`, _or_
            * provide a `sub_problem=` (consider maybe specifying `sub_state=` to specify the solver)
            """,
        )
    end
    dmdcpo = decorate_objective!(M, mdcpo; objective_type = objective_type, kwargs...)
    dmp = DefaultManoptProblem(M, dmdcpo)
    dcps = DifferenceOfConvexProximalState(
        M, sub_problem, sub_state;
        callbacks = process_callbacks_arg(callbacks, DifferenceOfConvexProximalState),
        p = p, X = X, stepsize = _produce_type(stepsize, M, p),
        stopping_criterion = stopping_criterion,
        inverse_retraction_method = inverse_retraction_method,
        retraction_method = retraction_method,
        λ = λ,
    )
    ddcps = decorate_state!(dcps; kwargs...)
    solve!(dmp, ddcps)
    return get_solver_return(get_objective(dmp), ddcps)
end
calls_with_kwargs(::typeof(difference_of_convex_proximal_point!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(::AbstractManoptProblem, dcps::DifferenceOfConvexProximalState)
    return dcps
end
#=
    Variant I: closed form of the prox
=#
function step_solver!(
        amp::AbstractManoptProblem,
        dcps::DifferenceOfConvexProximalState{P, T, F, ClosedFormSubSolverState},
        k,
    ) where {P, T, F <: Function}
    M = get_manifold(amp)
    # each line is one step in the documented solver steps. Note the reuse of `dcps.X`
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    retract!(M, dcps.q, dcps.p, dcps.λ(k) * dcps.X, dcps.retraction_method)
    callback(:BeforeSubsolver, amp, dcps, k)
    dcps.sub_problem(M, dcps.r, dcps.λ(k), dcps.q)
    callback(:Subsolver, amp, dcps, k)
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    s = dcps.stepsize(amp, dcps, k)
    callback(:Stepsize, amp, dcps, k)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    # store the gradient of `f` in `X` at the end of the iteration for the gradient norm stopping criterion
    !ismissing(get_gradient_function(get_objective(amp, true))) && get_gradient!(amp, dcps.X, dcps.p)
    return dcps
end
#=
    Variant II: subsolver variant of the prox
=#
function step_solver!(
        amp::AbstractManoptProblem,
        dcps::DifferenceOfConvexProximalState{P, T, <:AbstractManoptProblem, <:AbstractManoptSolverState},
        k,
    ) where {P, T}
    M = get_manifold(amp)
    # Evaluate gradient of h into X
    get_subtrahend_gradient!(amp, dcps.X, dcps.p)
    # do a step in that direction
    retract!(M, dcps.q, dcps.p, dcps.λ(k) * dcps.X, dcps.retraction_method)
    # use this point (q) for the proximal map
    set_parameter!(dcps.sub_problem, Val(:Objective), Val(:Cost), Val(:p), dcps.q)
    set_parameter!(dcps.sub_problem, Val(:Objective), Val(:Cost), Val(:λ), dcps.λ(k))
    set_parameter!(dcps.sub_problem, Val(:Objective), Val(:Gradient), Val(:p), dcps.q)
    set_parameter!(dcps.sub_problem, Val(:Objective), Val(:Gradient), Val(:λ), dcps.λ(k))
    set_iterate!(dcps.sub_state, M, copy(M, dcps.q))
    callback(:BeforeSubsolver, amp, dcps, k)
    solve!(dcps.sub_problem, dcps.sub_state)
    copyto!(M, dcps.r, get_solver_result(dcps.sub_state))
    callback(:Subsolver, amp, dcps, k)
    # use that direction
    inverse_retract!(M, dcps.X, dcps.p, dcps.r, dcps.inverse_retraction_method)
    # to determine a step size
    s = dcps.stepsize(amp, dcps, k)
    callback(:Stepsize, amp, dcps, k)
    retract!(M, dcps.p, dcps.p, s * dcps.X, dcps.retraction_method)
    # store the gradient of `f` in `X` at the end of the iteration for the gradient norm stopping criterion
    !ismissing(get_gradient_function(get_objective(amp, true))) && get_gradient!(amp, dcps.X, dcps.p)
    return dcps
end
