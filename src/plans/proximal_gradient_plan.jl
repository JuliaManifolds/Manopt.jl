@doc raw"""
    ManifoldProximalGradientObjective{E,<:AbstractEvaluationType, TC, TG, TP} <: AbstractManifoldObjective{E,TC,TG}

Model an objective of the form
```math
    f(p) = g(p) + h(p),\qquad p \in \mathcal M,
```
where ``g: \mathcal M → \bar{ℝ}`` is differentiable
and ``h: → \bar{ℝ}`` is convex, lower semicontinous, and proper.

This objective also provides ``\operatorname{grad} g`` and ``\operatorname{prox}_{λ} h``.

# Fields

* `cost`: the overall cost ``f```
* `gradient_g`: the ``\operatorname{grad} g``
* `proximal_map_h` and ``\operatorname{prox}_{λ} h``

# Constructor
    ManifoldProximalGradientObjective(f, prox_g, prox_h;
        evalauation=[`AllocatingEvaluation`](@ref)
    )

Generate the proximal gradient objective given the cost `f`, the gradient of the smooth
component `grad_g`, and the proximal map of the nonsmooth component `prox_h`.

## Keyword arguments

* `evaluation=`[`AllocatingEvaluation`](@ref): whether the gradient and proximal map
  is given as an allocation function or an in-place ([`InplaceEvaluation`](@ref)).
"""
struct ManifoldProximalGradientObjective{E<:AbstractEvaluationType,TC,TG,TP} <:
       AbstractManifoldCostObjective{E,TC}
    cost::TC
    gradient_g!!::TG
    proximal_map_h!!::TP
    function ManifoldProximalGradientObjective(
        f::TF, grad_g::TG, prox_h::TP; evaluation::E=AllocatingEvaluation()
    ) where {TF,TG,TP,E<:AbstractEvaluationType}
        return new{E,TF,TG,TP}(f, grad_g, prox_h)
    end
end

"""
    get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, mgo::ManifoldProximalGradientObjective, p)

evaluate the gradient of the smooth part of a [`ManifoldProximalGradientObjective`](@ref) `mgo` at `p`.
"""
get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)

function get_gradient(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, p
)
    return mpgo.gradient_g!!(M, p)
end
function get_gradient(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    mpgo.gradient_g!!(M, X, p)
    return X
end

function get_gradient!(
    M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, p
)
    copyto!(M, X, p, mpgo.gradient_g!!(M, p))
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, p
)
    mpgo.gradient_g!!(M, X, p)
    return X
end

@doc raw"""
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalGradientObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalGradientObjective, λ, p)

evaluate proximal map of the nonsmooth component ``h`` of the [`ManifoldProximalGradientObjective`](@ref)` mpo`
at the point `p` on `M` with parameter ``λ>0``.
"""
get_proximal_map(::AbstractManifold, ::ManifoldProximalGradientObjective, ::Any...)

function get_proximal_map(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, λ, p
)
    return mpgo.proximal_map_h!!(M, λ, p)
end
function get_proximal_map!(
    M::AbstractManifold,
    q,
    mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation},
    λ,
    p,
)
    copyto!(M, q, mpgo.proximal_map_h!!(M, λ, p))
    return q
end

function get_proximal_map(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, λ, p
)
    q = allocate_result(M, get_proximal_map, p)
    mpgo.proximal_map_h!!(M, q, λ, p)
    return q
end
function get_proximal_map!(
    M::AbstractManifold, q, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, λ, p
)
    mpgo.proximal_map_h!!(M, q, λ, p)
    return q
end
#
#
# Solver State
@doc """
    ProximalGradientMethodState <: AbstractManoptSolverState

stores options for the [`proximal_gradient_method`](@ref) solver

# Fields

$(_var(:Field, :p, "a"; add=" storing the acceleration step"))
$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :p, "q"; add=" storing the gradient step"))
$(_var(:Field, :retraction_method))
$(_var(:Field, :X))
$(_var(:Field, :stopping_criterion, "stop"))
* `acceleration`: a function `(problem, state, k) -> state` to compute an acceleration, that is performed before the gradient step
$(_var(:Field, :sub_problem, "sub_problem", "Union{AbstractManoptProblem, F}"; add="or nothing to take the proximal map from the [`ManifoldProximalGradientObjective`](@ref)"))
$(_var(:Field, :sub_state; add="This field is ignored, if the `sub_problem` is `Nothing`"))
* `λ`:                         a function for the values of ``λ_i`` per iteration ``ì``

# Constructor

    ProximalGradientMethodState(M::AbstractManifold; kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

## Input

$(_var(:Argument, :M; type=true))

# Keyword arguments

* `λ = k -> 0.5`
$(_var(:Keyword, :p; add=:as_Initial))
$(_var(:Keyword, :retraction_method))
* `acceleration=(p, s, k) -> (copyto!(get_manifold(M), s.a, s.p); s)` by default no acceleration is performed
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(100)`"))
$(_var(:Keyword, :sub_problem; default="nothing"))
$(_var(:Keyword, :sub_state; default=_var(:evaluation, :default)))
$(_var(:Keyword, :X; add=:as_Memory))
"""
mutable struct ProximalGradientMethodState{
    P,
    T,
    Pr<:Union{<:AbstractManoptProblem,F,Nothing} where {F},
    St<:AbstractManoptSolverState,
    A,
    S<:StoppingCriterion,
    Λ,
    RM<:AbstractRetractionMethod,
} <: AbstractManoptSolverState
    a::P
    acceleration::A
    λ::Λ
    p::P
    q::P
    stop::S
    X::T
    retraction_method::RM
    sub_problem::Pr
    sub_state::St
end
function ProximalGradientMethodState(
    M::AbstractManifold;
    acceleration::A=function (pr, st, k)
        copyto!(get_manifold(pr), s.a, s.p)
        return st
    end,
    p::P=rand(M),
    stopping_criterion::S=StopAfterIteration(100),
    λ::Λ=i -> 0.25,
    X::T=zero_vector(M, p),
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    sub_problem::Pr=nothing,
    sub_state::St=AllocatingEvaluation(),
) where {
    P,
    T,
    S<:StoppingCriterion,
    A,
    Λ,
    Pr<:Union{<:AbstractManoptProblem,F,Nothing} where {F},
    St<:Union{<:AbstractManoptSolverState,<:AbstractEvaluationType},
    RM<:AbstractRetractionMethod,
}
    _sub_state = if sub_state isa AbstractEvaluationType
        ClosedFormSubSolverState(; evaluation=sub_state)
    else
        sub_state
    end
    return ProximalGradientMethodState{P,T,Pr,typeof(_sub_state),A,S,Λ,RM}(
        copy(M, p),
        acceleration,
        λ,
        p,
        copy(M, p),
        stopping_criterion,
        X,
        retraction_method,
        sub_problem,
        _sub_state,
    )
end
get_iterate(pgms::ProximalGradientMethodState) = pgms.p
function set_iterate!(pgms::ProximalGradientMethodState, p)
    pgms.p = p
    return p
end
function show(io::IO, pgms::ProximalGradientMethodState)
    i = get_count(pgms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pgms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Proximal Gradient Method
    $Iter

    ## Parameters

    * retraction_method:              $(pgms.retraction_method)

    ## Stopping criterion

    $(status_summary(pgms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#
#
# Accelerations
@doc """
    ProxGradAcceleration{P, T, F}

Compute an acceleration step

```math
a^{(k)} = $(_tex(:retr))_{p^{(k)}}$(_tex(:bigl))(
  -β_k$(_tex(:invretr))_{p^{(k)}}(p)
```

where `p^{(k)}`` is the current iterate from the [`ProximalGradientMethodState`](@ref)s
field `p` and the result is stored in `state.a`
`p` is the internal field of this struct and stored the last iterate.

The retraction and its inverse are also taken from the state

# Fields

$(_var(:Field, :p; add="the last iterate"))
* `β::F`
$(_var(:Field, :inverse_retraction_method))
$(_var(:Field, :X))

# Constructor

    ProxGradAcceleration(M::AbstractManifold; kwargs...)

Generate the state for a given manifold `M` with initial iterate `p`.

## Input

$(_var(:Argument, :M; type=true))

# Keyword arguments

* `β = k -> (k-1)/(k+1)`
$(_var(:Keyword, :p; add=:as_Initial))
$(_var(:Keyword, :X; add=:as_Memory))
"""
struct ProxGradAcceleration{P,T,F,ITR}
    β::F
    inverse_retraction_method::ITR
    p::P
    X::T
end
function ProxGradAcceleration(
    M::AbstractManifold;
    p::P=rand(M),
    X::T=zero_vector(M, p),
    β::F=(k) -> (k - 1) / (k + 2),
    inverse_retraction_method::I=default_inverse_retraction_method(M, typeof(p)),
) where {P,T,F,I<:AbstractInverseRetractionMethod}
    return ProxGradAcceleration{P,T,F,I}(β, inverse_retraction_method, p, X)
end
function (pga::ProxGradAcceleration)(
    amp::AbstractManoptProblem, pgms::ProximalGradientMethodState, k
)
    # compute the step
    M = get_manifold(amp)
    inverse_retract!(M, pga.X, pgms.p, pga.p)
    # retract with step and store in a
    retract!(M, pgms.a, pgms.p, -pga.β(k) * pga.X)
    # save current p for nect time as last iterate
    copyto!(M, pga.p, pgms.p)
    return pgms
end
