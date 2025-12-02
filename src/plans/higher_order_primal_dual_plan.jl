@doc """
    PrimalDualManifoldSemismoothNewtonObjective{E<:AbstractEvaluationType, TC, LO, TALO, PF, DPF, PG, DPG, L} <: AbstractPrimalDualManifoldObjective{E, TC, PF}

Describes a Problem for the Primal-dual Riemannian semismooth Newton algorithm. [DiepeveenLellmann:2021](@cite)

# Fields

* `cost`:                        ``F + G(Λ(⋅))`` to evaluate interim cost function values
* `linearized_operator`:         the linearization ``DΛ(⋅)[⋅]`` of the operator ``Λ(⋅)``.
* `linearized_adjoint_operator`: the adjoint differential ``(DΛ)^* : $(_math(:M; M = "N")) → $(_math(:TM))``
* `prox_F`:                      the proximal map belonging to ``F``
* `diff_prox_F`:                 the (Clarke Generalized) differential of the proximal maps of ``F``
* `prox_G_dual`:                 the proximal map belonging to `G^$(_tex(:ast))_n``
* `diff_prox_dual_G`:            the (Clarke Generalized) differential of the proximal maps of ``G^$(_tex(:ast))_n``
* `Λ`:                           the exact forward operator. This operator is required if `Λ(m)=n` does not hold.

# Constructor

    PrimalDualManifoldSemismoothNewtonObjective(cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,Λ)
"""
mutable struct PrimalDualManifoldSemismoothNewtonObjective{
        E <: AbstractEvaluationType, TC, PF, DPF, PG, DPG, LFO, TALO, L,
    } <: AbstractPrimalDualManifoldObjective{E, TC, PF}
    cost::TC
    prox_f!!::PF
    diff_prox_f!!::DPF
    prox_g_dual!!::PG
    diff_prox_g_dual!!::DPG
    linearized_forward_operator!!::LFO
    adjoint_linearized_operator!!::TALO
    Λ!!::L
end
function PrimalDualManifoldSemismoothNewtonObjective(
        cost,
        prox_F,
        diff_prox_F,
        prox_G_dual,
        diff_prox_G_dual,
        linearized_forward_operator,
        adjoint_linearized_operator;
        Λ = missing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    )
    return PrimalDualManifoldSemismoothNewtonObjective{
        typeof(evaluation),
        typeof(cost),
        typeof(prox_F),
        typeof(diff_prox_F),
        typeof(prox_G_dual),
        typeof(diff_prox_G_dual),
        typeof(linearized_forward_operator),
        typeof(adjoint_linearized_operator),
        typeof(Λ),
    }(
        cost,
        prox_F,
        diff_prox_F,
        prox_G_dual,
        diff_prox_G_dual,
        linearized_forward_operator,
        adjoint_linearized_operator,
        Λ,
    )
end

@doc """
    PrimalDualSemismoothNewtonState <: AbstractPrimalDualSolverState

# Fields

$(_var(:Field, :p, "m"))
$(_var(:Field, :p, "n", "Q"; M = "N"))
$(_var(:Field, :p; add = [:as_Iterate]))
$(_var(:Field, :X))
* `primal_stepsize::Float64`:  proximal parameter of the primal prox
* `dual_stepsize::Float64`:    proximal parameter of the dual prox
* `reg_param::Float64`:        regularisation parameter for the Newton matrix
$(_var(:Field, :stopping_criterion, "stop"))
* `update_primal_base`:        function to update the primal base
* `update_dual_base`:          function to update the dual base
$(_var(:Field, :inverse_retraction_method))
$(_var(:Field, :retraction_method))
$(_var(:Field, :vector_transport_method))

where for the update functions a [`AbstractManoptProblem`](@ref) `amp`,
[`AbstractManoptSolverState`](@ref) `ams` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing`).

# Constructor

    PrimalDualSemismoothNewtonState(M::AbstractManifold; kwargs...)

Generate a state for the [`primal_dual_semismooth_Newton`](@ref).

## Keyword arguments

* `m=`$(Manopt._link(:rand))
* `n=`$(Manopt._link(:rand; M = "N"))
* `p=`$(Manopt._link(:rand))
* `X=`$(Manopt._link(:zero_vector))
* `primal_stepsize=1/sqrt(8)`
* `dual_stepsize=1/sqrt(8)`
* `reg_param=1e-5`
* `update_primal_base=(amp, ams, k) -> o.m`
* `update_dual_base=(amp, ams, k) -> o.n`
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :inverse_retraction_method))
$(_var(:Keyword, :stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(50)`"))
$(_var(:Keyword, :vector_transport_method))
"""
mutable struct PrimalDualSemismoothNewtonState{
        P,
        Q,
        T,
        RM <: AbstractRetractionMethod,
        IRM <: AbstractInverseRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
    } <: AbstractPrimalDualSolverState
    m::P
    n::Q
    p::P
    X::T
    primal_stepsize::Float64
    dual_stepsize::Float64
    regularization_parameter::Float64
    stop::StoppingCriterion
    update_primal_base::Union{Function, Missing}
    update_dual_base::Union{Function, Missing}
    retraction_method::RM
    inverse_retraction_method::IRM
    vector_transport_method::VTM

    function PrimalDualSemismoothNewtonState(
            M::AbstractManifold;
            m::P = rand(M),
            n::Q = rand(N),
            p::P = rand(M),
            X::T = zero_vector(M, p),
            primal_stepsize::Float64 = 1 / sqrt(8),
            dual_stepsize::Float64 = 1 / sqrt(8),
            regularization_parameter::Float64 = 1.0e-5,
            stopping_criterion::StoppingCriterion = StopAfterIteration(50),
            update_primal_base::Union{Function, Missing} = missing,
            update_dual_base::Union{Function, Missing} = missing,
            retraction_method::RM = default_retraction_method(M, typeof(p)),
            inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
            vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        ) where {
            P,
            Q,
            T,
            RM <: AbstractRetractionMethod,
            IRM <: AbstractInverseRetractionMethod,
            VTM <: AbstractVectorTransportMethod,
        }
        return new{P, Q, T, RM, IRM, VTM}(
            m,
            n,
            p,
            X,
            primal_stepsize,
            dual_stepsize,
            regularization_parameter,
            stopping_criterion,
            update_primal_base,
            update_dual_base,
            retraction_method,
            inverse_retraction_method,
            vector_transport_method,
        )
    end
end
function show(io::IO, pdsns::PrimalDualSemismoothNewtonState)
    i = get_count(pdsns, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pdsns.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s primal dual semismooth Newton
    $Iter
    ## Parameters
    * primal_stepsize:           $(pdsns.primal_stepsize)
    * dual_stepsize:             $(pdsns.dual_stepsize)
    * regularization_parameter:  $(pdsns.regularization_parameter)
    * retraction_method:         $(pdsns.retraction_method)
    * inverse_retraction_method: $(pdsns.inverse_retraction_method)
    * vector_transport_method:   $(pdsns.vector_transport_method)

    ## Stopping criterion

    $(status_summary(pdsns.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
get_iterate(pdsn::PrimalDualSemismoothNewtonState) = pdsn.p
function set_iterate!(pdsn::PrimalDualSemismoothNewtonState, p)
    pdsn.p = p
    return pdsn
end
@doc """
    y = get_differential_primal_prox(M::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective σ, x)
    get_differential_primal_prox!(p::TwoManifoldProblem, y, σ, x)

Evaluate the differential proximal map of ``F`` stored within [`AbstractPrimalDualManifoldObjective`](@ref)

```math
D$(_tex(:prox))_{σF}(x)[X]
```

which can also be computed in place of `y`.
"""
get_differential_primal_prox(
    M::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, ::Any...
)

function get_differential_primal_prox(tmo::TwoManifoldProblem, σ, p, X)
    M = get_manifold(tmo, 1)
    pdsno = get_objective(tmo)
    return get_differential_primal_prox(M, pdsno, σ, p, X)
end
function get_differential_primal_prox!(tmo::TwoManifoldProblem, Y, σ, p, X)
    M = get_manifold(tmo, 1)
    pdsno = get_objective(tmo)
    get_differential_primal_prox!(M, Y, pdsno, σ, p, X)
    return Y
end

function get_differential_primal_prox(
        M::AbstractManifold,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation},
        σ,
        p,
        X,
    )
    return pdsno.diff_prox_f!!(M, σ, p, X)
end
function get_differential_primal_prox(
        M::AbstractManifold,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation},
        σ,
        p,
        X,
    )
    Y = allocate_result(M, get_differential_primal_prox, p, X)
    pdsno.diff_prox_f!!(M, Y, σ, p, X)
    return Y
end
function get_differential_primal_prox(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, σ, p, X
    )
    return get_differential_primal_prox(M, get_objective(admo, false), σ, p, X)
end
function get_differential_primal_prox!(
        M::AbstractManifold,
        Y,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation},
        σ,
        p,
        X,
    )
    copyto!(M, Y, p, pdsno.diff_prox_f!!(M, σ, p, X))
    return Y
end
function get_differential_primal_prox!(
        M::AbstractManifold,
        Y,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation},
        σ,
        p,
        X,
    )
    pdsno.diff_prox_f!!(M, Y, σ, p, X)
    return Y
end
function get_differential_primal_prox!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, σ, p, X
    )
    return get_differential_primal_prox!(M, Y, get_objective(admo, false), σ, p, X)
end

@doc """
    η = get_differential_dual_prox(N::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, n, τ, X, ξ)
    get_differential_dual_prox!(N::AbstractManifold, pdsno::PrimalDualManifoldSemismoothNewtonObjective, η, n, τ, X, ξ)

Evaluate the differential proximal map of ``G_n^*`` stored within [`PrimalDualManifoldSemismoothNewtonObjective`](@ref)

```math
D$(_tex(:prox))_{τG_n^*}(X)[ξ]
```

which can also be computed in place of `η`.
"""
get_differential_dual_prox(
    ::AbstractManifold,
    ::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation},
    Any...,
)

function get_differential_dual_prox(tmo::TwoManifoldProblem, n, τ, X, ξ)
    N = get_manifold(tmo, 2)
    pdsno = get_objective(tmo)
    return get_differential_dual_prox(N, pdsno, n, τ, X, ξ)
end
function get_differential_dual_prox!(tmo::TwoManifoldProblem, η, n, τ, X, ξ)
    N = get_manifold(tmo, 2)
    pdsno = get_objective(tmo)
    get_differential_dual_prox!(N, η, pdsno, n, τ, X, ξ)
    return η
end

function get_differential_dual_prox(
        N::AbstractManifold,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation},
        n,
        τ,
        X,
        ξ,
    )
    return pdsno.diff_prox_g_dual!!(N, n, τ, X, ξ)
end
function get_differential_dual_prox(
        N::AbstractManifold,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation},
        n,
        τ,
        X,
        ξ,
    )
    η = allocate_result(N, get_differential_dual_prox, X, ξ)
    pdsno.diff_prox_g_dual!!(N, η, n, τ, X, ξ)
    return η
end
function get_differential_dual_prox(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, n, τ, X, ξ
    )
    return get_differential_dual_prox(M, get_objective(admo, false), n, τ, X, ξ)
end
function get_differential_dual_prox!(
        N::AbstractManifold,
        η,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation},
        n,
        τ,
        X,
        ξ,
    )
    copyto!(N, n, η, pdsno.diff_prox_g_dual!!(N, n, τ, X, ξ))
    return η
end
function get_differential_dual_prox!(
        N::AbstractManifold,
        η,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation},
        n,
        τ,
        X,
        ξ,
    )
    pdsno.diff_prox_g_dual!!(N, η, n, τ, X, ξ)
    return η
end
function get_differential_dual_prox!(
        M::AbstractManifold, η, admo::AbstractDecoratedManifoldObjective, n, τ, X, ξ
    )
    return get_differential_dual_prox!(M, η, get_objective(admo, false), n, τ, X, ξ)
end
