@doc """
    PrimalDualManifoldSemismoothNewtonObjective{E<:AbstractEvaluationType, TC, LO, TALO, PF, DPF, PG, DPG, L} <: AbstractPrimalDualManifoldObjective{E, TC, PF}

Describes a Problem for the Primal-dual Riemannian semismooth Newton algorithm. [DiepeveenLellmann:2021](@cite)

# Fields

* `cost`:                        ``F + G(Λ(⋅))`` to evaluate interim cost function values
* `linearized_operator`:         the linearization ``DΛ(⋅)[⋅]`` of the operator ``Λ(⋅)``.
* `linearized_adjoint_operator`: the adjoint differential ``(DΛ)^* : $(_math(:Manifold; M = "N")) → $(_math(:TangentBundle))``
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
        cost::C, prox_F::PF, diff_prox_F::DPF, prox_G_dual::PG, diff_prox_G_dual::DPG,
        linearized_forward_operator::LFO, adjoint_linearized_operator::AL;
        Λ::L = missing, evaluation::E = AllocatingEvaluation(),
    ) where {C, PF, DPF, PG, DPG, LFO, AL, L, E <: AbstractEvaluationType}
    return PrimalDualManifoldSemismoothNewtonObjective{
        E, C, PF, DPF, PG, DPG, LFO, AL, L,
    }(
        cost, prox_F, diff_prox_F, prox_G_dual,
        diff_prox_G_dual, linearized_forward_operator, adjoint_linearized_operator, Λ,
    )
end

@doc """
    PrimalDualSemismoothNewtonState <: AbstractPrimalDualSolverState

# Fields

$(_fields(:callbacks; add_properties = [:as_dict]))
* `dual_stepsize::Float64`:    proximal parameter of the dual prox
$(_fields(:inverse_retraction_method))
$(_fields(:p; name = "m"))
$(_fields(:p; type = "Q", name = "n", M = "N"))
$(_fields(:p; add_properties = [:as_Iterate]))
* `primal_stepsize::Float64`:  proximal parameter of the primal prox
* `reg_param::Float64`:        regularisation parameter for the Newton matrix
$(_fields(:retraction_method))
$(_fields(:stopping_criterion; name = "stop"))
* `update_dual_base`:          function to update the dual base
* `update_primal_base`:        function to update the primal base
$(_fields(:vector_transport_method))
$(_fields(:X))

where for the update functions a [`AbstractManoptProblem`](@ref) `amp`,
[`AbstractManoptSolverState`](@ref) `ams` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing`).

# Constructor

    PrimalDualSemismoothNewtonState(M::AbstractManifold; kwargs...)

Generate a state for the [`primal_dual_semismooth_Newton`](@ref).

## Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `dual_stepsize=1/sqrt(8)`
$(Manopt._kwargs([:inverse_retraction_method]))
* `m=`$(Manopt._link(:rand))
* `n=`$(Manopt._link(:rand; M = "N"))
* `p=`$(Manopt._link(:rand))
* `primal_stepsize=1/sqrt(8)`
* `reg_param=1e-5`
$(Manopt._kwargs([:retraction_method]))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(50)`"))
* `update_dual_base=(amp, ams, k) -> o.n`
* `update_primal_base=(amp, ams, k) -> o.m`
$(_kwargs(:vector_transport_method))
* `X=`$(Manopt._link(:zero_vector))
"""
mutable struct PrimalDualSemismoothNewtonState{
        P, Q, T, C <: AbstractDict{Symbol}, RM <: AbstractRetractionMethod,
        IRM <: AbstractInverseRetractionMethod, VTM <: AbstractVectorTransportMethod,
    } <: AbstractPrimalDualSolverState
    callbacks::C
    dual_stepsize::Float64
    inverse_retraction_method::IRM
    m::P
    n::Q
    p::P
    primal_stepsize::Float64
    regularization_parameter::Float64
    retraction_method::RM
    stop::StoppingCriterion
    update_dual_base::Union{Function, Missing}
    update_primal_base::Union{Function, Missing}
    vector_transport_method::VTM
    X::T
    function PrimalDualSemismoothNewtonState(
            M::AbstractManifold;
            callbacks::C = Dict{Symbol, Function}(),
            dual_stepsize::Float64 = 1 / sqrt(8),
            m::P = rand(M), n::Q = rand(N), p::P = rand(M),
            primal_stepsize::Float64 = 1 / sqrt(8),
            regularization_parameter::Float64 = 1.0e-5,
            stopping_criterion::StoppingCriterion = StopAfterIteration(50),
            update_dual_base::Union{Function, Missing} = missing,
            update_primal_base::Union{Function, Missing} = missing,
            # the following defaults depend on `p`, so they have to be keyword arguments
            # listed after `p` is bound above
            inverse_retraction_method::IRM = default_inverse_retraction_method(M, typeof(p)),
            retraction_method::RM = default_retraction_method(M, typeof(p)),
            vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
            X::T = zero_vector(M, p),
        ) where {
            P, Q, T, C <: AbstractDict{Symbol}, RM <: AbstractRetractionMethod,
            IRM <: AbstractInverseRetractionMethod, VTM <: AbstractVectorTransportMethod,
        }
        return new{P, Q, T, C, RM, IRM, VTM}(
            callbacks, dual_stepsize, inverse_retraction_method, m, n, p,
            primal_stepsize, regularization_parameter, retraction_method,
            stopping_criterion, update_dual_base, update_primal_base,
            vector_transport_method, X,
        )
    end
end
provided_callbacks(::Type{PrimalDualSemismoothNewtonState}) = _MANOPT_DEFAULT_CALLBACKS
get_callbacks(pdsn::PrimalDualSemismoothNewtonState) = pdsn.callbacks

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
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation}, σ, p, X,
    )
    return pdsno.diff_prox_f!!(M, σ, p, X)
end
function get_differential_primal_prox(
        M::AbstractManifold,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation}, σ, p, X,
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
        M::AbstractManifold, Y,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation}, σ, p, X,
    )
    copyto!(M, Y, p, pdsno.diff_prox_f!!(M, σ, p, X))
    return Y
end
function get_differential_primal_prox!(
        M::AbstractManifold, Y,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation}, σ, p, X,
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
    ::AbstractManifold, ::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation}, Any...,
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
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation}, n, τ, X, ξ,
    )
    return pdsno.diff_prox_g_dual!!(N, n, τ, X, ξ)
end
function get_differential_dual_prox(
        N::AbstractManifold,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation}, n, τ, X, ξ,
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
        N::AbstractManifold, η,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{AllocatingEvaluation}, n, τ, X, ξ,
    )
    copyto!(N, n, η, pdsno.diff_prox_g_dual!!(N, n, τ, X, ξ))
    return η
end
function get_differential_dual_prox!(
        N::AbstractManifold, η,
        pdsno::PrimalDualManifoldSemismoothNewtonObjective{InplaceEvaluation}, n, τ, X, ξ,
    )
    pdsno.diff_prox_g_dual!!(N, η, n, τ, X, ξ)
    return η
end
function get_differential_dual_prox!(
        M::AbstractManifold, η, admo::AbstractDecoratedManifoldObjective, n, τ, X, ξ
    )
    return get_differential_dual_prox!(M, η, get_objective(admo, false), n, τ, X, ξ)
end

get_iterate(pdsn::PrimalDualSemismoothNewtonState) = pdsn.p

function set_iterate!(pdsn::PrimalDualSemismoothNewtonState, p)
    pdsn.p = p
    return pdsn
end

function Base.show(io::IO, pdsns::PrimalDualSemismoothNewtonState)
    print(io, "PrimalDualSemismoothNewtonState(; ")
    print(io, "callbacks = ", pdsns.callbacks, ", ")
    print(io, "dual_stepsize = ", pdsns.dual_stepsize, ", ")
    print(io, "inverse_retraction_method = ", pdsns.inverse_retraction_method, ", ")
    print(io, "m = ", pdsns.m, ", n = ", pdsns.n, ", p = ", pdsns.p, ", ")
    print(io, "primal_stepsize = ", pdsns.primal_stepsize, ", ")
    print(io, "regularization_parameter = ", pdsns.regularization_parameter, ", ")
    print(io, "retraction_method = ", pdsns.retraction_method, ", ")
    print(io, "stopping_criterion = ", status_summary(pdsns.stop; context = :short), ", ")
    print(io, "update_dual_base = ", pdsns.update_dual_base, ", update_primal_base = ", pdsns.update_primal_base, ", ")
    print(io, "vector_transport_method = ", pdsns.vector_transport_method, ", X = ", pdsns.X)
    return print(io, ")")
end

function status_summary(pdsns::PrimalDualSemismoothNewtonState; context::Symbol = :default)
    (context === :short) && return repr(pdsns)
    i = get_count(pdsns, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(pdsns.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the primal dual semismooth Newton solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pdsns.stop) ? "Yes" : "No"
    as = _callbacks_summary(pdsns)
    s = """
    # Solver state for `Manopt.jl`s primal dual semismooth Newton
    $Iter
    ## Parameters$(as)
    * primal_stepsize:          $(_MANOPT_INDENT)$(pdsns.primal_stepsize)
    * dual_stepsize:            $(_MANOPT_INDENT)$(pdsns.dual_stepsize)
    * regularization_parameter: $(_MANOPT_INDENT)$(pdsns.regularization_parameter)
    * retraction_method:        $(_MANOPT_INDENT)$(pdsns.retraction_method)
    * inverse_retraction_method:$(_MANOPT_INDENT)$(pdsns.inverse_retraction_method)
    * vector_transport_method:  $(_MANOPT_INDENT)$(pdsns.vector_transport_method)

    ## Stopping criterion
    $(_in_str(status_summary(pdsns.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv"""
    return s
end
function Base.show(io::IO, pdmssno::PrimalDualManifoldSemismoothNewtonObjective{E}) where {E}
    print(io, "PrimalDualManifoldSemismoothNewtonObjective(")
    print(io, pdmssno.cost); print(io, ", ")
    print(io, pdmssno.prox_f!!); print(io, ", ")
    print(io, pdmssno.diff_prox_f!!); print(io, ", ")
    print(io, pdmssno.prox_g_dual!!); print(io, ", ")
    print(io, pdmssno.diff_prox_g_dual!!); print(io, ", ")
    print(io, pdmssno.linearized_forward_operator!!); print(io, ", ")
    print(io, pdmssno.adjoint_linearized_operator!!); print(io, "; ")
    if !ismissing(pdmssno.Λ!!)
        print(io, "Λ = "); print(io, pdmssno.Λ!!); print(io, ", ")
    end
    print(io, _to_kw(E))
    return print(io, ")")
end
function status_summary(pdmssno::PrimalDualManifoldSemismoothNewtonObjective; context::Symbol = :default)
    (context === :short) && return repr(pdmssno)
    (context === :inline) && return "A primal dual semismooth Newton objective"
    Λs = ismissing(pdmssno.Λ!!) ? "" : "\n* Λ:                $(_MANOPT_INDENT)$(pdmssno.Λ!!)"
    return """
    A primal dual semismooth Newton objective

    ## Functions
    * cost:             $(_MANOPT_INDENT)$(pdmssno.cost)
    * prox_f:           $(_MANOPT_INDENT)$(pdmssno.prox_f!!)
    * D prox_f:         $(_MANOPT_INDENT)$(pdmssno.diff_prox_f!!)
    * prox_g*:          $(_MANOPT_INDENT)$(pdmssno.prox_g_dual!!)
    * D prox_g*:        $(_MANOPT_INDENT)$(pdmssno.diff_prox_g_dual!!)
    * lin. forward Op:  $(_MANOPT_INDENT)$(pdmssno.linearized_forward_operator!!)
    * adj. lin. fw. Op.:$(_MANOPT_INDENT)$(pdmssno.adjoint_linearized_operator!!)$(Λs)"""
end
