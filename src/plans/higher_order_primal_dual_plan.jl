# TODO: change documentation
# DONE: addapted struct and functions
@doc raw"""
    PrimalDualProblem {mT <: AbstractManifold, nT <: AbstractManifold} <: PrimalDualProblem} <: Problem

Describes a Problem for the linearized Chambolle-Pock algorithm.

# Fields

* `M`, `N` – two manifolds $\mathcal M$, $\mathcal N$
* `cost` $F + G(Λ(⋅))$ to evaluate interims cost function values
* `forward_oprator` the operator for the forward operation in the algorthm, either $Λ$ (exact) or $DΛ$ (linearized).
* `linearized_adjoint_operator` The adjoint differential $(DΛ)^* \colon \mathcal N \to T\mathcal M$
* `prox_F` the proximal map belonging to $f$
* `prox_G_dual` the proximal map belonging to $g_n^*$
* `Λ` – (`fordward_operator`) for the linearized variant, this has to be set to the exact forward operator.
  This operator is required in several variants of the linearized algorithm.
  Since the exact variant is the default, `Λ` is by default set to `forward_operator`.

# Constructor

    LinearizedPrimalDualProblem(M, N, cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,Λ=forward_operator)

"""
mutable struct PrimalDualSemismoothNewtonProblem{
    T,mT<:AbstractManifold,nT<:AbstractManifold
} <: Problem{T}
    M::mT
    N::nT
    cost::Function
    prox_F!!::Function
    diff_prox_F!!::Function
    prox_G_dual!!::Function
    diff_prox_G_dual!!::Function
    forward_operator!!::Function
    adjoint_linearized_operator!!::Function
    Λ!!::Union{Function,Missing}
end
function PrimalDualSemismoothNewtonProblem(
    M::mT,
    N::nT,
    cost,
    prox_F,
    diff_prox_F,
    prox_G_dual,
    diff_prox_G_dual,
    forward_operator,
    adjoint_linearized_operator,
    Λ=forward_operator,
) where {mT<:AbstractManifold,nT<:AbstractManifold}
    return PrimalDualSemismoothNewtonProblem{mT,nT}(
        M,
        N,
        cost,
        prox_F,
        diff_prox_F,
        prox_G_dual,
        diff_prox_G_dual,
        forward_operator,
        adjoint_linearized_operator,
        Λ,
    )
end

@doc raw"""
    PrimalDualSemismoothNewtonOptions <: PrimalDualOptions

* `m` - base point on $ \mathcal M $
* `n` - base point on $ \mathcal N $
* `x` - an initial point on $x^{(0)} \in \mathcal M$ (and its previous iterate)
* `ξ` - an initial tangent vector $\xi^{(0)}\in T^*\mathcal N$ (and its previous iterate)
* `xbar` - the relaxed iterate used in the next dual update step (when using `:primal` relaxation)
* `ξbar` - the relaxed iterate used in the next primal update step (when using `:dual` relaxation)
* `Θ` – factor to damp the helping $\tilde x$
* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `dual_stepsize` – (`1/sqrt(8)`) proximnal parameter of the dual prox
* `acceleration` – (`0.`) acceleration factor due to Chambolle & Pock
* `relaxation` – (`1.`) relaxation in the primal relaxation step (to compute `xbar`)
* `relax` – (`_primal`) which variable to relax (`:primal` or `:dual`)
* `stop` - a [`StoppingCriterion`](@ref)
* `type` – (`exact`) whether to perform an `:exact` or `:linearized` Chambolle-Pock
* `update_primal_base` (`(p,o,i) -> o.m`) function to update the primal base
* `update_dual_base` (`(p,o,i) -> o.n`) function to update the dual base
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use
* `inverse_retraction_method` - (`LogarithmicInverseRetraction()`) an inverse retraction to use.
* `vector_transport_method` - (`ParallelTransport()`) a vector transport to use

where for the last two the functions a [`Problem`](@ref)` p`,
[`Options`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing` in the linearized case).

# Constructor
    PrimalDualSemismoothNewtonOptions(m::P, n::Q, x::P, ξ::T, primal_stepsize::Float64, dual_stepsize::Float64;
        acceleration::Float64 = 0.0,
        relaxation::Float64 = 1.0,
        relax::Symbol = :primal,
        stopping_criterion::StoppingCriterion = StopAfterIteration(300),
        variant::Symbol = :exact,
        update_primal_base::Union{Function,Missing} = missing,
        update_dual_base::Union{Function,Missing} = missing,
        retraction_method = ExponentialRetraction(),
        inverse_retraction_method = LogarithmicInverseRetraction(),
        vector_transport_method = ParallelTransport(),
    )
"""
mutable struct PrimalDualSemismoothNewtonOptions{
    P,
    Q,
    T,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
} <: PrimalDualOptions
    m::P
    n::Q
    x::P
    # xbar::P
    ξ::T
    # ξbar::T
    primal_stepsize::Float64
    dual_stepsize::Float64
    # acceleration::Float64
    # relaxation::Float64
    # relax::Symbol
    stop::StoppingCriterion
    # variant::Symbol
    update_primal_base::Union{Function,Missing}
    update_dual_base::Union{Function,Missing}
    retraction_method::RM
    inverse_retraction_method::IRM
    vector_transport_method::VTM

    function PrimalDualSemismoothNewtonOptions(
        m::P,
        n::Q,
        x::P,
        ξ::T,
        primal_stepsize::Float64=1 / sqrt(8),
        dual_stepsize::Float64=1 / sqrt(8);
        # acceleration::Float64=0.0,
        # relaxation::Float64=1.0,
        # relax::Symbol=:primal,
        stopping_criterion::StoppingCriterion=StopAfterIteration(50),
        # variant::Symbol=:exact,
        update_primal_base::Union{Function,Missing}=missing,
        update_dual_base::Union{Function,Missing}=missing,
        retraction_method::RM=ExponentialRetraction(),
        inverse_retraction_method::IRM=LogarithmicInverseRetraction(),
        vector_transport_method::VTM=ParallelTransport(),
    ) where {
        P,
        Q,
        T,
        RM<:AbstractRetractionMethod,
        IRM<:AbstractInverseRetractionMethod,
        VTM<:AbstractVectorTransportMethod,
    }
        return new{P,Q,T,RM,IRM,VTM}(
            m,
            n,
            x,
            # deepcopy(x),
            ξ,
            # deepcopy(ξ),
            primal_stepsize,
            dual_stepsize,
            # acceleration,
            # relaxation,
            # relax,
            stopping_criterion,
            # variant,
            update_primal_base,
            update_dual_base,
            retraction_method,
            inverse_retraction_method,
            vector_transport_method,
        )
    end
end
