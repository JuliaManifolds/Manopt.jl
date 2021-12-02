# TODO: change documentation
# DONE: addapted struct and functions
@doc raw"""
    PrimalDualSemismoothNewtonProblem {mT <: AbstractManifold, nT <: AbstractManifold} <: PrimalDualProblem} <: Problem

Describes a Problem for the linearized Chambolle-Pock algorithm.

# Fields

* `M`, `N` – two manifolds $\mathcal M$, $\mathcal N$
* `cost` $F + G(Λ(⋅))$ to evaluate interims cost function values
* `linearized_operator` the linearization $DΛ(⋅)[⋅]$ of the operator $Λ(⋅)$.
* `linearized_adjoint_operator` The adjoint differential $(DΛ)^* \colon \mathcal N \to T\mathcal M$
* `prox_F` the proximal map belonging to $f$
* `diff_prox_F` the (Clarke Generalized) differential of the proximal maps of $F$
* `prox_G_dual` the proximal map belonging to $g_n^*$
* `diff_prox_dual_G` the (Clarke Generalized) differential of the proximal maps of $G^\ast_n$
* `Λ` – the exact forward operator.
  This operator is required if `Λ(m)=n` does not hold.

# Constructor

    PrimalDualSemismoothNewtonProblem(M, N, cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,Λ)

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
    Λ::Union{Function,Missing}=missing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),  # @Ronny: do we need this?
) where {mT<:AbstractManifold,nT<:AbstractManifold}
    return PrimalDualSemismoothNewtonProblem{typeof(evaluation),mT,nT}(
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
* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `dual_stepsize` – (`1/sqrt(8)`) proximnal parameter of the dual prox
* `stop` - a [`StoppingCriterion`](@ref)
* `update_primal_base` (`(p,o,i) -> o.m`) function to update the primal base
* `update_dual_base` (`(p,o,i) -> o.n`) function to update the dual base
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use
* `inverse_retraction_method` - (`LogarithmicInverseRetraction()`) an inverse retraction to use.
* `vector_transport_method` - (`ParallelTransport()`) a vector transport to use

where for the last two the functions a [`Problem`](@ref)` p`,
[`Options`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing`).

# Constructor
    PrimalDualSemismoothNewtonOptions(m::P, n::Q, x::P, ξ::T, primal_stepsize::Float64, dual_stepsize::Float64;
        stopping_criterion::StoppingCriterion = StopAfterIteration(300),
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
    ξ::T
    primal_stepsize::Float64
    dual_stepsize::Float64
    stop::StoppingCriterion
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
        stopping_criterion::StoppingCriterion=StopAfterIteration(50),
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
            ξ,
            primal_stepsize,
            dual_stepsize,
            stopping_criterion,
            update_primal_base,
            update_dual_base,
            retraction_method,
            inverse_retraction_method,
            vector_transport_method,
        )
    end
end
