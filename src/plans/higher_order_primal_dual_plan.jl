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
        M, N, cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator, Λ
    )
end

# TODO: adapt documentation
# TODO: do we actually need this here as well? Also in ChambollePock
@doc raw"""
    PrimalDualOptions

A general type for all primal dual based options to be used within primal dual
based algorithms
"""
abstract type PrimalDualOptions <: Options end

@doc raw"""
    ChambollePockOptions <: PrimalDualOptions

stores all options and variables within a linearized or exact Chambolle Pock.
The following list provides the order for the constructor, where the previous iterates are
initialized automatically and values with a default may be left out.

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
    ChambollePockOptions(m::P, n::Q, x::P, ξ::T, primal_stepsize::Float64, dual_stepsize::Float64;
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
get_solver_result(o::PrimalDualSemismoothNewtonOptions) = o.x
@doc raw"""
    primal_residual(p, o, x_old, ξ_old, n_old)

Compute the primal residual at current iterate $k$ given the necessary values $x_{k-1},
ξ_{k-1}, and $n_{k-1}$ from the previous iterate.
```math
\Bigl\lVert
\frac{1}{σ}\operatorname{retr}^{-1}_{x_{k}}x_{k-1} -
V_{x_k\gets m_k}\bigl(DΛ^*(m_k)\bigl[V_{n_k\gets n_{k-1}}ξ_{k-1} - ξ_k \bigr]
\Bigr\rVert
```
where $V_{\cdot\gets\cdot}$ is the vector transport used in the [`ChambollePockOptions`](@ref)
"""
function primal_residual(
    p::PrimalDualSemismoothNewtonProblem,
    o::PrimalDualSemismoothNewtonOptions,
    x_old,
    ξ_old,
    n_old,
)
    return norm(
        p.M,
        o.x,
        1 / o.primal_stepsize *
        inverse_retract(p.M, o.x, x_old, o.inverse_retraction_method) -
        vector_transport_to(
            p.M,
            o.m,
            p.adjoint_linearized_operator(
                o.m,
                vector_transport_to(p.N, n_old, ξ_old, o.n, o.vector_transport_method) -
                o.ξ,
            ),
            o.x,
            o.vector_transport_method,
        ),
    )
end
@doc raw"""
    dual_residual(p, o, x_old, ξ_old, n_old)

Compute the dual residual at current iterate $k$ given the necessary values $x_{k-1},
ξ_{k-1}, and $n_{k-1}$ from the previous iterate. The formula is slightly different depending
on the `o.variant` used:

For the `:lineaized` it reads
```math
\Bigl\lVert
\frac{1}{τ}\bigl(
V_{n_{k}\gets n_{k-1}}(ξ_{k-1})
- ξ_k
\bigr)
-
DΛ(m_k)\bigl[
V_{m_k\gets x_k}\operatorname{retr}^{-1}_{x_{k}}x_{k-1}
\bigr]
\Bigr\rVert
```

and for the `:exact` variant

```math
\Bigl\lVert
\frac{1}{τ} V_{n_{k}\gets n_{k-1}}(ξ_{k-1})
-
\operatorname{retr}^{-1}_{n_{k}}\bigl(
Λ(\operatorname{retr}_{m_{k}}(V_{m_k\gets x_k}\operatorname{retr}^{-1}_{x_{k}}x_{k-1}))
\bigr)
\Bigr\rVert
```

where in both cases $V_{\cdot\gets\cdot}$ is the vector transport used in the [`ChambollePockOptions`](@ref).
"""
function dual_residual(
    p::PrimalDualSemismoothNewtonProblem,
    o::PrimalDualSemismoothNewtonOptions,
    x_old,
    ξ_old,
    n_old,
)
    # if o.variant === :linearized (=> Always lineaized)
    return norm(
        p.N,
        o.n,
        1 / o.dual_stepsize *
        (vector_transport_to(p.N, n_old, ξ_old, o.n, o.vector_transport_method) - o.ξ) -
        p.forward_operator(
            o.m,
            vector_transport_to(
                p.M,
                o.x,
                inverse_retract(p.M, o.x, x_old, o.inverse_retraction_method),
                o.m,
                o.vector_transport_method,
            ),
        ),
    )
end
#
# Special Debuggers - we just have to define how they act
#
function (d::DebugDualResidual)(
    p::PrimalDualSemismoothNewtonProblem, o::PrimalDualSemismoothNewtonOptions, i::Int
)
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(d.io, d.prefix * string(dual_residual(p, o, xOld, ξOld, nOld)))
    end
    return d.storage(p, o, i)
end

function (d::DebugPrimalResidual)(
    p::P, o::PrimalDualSemismoothNewtonOptions, i::Int
) where {P<:PrimalDualSemismoothNewtonProblem}
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(d.io, d.prefix * string(primal_residual(p, o, xOld, ξOld, nOld)))
    end
    return d.storage(p, o, i)
end

function (d::DebugPrimalDualResidual)(
    p::P, o::PrimalDualSemismoothNewtonOptions, i::Int
) where {P<:PrimalDualSemismoothNewtonProblem}
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(
            d.io,
            d.prefix * string(
                (
                    primal_residual(p, o, xOld, ξOld, nOld) +
                    dual_residual(p, o, xOld, ξOld, nOld)
                ) / manifold_dimension(p.M),
            ),
        )
    end
    return d.storage(p, o, i)
end
function (d::DebugDualChange)(
    p::P, o::PrimalDualSemismoothNewtonOptions, i::Int
) where {P<:PrimalDualSemismoothNewtonProblem}
    if all(has_storage.(Ref(d.storage), [:ξ, :n])) && i > 0 # all values stored
        ξOld, nOld = get_storage.(Ref(d.storage), [:ξ, :n]) #fetch
        print(
            d.io,
            d.prefix * string(
                norm(
                    p.N,
                    o.n,
                    vector_transport_to(p.N, nOld, ξOld, o.n, o.vector_transport_method) -
                    o.ξ,
                ),
            ),
        )
    end
    return d.storage(p, o, i)
end