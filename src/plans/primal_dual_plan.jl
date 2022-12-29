@doc raw"""
    TwoManifoldProblem{
    MT<:AbstractManifold,NT<:AbstractManifold,O<:AbstractManifoldObjective
} <: AbstractManoptProblem{MT}

An abstract type for primal-dual-based problems.
"""
struct TwoManifoldProblem{
    MT<:AbstractManifold,NT<:AbstractManifold,S<:AbstractManifoldObjective
} <: AbstractManoptProblem{MT}
    first_manifold::MT
    second_manifold::NT
    objective::S
end
get_manifold(tmp::TwoManifoldProblem) = get_manifold(tmp, 1)
get_manifold(tmp::TwoManifoldProblem, i) = _get_manifold(tmp, Val(i))
_get_manifold(tmp::TwoManifoldProblem, ::Val{1}) = tmp.first_manifold
_get_manifold(tmp::TwoManifoldProblem, ::Val{2}) = tmp.second_manifold

get_objective(tmo::TwoManifoldProblem) = tmo.objective

function TwoManifoldProblem(
    M::MT, obj::O
) where {MT<:AbstractManifold,O<:AbstractManifoldObjective}
    return TwoManifoldProblem{MT,MT,O}(M, M, obj)
end

abstract type AbstractPrimalDualManifoldObjective{E<:AbstractEvaluationType,C,P} <:
              AbstractManifoldCostObjective{E,C} end

@doc raw"""
    PrimalDualManifoldObjective{E<:AbstractEvaluationType} <: AbstractPrimalDualManifoldObjective{E}

Describes an Objective linearized or exact Chambolle-Pock algorithm.[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020][^ChambollePock2011]

# Fields

All fields with !! can either be mutating or nonmutating functions, which should be set
depenting on the parameter `T <: AbstractEvaluationType`.

* `cost` ``F + G(Λ(⋅))`` to evaluate interims cost function values
* `linearized_forward_operator!!` linearized operator for the forward operation in the algorithm ``DΛ``
* `linearized_adjoint_operator!!` The adjoint differential ``(DΛ)^* : \mathcal N → T\mathcal M``
* `prox_f!!` the proximal map belonging to ``f``
* `prox_G_dual!!` the proximal map belonging to ``g_n^*``
* `Λ!!` – (`fordward_operator`) the  forward operator (if given) ``Λ: \mathcal M → \mathcal N``

Either the linearized operator ``DΛ`` or ``Λ`` are required usually.

# Constructor

    PrimalDualManifoldObjective(cost, prox_f, prox_G_dual, adjoint_linearized_operator;
        linearized_forward_operator::Union{Function,Missing}=missing,
        Λ::Union{Function,Missing}=missing,
        evaluation::AbstractEvaluationType=AllocatingEvaluation()
    )

The last optional argument can be used to provide the 4 or 5 functions as allocating or
mutating (in place computation) ones.
Note that the first argument is always the manifold under consideration, the mutated one is
the second.

[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020]:
    > R. Bergmann, R. Herzog, M. Silva Louzeiro, D. Tenbrinck, J. Vidal-Núñez:
    > _Fenchel Duality Theory and a Primal-Dual Algorithm on Riemannian Manifolds_,
    > Foundations of Computational Mathematics, 2021.
    > doi: [10.1007/s10208-020-09486-5](http://dx.doi.org/10.1007/s10208-020-09486-5)
    > arXiv: [1908.02022](http://arxiv.org/abs/1908.02022)
[^ChambollePock2011]:
    > A. Chambolle, T. Pock:
    > _A first-order primal-dual algorithm for convex problems with applications to imaging_,
    > Journal of Mathematical Imaging and Vision 40(1), 120–145, 2011.
    > doi: [10.1007/s10851-010-0251-1](https://dx.doi.org/10.1007/s10851-010-0251-1)
"""
mutable struct PrimalDualManifoldObjective{
    T<:AbstractEvaluationType,TC,TP,TDP,LFO,ALFO,L
} <: AbstractPrimalDualManifoldObjective{T,TC,TP}
    cost::TC
    prox_f!!::TP
    prox_g_dual!!::TDP
    linearized_forward_operator!!::LFO
    adjoint_linearized_operator!!::ALFO
    Λ!!::L
end
function PrimalDualManifoldObjective(
    cost,
    prox_f,
    prox_g_dual,
    adjoint_linearized_operator;
    linearized_forward_operator::Union{Function,Missing}=missing,
    Λ::Union{Function,Missing}=missing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
)
    return PrimalDualManifoldObjective{
        typeof(evaluation),
        typeof(cost),
        typeof(prox_f),
        typeof(prox_g_dual),
        typeof(linearized_forward_operator),
        typeof(adjoint_linearized_operator),
        typeof(Λ),
    }(
        cost,
        prox_f,
        prox_g_dual,
        linearized_forward_operator,
        adjoint_linearized_operator,
        Λ,
    )
end

@doc raw"""
    q = get_primal_prox(M::AbstractManifold, p::AbstractPrimalDualManifoldObjective, σ, p)
    get_primal_prox!(M::AbstractManifold, p::AbstractPrimalDualManifoldObjective, q, σ, p)

Evaluate the proximal map of ``F`` stored within [`AbstractPrimalDualManifoldObjective`](@ref)

```math
\operatorname{prox}_{σF}(x)
```

which can also be computed in place of `y`.
"""
get_primal_prox(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function get_primal_prox(tmp::TwoManifoldProblem, σ, p)
    return get_primal_prox(get_manifold(tmp, 1), get_objective(tmp), σ, p)
end
function get_primal_prox!(tmp::TwoManifoldProblem, q, σ, p)
    get_primal_prox!(get_manifold(tmp, 1), q, get_objective(tmp), σ, p)
    return q
end

function get_primal_prox(
    M::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    σ,
    p,
)
    return apdmo.prox_f!!(M, σ, p)
end
function get_primal_prox(
    M::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation}, σ, p
)
    q = allocate_result(M, get_primal_prox, p)
    return apdmo.prox_f!!(M, q, σ, p)
end
function get_primal_prox!(
    M::AbstractManifold,
    q,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    σ,
    p,
)
    copyto!(M, q, apdmo.prox_f!!(M, σ, p))
    return q
end
function get_primal_prox!(
    M::AbstractManifold,
    q,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    σ,
    p,
)
    apdmo.prox_f!!(M, q, σ, p)
    return q
end

@doc raw"""
    Y = get_dual_prox(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, n, τ, X)
    get_dual_prox!(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, Y, n, τ, X)

Evaluate the proximal map of ``g_n^*`` stored within [`AbstractPrimalDualManifoldObjective`](@ref)

```math
  Y = \operatorname{prox}_{τG_n^*}(X)
```

which can also be computed in place of `Y`.
"""
get_dual_prox(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function get_dual_prox(tmp::TwoManifoldProblem, n, τ, X)
    return get_dual_prox(get_manifold(tmp, 2), get_objective(tmp), n, τ, X)
end
function get_dual_prox!(tmp::TwoManifoldProblem, Y, n, τ, X)
    return get_dual_prox!(get_manifold(tmp, 2), Y, get_objective(tmp), n, τ, X)
    return Y
end

function get_dual_prox(
    M::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    n,
    τ,
    X,
)
    return apdmo.prox_g_dual!!(M, n, τ, X)
end
function get_dual_prox(
    M::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    n,
    τ,
    X,
)
    Y = allocate_result(M, get_dual_prox, X)
    return apdmo.prox_g_dual!!(M, Y, n, τ, X)
end
function get_dual_prox!(
    M::AbstractManifold,
    Y,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    n,
    τ,
    X,
)
    copyto!(M, Y, apdmo.prox_g_dual!!(M, n, τ, X))
    return Y
end
function get_dual_prox!(
    M::AbstractManifold,
    Y,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    n,
    τ,
    X,
)
    apdmo.prox_g_dual!!(M, Y, n, τ, X)
    return Y
end

@doc raw"""
    Y = linearized_forward_operator(M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, X, n)
    linearized_forward_operator!(M::AbstractManifold, N::AbstractManifold, Y, apdmo::AbstractPrimalDualManifoldObjective, m, X, n)

Evaluate the linearized operator (differential) ``DΛ(m)[X]`` stored within
the [`AbstractPrimalDualManifoldObjective`](@ref) (in place of `Y`), where `n = Λ(m)`.
"""
linearized_forward_operator(
    ::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...
)

function linearized_forward_operator(tmp::TwoManifoldProblem, m, X, n)
    return linearized_forward_operator(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), m, X, n
    )
end
function linearized_forward_operator!(tmp::TwoManifoldProblem, Y, m, X, n)
    linearized_forward_operator!(
        get_manifold(tmp, 1), get_manifold(tmp, 2), Y, get_objective(tmp), m, X, n
    )
    return Y
end

function linearized_forward_operator(
    M::AbstractManifold,
    ::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    m,
    X,
    ::Any,
)
    return apdmo.linearized_forward_operator!!(M, m, X)
end
function linearized_forward_operator(
    M::AbstractManifold,
    N::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    m,
    X,
    n,
)
    Y = zero_vector(N, n)
    apdmo.linearized_forward_operator!!(M, Y, m, X)
    return Y
end
function linearized_forward_operator!(
    M::AbstractManifold,
    N::AbstractManifold,
    Y,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    m,
    X,
    n,
)
    copyto!(N, Y, n, apdmo.linearized_forward_operator!!(M, m, X))
    return Y
end
function linearized_forward_operator!(
    M::AbstractManifold,
    ::AbstractManifold,
    Y,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    m,
    X,
    ::Any,
)
    apdmo.linearized_forward_operator!!(M, Y, m, X)
    return Y
end

@doc raw"""
    q = forward_operator(M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, p)
    forward_operator!(M::AbstractManifold, N::AbstractManifold, q, apdmo::AbstractPrimalDualManifoldObjective, p)

Evaluate the forward operator of ``Λ(x)`` stored within the [`AbstractTwoManifoldProblem`](@ref)
(in place of `q`).
"""
forward_operator(::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...)

function forward_operator(tmp::TwoManifoldProblem, p)
    return forward_operator(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), p
    )
end
function forward_operator!(tmp::TwoManifoldProblem, q, p)
    return forward_operator!(
        get_manifold(tmp, 1), get_manifold(tmp, 2), q, get_objective(tmp), p
    )
end

function forward_operator(
    M::AbstractManifold,
    ::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    p,
)
    return apdmo.Λ!!(M, p)
end
function forward_operator(
    M::AbstractManifold,
    N::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    p,
)
    q = random_point(N)
    apdmo.Λ!!(M, q, p)
    return q
end
function forward_operator!(
    M::AbstractManifold,
    N::AbstractManifold,
    q,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    p,
)
    copyto!(N, q, apdmo.Λ!!(M, p))
    return q
end
function forward_operator!(
    M::AbstractManifold,
    ::AbstractManifold,
    q,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    p,
)
    apdmo.Λ!!(M, q, p)
    return q
end

@doc raw"""
    X = adjoint_linearized_operator(N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y)
    adjoint_linearized_operator(N::AbstractManifold, X, apdmo::AbstractPrimalDualManifoldObjective, m, n, Y)

Evaluate the adjoint of the linearized forward operator of ``(DΛ(m))^*[Y]`` stored within
the [`AbstractPrimalDualManifoldObjective`](@ref) (in place of `X`).
Since ``Y∈T_n\mathcal N``, both ``m`` and ``n=Λ(m)`` are necessary arguments, mainly because
the forward operator ``Λ`` might be `missing` in `p`.
"""
adjoint_linearized_operator(
    ::AbstractManifold, ::AbstractPrimalDualManifoldObjective, ::Any...
)

function adjoint_linearized_operator(tmp::TwoManifoldProblem, m, n, Y)
    return adjoint_linearized_operator(
        get_manifold(tmp, 1), get_manifold(tmp, 2), get_objective(tmp), m, n, Y
    )
end
function adjoint_linearized_operator!(tmp::TwoManifoldProblem, X, m, n, Y)
    return adjoint_linearized_operator!(
        get_manifold(tmp, 1), get_manifold(tmp, 2), X, get_objective(tmp), m, n, Y
    )
end

function adjoint_linearized_operator(
    ::AbstractManifold,
    N::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    m,
    n,
    Y,
)
    return apdmo.adjoint_linearized_operator!!(N, m, n, Y)
end
function adjoint_linearized_operator(
    M::AbstractManifold,
    N::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    m,
    n,
    Y,
)
    X = zero_vector(M, m)
    apdmo.adjoint_linearized_operator!!(N, X, m, n, Y)
    return X
end
function adjoint_linearized_operator!(
    M::AbstractManifold,
    N::AbstractManifold,
    X,
    apdmo::AbstractPrimalDualManifoldObjective{AllocatingEvaluation},
    m,
    n,
    Y,
)
    copyto!(M, X, apdmo.adjoint_linearized_operator!!(N, m, n, Y))
    return X
end
function adjoint_linearized_operator!(
    ::AbstractManifold,
    N::AbstractManifold,
    X,
    apdmo::AbstractPrimalDualManifoldObjective{InplaceEvaluation},
    m,
    n,
    Y,
)
    apdmo.adjoint_linearized_operator!!(N, X, m, n, Y)
    return X
end

@doc raw"""
    AbstractPrimalDualSolverState

A general type for all primal dual based options to be used within primal dual
based algorithms
"""
abstract type AbstractPrimalDualSolverState <: AbstractManoptSolverState end

@doc raw"""
    ChambollePockState <: AbstractPrimalDualSolverState

stores all options and variables within a linearized or exact Chambolle Pock.
The following list provides the order for the constructor, where the previous iterates are
initialized automatically and values with a default may be left out.

* `m` - base point on ``\mathcal M``
* `n` - base point on ``\mathcal N``
* `p` - an initial point on ``x^{(0)} ∈\mathcal M`` (and its previous iterate)
* `X` - an initial tangent vector ``X^{(0)}∈T^*\mathcal N`` (and its previous iterate)
* `pbar` - the relaxed iterate used in the next dual update step (when using `:primal` relaxation)
* `Xbar` - the relaxed iterate used in the next primal update step (when using `:dual` relaxation)
* `Θ` – factor to damp the helping ``\tilde x``
* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `dual_stepsize` – (`1/sqrt(8)`) proximal parameter of the dual prox
* `acceleration` – (`0.`) acceleration factor due to Chambolle & Pock
* `relaxation` – (`1.`) relaxation in the primal relaxation step (to compute `pbar`)
* `relax` – (`_primal`) which variable to relax (`:primal` or `:dual`)
* `stop` - a [`StoppingCriterion`](@ref)
* `type` – (`exact`) whether to perform an `:exact` or `:linearized` Chambolle-Pock
* `update_primal_base` (`(p,o,i) -> o.m`) function to update the primal base
* `update_dual_base` (`(p,o,i) -> o.n`) function to update the dual base
* `retraction_method` – (`default_retraction_method(M)`) the retraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an inverse
  retraction to use on the manifold ``\mathcal M``.
* `inverse_retraction_method_dual` - (`default_inverse_retraction_method(N)`)
  an inverse retraction to use on manifold ``\mathcal N``.
* `vector_transport_method` - (`default_vector_transport_method(M)`) a vector transport to
  use on the manifold ``\mathcal M``.
* `vector_transport_method_dual` - (`default_vector_transport_method(N)`) a
  vector transport to use on manifold ``\mathcal N``.

where for the last two the functions a [`Problem`](@ref)` p`,
[`AbstractManoptSolverState`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing` in the linearized case).

# Constructor
    ChambollePockState(M::AbstractManifold,
        m::P, n::Q, p::P, X::T, primal_stepsize::Float64, dual_stepsize::Float64;
        kwargs...
    )
where all other fields from above are keyword arguments with their default values given in brackets,
as well as `N=TangentBundle(M)`
"""
mutable struct ChambollePockState{
    P,
    Q,
    T,
    RM<:AbstractRetractionMethod,
    IRM<:AbstractInverseRetractionMethod,
    IRM_Dual<:AbstractInverseRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
    VTM_Dual<:AbstractVectorTransportMethod,
} <: AbstractPrimalDualSolverState
    m::P
    n::Q
    p::P
    pbar::P
    X::T
    Xbar::T
    primal_stepsize::Float64
    dual_stepsize::Float64
    acceleration::Float64
    relaxation::Float64
    relax::Symbol
    stop::StoppingCriterion
    variant::Symbol
    update_primal_base::Union{Function,Missing}
    update_dual_base::Union{Function,Missing}
    retraction_method::RM
    inverse_retraction_method::IRM
    inverse_retraction_method_dual::IRM_Dual
    vector_transport_method::VTM
    vector_transport_method_dual::VTM_Dual

    function ChambollePockState(
        M::AbstractManifold,
        m::P,
        n::Q,
        p::P,
        X::T;
        N=TangentBundle(M),
        primal_stepsize::Float64=1 / sqrt(8),
        dual_stepsize::Float64=1 / sqrt(8),
        acceleration::Float64=0.0,
        relaxation::Float64=1.0,
        relax::Symbol=:primal,
        stopping_criterion::StoppingCriterion=StopAfterIteration(300),
        variant::Symbol=:exact,
        update_primal_base::Union{Function,Missing}=missing,
        update_dual_base::Union{Function,Missing}=missing,
        retraction_method::RM=default_retraction_method(M),
        inverse_retraction_method::IRM=default_inverse_retraction_method(M),
        inverse_retraction_method_dual::IRM_Dual=default_inverse_retraction_method(N),
        vector_transport_method::VTM=default_vector_transport_method(M),
        vector_transport_method_dual::VTM_Dual=default_vector_transport_method(N),
    ) where {
        P,
        Q,
        T,
        RM<:AbstractRetractionMethod,
        IRM<:AbstractInverseRetractionMethod,
        IRM_Dual<:AbstractInverseRetractionMethod,
        VTM<:AbstractVectorTransportMethod,
        VTM_Dual<:AbstractVectorTransportMethod,
    }
        return new{P,Q,T,RM,IRM,IRM_Dual,VTM,VTM_Dual}(
            m,
            n,
            p,
            copy(M, p),
            X,
            copy(N, X),
            primal_stepsize,
            dual_stepsize,
            acceleration,
            relaxation,
            relax,
            stopping_criterion,
            variant,
            update_primal_base,
            update_dual_base,
            retraction_method,
            inverse_retraction_method,
            inverse_retraction_method_dual,
            vector_transport_method,
            vector_transport_method_dual,
        )
    end
end
get_solver_result(apds::AbstractPrimalDualSolverState) = get_iterate(apds)
get_iterate(apds::AbstractPrimalDualSolverState) = apds.p
function set_iterate!(apds::AbstractPrimalDualSolverState, p)
    apds.p = p
    return apds
end
@doc raw"""
    primal_residual(p, o, x_old, X_old, n_old)

Compute the primal residual at current iterate ``k`` given the necessary values ``x_{k-1},
X_{k-1}``, and ``n_{k-1}`` from the previous iterate.

```math
\Bigl\lVert
\frac{1}{σ}\operatorname{retr}^{-1}_{x_{k}}x_{k-1} -
V_{x_k\gets m_k}\bigl(DΛ^*(m_k)\bigl[V_{n_k\gets n_{k-1}}X_{k-1} - X_k \bigr]
\Bigr\rVert
```
where ``V_{⋅\gets⋅}`` is the vector transport used in the [`ChambollePockState`](@ref)
"""
function primal_residual(
    tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, p_old, X_old, n_old
)
    return primal_residual(
        get_manifold(tmp, 1),
        get_manifold(tmp, 2),
        get_objective(tmp),
        apds,
        p_old,
        X_old,
        n_old,
    )
end
function primal_residual(
    M::AbstractManifold,
    N::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective,
    apds::AbstractPrimalDualSolverState,
    p_old,
    X_old,
    n_old,
)
    return norm(
        M,
        apds.p,
        1 / apds.primal_stepsize *
        inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method) -
        vector_transport_to(
            M,
            apds.m,
            adjoint_linearized_operator(
                M,
                N,
                apdmo,
                apds.m,
                apds.n,
                vector_transport_to(
                    N, n_old, X_old, apds.n, apds.vector_transport_method_dual
                ) - apds.X,
            ),
            apds.p,
            apds.vector_transport_method,
        ),
    )
end
@doc raw"""
    dual_residual(p, o, x_old, X_old, n_old)

Compute the dual residual at current iterate ``k`` given the necessary values ``x_{k-1},
X_{k-1}``, and ``n_{k-1}`` from the previous iterate. The formula is slightly different depending
on the `o.variant` used:

For the `:lineaized` it reads
```math
\Bigl\lVert
\frac{1}{τ}\bigl(
V_{n_{k}\gets n_{k-1}}(X_{k-1})
- X_k
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
\frac{1}{τ} V_{n_{k}\gets n_{k-1}}(X_{k-1})
-
\operatorname{retr}^{-1}_{n_{k}}\bigl(
Λ(\operatorname{retr}_{m_{k}}(V_{m_k\gets x_k}\operatorname{retr}^{-1}_{x_{k}}x_{k-1}))
\bigr)
\Bigr\rVert
```

where in both cases ``V_{⋅\gets⋅}`` is the vector transport used in the [`ChambollePockState`](@ref).
"""
function dual_residual(
    tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, p_old, X_old, n_old
)
    return dual_residual(
        get_manifold(tmp, 1),
        get_manifold(tmp, 2),
        get_objective(tmp),
        apds,
        p_old,
        X_old,
        n_old,
    )
end

function dual_residual(
    M::AbstractManifold,
    N::AbstractManifold,
    apdmo::AbstractPrimalDualManifoldObjective,
    apds::AbstractPrimalDualSolverState,
    p_old,
    X_old,
    n_old,
)
    if apds.variant === :linearized
        return norm(
            N,
            apds.n,
            1 / apds.dual_stepsize * (
                vector_transport_to(
                    N, n_old, X_old, apds.n, apds.vector_transport_method_dual
                ) - apds.X
            ) - linearized_forward_operator(
                M,
                N,
                apdmo,
                apds.m,
                vector_transport_to(
                    M,
                    apds.p,
                    inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method),
                    apds.m,
                    apds.vector_transport_method,
                ),
                apds.n,
            ),
        )
    elseif apds.variant === :exact
        return norm(
            N,
            apds.n,
            1 / apds.dual_stepsize * (
                vector_transport_to(
                    N, n_old, X_old, apds.n, apds.vector_transport_method_dual
                ) - apds.n
            ) - inverse_retract(
                N,
                apds.n,
                forward_operator(
                    M,
                    N,
                    apdmo,
                    retract(
                        M,
                        apds.m,
                        vector_transport_to(
                            M,
                            apds.p,
                            inverse_retract(
                                M, apds.p, p_old, apds.inverse_retraction_method
                            ),
                            apds.m,
                            apds.vector_transport_method,
                        ),
                        apds.retraction_method,
                    ),
                ),
                apds.inverse_retraction_method_dual,
            ),
        )
    else
        throw(
            DomainError(
                apds.variant,
                "Unknown Chambolle–Pock variant, allowed are `:exact` or `:linearized`.",
            ),
        )
    end
end
#
# Special Debuggers
#
@doc raw"""
    DebugDualResidual <: DebugAction

A Debug action to print the dual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor
DebugDualResidual()

with the keywords
* `io` (`stdout`) - stream to perform the debug to
* format (`"$prefix%s"`) format to print the dual residual, using the
* `prefix` (`"Dual Residual: "`) short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugDualResidual(;
        storage::StoreStateAction=StoreStateAction((:Iterate, :X, :n)),
        io::IO=stdout,
        prefix="Dual Residual: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugDualResidual(
        initial_values::Tuple{P,T,Q};
        storage::StoreStateAction=StoreStateAction((:Iterate, :X, :n)),
        io::IO=stdout,
        prefix="Dual Residual: ",
        format="$prefix%s",
    ) where {P,T,Q}
        update_storage!(
            storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), initial_values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualResidual)(
    tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, i::Int
)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && i > 0 # all values stored
        p_old, X_old, n_old = get_storage.(Ref(d.storage), [:Iterate, :X, :n]) #fetch
        Printf.format(
            d.io,
            Printf.Format(d.format),
            dual_residual(M, N, apdmo, apds, p_old, X_old, n_old),
        )
    end
    return d.storage(tmp, apds, i)
end
@doc raw"""
    DebugPrimalResidual <: DebugAction

A Debug action to print the primal residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor

    DebugPrimalResidual()

with the keywords
* `io` (`stdout`) - stream to perform the debug to
* format (`"$prefix%s"`) format to print the dual residual, using the
* `prefix` (`"Primal Residual: "`) short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugPrimalResidual(;
        storage::StoreStateAction=StoreStateAction((:Iterate, :X, :n)),
        io::IO=stdout,
        prefix="Primal Residual: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugPrimalResidual(
        values::Tuple{P,T,Q};
        storage::StoreStateAction=StoreStateAction((:Iterate, :X, :n)),
        io::IO=stdout,
        prefix="Primal Residual: ",
        format="$prefix%s",
    ) where {P,T,Q}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), values)))
        return new(io, format, storage)
    end
end
function (d::DebugPrimalResidual)(
    tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, i::Int
)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && i > 0 # all values stored
        xOld, XOld, nOld = get_storage.(Ref(d.storage), [:Iterate, :X, :n]) #fetch
        Printf.format(
            d.io,
            Printf.Format(d.format),
            primal_residual(M, N, apdmo, apds, xOld, XOld, nOld),
        )
    end
    return d.storage(tmp, apds, i)
end
@doc raw"""
    DebugPrimalDualResidual <: DebugAction

A Debug action to print the primaldual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor

    DebugPrimalDualResidual()

with the keywords
* `io` (`stdout`) - stream to perform the debug to
* format (`"$prefix%s"`) format to print the dual residual, using the
* `prefix` (`"Primal Residual: "`) short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugPrimalDualResidual(;
        storage::StoreStateAction=StoreStateAction((:Iterate, :X, :n)),
        io::IO=stdout,
        prefix="PD Residual: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugPrimalDualResidual(
        values::Tuple{P,T,Q};
        storage::StoreStateAction=StoreStateAction((:Iterate, :X, :n)),
        io::IO=stdout,
        prefix="PD Residual: ",
        format="$prefix%s",
    ) where {P,Q,T}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), values)))
        return new(io, format, storage)
    end
end
function (d::DebugPrimalDualResidual)(
    tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, i::Int
)
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && i > 0 # all values stored
        p_old, X_old, n_old = get_storage.(Ref(d.storage), [:Iterate, :X, :n]) #fetch
        v =
            primal_residual(M, N, apdmo, apds, p_old, X_old, n_old) +
            dual_residual(tmp, apds, p_old, X_old, n_old)
        Printf.format(d.io, Printf.Format(d.format), v / manifold_dimension(M))
    end
    return d.storage(tmp, apds, i)
end

#
# Debugs
#
"""
    DebugPrimalChange(opts...)

Print the change of the primal variable by using [`DebugChange`](@ref),
see their constructors for detail.
"""
function DebugPrimalChange(;
    storage::StoreStateAction=StoreStateAction((:Iterate,)),
    prefix="Primal Change: ",
    kwargs...,
)
    return DebugChange(; storage=storage, prefix=prefix, kwargs...)
end

"""
    DebugPrimalIterate(opts...;kwargs...)

Print the change of the primal variable by using [`DebugIterate`](@ref),
see their constructors for detail.
"""
DebugPrimalIterate(opts...; kwargs...) = DebugIterate(opts...; kwargs...)

"""
    DebugDualIterate(e)

Print the dual variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.X`.
"""
DebugDualIterate(opts...; kwargs...) = DebugEntry(:X, opts...; kwargs...)

"""
    DebugDualChange(opts...)

Print the change of the dual variable, similar to [`DebugChange`](@ref),
see their constructors for detail, but with a different calculation of the change,
since the dual variable lives in (possibly different) tangent spaces.
"""
mutable struct DebugDualChange <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    function DebugDualChange(;
        storage::StoreStateAction=StoreStateAction((:X, :n)),
        io::IO=stdout,
        prefix="Dual Change: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugDualChange(
        values::Tuple{T,P};
        storage::StoreStateAction=StoreStateAction((:X, :n)),
        io::IO=stdout,
        prefix="Dual Change: ",
        format="$prefix%s",
    ) where {P,T}
        update_storage!(
            storage, Dict{Symbol,Any}(k => v for (k, v) in zip((:X, :n), values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualChange)(
    tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, i::Int
)
    N = get_manifold(tmp, 2)
    if all(has_storage.(Ref(d.storage), [:X, :n])) && i > 0 # all values stored
        X_old, n_old = get_storage.(Ref(d.storage), [:X, :n]) #fetch
        v = norm(
            N,
            apds.n,
            vector_transport_to(
                N, n_old, X_old, apds.n, apds.vector_transport_method_dual
            ) - apds.X,
        )
        Printf.format(d.io, Printf.Format(d.format), v)
    end
    return d.storage(tmp, apds, i)
end

"""
    DebugDualBaseIterate(io::IO=stdout)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.n`.
"""
DebugDualBaseIterate(; kwargs...) = DebugEntry(:n; kwargs...)

"""
    DebugDualChange(; storage=StoreStateAction((:X)), io::IO=stdout)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugDualBaseChange(;
    storage::StoreStateAction=StoreStateAction((:n)), prefix="Dual Base Change:", kwargs...
)
    return DebugEntryChange(
        :n,
        (p, o, x, y) ->
            distance(get_manifold(p, 2), x, y, o.inverse_retraction_method_dual);
        storage=storage,
        prefix=prefix,
        kwargs...,
    )
end

"""
    DebugPrimalBaseIterate()

Print the primal base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.m`.
"""
DebugPrimalBaseIterate(opts...; kwargs...) = DebugEntry(:m, opts...; kwargs...)

"""
    DebugPrimalBaseChange(a::StoreStateAction=StoreStateAction((:m)),io::IO=stdout)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugPrimalBaseChange(opts...; prefix="Primal Base Change:", kwargs...)
    return DebugEntryChange(
        :m,
        (p, o, x, y) -> distance(get_manifold(p, 1), x, y),
        opts...;
        prefix=prefix,
        kwargs...,
    )
end

#
# Records
#

# Primals are just the entries
"""
    RecordPrimalChange(a)

Create an [`RecordAction`](@ref) that records the primal value change,
i.e. [`RecordChange`](@ref), since we just record the change of `o.x`.
"""
RecordPrimalChange() = RecordChange()

"""
    RecordDualBaseIterate(x)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordIterate`](@ref), i.e. `o.x`.
"""
RecordPrimalIterate(p) = RecordIterate(p)

"""
    RecordDualIterate(X)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordEntry`](@ref) of `o.X`, so .
"""
RecordDualIterate(X) = RecordEntry(X, :X)

"""
    RecordDualChange()

Create the action either with a given (shared) Storage, which can be set to the
`values` Tuple, if that is provided).
"""
function RecordDualChange()
    return RecordEntryChange(:X, (p, o, x, y) -> distance(get_manifold(p, 2), x, y))
end

"""
    RecordDualBaseIterate(n)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordEntry`](@ref) of `o.n`.
"""
RecordDualBaseIterate(n) = RecordEntry(n, :n)

"""
    RecordDualBaseChange(e)

Create an [`RecordAction`](@ref) that records the dual base point change,
i.e. [`RecordEntryChange`](@ref) of `o.n` with distance to the last value to store a value.
"""
function RecordDualBaseChange()
    return RecordEntryChange(:n, (p, o, x, y) -> distance(get_manifold(p, 2), x, y))
end

"""
    RecordPrimalBaseIterate(x)

Create an [`RecordAction`](@ref) that records the primal base point,
i.e. [`RecordEntry`](@ref) of `o.m`.
"""
RecordPrimalBaseIterate(m) = RecordEntry(m, :m)
"""
    RecordPrimalBaseChange()

Create an [`RecordAction`](@ref) that records the primal base point change,
i.e. [`RecordEntryChange`](@ref) of `o.m` with distance to the last value to store a value.
"""
function RecordPrimalBaseChange()
    return RecordEntryChange(:m, (p, o, x, y) -> distance(get_manifold(p, 1), x, y))
end
