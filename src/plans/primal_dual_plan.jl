@doc raw"""
An abstract type for primal-dual-based problems.
"""
abstract type AbstractPrimalDualProblem{T} <: Problem{T} end

@doc raw"""
    PrimalDualProblem {T, mT <: AbstractManifold, nT <: AbstractManifold} <: AbstractPrimalDualProblem

Describes a Problem for the linearized or exact Chambolle-Pock algorithm.[^BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez2020][^ChambollePock2011]

# Fields

All fields with !! can either be mutating or nonmutating functions, which should be set
depenting on the parameter `T <: AbstractEvaluationType`.

* `M`, `N` – two manifolds ``\mathcal M``, ``\mathcal N``
* `cost` ``F + G(Λ(⋅))`` to evaluate interims cost function values
* `linearized_forward_operator!!` linearized operator for the forward operation in the algorithm ``DΛ``
* `linearized_adjoint_operator!!` The adjoint differential ``(DΛ)^* : \mathcal N → T\mathcal M``
* `prox_F!!` the proximal map belonging to ``f``
* `prox_G_dual!!` the proximal map belonging to ``g_n^*``
* `Λ!!` – (`fordward_operator`) the  forward operator (if given) ``Λ: \mathcal M → \mathcal N``

Either ``DΛ`` (for the linearized) or ``Λ`` are required usually.

# Constructor

    LinearizedPrimalDualProblem(M, N, cost, prox_F, prox_G_dual, adjoint_linearized_operator;
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
mutable struct PrimalDualProblem{T,mT<:AbstractManifold,nT<:AbstractManifold} <:
               AbstractPrimalDualProblem{T}
    M::mT
    N::nT
    cost::Function
    prox_F!!::Function
    prox_G_dual!!::Function
    linearized_forward_operator!!::Union{Function,Missing}
    adjoint_linearized_operator!!::Function
    Λ!!::Union{Function,Missing}
end
function PrimalDualProblem(
    M::mT,
    N::nT,
    cost,
    prox_F,
    prox_G_dual,
    adjoint_linearized_operator;
    linearized_forward_operator::Union{Function,Missing}=missing,
    Λ::Union{Function,Missing}=missing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {mT<:AbstractManifold,nT<:AbstractManifold}
    return PrimalDualProblem{typeof(evaluation),mT,nT}(
        M,
        N,
        cost,
        prox_F,
        prox_G_dual,
        linearized_forward_operator,
        adjoint_linearized_operator,
        Λ,
    )
end

@doc raw"""
    y = get_primal_prox(p::AbstractPrimalDualProblem, σ, x)
    get_primal_prox!(p::AbstractPrimalDualProblem, y, σ, x)

Evaluate the proximal map of ``F`` stored within [`AbstractPrimalDualProblem`](@ref)

```math
\operatorname{prox}_{σF}(x)
```

which can also be computed in place of `y`.
"""
get_primal_prox(::AbstractPrimalDualProblem, ::Any...)

function get_primal_prox(p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, σ, x)
    return p.prox_F!!(p.M, σ, x)
end
function get_primal_prox(p::AbstractPrimalDualProblem{<:MutatingEvaluation}, σ, x)
    y = allocate_result(p.M, get_primal_prox, x)
    return p.prox_F!!(p.M, y, σ, x)
end
function get_primal_prox!(p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, y, σ, x)
    return copyto!(p.M, y, p.prox_F!!(p.M, σ, x))
end
function get_primal_prox!(p::AbstractPrimalDualProblem{<:MutatingEvaluation}, y, σ, x)
    return p.prox_F!!(p.M, y, σ, x)
end

@doc raw"""
    y = get_dual_prox(p::AbstractPrimalDualProblem, n, τ, ξ)
    get_dual_prox!(p::AbstractPrimalDualProblem, y, n, τ, ξ)

Evaluate the proximal map of ``G_n^*`` stored within [`AbstractPrimalDualProblem`](@ref)

```math
\operatorname{prox}_{τG_n^*}(ξ)
```

which can also be computed in place of `y`.
"""
get_dual_prox(::AbstractPrimalDualProblem, ::Any...)

function get_dual_prox(p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, n, τ, ξ)
    return p.prox_G_dual!!(p.N, n, τ, ξ)
end
function get_dual_prox(p::AbstractPrimalDualProblem{<:MutatingEvaluation}, n, τ, ξ)
    η = allocate_result(p.N, get_dual_prox, ξ)
    return p.prox_G_dual!!(p.N, η, n, τ, ξ)
end
function get_dual_prox!(p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, η, n, τ, ξ)
    return copyto!(p.N, η, p.prox_G_dual!!(p.N, n, τ, ξ))
end
function get_dual_prox!(p::AbstractPrimalDualProblem{<:MutatingEvaluation}, η, n, τ, ξ)
    return p.prox_G_dual!!(p.N, η, n, τ, ξ)
end
@doc raw"""
    Y = linearized_forward_operator(p::AbstractPrimalDualProblem, m X, n)
    linearized_forward_operator!(p::AbstractPrimalDualProblem, Y, m, X, n)

Evaluate the linearized operator (differential) ``DΛ(m)[X]`` stored within
the [`AbstractPrimalDualProblem`](@ref) (in place of `Y`), where `n = Λ(m)`.
"""
linearized_forward_operator(::AbstractPrimalDualProblem, ::Any...)

function linearized_forward_operator(
    p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, m, X, ::Any
)
    return p.linearized_forward_operator!!(p.M, m, X)
end
function linearized_forward_operator(
    p::AbstractPrimalDualProblem{<:MutatingEvaluation}, m, X, ::Any
)
    y = random_point(p.N)
    forward_operator!(p, y, m)
    Y = zero_vector(p.N, y)
    return p.linearized_forward_operator!!(p.M, Y, m, X)
end
function linearized_forward_operator!(
    p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, Y, m, X, n
)
    return copyto!(p.N, Y, n, p.linearized_forward_operator!!(p.M, m, X))
end
function linearized_forward_operator!(
    p::AbstractPrimalDualProblem{<:MutatingEvaluation}, Y, m, X, ::Any
)
    return p.linearized_forward_operator!!(p.M, Y, m, X)
end

@doc raw"""
    y = forward_operator(p::AbstractPrimalDualProblem, x)
    forward_operator!(p::AbstractPrimalDualProblem, y, x)

Evaluate the forward operator of ``Λ(x)`` stored within the [`AbstractPrimalDualProblem`](@ref)
(in place of `y`).
"""
forward_operator(::AbstractPrimalDualProblem{<:AllocatingEvaluation}, ::Any...)

function forward_operator(p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, x)
    return p.Λ!!(p.M, x)
end
function forward_operator(p::AbstractPrimalDualProblem{<:MutatingEvaluation}, x)
    y = random_point(p.N)
    return p.Λ!!(p.M, y, x)
end
function forward_operator!(p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, y, x)
    return copyto!(p.N, y, p.Λ!!(p.M, x))
end
function forward_operator!(p::AbstractPrimalDualProblem{<:MutatingEvaluation}, y, x)
    return p.Λ!!(p.M, y, x)
end

@doc raw"""
    X = adjoint_linearized_operator(p::AbstractPrimalDualProblem, m, n, Y)
    adjoint_linearized_operator(p::AbstractPrimalDualProblem, X, m, n, Y)

Evaluate the adjoint of the linearized forward operator of ``(DΛ(m))^*[Y]`` stored within
the [`AbstractPrimalDualProblem`](@ref) (in place of `X`).
Since ``Y∈T_n\mathcal N``, both ``m`` and ``n=Λ(m)`` are necessary arguments, mainly because
the forward operator ``Λ`` might be `missing` in `p`.
"""
adjoint_linearized_operator(::AbstractPrimalDualProblem{<:AllocatingEvaluation}, ::Any...)

function adjoint_linearized_operator(
    p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, m, n, Y
)
    return p.adjoint_linearized_operator!!(p.N, m, n, Y)
end
function adjoint_linearized_operator(
    p::AbstractPrimalDualProblem{<:MutatingEvaluation}, m, n, Y
)
    X = zero_vector(p.M, m)
    return p.adjoint_linearized_operator!!(p.N, X, m, n, Y)
end
function adjoint_linearized_operator!(
    p::AbstractPrimalDualProblem{<:AllocatingEvaluation}, X, m, n, Y
)
    return copyto!(p.M, X, p.adjoint_linearized_operator!!(p.N, m, n, Y))
end
function adjoint_linearized_operator!(
    p::AbstractPrimalDualProblem{<:MutatingEvaluation}, X, m, n, Y
)
    return p.adjoint_linearized_operator!!(p.N, X, m, n, Y)
end

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

* `m` - base point on ``\mathcal M``
* `n` - base point on ``\mathcal N``
* `x` - an initial point on ``x^{(0)} ∈\mathcal M`` (and its previous iterate)
* `ξ` - an initial tangent vector ``ξ^{(0)}∈T^*\mathcal N`` (and its previous iterate)
* `xbar` - the relaxed iterate used in the next dual update step (when using `:primal` relaxation)
* `ξbar` - the relaxed iterate used in the next primal update step (when using `:dual` relaxation)
* `Θ` – factor to damp the helping ``\tilde x``
* `primal_stepsize` – (`1/sqrt(8)`) proximal parameter of the primal prox
* `dual_stepsize` – (`1/sqrt(8)`) proximal parameter of the dual prox
* `acceleration` – (`0.`) acceleration factor due to Chambolle & Pock
* `relaxation` – (`1.`) relaxation in the primal relaxation step (to compute `xbar`)
* `relax` – (`_primal`) which variable to relax (`:primal` or `:dual`)
* `stop` - a [`StoppingCriterion`](@ref)
* `type` – (`exact`) whether to perform an `:exact` or `:linearized` Chambolle-Pock
* `update_primal_base` (`(p,o,i) -> o.m`) function to update the primal base
* `update_dual_base` (`(p,o,i) -> o.n`) function to update the dual base
* `retraction_method` – (`ExponentialRetraction()`) the retraction to use
* `inverse_retraction_method` - (`LogarithmicInverseRetraction()`) an inverse retraction to use.
* `vector_transport_method` - (`ParallelTransport()`) a vector transport to use

where for the last two the functions a [`Problem`](@ref)` p`,
[`Options`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing` in the linearized case).

# Constructor
    ChambollePockOptions(M::AbstractManifold,
        m::P, n::Q, x::P, ξ::T, primal_stepsize::Float64, dual_stepsize::Float64;
        acceleration::Float64 = 0.0,
        relaxation::Float64 = 1.0,
        relax::Symbol = :primal,
        stopping_criterion::StoppingCriterion = StopAfterIteration(300),
        variant::Symbol = :exact,
        update_primal_base::Union{Function,Missing} = missing,
        update_dual_base::Union{Function,Missing} = missing,
        retraction_method = default_retraction_method(M),
        inverse_retraction_method = default_inverse_retraction_method(M),
        vector_transport_method = default_vector_transport_method(M),
    )
"""
mutable struct ChambollePockOptions{
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
    xbar::P
    ξ::T
    ξbar::T
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
    vector_transport_method::VTM

    @deprecate ChambollePockOptions(
        m,
        n,
        x,
        ξ,
        primal_stepsize::Float64=1 / sqrt(8),
        dual_stepsize::Float64=1 / sqrt(8);
        retraction_method=ExponentialRetraction(),
        inverse_retraction_method=LogarithmicInverseRetraction(),
        vector_transport_method=ParallelTransport(),
        kwargs...,
    ) ChambollePockOptions(
        DefaultManifold(2),
        m,
        n,
        x,
        ξ,
        primal_stepsize,
        dual_stepsize;
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        vector_transport_method=vector_transport_method,
        kwargs...,
    )

    function ChambollePockOptions(
        M::AbstractManifold,
        m::P,
        n::Q,
        x::P,
        ξ::T,
        primal_stepsize::Float64=1 / sqrt(8),
        dual_stepsize::Float64=1 / sqrt(8);
        acceleration::Float64=0.0,
        relaxation::Float64=1.0,
        relax::Symbol=:primal,
        stopping_criterion::StoppingCriterion=StopAfterIteration(300),
        variant::Symbol=:exact,
        update_primal_base::Union{Function,Missing}=missing,
        update_dual_base::Union{Function,Missing}=missing,
        retraction_method::RM=default_retraction_method(M),
        inverse_retraction_method::IRM=default_inverse_retraction_method(M),
        vector_transport_method::VTM=default_vector_transport_method(M),
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
            deepcopy(x),
            ξ,
            deepcopy(ξ),
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
            vector_transport_method,
        )
    end
end
get_solver_result(o::PrimalDualOptions) = o.x
@doc raw"""
    primal_residual(p, o, x_old, ξ_old, n_old)

Compute the primal residual at current iterate ``k`` given the necessary values ``x_{k-1},
ξ_{k-1}``, and ``n_{k-1}`` from the previous iterate.

```math
\Bigl\lVert
\frac{1}{σ}\operatorname{retr}^{-1}_{x_{k}}x_{k-1} -
V_{x_k\gets m_k}\bigl(DΛ^*(m_k)\bigl[V_{n_k\gets n_{k-1}}ξ_{k-1} - ξ_k \bigr]
\Bigr\rVert
```
where ``V_{⋅\gets⋅}`` is the vector transport used in the [`ChambollePockOptions`](@ref)
"""
function primal_residual(
    p::AbstractPrimalDualProblem, o::PrimalDualOptions, x_old, ξ_old, n_old
)
    return norm(
        p.M,
        o.x,
        1 / o.primal_stepsize *
        inverse_retract(p.M, o.x, x_old, o.inverse_retraction_method) -
        vector_transport_to(
            p.M,
            o.m,
            adjoint_linearized_operator(
                p,
                o.m,
                o.n,
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

Compute the dual residual at current iterate ``k`` given the necessary values ``x_{k-1},
ξ_{k-1}``, and ``n_{k-1}`` from the previous iterate. The formula is slightly different depending
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

where in both cases ``V_{⋅\gets⋅}`` is the vector transport used in the [`ChambollePockOptions`](@ref).
"""
function dual_residual(
    p::AbstractPrimalDualProblem, o::PrimalDualOptions, x_old, ξ_old, n_old
)
    if o.variant === :linearized
        return norm(
            p.N,
            o.n,
            1 / o.dual_stepsize *
            (vector_transport_to(p.N, n_old, ξ_old, o.n, o.vector_transport_method) - o.ξ) -
            linearized_forward_operator(
                p,
                o.m,
                vector_transport_to(
                    p.M,
                    o.x,
                    inverse_retract(p.M, o.x, x_old, o.inverse_retraction_method),
                    o.m,
                    o.vector_transport_method,
                ),
                o.n,
            ),
        )
    elseif o.variant === :exact
        return norm(
            p.N,
            o.n,
            1 / o.dual_stepsize *
            (vector_transport_to(p.N, n_old, ξ_old, o.n, o.vector_transport_method) - o.n) -
            inverse_retract(
                p.N,
                o.n,
                forward_operator(
                    p,
                    retract(
                        p.M,
                        o.m,
                        vector_transport_to(
                            p.M,
                            o.x,
                            inverse_retract(p.M, o.x, x_old, o.inverse_retraction_method),
                            o.m,
                            o.vector_transport_method,
                        ),
                        o.retraction_method,
                    ),
                ),
                o.inverse_retraction_method,
            ),
        )
    else
        throw(
            DomainError(
                o.variant,
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
should at least record `:x`, `:ξ` and `:n`.

# Constructor
DebugDualResidual()

with the keywords
* `io` (`stdout`) - stream to perform the debug to
* format (`"$prefix%s"`) format to print the dual residual, using the
* `prefix` (`"Dual Residual: "`) short form to just set the prefix
* `storage` (a new [`StoreOptionsAction`](@ref)) to store values for the debug.
"""
mutable struct DebugDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreOptionsAction
    function DebugDualResidual(;
        storage::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
        prefix="Dual Residual: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugDualResidual(
        initial_values::Tuple{P,T,Q};
        storage::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
        prefix="Dual Residual: ",
        format="$prefix%s",
    ) where {P,T,Q}
        update_storage!(
            storage, Dict(k => v for (k, v) in zip((:x, :ξ, :n), initial_values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualResidual)(p::AbstractPrimalDualProblem, o::PrimalDualOptions, i::Int)
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        Printf.format(d.io, Printf.Format(d.format), dual_residual(p, o, xOld, ξOld, nOld))
    end
    return d.storage(p, o, i)
end
@doc raw"""
    DebugPrimalResidual <: DebugAction

A Debug action to print the primal residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:x`, `:ξ` and `:n`.

# Constructor

    DebugPrimalResidual()

with the keywords
* `io` (`stdout`) - stream to perform the debug to
* format (`"$prefix%s"`) format to print the dual residual, using the
* `prefix` (`"Primal Residual: "`) short form to just set the prefix
* `storage` (a new [`StoreOptionsAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalResidual <: DebugAction
    io::IO
    format::String
    storage::StoreOptionsAction
    function DebugPrimalResidual(;
        storage::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
        prefix="Primal Residual: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugPrimalResidual(
        values::Tuple{P,T,Q};
        storage::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
        prefix="Primal Residual: ",
        format="$prefix%s",
    ) where {P,T,Q}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(io, format, storage)
    end
end
function (d::DebugPrimalResidual)(
    p::AbstractPrimalDualProblem, o::PrimalDualOptions, i::Int
)
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        Printf.format(
            d.io, Printf.Format(d.format), primal_residual(p, o, xOld, ξOld, nOld)
        )
    end
    return d.storage(p, o, i)
end
@doc raw"""
    DebugPrimalDualResidual <: DebugAction

A Debug action to print the primaldual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:x`, `:ξ` and `:n`.

# Constructor

    DebugPrimalDualResidual()

with the keywords
* `io` (`stdout`) - stream to perform the debug to
* format (`"$prefix%s"`) format to print the dual residual, using the
* `prefix` (`"Primal Residual: "`) short form to just set the prefix
* `storage` (a new [`StoreOptionsAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreOptionsAction
    function DebugPrimalDualResidual(;
        storage::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
        prefix="PD Residual: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugPrimalDualResidual(
        values::Tuple{P,T,Q};
        storage::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
        prefix="PD Residual: ",
        format="$prefix%s",
    ) where {P,Q,T}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(io, format, storage)
    end
end
function (d::DebugPrimalDualResidual)(
    p::AbstractPrimalDualProblem, o::PrimalDualOptions, i::Int
)
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        v = primal_residual(p, o, xOld, ξOld, nOld) + dual_residual(p, o, xOld, ξOld, nOld)
        Printf.format(d.io, Printf.Format(d.format), v / manifold_dimension(p.M))
    end
    return d.storage(p, o, i)
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
    storage::StoreOptionsAction=StoreOptionsAction((:x,)),
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
This method is further set display `o.ξ`.
"""
DebugDualIterate(opts...; kwargs...) = DebugEntry(:ξ, opts...; kwargs...)

"""
    DebugDualChange(opts...)

Print the change of the dual variable, similar to [`DebugChange`](@ref),
see their constructors for detail, but with a different calculation of the change,
since the dual variable lives in (possibly different) tangent spaces.
"""
mutable struct DebugDualChange <: DebugAction
    io::IO
    format::String
    storage::StoreOptionsAction
    function DebugDualChange(;
        storage::StoreOptionsAction=StoreOptionsAction((:ξ, :n)),
        io::IO=stdout,
        prefix="Dual Change: ",
        format="$prefix%s",
    )
        return new(io, format, storage)
    end
    function DebugDualChange(
        values::Tuple{T,P};
        storage::StoreOptionsAction=StoreOptionsAction((:ξ, :n)),
        io::IO=stdout,
        prefix="Dual Change: ",
        format="$prefix%s",
    ) where {P,T}
        update_storage!(
            storage, Dict{Symbol,Any}(k => v for (k, v) in zip((:ξ, :n), values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualChange)(p::AbstractPrimalDualProblem, o::PrimalDualOptions, i::Int)
    if all(has_storage.(Ref(d.storage), [:ξ, :n])) && i > 0 # all values stored
        ξOld, nOld = get_storage.(Ref(d.storage), [:ξ, :n]) #fetch
        v = norm(
            p.N,
            o.n,
            vector_transport_to(p.N, nOld, ξOld, o.n, o.vector_transport_method) - o.ξ,
        )
        Printf.format(d.io, Printf.Format(d.format), v)
    end
    return d.storage(p, o, i)
end

"""
    DebugDualBaseIterate(io::IO=stdout)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.n`.
"""
DebugDualBaseIterate(; kwargs...) = DebugEntry(:n; kwargs...)

"""
    DebugDualChange(; storage=StoreOptionsAction((:ξ)), io::IO=stdout)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugDualBaseChange(;
    storage::StoreOptionsAction=StoreOptionsAction((:n)),
    prefix="Dual Base Change:",
    kwargs...,
)
    return DebugEntryChange(
        :n, (p, o, x, y) -> distance(p.N, x, y); storage=storage, prefix=prefix, kwargs...
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
    DebugPrimalBaseChange(a::StoreOptionsAction=StoreOptionsAction((:m)),io::IO=stdout)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugPrimalBaseChange(opts...; prefix="Primal Base Change:", kwargs...)
    return DebugEntryChange(
        :m, (p, o, x, y) -> distance(p.M, x, y), opts...; prefix=prefix, kwargs...
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
RecordPrimalIterate(x) = RecordIterate(x)

"""
    RecordDualIterate(ξ)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordEntry`](@ref) of `o.ξ`, so .
"""
RecordDualIterate(ξ) = RecordEntry(ξ, :ξ)

"""
    RecordDualChange()

Create the action either with a given (shared) Storage, which can be set to the
`values` Tuple, if that is provided).
"""
RecordDualChange() = RecordEntryChange(:ξ, (p, o, x, y) -> distance(p.N, x, y))

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
RecordDualBaseChange() = RecordEntryChange(:n, (p, o, x, y) -> distance(p.N, x, y))

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
RecordPrimalBaseChange() = RecordEntryChange(:m, (p, o, x, y) -> distance(p.M, x, y))
