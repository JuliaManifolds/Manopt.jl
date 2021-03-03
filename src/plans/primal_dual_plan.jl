@doc raw"""
    PrimalDualProblem {T, mT <: Manifold, nT <: Manifold} <: PrimalDualProblem} <: Problem{T}

Describes a Problem for the linearized Chambolle-Pock algorithm.

# Fields

All fields with !! can either be mutating or nonmutating functions, which should be set
depenting on the parameter `T <: AbstractEvaluationType`.

* `M`, `N` – two manifolds $\mathcal M$, $\mathcal N$
* `cost` $F + G(Λ(⋅))$ to evaluate interims cost function values
* `linearized_forward_operator!!` linearized operator for the forward operation in the algorthm $DΛ$ (linearized).
* `linearized_adjoint_operator!!` The adjoint differential $(DΛ)^* \colon \mathcal N → T\mathcal M$
* `prox_F!!` the proximal map belonging to $f$
* `prox_G_dual!!` the proximal map belonging to $g_n^*$
* `Λ!!` – (`fordward_operator`) for the linearized variant, this has to be set to the exact forward operator.
  This operator is required in several variants of the linearized algorithm.
  Since the exact variant is the default, `Λ` is by default set to `forward_operator`.

# Constructor

    LinearizedPrimalDualProblem(M, N, cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,Λ=forward_operator)

"""
mutable struct PrimalDualProblem{T,mT<:Manifold,nT<:Manifold} <: Problem{T}
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
) where {mT<:Manifold,nT<:Manifold}
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

function get_primal_prox(p::PrimalDualProblem{AllocatingEvaluation}, m, τ, x)
    return p.prox_F!!(p.M, m, τ, x)
end
function get_primal_prox(p::PrimalDualProblem{MutatingEvaluation}, m, σ, x)
    y = allocate_result(p.M, get_primal_prox, x)
    return p.prox_F!!(p.M, y, m, σ, x)
end
function get_primal_prox!(p::PrimalDualProblem{AllocatingEvaluation}, y, m, σ, x)
    return copyto!(y, p.prox_F!!(p.M, m, σ, x))
end
function get_primal_prox!(p::PrimalDualProblem{MutatingEvaluation}, y, m, σ, x)
    return p.prox_F!!(p.M, y, m, σ, x)
end

function get_dual_prox(p::PrimalDualProblem{AllocatingEvaluation}, n, τ, ξ)
    return p.prox_G_dual!!(p.N, n, τ, ξ)
end
function get_dual_prox(p::PrimalDualProblem{MutatingEvaluation}, n, τ, ξ)
    η = allocate_result(p.N, get_dual_prox, ξ)
    return p.prox_G_dual!!(p.N, η, n, τ, ξ)
end
function get_dual_prox!(p::PrimalDualProblem{AllocatingEvaluation}, η, n, τ, ξ)
    return copyto!(η, p.prox_G_dual!!(p.N, n, τ, ξ))
end
function get_dual_prox!(p::PrimalDualProblem{MutatingEvaluation}, η, n, τ, ξ)
    return p.prox_G_dual!!(p.N, η, n, τ, ξ)
end

function linearized_forward_operator(p::PrimalDualProblem{AllocatingEvaluation}, m, X)
    return p.linearized_forward_operator!!(p.M, m, X)
end
function linearized_forward_operator(p::PrimalDualProblem{MutatingEvaluation}, m, X)
    y = random_point(p.N)
    forward_operator!(p, y, m)
    Y = zero_tangent_vector(p.N, y)
    return p.linearized_forward_operator!!(p.M, Y, m, X)
end
function linearized_forward_operator!(p::PrimalDualProblem{AllocatingEvaluation}, Y, m, X)
    return copyto!(Y, p.linearized_forward_operator!!(p.M, m, X))
end
function linearized_forward_operator!(p::PrimalDualProblem{MutatingEvaluation}, Y, m, X)
    return p.linearized_forward_operator!!(p.M, Y, m, X)
end

function forward_operator(p::PrimalDualProblem{AllocatingEvaluation}, x)
    return p.Λ!!(p.M, x)
end
function forward_operator(p::PrimalDualProblem{MutatingEvaluation}, x)
    y = random_point(p.N)
    return p.Λ!!(p.M, y, x)
end
function forward_operator!(p::PrimalDualProblem{AllocatingEvaluation}, y, x)
    return copyto!(y, p.Λ!!(p.M, x))
end
function forward_operator!(p::PrimalDualProblem{MutatingEvaluation}, y, x)
    return p.Λ!!(p.M, y, x)
end

function adjoint_linearized_operator(p::PrimalDualProblem{AllocatingEvaluation}, m, n, Y)
    return p.adjoint_linearized_operator!!(p.N, m, n, Y)
end
function adjoint_linearized_operator(p::PrimalDualProblem{MutatingEvaluation}, m, n, Y)
    X = zero_tangent_vector(p.M, m)
    return p.adjoint_linearized_operator!!(p.N, X, m, n, Y)
end
function adjoint_linearized_operator!(
    p::PrimalDualProblem{AllocatingEvaluation}, X, m, n, Y
)
    return copyto!(X, p.adjoint_linearized_operator!!(p.N, m, n, Y))
end
function adjoint_linearized_operator!(p::PrimalDualProblem{MutatingEvaluation}, X, m, n, Y)
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

* `m` - base point on $ \mathcal M $
* `n` - base point on $ \mathcal N $
* `x` - an initial point on $x^{(0)} ∈\mathcal M$ (and its previous iterate)
* `ξ` - an initial tangent vector $ξ^{(0)}∈T^*\mathcal N$ (and its previous iterate)
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

    function ChambollePockOptions(
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
get_solver_result(o::ChambollePockOptions) = o.x
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
where $V_{⋅\gets⋅}$ is the vector transport used in the [`ChambollePockOptions`](@ref)
"""
function primal_residual(p::PrimalDualProblem, o::ChambollePockOptions, x_old, ξ_old, n_old)
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

where in both cases $V_{⋅\gets⋅}$ is the vector transport used in the [`ChambollePockOptions`](@ref).
"""
function dual_residual(p::PrimalDualProblem, o::ChambollePockOptions, x_old, ξ_old, n_old)
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
"""
mutable struct DebugDualResidual <: DebugAction
    io::IO
    prefix::String
    storage::StoreOptionsAction
    function DebugDualResidual(
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)), io::IO=stdout
    )
        return new(io, "Dual Residual: ", a)
    end
    function DebugDualResidual(
        values::Tuple{P,T,Q},
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
    ) where {P,T,Q}
        update_storage!(a, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(io, "Dual Residual: ", a)
    end
end
function (d::DebugDualResidual)(p::PrimalDualProblem, o::ChambollePockOptions, i::Int)
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(d.io, d.prefix * string(dual_residual(p, o, xOld, ξOld, nOld)))
    end
    return d.storage(p, o, i)
end

@doc raw"""
    DebugPrimalResidual <: DebugAction

A Debug action to print the primal residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:x`, `:ξ` and `:n`.
"""
mutable struct DebugPrimalResidual <: DebugAction
    io::IO
    prefix::String
    storage::StoreOptionsAction
    function DebugPrimalResidual(
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)), io::IO=stdout
    )
        return new(io, "Primal Residual: ", a)
    end
    function DebugPrimalResidual(
        values::Tuple{P,T,Q},
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
    ) where {P,T,Q}
        update_storage!(a, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(io, "Primal Residual: ", a)
    end
end
function (d::DebugPrimalResidual)(
    p::P, o::ChambollePockOptions, i::Int
) where {P<:PrimalDualProblem}
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(d.io, d.prefix * string(primal_residual(p, o, xOld, ξOld, nOld)))
    end
    return d.storage(p, o, i)
end
@doc raw"""
    DebugPrimalDualResidual <: DebugAction

A Debug action to print the primaldual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:x`, `:ξ` and `:n`.
"""
mutable struct DebugPrimalDualResidual <: DebugAction
    io::IO
    prefix::String
    storage::StoreOptionsAction
    function DebugPrimalDualResidual(
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)), io::IO=stdout
    )
        return new(io, "PD Residual: ", a)
    end
    function DebugPrimalDualResidual(
        values::Tuple{P,T,Q},
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        io::IO=stdout,
    ) where {P,Q,T}
        update_storage!(a, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(io, "PD Residual: ", a)
    end
end
function (d::DebugPrimalDualResidual)(
    p::P, o::ChambollePockOptions, i::Int
) where {P<:PrimalDualProblem}
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

#
# Debugs
#
"""
    DebugPrimalChange(opts...)

Print the change of the primal variable by using [`DebugChange`](@ref),
see their constructors for detail.
"""
DebugPrimalChange(opts...) = DebugChange(opts[1], "Primal Change: ", opts[2:end]...)

"""
    DebugPrimalIterate(opts...)

Print the change of the primal variable by using [`DebugIterate`](@ref),
see their constructors for detail.
"""
DebugPrimalIterate(opts...) = DebugIterate(opts...)

"""
    DebugDualIterate(e)

Print the dual variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.ξ`.
"""
DebugDualIterate(opts...) = DebugEntry(:ξ, "ξ:", opts...)

"""
    DebugDualChange(opts...)

Print the change of the dual variable, similar to [`DebugChange`](@ref),
see their constructors for detail, but with a different calculation of the change,
since the dual variable lives in (possibly different) tangent spaces.
"""
mutable struct DebugDualChange <: DebugAction
    io::IO
    prefix::String
    storage::StoreOptionsAction
    function DebugDualChange(
        a::StoreOptionsAction=StoreOptionsAction((:ξ, :n)), io::IO=stdout
    )
        return new(io, "Dual Change: ", a)
    end
    function DebugDualChange(
        values::Tuple{T,P},
        a::StoreOptionsAction=StoreOptionsAction((:ξ, :n)),
        io::IO=stdout,
    ) where {P,T}
        update_storage!(a, Dict{Symbol,Any}(k => v for (k, v) in zip((:ξ, :n), values)))
        return new(io, "Dual Change: ", a)
    end
end
function (d::DebugDualChange)(
    p::P, o::ChambollePockOptions, i::Int
) where {P<:PrimalDualProblem}
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

"""
    DebugDualBaseIterate(io::IO=stdout)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.n`.
"""
DebugDualBaseIterate(io::IO=stdout) = DebugEntry(:n, "n:", io)
"""
    DebugDualChange(a=StoreOptionsAction((:ξ)),io::IO=stdout)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugDualBaseChange(a::StoreOptionsAction=StoreOptionsAction((:n)), io::IO=stdout)
    return DebugEntryChange(
        :n, (p, o, x, y) -> distance(p.N, x, y), a, "Dual Base Change:", io
    )
end

"""
    DebugPrimalBaseIterate(io::IO=stdout)

Print the primal base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.m`.
"""
DebugPrimalBaseIterate(io::IO=stdout) = DebugEntry(:m, "m:", io)

"""
    DebugPrimalBaseChange(a::StoreOptionsAction=StoreOptionsAction((:m)),io::IO=stdout)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
function DebugPrimalBaseChange(
    a::StoreOptionsAction=StoreOptionsAction((:m)), io::IO=stdout
)
    return DebugEntryChange(
        :m, (p, o, x, y) -> distance(p.M, x, y), a, "Primal Base Change:", io
    )
end

#
# Records
#

# Primals are just the entries
"""
    RecordPrimalChange(a)

Create an [`RecordAction`](@ref) that records the primal value change,
i.e. [`RecordChange`](@ref), since we just redord the change of `o.x`.
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
