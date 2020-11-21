@doc raw"""
    PrimalDualProblem {mT <: Manifold, nT <: Manifold} <: PrimalDualProblem} <: Problem

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
mutable struct PrimalDualProblem{mT<:Manifold,nT<:Manifold} <: Problem
    M::mT
    N::nT
    cost::Function
    prox_F::Function
    prox_G_dual::Function
    forward_operator::Function
    adjoint_linearized_operator::Function
    Λ::Union{Function,Missing}
end
function PrimalDualProblem(
    M::mT,
    N::nT,
    cost,
    prox_F,
    prox_G_dual,
    forward_operator,
    adjoint_linearized_operator,
    Λ=forward_operator,
) where {mT<:Manifold,nT<:Manifold}
    return PrimalDualProblem{mT,nT}(
        M, N, cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator, Λ
    )
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
        primal_stepsize::Float64=1/sqrt(8),
        dual_stepsize::Float64=1/sqrt(8);
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
where $V_{\cdot\gets\cdot}$ is the vector transport used in the [`ChambollePockOptions`](@ref)
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
            p.adjoint_linearized_operator(
                o.m,
                vector_transport_to(p.N, n_old, ξ_old, o.n, o.vector_transport_method) - o.ξ,
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
function dual_residual(p::PrimalDualProblem, o::ChambollePockOptions, x_old, ξ_old, n_old)
    if o.variant === :linearized
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
    elseif o.variant === :exact
        return norm(
            p.N,
            o.n,
            1 / o.dual_stepsize * (
                vector_transport_to(p.N, n_old, ξ_old, o.n, o.vector_transport_method) -
                o.n
            ) - inverse_retract(
                p.N,
                o.n,
                p.Λ(retract(
                    p.M,
                    o.m,
                    vector_transport_to(
                        p.M,
                        o.x,
                        inverse_retract(p.M, o.x, x_old, o.inverse_retraction_method),
                        o.m,
                        ParallelTransport(),
                    ),
                    o.retraction_method,
                )),
                o.inverse_retraction_method,
            ),
        )
    else
        throw(DomainError(o.variant, "Unknown Chambolle–Pock variant, allowed are `:exact` or `:linearized`."))
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
    print::Function
    prefix::String
    storage::StoreOptionsAction
    function DebugDualResidual(
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)), print::Function=print
    )
        return new(print, "Dual Residual: ", a)
    end
    function DebugDualResidual(
        values::Tuple{P,P,T},
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        print::Function=print,
    ) where {P,T}
        update_storage!(a, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(print, "Dual Residual: ", a)
    end
end
function (d::DebugDualResidual)(p::PrimalDualProblem, o::ChambollePockOptions, i::Int)
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(d.prefix * string(dual_residual(p, o, xOld, ξOld, nOld)))
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
    print::Function
    prefix::String
    storage::StoreOptionsAction
    function DebugPrimalResidual(
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)), print::Function=print
    )
        return new(print, "Primal Residual: ", a)
    end
    function DebugPrimalResidual(
        values::Tuple{P,T,Q},
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        print::Function=print,
    ) where {P,T,Q}
        update_storage!(a, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(print, "Primal Residual: ", a)
    end
end
function (d::DebugPrimalResidual)(
    p::P, o::ChambollePockOptions, i::Int
) where {P<:PrimalDualProblem}
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(d.prefix * string(primal_residual(p, o, xOld, ξOld, nOld)))
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
    print::Function
    prefix::String
    storage::StoreOptionsAction
    function DebugPrimalDualResidual(
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)), print::Function=print
    )
        return new(print, "Dual Residual: ", a)
    end
    function DebugPrimalDualResidual(
        values::Tuple{P,T,Q},
        a::StoreOptionsAction=StoreOptionsAction((:x, :ξ, :n)),
        print::Function=print,
    ) where {P,Q,T}
        update_storage!(a, Dict(k => v for (k, v) in zip((:x, :ξ, :n), values)))
        return new(print, "Dual Residual", a)
    end
end
function (d::DebugPrimalDualResidual)(
    p::P, o::ChambollePockOptions, i::Int
) where {P<:PrimalDualProblem}
    if all(has_storage.(Ref(d.storage), [:x, :ξ, :n])) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print(
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
DebugPrimalChange(opts...) = DebugChange(opts...)

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
DebugDualIterate(e) = DebugEntry(e, :ξ)

"""
    DebugDualChange(opts...)

Print the change of the dual variable, similar to [`DebugChange`](@ref),
see their constructors for detail, but with a different calculation of the change,
since the dual variable lives in (possibly different) tangent spaces.
"""
mutable struct DebugDualChange <: DebugAction
    print::Function
    prefix::String
    storage::StoreOptionsAction
    function DebugDualChange(
        a::StoreOptionsAction=StoreOptionsAction((:ξ, :n)), print::Function=print
    )
        return new(print, "Dual Change: ", a)
    end
    function DebugDualChange(
        values::Tuple{T,P},
        a::StoreOptionsAction=StoreOptionsAction((:ξ, :n)),
        print::Function=print,
    ) where {P,T}
        update_storage!(a, Dict{Symbol,Any}(k => v for (k, v) in zip((:ξ, :n), values)))
        return new(print, "Dual Change: ", a)
    end
end
function (d::DebugDualChange)(
    p::P, o::ChambollePockOptions, i::Int
) where {P<:PrimalDualProblem}
    if all(has_storage.(Ref(d.storage), [:ξ, :n])) && i > 0 # all values stored
        ξOld, nOld = get_storage.(Ref(d.storage), [:ξ, :n]) #fetch
        print(
            d.prefix * string(norm(
                p.N, o.n, vector_transport(p.N, nOld, ξOld, o.n, ParallelTransport()) - o.ξ
            )),
        )
    end
    return d.storage(p, o, i)
end

"""
    DebugDualBaseIterate(e)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.n`.
"""
DebugDualBaseIterate(e) = DebugEntry(e, :n)
"""
    DebugDualChange(opts...)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
DebugDualBaseChange(e) = DebugEntryChange(e, :n, (p, o, x, y) -> distance(p.N, x, y))

"""
    DebugPrimalBaseIterate(e)

Print the primal base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.m`.
"""
DebugPrimalBaseIterate(e) = DebugEntry(e, :m)
"""
    DebugPrimalBaseChange(opts...)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
DebugPrimalBaseChange(e) = DebugEntryChange(e, :m, (p, o, x, y) -> distance(p.M, x, y))

#
# Records
#

# Primals are just the entries
"""
    RecordPrimalChange(a)

Create an [`RecordAction`](@ref) that records the primal value change,
i.e. [`RecordChange`](@ref), since we just redord the change of `o.x`.
"""
RecordPrimalChange(a) = RecordChange(a)

"""
    RecordDualBaseIterate(e)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordIterate`](@ref), i.e. `o.x`.
"""
RecordPrimalIterate(e) = RecordIterate(e)

"""
    RecordDualIterate(e)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordEntry`](@ref) of `o.ξ`, so .
"""
RecordDualIterate(e) = RecordEntry(e, :ξ)

# RecordDualChange(e) = RecordEntryChange(e, :ξ, (p,o,ξ,ν) -> norm(p.M,o.n,ξ,ν) )
"""
    RecordDualChange <: RecordAction

Create an [`RecordAction`](@ref) that records the dual value change.
While this is similar to a [`RecordEntry`](@ref), we further have to store
the base point to compute the change.

# constructor

    RecordDualChange(a=StoreOptionsAction((:ξ,:n))
    RecordDualChange(values::Tuple{T,P},a=StoreOptionsAction((:ξ,:n))

Create the action either with a given (shared) Storage, which can be set to the
`values` Tuple, if that is provided).
"""
mutable struct RecordDualChange <: RecordAction
    recordedValues::Array{Float64,1}
    storage::StoreOptionsAction
    function RecordDualChange(a::StoreOptionsAction=StoreOptionsAction((:ξ, :n)))
        return new(Array{Float64,1}(), a)
    end
    function RecordDualChange(
        values::Tuple{T,P}, a::StoreOptionsAction=StoreOptionsAction((:ξ, :n))
    ) where {T,P}
        update_storage!(a, Dict{Symbol,Any}(k => v for (k, v) in zip((:ξ, :n), values)))
        return new(Array{Float64,1}(), a)
    end
end
function (r::RecordDualChange)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    v = 0.0
    if all(has_storage.(Ref(r.storage), [:n, :ξ])) # both old values stored
        n_old = get_storage(r.storage, :n)
        ξ_old = get_storage(r.storage, :ξ)
        v = norm(p.N, o.n, vector_transport_to(p.N, n_old, ξ_old, o.n) - o.ξ)
    end
    Manopt.record_or_reset!(r, v, i)
    return r.storage(p, o, i)
end

"""
    RecordDualBaseIterate(e)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordEntry`](@ref) of `o.n`.
"""
RecordDualBaseIterate(e) = RecordEntry(e, :n)

"""
    RecordDualBaseChange(e)

Create an [`RecordAction`](@ref) that records the dual base point change,
i.e. [`RecordEntryChange`](@ref) of `o.n` with distance to the last value to store a value.
"""
RecordDualBaseChange(e) = RecordEntryChange(e, :n, (p, o, x, y) -> distance(p.N, x, y))

"""
    RecordPrimalBaseIterate(e)

Create an [`RecordAction`](@ref) that records the primal base point,
i.e. [`RecordEntry`](@ref) of `o.m`.
"""
RecordPrimalBaseIterate(e) = RecordEntry(e, :m)
"""
    RecordPrimalBaseChange(e)

Create an [`RecordAction`](@ref) that records the primal base point change,
i.e. [`RecordEntryChange`](@ref) of `o.m` with distance to the last value to store a value.
"""
RecordPrimalBaseChange(e) = RecordEntryChange(e, :m, (p, o, x, y) -> distance(p.M, x, y))
