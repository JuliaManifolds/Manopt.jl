@doc raw"""
   PrimalDualProblem {mT <: Manifold, nT <: Manifold} <: PrimalDualProblem} <: Problem

Describes a Problem for the linearized Chambolle-Pock algorithm.

# Fields

* two manifolds $\mathcal M$, $\mathcal N$
* `M`, `N` – two manifolds.
* `cost` $F + G(Λ(⋅))$ to evaluate interims cost function values
* `forward_oprator` the operator for the forward operation in the algorthm, either $Λ$ (exact) or $DΛ$ (linearized).
* `linearized_adjoint_operator` The adjoint differential $(DΛ)^* \colon \mathcal N \to T\mathcal M$
* `prox_F` the proximal map belonging to $f$
* `prox_G_dual` the proximal map belonging to $g_n^*$
* `Λ` – (`fordward_operator`) for the linearized variant, this has to be set to the exact forward operator.
  This operator is required in several variants of the linearized algorithm

# Constructor

    LinearizedPrimalDualProblem(M, N, cost, prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,proxF,proxGDual,Λ=missing)

"""
mutable struct PrimalDualProblem{mT <: Manifold, nT <: Manifold} <: Problem
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
    Λ=forward_operator
) where {mT <: Manifold, nT <: Manifold}
    return PrimalDualProblem{mT,nT}(M,N,cost,prox_F, prox_G_dual, forward_operator, adjoint_linearized_operator,Λ)
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
* `x` - an initial point on $x^{(0)} \in \mathcal M $ (and its previous iterate)
* `ξ` - an initial tangent vector $\xi^{(0)}\in T^*\mathcal N $ (and its previous iterate)
* `xbar` - the relaxed iterate used in the next dual update step
* `Θ` – factor to damp the helping $\tilde x$
* `primal_stepsize` – proximal parameter of the primal prox
* `dual_stepsize` – proximnal parameter of the dual prox
* `acceleration` – (`0.`) acceleration factor due to Chambolle & Pock
* `relaxation` – (`1.`) relaxation in the primal relaxation step (to compute `xbar`)
* `relax` – (`_primal`) which variable to relax (`:primal` or `:dual`)
* `stop` - a [`StoppingCriterion`](@ref)
* `type` – (`exact`) whether to perform an `:exact` or `:linearized` Chambolle-Pock
* `update_primal_base` (`(p,o,i) -> o.m`) function to update the primal base
* `update_dual_base` (`(p,o,i) -> o.n`) function to update the dual base

where for the last two the functions a [`Problem`](@ref)` p`,
[`Options`](@ref)` o` and the current iterate `i` are the arguments.
If you activate these to be different from the default identity, you have to provide
`p.Λ` for the algorithm to work (which might be `missing` in the linearized case).
"""
mutable struct ChambollePockOptions{P,Q,T} <: PrimalDualOptions
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
  function ChambollePockOptions(
    m::P,
    n::Q,
    x::P,
    ξ::T,
    primal_stepsize::Float64,
    dual_stepsize::Float64;
    acceleration::Float64 = 0.0,
    relaxation::Float64 = 1.0,
    relax::Symbol = :primal,
    stopping_criterion::StoppingCriterion = StopAfterIteration(300),
    variant::Symbol = :exact,
    update_primal_base::Union{Function,Missing} = missing,
    update_dual_base::Union{Function,Missing} = missing,
    ) where {P,Q,T}
       return new{P,Q,T}(
            m,n,x,deepcopy(x),ξ,deepcopy(ξ),primal_stepsize,dual_stepsize,
            acceleration,relaxation,relax,stopping_criterion,variant,
            update_primal_base,update_dual_base
        )
    end
end
get_solver_result(o::ChambollePockOptions) = o.x

function primal_residual(p::PrimalDualProblem,o::ChambollePockOptions,xOld,ξOld,nOld)
    return norm(p.M, o.x,
        1/o.primal_stepsize*log(p.M, o.x, xOld) -
        parallelTransport(p.M,o.m,o.x,
            p.adjoint_linearized_operator(o.m, vector_transpor_tot(p.N,nOld,ξOld,o.n) - o.ξ)
        )
    )
end
function dual_residual(p::PrimalDualProblem,o::ChambollePockOptions,xOld,ξOld,nOld)
    if o.variant === :linearized
        return norm(p.N, o.n,
            1/o.dual_stepsize * (vector_transport_to(p.N,o.nOld,o.ξOld,o.n) - o.ξ) -
            p.fordward_operator(o.m, vector_transport_to(p.M, o.x, log(p.M,o.x,xOld), o.m) )
        )
    elseif o.variant === :exact
        return norm(p.N, o.n,
            1/o.dual_stepsize * (vector_transport_to(p.N,o.nOld,ξOld,o.n, ParallelTransport()) - o.n) -
            log(p.N,o.n,
                p.Λ( exp(p.M, o.m, vector_transport_to(p.M,o.x,log(p.M,o.x,xOld),o.m, ParallelTransport())))
            )
        )
    else
        error("Unknown ChambollePock type $(o.variant).")
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
    DebugDualResidual(a::StoreOptionsAction=StoreOptionsAction( (:x,:ξ,:n) ),
        print::Function=print) = new(print,"Dual Residual: ",a)
    function DebugDualResidual(
            values::Tuple{P,P,T},
            a::StoreOptionsAction=StoreOptionsAction( (:x,:ξ,:n) ),
            print::Function=print
        ) where {P,T}
        update_storage!(a, Dict( k=>v for (k,v) in zip( (:x, :ξ, :n), values ) ) )
        return new(print,"Dual Residual: ",a)
    end
end
function (d::DebugDualResidual)(p::P,o::ChambollePockOptions, i::Int) where {P <: PrimalDualProblem}
    if all( has_storage.(Ref(d.storage), [:x, :ξ, :n] ) ) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print( d.prefix * string(dual_residual(p,o,xOld,ξOld,nOld)) )
    end
    d.storage(p ,o, i)
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
    DebugPrimalResidual(a::StoreOptionsAction=StoreOptionsAction( (:x,:ξ,:n) ),
        print::Function=print) = new(print,"Primal Residual: ",a)
    function DebugPrimalResidual(
            values::Tuple{P,T,Q},
            a::StoreOptionsAction=StoreOptionsAction( (:x,:ξ,:n) ),
            print::Function=print
        ) where {P,T,Q}
        update_storage!(a, Dict( k=>v for (k,v) in zip( (:x, :ξ, :n), values ) ) )
        return new(print,"Primal Residual: ",a)
    end
end
function (d::DebugPrimalResidual)(p::P,o::ChambollePockOptions, i::Int) where {P <: PrimalDualProblem}
    if all( has_storage.(Ref(d.storage), [:x, :ξ, :n] ) ) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print( d.prefix * string(primal_residual(p,o,xOld,ξOld,nOld)) )
    end
    d.storage(p,o,i)
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
    DebugPrimalDualResidual(a::StoreOptionsAction=StoreOptionsAction( (:x, :ξ, :n) ),
        print::Function=print) = new(print,"Dual Residual: ",a)
    function DebugPrimalDualResidual(
            values::Tuple{P,T,Q},
            a::StoreOptionsAction=StoreOptionsAction( (:x, :ξ, :n) ),
            print::Function=print
        ) where {P,Q,T}
        update_storage!(a, Dict( k=>v for (k,v) in zip( (:x, :ξ, :n), values ) ) )
        return new(print,"Dual Residual",a)
    end
end
function (d::DebugPrimalDualResidual)(p::P,o::ChambollePockOptions, i::Int) where {P <: PrimalDualProblem}
    if all( has_storage.(Ref(d.storage), [:x, :ξ, :n] ) ) && i > 0 # all values stored
        xOld, ξOld, nOld = get_storage.(Ref(d.storage), [:x, :ξ, :n]) #fetch
        print( d.prefix * string( (primal_residual(p,o,xOld,ξOld,nOld) + dual_residual(p,o,xOld,ξOld,nOld))/manifold_dimension(p.M) ) )
    end
    d.storage(p,o,i)
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
DebugDualIterate(e) = DebugEntry(e,:ξ)

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
    DebugDualChange(
        a::StoreOptionsAction=StoreOptionsAction( (:ξ,:n) ),
        print::Function=print) = new(print,"Dual Change: ",a)
    function DebugDualChange(
        values::Tuple{T,P},
        a::StoreOptionsAction=StoreOptionsAction( (:ξ,:n) ),
        print::Function=print
    ) where {P,T}
        update_storage!(a, Dict{Symbol,Any}( k=>v for (k,v) in zip( (:ξ,:n), values ) ) )
        return new(print,"Dual Change: ",a)
    end
end
function (d::DebugDualChange)(p::P,o::ChambollePockOptions, i::Int) where {P <: PrimalDualProblem}
    if all( has_storage.(Ref(d.storage), [:ξ, :n] ) ) && i > 0 # all values stored
        ξOld, nOld = get_storage.(Ref(d.storage), [:ξ, :n]) #fetch
        print( d.prefix * string( norm(p.N, o.n, vector_transport(p.N,nOld,ξOld,o.n, ParallelTransport()) - o.ξ) ) )
    end
    d.storage(p,o,i)
end

"""
    DebugDualBaseIterate(e)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.n`.
"""
DebugDualBaseIterate(e) = DebugEntry(e,:n)
"""
    DebugDualChange(opts...)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
DebugDualBaseChange(e) = DebugEntryChange(e, :n, (p,o,x,y) -> distance(p.N,x,y) )

"""
    DebugPrimalBaseIterate(e)

Print the primal base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.m`.
"""
DebugPrimalBaseIterate(e) = DebugEntry(e,:m)
"""
    DebugPrimalBaseChange(opts...)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on `o.n`.
"""
DebugPrimalBaseChange(e) = DebugEntryChange(e, :m, (p,o,x,y) -> distance(p.M,x,y) )

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
RecordDualIterate(e) = RecordEntry(e,:ξ)

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
    RecordDualChange(a::StoreOptionsAction=StoreOptionsAction( (:ξ,:n) ) ) =
        new(Array{Float64,1}(), a)
    function RecordDualChange(
        values::Tuple{T,P},
        a::StoreOptionsAction=StoreOptionsAction( (:ξ,:n) )
    ) where {T,P}
        update_storage!(a, Dict{Symbol,Any}( k=>v for (k,v) in zip( (:ξ, :n), values ) ) )
        return new(Array{Float64,1}(), a)
    end
end
function (r::RecordDualChange)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    v=0.0
    if all( has_storage.(Ref(r.storage), [:n, :ξ] ) ) # both old values stored
        n_old = get_storage(r.storage, :n)
        ξ_old = get_storage(r.storage, :ξ)
        v = norm(p.N, o.n, vector_transport_to(p.N, n_old, ξ_old, o.n) - o.ξ)
    end
    Manopt.record_or_reset!(r, v, i)
    r.storage(p ,o, i)
end

"""
    RecordDualBaseIterate(e)

Create an [`RecordAction`](@ref) that records the dual base point,
i.e. [`RecordEntry`](@ref) of `o.n`.
"""
RecordDualBaseIterate(e) = RecordEntry(e,:n)

"""
    RecordDualBaseChange(e)

Create an [`RecordAction`](@ref) that records the dual base point change,
i.e. [`RecordEntryChange`](@ref) of `o.n` with distance to the last value to store a value.
"""
RecordDualBaseChange(e) = RecordEntryChange(e, :n, (p,o,x,y) -> distance(p.N,x,y) )

"""
    RecordPrimalBaseIterate(e)

Create an [`RecordAction`](@ref) that records the primal base point,
i.e. [`RecordEntry`](@ref) of `o.m`.
"""
RecordPrimalBaseIterate(e) = RecordEntry(e,:m)
"""
    RecordPrimalBaseChange(e)

Create an [`RecordAction`](@ref) that records the primal base point change,
i.e. [`RecordEntryChange`](@ref) of `o.m` with distance to the last value to store a value.
"""
RecordPrimalBaseChange(e) = RecordEntryChange(e, :m, (p,o,x,y) -> distance(p.M,x,y) )
