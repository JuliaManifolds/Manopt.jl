@doc raw"""
    BundleMethodState <: AbstractManoptSolverState
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `index_set` - the index set that keeps track of the strictly positive convex coefficients of the subproblem
* `inverse_retraction_method` - the inverse retraction to use within
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
    `p` that was last evaluated.
* `ξ` - the stopping parameter given by ξ = -\norm{g}^2 - ε

# Constructor

BundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_last_serious` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

"""
mutable struct BundleMethodState{
    R,
    P,
    T,
    A<:AbstractVector{<:R},
    B<:AbstractVector{Tuple{<:P,<:T}},
    C<:AbstractVector{T},
    I,
    D<:AbstractVector{<:I},
    IR<:AbstractInverseRetractionMethod,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState where {R<:Float64,P,T,I<:Int64}
    bundle::B
    bundle_size::I
    indices::D
    indices_ref::D
    inverse_retraction_method::IR
    j::I
    lin_errors::A
    p::P
    p_last_serious::P
    p0::P
    X::T
    retraction_method::TR
    stop::TSC
    v::D
    vector_transport_method::VT
    m::R
    ξ::R
    diam::R
    λ::A
    g::T
    ε::R
    transported_subgradients::C
    filter1::R
    filter2::R
    δ::R
    function BundleMethodState(
        M::TM,
        p::P;
        bundle_size::Integer=50,
        m::R=1e-2,
        diam::R=1.0,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenBundleLess(1e-8),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        filter1::R=eps(Float64),
        filter2::R=eps(Float64),
        δ::R=√2,
    ) where {
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        VT<:AbstractVectorTransportMethod,
        R<:Real,
    }
        # Initialize indes set, bundle points, linearization errors, and stopping parameter
        bundle = [(zero.(p), zero.(X)) for _ in 1:bundle_size]
        indices = [0 for _ in 1:bundle_size]
        indices[1] = 1
        v = [1]
        indices_ref = copy(indices)
        j = 1
        lin_errors = zeros(bundle_size)
        ξ = 0.0
        λ = [Inf for _ in 1:bundle_size]
        λ[1] = 1.0
        g = zero.(X)
        ε = 0.0
        transported_subgradients = [zero.(X) for _ in 1:bundle_size]
        return new{
            typeof(m),
            P,
            T,
            typeof(lin_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            typeof(bundle_size),
            typeof(indices),
            IR,
            TR,
            SC,
            VT,
        }(
            bundle,
            bundle_size,
            indices,
            indices_ref,
            inverse_retraction_method,
            j,
            lin_errors,
            p,
            copy(M, p),
            copy(M, p),
            X,
            retraction_method,
            stopping_criterion,
            v,
            vector_transport_method,
            m,
            ξ,
            diam,
            λ,
            g,
            ε,
            transported_subgradients,
            filter1,
            filter2,
            δ,
        )
    end
end
get_iterate(bms::BundleMethodState) = bms.p_last_serious
get_subgradient(bms::BundleMethodState) = bms.g

@doc raw"""
    bundle_method(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)``,

where ``g_k = \sum_{j\in J_k} λ_j^k \mathrm{P}_{p_k←q_j}X_{q_j}``,

with ``X_{q_j}\in∂f(q_j)``, and

where ``\mathrm{retr}`` is a retraction and ``p_k`` is the last serious iterate.
Though the subgradient might be set valued, the argument `∂f` should always
return _one_ element from the subgradient, but not necessarily deterministic.

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `∂f`– the (sub)gradient ``\partial f: \mathcal M→ T\mathcal M`` of f
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

# Optional
* `m` - a real number that controls the decrease of the cost function
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂f(M, q)` or [`MutatingEvaluation`](@ref) in place, i.e. is
   of the form `∂f!(M, X, p)`.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M, typeof(p))`) an inverse retraction method to use
* `retraction` – (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.
* `stopping_criterion` – ([`StopWhenBundleLess`](@ref)`(1e-8)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `vector_transport_method` - (`default_vector_transport_method(M, typeof(p))`) a vector transport method to use
...
and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function bundle_method(M::AbstractManifold, f::TF, ∂f::TdF, p; kwargs...) where {TF,TdF}
    p_star = copy(M, p)
    return bundle_method!(M, f, ∂f, p_star; kwargs...)
end
@doc raw"""
    bundle_method!(M, f, ∂f, p)

perform a bundle method ``p_{j+1} = \mathrm{retr}(p_k, -g_k)`` in place of `p`

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``f:\mathcal M→ℝ`` to minimize
* `∂f`- the (sub)gradient ``\partial f:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subdifferential.
  This function can be passed as an allocation function `(M, p) -> X` or
  a mutating function `(M, X, p) -> X`, see `evaluation`.
* `p` – an initial value ``p_0=p ∈ \mathcal M``

for more details and all optional parameters, see [`bundle_method`](@ref).
"""
function bundle_method!(
    M::AbstractManifold,
    f::TF,
    ∂f!!::TdF,
    p;
    bundle_size=50,
    m=1e-2,
    diam=1.0,
    filter1=eps(),
    filter2=eps(),
    δ=1.0,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
    retraction_method::TRetr=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopWhenBundleLess(1e-4),
    vector_transport_method::VTransp=default_vector_transport_method(M, typeof(p)),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation=evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    bms = BundleMethodState(
        M,
        p;
        bundle_size=bundle_size,
        m=m,
        diam=diam,
        filter1=filter1,
        filter2=filter2,
        δ=δ,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
    )
    bms = decorate_state!(bms; kwargs...)
    return get_solver_return(solve!(mp, bms))
end
function initialize_solver!(mp::AbstractManoptProblem, bms::BundleMethodState)
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    get_subgradient!(mp, bms.X, bms.p)
    bms.bundle[1] = (copy(M, bms.p), copy(M, bms.p, bms.X))
    return bms
end
function bundle_method_sub_solver(::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function _zero_indices!(bms::BundleMethodState)
    for k in 1:length(bms.indices)
        if bms.indices[k] != 0 &&  bms.λ[bms.indices[k]] ≤ bms.filter1
            bms.indices[k] = 0
        end
    end
    return bms.indices
end
function _update_indices!(bms::BundleMethodState, i::Int)
    if i ≤ bms.bundle_size
        bms.indices[i] = i
        bms.indices_ref[i] = i
    else
        circshift!(bms.indices, bms.indices_ref, -bms.j)
    end
    _zero_indices!(bms)
    return bms.indices
end
function step_solver!(mp::AbstractManoptProblem, bms::BundleMethodState, i)
    M = get_manifold(mp)
    bms.v = findall(x -> x != 0, bms.indices)
    for l in bms.v
        vector_transport_to!(
            M,
            bms.transported_subgradients[l],
            bms.bundle[l][1],
            bms.bundle[l][2],
            bms.p_last_serious,
            bms.vector_transport_method,
        )
    end
    bms.λ[bms.v] .= bundle_method_sub_solver(M, bms)
    bms.g .= sum(bms.λ[bms.v] .* bms.transported_subgradients[bms.v])
    bms.ε = sum(bms.λ[bms.v] .* bms.lin_errors[bms.v])
    bms.ξ = -norm(M, bms.p_last_serious, bms.g)^2 - bms.ε
    retract!(M, bms.p, bms.p_last_serious, -bms.g, bms.retraction_method)
    get_subgradient!(mp, bms.X, bms.p)
    if get_cost(mp, bms.p) ≤ (get_cost(mp, bms.p_last_serious) + bms.m * bms.ξ)
        copyto!(M, bms.p_last_serious, bms.p)
    end
    bms.j = mod1(i, bms.bundle_size)
    bms.p0 .= bms.bundle[bms.indices[bms.v[1]]][1]
    _update_indices!(bms, i)
    bms.v = findall(x -> x != 0, bms.indices)
    copyto!(M, bms.bundle[bms.j][1], bms.p)
    copyto!(M, bms.bundle[bms.j][2], bms.p, bms.X)
    if i > bms.bundle_size
        bms.diam = max(
            0.0, bms.diam - bms.δ * distance(M, bms.bundle[bms.v[1]][1], bms.p0)
        )
    end
    for l in bms.v
        logs = inverse_retract(
            M, bms.bundle[l][1], bms.p_last_serious, bms.inverse_retraction_method
        )
        bms.lin_errors[l] =
            get_cost(mp, bms.p_last_serious) - get_cost(mp, bms.bundle[l][1]) -
            inner(M, bms.bundle[l][1], bms.bundle[l][2], logs) +
            bms.diam *
            sqrt(2 * norm(M, bms.bundle[l][1], logs)) *
            norm(M, bms.bundle[l][1], bms.bundle[l][2])
        (0 ≥ bms.lin_errors[l] ≥ -bms.filter2) && (bms.lin_errors[l] == 0)
    end
    return bms
end
get_solver_result(bms::BundleMethodState) = bms.p_last_serious

"""
    StopWhenBundleLess <: StoppingCriterion

Two stopping criteria for [`bundle_method`](@ref) to indicate to stop when either

* the parameters ε and |g|

are less than given tolerances tole and tolg respectively, or

* the parameter -ξ = - |g|^2 - ε

is less than a given tolerance tolxi.

# Constructors

    StopWhenBundleLess(tole=1e-4, tolg=1e-2)

    StopWhenBundleLess(tolxi=1e-4)

"""
mutable struct StopWhenBundleLess{T,R} <: StoppingCriterion
    tole::T
    tolg::T
    tolxi::R
    reason::String
    at_iteration::Int
    function StopWhenBundleLess{Real,Nothing}(tole=1e-4, tolg=1e-2)
        return new{typeof(tole),Nothing}(tole, tolg, nothing, "", 0)
    end
    function StopWhenBundleLess(tole::Real, tolg::Real)
        return StopWhenBundleLess{Real,Nothing}(tole, tolg)
    end
    function StopWhenBundleLess{Nothing,Real}(tolxi=1e-4)
        return new{Nothing,typeof(tolxi)}(nothing, nothing, tolxi, "", 0)
    end
    StopWhenBundleLess(tolxi::Real) = StopWhenBundleLess{Nothing,Real}(tolxi)
end
function (b::StopWhenBundleLess)(mp::AbstractManoptProblem, bms::BundleMethodState, i::Int)
    if i == 0 # reset on init
        b.reason = ""
        b.at_iteration = 0
    end
    M = get_manifold(mp)
    if b.tolxi == nothing
        if (bms.ε ≤ b.tole && norm(M, bms.p_last_serious, bms.g) ≤ b.tolg) && i > 0
            b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter ε = $(bms.ε) is less than $(b.tole) and |g| = $(norm(M, bms.p_last_serious, bms.g)) is less than $(b.tolg).\n"
            b.at_iteration = i
            return true
        end
    elseif -bms.ξ ≤ b.tolxi && i > 0
        b.reason = "After $i iterations the algorithm reached an approximate critical point: the parameter -ξ = $(-bms.ξ) is less than $(b.tolxi).\n"
        b.at_iteration = i
        return true
    end
    return false
end
function status_summary(b::StopWhenBundleLess)
    has_stopped = length(b.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    if b.tolxi == nothing
        return "Stopping parameter: ε ≤ $(b.tole), |g| ≤ $(b.tolg):\t$s"
    else
        return "Stopping parameter: -ξ ≤ $(b.tolxi):\t$s"
    end
end
function show(io::IO, b::StopWhenBundleLess)
    if b.tolxi == nothing
        return print(
            io, "StopWhenBundleLess($(b.tole), $(b.tolg)\n    $(status_summary(b))"
        )
    else
        return print(io, "StopWhenBundleLess($(b.tolxi)\n    $(status_summary(b))")
    end
end
