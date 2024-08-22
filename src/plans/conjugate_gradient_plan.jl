
struct DirectionUpdateRuleStorage{TC<:DirectionUpdateRule,TStorage<:StoreStateAction}
    coefficient::TC
    storage::TStorage
end

function DirectionUpdateRuleStorage(
    M::AbstractManifold,
    dur::DirectionUpdateRule;
    p_init=rand(M),
    X_init=zero_vector(M, p_init),
)
    ursp = update_rule_storage_points(dur)
    ursv = update_rule_storage_vectors(dur)
    # StoreStateAction makes a copy
    sa = StoreStateAction(
        M; store_points=ursp, store_vectors=ursv, p_init=p_init, X_init=X_init
    )
    return DirectionUpdateRuleStorage{typeof(dur),typeof(sa)}(dur, sa)
end

@doc """
    ConjugateGradientState <: AbstractGradientSolverState

specify options for a conjugate gradient descent algorithm, that solves a
[`DefaultManoptProblem`].

# Fields

$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :X))
* `δ`:                       the current descent direction, also a tangent vector
* `β`:                       the current update coefficient rule, see .
* `coefficient`:             function to determine the new `β`
* $_field_step
* $_field_stop
$(_var(:Field, :retraction_method))
$(_var(:Field, :vector_transport_method))

# Constructor

    ConjugateGradientState(M; kwargs...)

where the last five fields can be set by their names as keyword and the
`X` can be set to a tangent vector type using the keyword `initial_gradient` which defaults to `zero_vector(M,p)`,
and `δ` is initialized to a copy of this vector.

## Keyword arguments

The following fields from above <re keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
* `coefficient=[`ConjugateDescentCoefficient`](@ref)`()`
* `stepsize=[`default_stepsize`](@ref)`(M, ConjugateGradientDescentState; retraction_method=retraction_method)`)
* `stop=[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)`)
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))

# See also

[`conjugate_gradient_descent`](@ref), [`DefaultManoptProblem`](@ref), [`ArmijoLinesearch`](@ref)
"""
mutable struct ConjugateGradientDescentState{
    P,
    T,
    F,
    TCoeff<:DirectionUpdateRuleStorage,
    TStepsize<:Stepsize,
    TStop<:StoppingCriterion,
    TRetr<:AbstractRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
} <: AbstractGradientSolverState
    p::P
    p_old::P
    X::T
    δ::T
    β::F
    coefficient::TCoeff
    stepsize::TStepsize
    stop::TStop
    retraction_method::TRetr
    vector_transport_method::TVTM
    function ConjugateGradientDescentState{P,T}(
        M::AbstractManifold,
        p::P,
        sC::StoppingCriterion,
        s::Stepsize,
        dC::DirectionUpdateRule,
        retr::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
        vtr::AbstractVectorTransportMethod=default_vector_transport_method(M),
        initial_gradient::T=zero_vector(M, p),
    ) where {P,T}
        coef = DirectionUpdateRuleStorage(M, dC; p_init=p, X_init=initial_gradient)
        βT = allocate_result_type(M, ConjugateGradientDescentState, (p, initial_gradient))
        cgs = new{P,T,βT,typeof(coef),typeof(s),typeof(sC),typeof(retr),typeof(vtr)}()
        cgs.p = p
        cgs.p_old = copy(M, p)
        cgs.X = initial_gradient
        cgs.δ = copy(M, p, initial_gradient)
        cgs.stop = sC
        cgs.retraction_method = retr
        cgs.stepsize = s
        cgs.coefficient = coef
        cgs.vector_transport_method = vtr
        cgs.β = zero(βT)
        return cgs
    end
end

function ConjugateGradientDescentState(
    M::AbstractManifold;
    p::P=rand(M),
    coefficient::DirectionUpdateRule=ConjugateDescentCoefficientRule(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stepsize::Stepsize=default_stepsize(
        M, ConjugateGradientDescentState; retraction_method=retraction_method
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(500) |
                                          StopWhenGradientNormLess(1e-8),
    vector_transport_method=default_vector_transport_method(M, typeof(p)),
    initial_gradient::T=zero_vector(M, p),
) where {P,T}
    return ConjugateGradientDescentState{P,T}(
        M,
        p,
        stopping_criterion,
        stepsize,
        coefficient,
        retraction_method,
        vector_transport_method,
        initial_gradient,
    )
end

function get_message(cgs::ConjugateGradientDescentState)
    # for now only step size is quipped with messages
    return get_message(cgs.stepsize)
end

@doc raw"""
    ConjugateDescentCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [Fletcher:1987](@cite) adapted to manifolds:

```math
β_k =
\frac{ \lVert X_{k+1} \rVert_{p_{k+1}}^2 }
{\langle -\delta_k,X_k \rangle_{p_k}}.
```

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    ConjugateDescentCoefficientRule()

Construct the conjugate descent coefficient update rule, a new storage is created by default.
"""
struct ConjugateDescentCoefficientRule <: DirectionUpdateRule end

## TODO: continue here with the docs.
"""
    ConjugateDescentCoefficient()


"""
function ConjugateDescentCoefficient()
    return ManifoldDefaultsFactory(
        Manopt.ConjugateDescentCoefficientRule; requires_manifold=false
    )
end

update_rule_storage_points(::ConjugateDescentCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::ConjugateDescentCoefficientRule) = Tuple{:Gradient}

function (u::DirectionUpdateRuleStorage{ConjugateDescentCoefficientRule})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    coef = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, -cgs.δ, X_old)
    update_storage!(u.storage, amp, cgs)
    return coef
end
function show(io::IO, ::ConjugateDescentCoefficientRule)
    return print(io, "ConjugateDescentCoefficientRule()")
end

@doc raw"""
    DaiYuanCoefficientRule <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``, based on [DaiYuan:1999](@cite) adapted to manifolds:

Let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the coefficient reads

````math
β_k =
\frac{ \lVert X_{k+1} \rVert_{p_{k+1}}^2 }
{\langle P_{p_{k+1}\gets p_k}\delta_k, \nu_k \rangle_{p_{k+1}}}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    function DaiYuanCoefficientRule(
        M::AbstractManifold=DefaultManifold(2);
        t::AbstractVectorTransportMethod=default_vector_transport_method(M)
    )

Construct the Dai—Yuan coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.
"""
struct DaiYuanCoefficientRule{TVTM<:AbstractVectorTransportMethod} <: DirectionUpdateRule
    transport_method::TVTM
    function DaiYuanCoefficientRule(t::AbstractVectorTransportMethod)
        return new{typeof(t)}(t)
    end
end
function DaiYuanCoefficientRule(
    M::AbstractManifold; vector_transport_method=default_vector_transport_method(M)
)
    return DaiYuanCoefficientRule(vector_transport_method)
end
## TODO: continue here with the docs.
function DaiYuanCoefficient(args; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DaiYuanCoefficientRule, args...; kwargs...)
end

update_rule_storage_points(::DaiYuanCoefficientRule) = Tuple{:Iterate}
update_rule_storage_vectors(::DaiYuanCoefficientRule) = Tuple{:Gradient,:δ}

function (u::DirectionUpdateRuleStorage{<:DaiYuanCoefficientRule})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
        !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.transport_method)
    coef = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, δtr, ν)
    update_storage!(u.storage, amp, cgs)
    return coef
end
function show(io::IO, u::DaiYuanCoefficientRule)
    return print(io, "DaiYuanCoefficient($(u.transport_method))")
end

@doc raw"""
    FletcherReevesCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [FletcherReeves:1964](@cite) adapted to manifolds:

````math
β_k =
\frac{\lVert X_{k+1}\rVert_{p_{k+1}}^2}{\lVert X_k\rVert_{x_{k}}^2}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    FletcherReevesCoefficient(a::StoreStateAction=())

Construct the Fletcher—Reeves coefficient update rule, a new storage is created by default.
"""
struct FletcherReevesCoefficient <: DirectionUpdateRule end

update_rule_storage_points(::FletcherReevesCoefficient) = Tuple{:Iterate}
update_rule_storage_vectors(::FletcherReevesCoefficient) = Tuple{:Gradient}

function (u::DirectionUpdateRuleStorage{FletcherReevesCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    coef = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, X_old, X_old)
    update_storage!(u.storage, amp, cgs)
    return coef
end
function show(io::IO, ::FletcherReevesCoefficient)
    return print(io, "FletcherReevesCoefficient()")
end

@doc raw"""
    HagerZhangCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``, based on [HagerZhang:2005](@cite).
adapted to manifolds: let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

````math
β_k = \Bigl\langle\nu_k -
\frac{ 2\lVert \nu_k\rVert_{p_{k+1}}^2 }{ \langle P_{p_{k+1}\gets p_k}\delta_k, \nu_k \rangle_{p_{k+1}} }
P_{p_{k+1}\gets p_k}\delta_k,
\frac{X_{k+1}}{ \langle P_{p_{k+1}\gets p_k}\delta_k, \nu_k \rangle_{p_{k+1}} }
\Bigr\rangle_{p_{k+1}}.
````

This method includes a numerical stability proposed by those authors.

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    function HagerZhangCoefficient(t::AbstractVectorTransportMethod)
    function HagerZhangCoefficient(M::AbstractManifold = DefaultManifold(2))

Construct the Hager Zhang coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.
"""
mutable struct HagerZhangCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM

    function HagerZhangCoefficient(t::AbstractVectorTransportMethod)
        return new{typeof(t)}(t)
    end
end
function HagerZhangCoefficient(M::AbstractManifold=DefaultManifold(2))
    return HagerZhangCoefficient(default_vector_transport_method(M))
end

update_rule_storage_points(::HagerZhangCoefficient) = Tuple{:Iterate}
update_rule_storage_vectors(::HagerZhangCoefficient) = Tuple{:Gradient,:δ}

function (u::DirectionUpdateRuleStorage{<:HagerZhangCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
        !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.transport_method)
    denom = inner(M, cgs.p, δtr, ν)
    νknormsq = inner(M, cgs.p, ν, ν)
    β =
        inner(M, cgs.p, ν, cgs.X) / denom -
        2 * νknormsq * inner(M, cgs.p, δtr, cgs.X) / denom^2
    # Numerical stability from Manopt / Hager-Zhang paper
    ξn = norm(M, cgs.p, cgs.X)
    η = -1 / (ξn * min(0.01, norm(M, p_old, X_old)))
    coef = max(β, η)
    update_storage!(u.storage, amp, cgs)
    return coef
end
function show(io::IO, u::HagerZhangCoefficient)
    return print(io, "HagerZhangCoefficient($(u.transport_method))")
end

@doc raw"""
    HestenesStiefelCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [HestenesStiefel:1952](@cite)
adapted to manifolds as follows:

Let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``.
Then the update reads

````math
β_k = \frac{\langle X_{k+1}, \nu_k \rangle_{p_{k+1}} }
    { \langle P_{p_{k+1}\gets p_k} \delta_k, \nu_k\rangle_{p_{k+1}} },
````

where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

# Constructor
    function HestenesStiefelCoefficient(transport_method::AbstractVectorTransportMethod)
    function HestenesStiefelCoefficient(M::AbstractManifold = DefaultManifold(2))

Construct the Heestens Stiefel coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

See also [`conjugate_gradient_descent`](@ref)
"""
struct HestenesStiefelCoefficient{TVTM<:AbstractVectorTransportMethod} <:
       DirectionUpdateRule
    transport_method::TVTM
    function HestenesStiefelCoefficient(t::AbstractVectorTransportMethod)
        return new{typeof(t)}(t)
    end
end
function HestenesStiefelCoefficient(M::AbstractManifold=DefaultManifold(2))
    return HestenesStiefelCoefficient(default_vector_transport_method(M))
end

update_rule_storage_points(::HestenesStiefelCoefficient) = Tuple{:Iterate}
update_rule_storage_vectors(::HestenesStiefelCoefficient) = Tuple{:Gradient,:δ}

function (u::DirectionUpdateRuleStorage{<:HestenesStiefelCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
        !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr #notation from [HZ06]
    β = inner(M, cgs.p, cgs.X, ν) / inner(M, cgs.p, δtr, ν)
    update_storage!(u.storage, amp, cgs)
    return max(0, β)
end
function show(io::IO, u::HestenesStiefelCoefficient)
    return print(io, "HestenesStiefelCoefficient($(u.transport_method))")
end

@doc raw"""
    LiuStoreyCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [LiuStorey:1991](@cite)
adapted to manifolds:

Let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the coefficient reads

````math
β_k = -
\frac{ \langle X_{k+1},\nu_k \rangle_{p_{k+1}} }
{\langle \delta_k,X_k \rangle_{p_k}}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    function LiuStoreyCoefficient(t::AbstractVectorTransportMethod)
    function LiuStoreyCoefficient(M::AbstractManifold = DefaultManifold(2))

Construct the Lui Storey coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.
"""
struct LiuStoreyCoefficient{TVTM<:AbstractVectorTransportMethod} <: DirectionUpdateRule
    transport_method::TVTM
    function LiuStoreyCoefficient(t::AbstractVectorTransportMethod)
        return new{typeof(t)}(t)
    end
end
function LiuStoreyCoefficient(M::AbstractManifold=DefaultManifold(2))
    return LiuStoreyCoefficient(default_vector_transport_method(M))
end

update_rule_storage_points(::LiuStoreyCoefficient) = Tuple{:Iterate}
update_rule_storage_vectors(::LiuStoreyCoefficient) = Tuple{:Gradient,:δ}

function (u::DirectionUpdateRuleStorage{<:LiuStoreyCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient)) ||
        !has_storage(u.storage, VectorStorageKey(:δ))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))
    δ_old = get_storage(u.storage, VectorStorageKey(:δ))
    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr # notation y from [HZ06]
    coef = inner(M, cgs.p, cgs.X, ν) / inner(M, p_old, -δ_old, X_old)
    update_storage!(u.storage, amp, cgs)
    return coef
end
function show(io::IO, u::LiuStoreyCoefficient)
    return print(io, "LiuStoreyCoefficient($(u.transport_method))")
end

@doc raw"""
    PolakRibiereCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [PolakRibiere:1969](@cite)
and [Polyak:1969](@cite) adapted to manifolds:

Let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the update reads

````math
β_k =
\frac{ \langle X_{k+1}, \nu_k \rangle_{p_{k+1}} }
{\lVert X_k \rVert_{p_k}^2 }.
````

# Constructor

    function PolakRibiereCoefficient(
        M::AbstractManifold=DefaultManifold(2);
        t::AbstractVectorTransportMethod=default_vector_transport_method(M)
    )

Construct the PolakRibiere coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

See also [`conjugate_gradient_descent`](@ref)
"""
struct PolakRibiereCoefficient{TVTM<:AbstractVectorTransportMethod} <: DirectionUpdateRule
    transport_method::TVTM
    function PolakRibiereCoefficient(t::AbstractVectorTransportMethod)
        return new{typeof(t)}(t)
    end
end
function PolakRibiereCoefficient(M::AbstractManifold=DefaultManifold(2))
    return PolakRibiereCoefficient(default_vector_transport_method(M))
end

update_rule_storage_points(::PolakRibiereCoefficient) = Tuple{:Iterate}
update_rule_storage_vectors(::PolakRibiereCoefficient) = Tuple{:Gradient}

function (u::DirectionUpdateRuleStorage{<:PolakRibiereCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr
    β = real(inner(M, cgs.p, cgs.X, ν)) / real(inner(M, p_old, X_old, X_old))
    update_storage!(u.storage, amp, cgs)
    return max(zero(β), β)
end
function show(io::IO, u::PolakRibiereCoefficient)
    return print(io, "PolakRibiereCoefficient($(u.transport_method))")
end

@doc raw"""
    SteepestDirectionUpdateRule <: DirectionUpdateRule

The simplest rule to update is to have no influence of the last direction and
hence return an update ``β = 0`` for all [`ConjugateGradientDescentState`](@ref)` cgds`

See also [`conjugate_gradient_descent`](@ref)
"""
struct SteepestDirectionUpdateRule <: DirectionUpdateRule end

update_rule_storage_points(::SteepestDirectionUpdateRule) = Tuple{}
update_rule_storage_vectors(::SteepestDirectionUpdateRule) = Tuple{}

function (u::DirectionUpdateRuleStorage{SteepestDirectionUpdateRule})(
    ::DefaultManoptProblem, ::ConjugateGradientDescentState, i
)
    return 0.0
end

@doc raw"""
    ConjugateGradientBealeRestartRule <: DirectionUpdateRule

An update rule might require a restart, that is using pure gradient as descent direction,
if the last two gradients are nearly orthogonal, see [HagerZhang:2006; page 12](@cite) (in the preprint, page 46 in Journal page numbers).
This method is named after E. Beale from his proceedings paper in 1972 [Beale:1972](@cite).
This method acts as a _decorator_ to any existing [`DirectionUpdateRule`](@ref) `direction_update`.

When obtain from the [`ConjugateGradientDescentState`](@ref)` cgs` the last
``p_k,X_k`` and the current ``p_{k+1},X_{k+1}`` iterate and the gradient, respectively.

Then a restart is performed, hence ``β_k = 0`` returned if

```math
    \frac{ ⟨X_{k+1}, P_{p_{k+1}\gets p_k}X_k⟩}{\lVert X_k \rVert_{p_k}} > ξ,
```
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``,
and ``ξ`` is the `threshold`.
The default threshold is chosen as `0.2` as recommended in [Powell:1977](@cite)

# Constructor

    ConjugateGradientBealeRestartRule(
        M::AbstractManifold,
        direction_update::D;
        threshold=0.2,
        vector_transport_method::V=default_vector_transport_method(M),
    )
"""
mutable struct ConjugateGradientBealeRestartRule{
    DUR<:DirectionUpdateRule,VT<:AbstractVectorTransportMethod,F
} <: DirectionUpdateRule
    direction_update::DUR
    threshold::F
    vector_transport_method::VT
end
function ConjugateGradientBealeRestartRule(
    M::AbstractManifold,
    direction_update::Union{DirectionUpdateRule,ManifoldDefaultsFactory};
    threshold=0.2,
    vector_transport_method::V=default_vector_transport_method(M),
) where {V<:AbstractVectorTransportMethod}
    dir = _produce_type(direction_update, M)
    return ConjugateGradientBealeRestartRule{typeof(dir),V,typeof(threshold)}(
        dir, threshold, vector_transport_method
    )
end
function ConjugateGradientBealeRestart(args...; kwargs...)
    return ManifoldDefaultsFactory(ConjugateGradientBealeRestartRule, args...; kwargs...)
end

@inline function update_rule_storage_points(dur::ConjugateGradientBealeRestartRule)
    dur_p = update_rule_storage_points(dur.direction_update)
    return :Iterate in dur_p.parameters ? dur_p : Tuple{:Iterate,dur_p.parameters...}
end
@inline function update_rule_storage_vectors(dur::ConjugateGradientBealeRestartRule)
    dur_X = update_rule_storage_vectors(dur.direction_update)
    return :Gradient in dur_X.parameters ? dur_X : Tuple{:Gradient,dur_X.parameters...}
end

function (u::DirectionUpdateRuleStorage{<:ConjugateGradientBealeRestartRule})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k
)
    M = get_manifold(amp)
    if !has_storage(u.storage, PointStorageKey(:Iterate)) ||
        !has_storage(u.storage, VectorStorageKey(:Gradient))
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_storage(u.storage, PointStorageKey(:Iterate))
    X_old = get_storage(u.storage, VectorStorageKey(:Gradient))

    # call actual rule
    β = u.coefficient.direction_update(amp, cgs, k)

    denom = norm(M, cgs.p, cgs.X)
    Xoldpk = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    num = inner(M, cgs.p, cgs.X, Xoldpk)
    # update storage only after that in case they share
    update_storage!(u.storage, amp, cgs)
    return real(num / denom) > u.coefficient.threshold ? zero(β) : β
end
function show(io::IO, u::ConjugateGradientBealeRestartRule)
    return print(
        io,
        "ConjugateGradientBealeRestartRule($(u.direction_update), $(u.threshold), $(u.vector_transport_method))",
    )
end
