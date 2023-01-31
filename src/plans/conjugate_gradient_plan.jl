
struct DirectionUpdateRuleStorage{TC<:DirectionUpdateRule,TStorage<:StoreStateAction}
    coefficient::TC
    storage::TStorage
end

function DirectionUpdateRuleStorage(
    M::AbstractManifold, dur::DirectionUpdateRule, p0=rand(M), X0=zero_vector(M, p0)
)
    ursp = update_rule_storage_points(dur)
    ursv = update_rule_storage_vectors(dur)
    sa = StoreStateAction(
        Symbol[],
        NamedTuple{ursp}(map(x -> copy(M, p0), ursp)),
        NamedTuple{ursv}(map(x -> copy(M, p0, X0), ursv)),
    )
    return DirectionUpdateRuleStorage{typeof(dur),typeof(sa)}(dur, sa)
end

@doc raw"""
    ConjugateGradientState <: AbstractGradientSolverState

specify options for a conjugate gradient descent algorithm, that solves a
[`DefaultManoptProblem`].

# Fields
* `p` – the current iterate, a point on a manifold
* `X` – the current gradient, also denoted as ``ξ`` or ``X_k`` for the gradient in the ``k``th step.
* `δ` – the current descent direction, i.e. also tangent vector
* `β` – the current update coefficient rule, see .
* `coefficient` – a [`DirectionUpdateRule`](@ref) function to determine the new `β`
* `stepsize` – a [`Stepsize`](@ref) function
* `stop` – a [`StoppingCriterion`](@ref)
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a type of retraction


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
        dC::DirectionUpdateRuleStorage,
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        initial_gradient::T=zero_vector(M, p),
    ) where {P,T}
        βT = allocate_result_type(M, ConjugateGradientDescentState, (p, initial_gradient))
        cgs = new{P,T,βT,typeof(dC),typeof(s),typeof(sC),typeof(retr),typeof(vtr)}()
        cgs.p = p
        cgs.X = initial_gradient
        cgs.δ = copy(M, p, initial_gradient)
        cgs.stop = sC
        cgs.retraction_method = retr
        cgs.stepsize = s
        cgs.coefficient = dC
        cgs.vector_transport_method = vtr
        cgs.β = zero(βT)
        return cgs
    end
end
function ConjugateGradientDescentState(
    M::AbstractManifold,
    p::P,
    sC::StoppingCriterion,
    s::Stepsize,
    dU::DirectionUpdateRuleStorage,
    retr::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    vtr::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
    initial_gradient::T=zero_vector(M, p),
) where {P,T}
    return ConjugateGradientDescentState{P,T}(M, p, sC, s, dU, retr, vtr, initial_gradient)
end

@doc raw"""
    ConjugateDescentCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [^Flethcer1987] adapted to manifolds:

```math
β_k =
\frac{ \lVert X_{k+1} \rVert_{p_{k+1}}^2 }
{\langle -\delta_k,X_k \rangle_{p_k}}.
```

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    ConjugateDescentCoefficient(a::StoreStateAction=())

Construct the conjugate descent coefficient update rule, a new storage is created by default.

[^Flethcer1987]:
    > R. Fletcher, __Practical Methods of Optimization vol. 1: Unconstrained Optimization__
    > John Wiley & Sons, New York, 1987. doi [10.1137/1024028](https://doi.org/10.1137/1024028)
"""
struct ConjugateDescentCoefficient <: DirectionUpdateRule end

update_rule_storage_points(::ConjugateDescentCoefficient) = (:Iterate,)
update_rule_storage_vectors(::ConjugateDescentCoefficient) = (:Gradient,)

function (u::DirectionUpdateRuleStorage{ConjugateDescentCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) || !has_tangent_storage(u.storage, :Gradient)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)
    coef = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, -cgs.δ, X_old)
    update_storage!(u.storage, amp, cgs)
    return coef
end

@doc raw"""
    DaiYuanCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``, based on [^DaiYuan1999] adapted to manifolds:

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
    DaiYuanCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreStateAction=(),
    )

Construct the Dai Yuan coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

[^DaiYuan1999]:
    > [Y. H. Dai and Y. Yuan, A nonlinear conjugate gradient method with a strong global convergence property,
    > SIAM J. Optim., 10 (1999), pp. 177–182.
    > doi: [10.1137/S1052623497318992](https://doi.org/10.1137/S1052623497318992)
"""
struct DaiYuanCoefficient{TVTM<:AbstractVectorTransportMethod} <: DirectionUpdateRule
    transport_method::TVTM
    function DaiYuanCoefficient(t::AbstractVectorTransportMethod=ParallelTransport())
        return new{typeof(t)}(t)
    end
end

update_rule_storage_points(::DaiYuanCoefficient) = (:Iterate,)
update_rule_storage_vectors(::DaiYuanCoefficient) = (:Gradient, :δ)

function (u::DirectionUpdateRuleStorage{<:DaiYuanCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) ||
        !has_tangent_storage(u.storage, :Gradient) ||
        !has_tangent_storage(u.storage, :δ)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)
    δ_old = get_tangent_storage(u.storage, :δ)

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.transport_method)
    coef = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, δtr, ν)
    update_storage!(u.storage, amp, cgs)
    return coef
end

@doc raw"""
    FletcherReevesCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [^FletcherReeves1964] adapted to manifolds:

````math
β_k =
\frac{\lVert X_{k+1}\rVert_{p_{k+1}}^2}{\lVert X_k\rVert_{x_{k}}^2}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    FletcherReevesCoefficient(a::StoreStateAction=())

Construct the Fletcher Reeves coefficient update rule, a new storage is created by default.

[^FletcherReeves1964]:
    > R. Fletcher and C. Reeves, Function minimization by conjugate gradients,
    > Comput. J., 7 (1964), pp. 149–154.
    > doi: [10.1093/comjnl/7.2.149](http://dx.doi.org/10.1093/comjnl/7.2.149)
"""
struct FletcherReevesCoefficient <: DirectionUpdateRule end

update_rule_storage_points(::FletcherReevesCoefficient) = (:Iterate,)
update_rule_storage_vectors(::FletcherReevesCoefficient) = (:Gradient,)

function (u::DirectionUpdateRuleStorage{FletcherReevesCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) || !has_tangent_storage(u.storage, :Gradient)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)
    coef = inner(M, cgs.p, cgs.X, cgs.X) / inner(M, p_old, X_old, X_old)
    update_storage!(u.storage, amp, cgs)
    return coef
end

@doc raw"""
    HagerZhangCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``, based on [^HagerZhang2005]
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
    HagerZhangCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreStateAction=(),
    )

Construct the Hager Zhang coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

[^HagerZhang2005]:
    > [W. W. Hager and H. Zhang, __A new conjugate gradient method with guaranteed descent and an efficient line search__,
    > SIAM J. Optim, (16), pp. 170-192, 2005.
    > doi: [10.1137/030601880](https://doi.org/10.1137/030601880)
"""
mutable struct HagerZhangCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    function HagerZhangCoefficient(t::AbstractVectorTransportMethod=ParallelTransport())
        return new{typeof(t)}(t)
    end
end

update_rule_storage_points(::HagerZhangCoefficient) = (:Iterate,)
update_rule_storage_vectors(::HagerZhangCoefficient) = (:Gradient, :δ)

function (u::DirectionUpdateRuleStorage{<:HagerZhangCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) ||
        !has_tangent_storage(u.storage, :Gradient) ||
        !has_tangent_storage(u.storage, :δ)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)
    δ_old = get_tangent_storage(u.storage, :δ)

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

@doc raw"""
    HestenesStiefelCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [^HeestensStiefel1952]
adapted to manifolds as follows:

Let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``.
Then the update reads

````math
β_k = \frac{\langle X_{k+1}, \nu_k \rangle_{p_{k+1}} }
    { \langle P_{p_{k+1}\gets p_k} \delta_k, \nu_k\rangle_{p_{k+1}} },
````

where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

# Constructor
    HestenesStiefelCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreStateAction=()
    )

Construct the Heestens Stiefel coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

See also [`conjugate_gradient_descent`](@ref)

[^HeestensStiefel1952]:
    > M.R. Hestenes, E.L. Stiefel, Methods of conjugate gradients for solving linear systems,
    > J. Research Nat. Bur. Standards, 49 (1952), pp. 409–436.
    > doi: [10.6028/jres.049.044](http://dx.doi.org/10.6028/jres.049.044)
"""
struct HestenesStiefelCoefficient{TVTM<:AbstractVectorTransportMethod} <:
       DirectionUpdateRule
    transport_method::TVTM
    function HestenesStiefelCoefficient(
        transport_method::AbstractVectorTransportMethod=ParallelTransport()
    )
        return new{typeof(transport_method)}(transport_method)
    end
end

update_rule_storage_points(::HestenesStiefelCoefficient) = (:Iterate,)
update_rule_storage_vectors(::HestenesStiefelCoefficient) = (:Gradient, :δ)

function (u::DirectionUpdateRuleStorage{<:HestenesStiefelCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) ||
        !has_tangent_storage(u.storage, :Gradient) ||
        !has_tangent_storage(u.storage, :δ)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
        return 0.0
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)
    δ_old = get_tangent_storage(u.storage, :δ)

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    δtr = vector_transport_to(M, p_old, δ_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr #notation from [HZ06]
    β = inner(M, cgs.p, cgs.X, ν) / inner(M, cgs.p, δtr, ν)
    update_storage!(u.storage, amp, cgs)
    return max(0, β)
end

@doc raw"""
    LiuStoreyCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [^LuiStorey1991]
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
    LiuStoreyCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreStateAction=()
    )

Construct the Lui Storey coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

[^LuiStorey1991]:
    > Y. Liu and C. Storey, Efficient generalized conjugate gradient algorithms, Part 1: Theory
    > J. Optim. Theory Appl., 69 (1991), pp. 129–137.
    > doi: [10.1007/BF00940464](https://doi.org/10.1007/BF00940464)
"""
struct LiuStoreyCoefficient{TVTM<:AbstractVectorTransportMethod} <: DirectionUpdateRule
    transport_method::TVTM
    function LiuStoreyCoefficient(t::AbstractVectorTransportMethod=ParallelTransport())
        return new{typeof(t)}(t)
    end
end

update_rule_storage_points(::LiuStoreyCoefficient) = (:Iterate,)
update_rule_storage_vectors(::LiuStoreyCoefficient) = (:Gradient, :δ)

function (u::DirectionUpdateRuleStorage{<:LiuStoreyCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) ||
        !has_tangent_storage(u.storage, :Gradient) ||
        !has_tangent_storage(u.storage, :δ)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)
    δ_old = get_tangent_storage(u.storage, :δ)
    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr # notation y from [HZ06]
    coef = inner(M, cgs.p, cgs.X, ν) / inner(M, p_old, -δ_old, X_old)
    update_storage!(u.storage, amp, cgs)
    return coef
end

@doc raw"""
    PolakRibiereCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentState`](@ref)` cgds` include the last iterates
``p_k,X_k``, the current iterates ``p_{k+1},X_{k+1}`` of the iterate and the gradient, respectively,
and the last update direction ``\delta=\delta_k``,  based on [^PolakRibiere1969][^Polyak1969]
adapted to manifolds:

Let ``\nu_k = X_{k+1} - P_{p_{k+1}\gets p_k}X_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the update reads

````math
β_k =
\frac{ \langle X_{k+1}, \nu_k \rangle_{p_{k+1}} }
{\lVert X_k \rVert_{p_k}^2 }.
````

# Constructor

    PolakRibiereCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreStateAction=()
    )

Construct the PolakRibiere coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

See also [`conjugate_gradient_descent`](@ref)

[^PolakRibiere1969]:
    > E. Polak, G. Ribiere, Note sur la convergence de méthodes de directions conjuguées
    > ESAIM: Mathematical Modelling and Numerical Analysis - Modélisation Mathématique et Analyse Numérique, Tome 3 (1969) no. R1, p. 35-43,
    > url: [http://www.numdam.org/item/?id=M2AN_1969__3_1_35_0](http://www.numdam.org/item/?id=M2AN_1969__3_1_35_0)

[^Polyak1969]:
    > B. T. Polyak, The conjugate gradient method in extreme problems,
    > USSR Comp. Math. Math. Phys., 9 (1969), pp. 94–112.
    > doi: [10.1016/0041-5553(69)90035-4](https://doi.org/10.1016/0041-5553(69)90035-4)
"""
struct PolakRibiereCoefficient{TVTM<:AbstractVectorTransportMethod} <: DirectionUpdateRule
    transport_method::TVTM
    function PolakRibiereCoefficient(t::AbstractVectorTransportMethod=ParallelTransport())
        return new{typeof(t)}(t)
    end
end

update_rule_storage_points(::PolakRibiereCoefficient) = (:Iterate,)
update_rule_storage_vectors(::PolakRibiereCoefficient) = (:Gradient,)

function (u::DirectionUpdateRuleStorage{<:PolakRibiereCoefficient})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) || !has_tangent_storage(u.storage, :Gradient)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)

    gradienttr = vector_transport_to(M, p_old, X_old, cgs.p, u.coefficient.transport_method)
    ν = cgs.X - gradienttr
    β = inner(M, cgs.p, cgs.X, ν) / inner(M, p_old, X_old, X_old)
    update_storage!(u.storage, amp, cgs)
    return max(0, β)
end

@doc raw"""
    SteepestDirectionUpdateRule <: DirectionUpdateRule

The simplest rule to update is to have no influence of the last direction and
hence return an update ``β = 0`` for all [`ConjugateGradientDescentState`](@ref)` cgds`

See also [`conjugate_gradient_descent`](@ref)
"""
struct SteepestDirectionUpdateRule <: DirectionUpdateRule end

update_rule_storage_points(::SteepestDirectionUpdateRule) = ()
update_rule_storage_vectors(::SteepestDirectionUpdateRule) = ()

function (u::DirectionUpdateRuleStorage{SteepestDirectionUpdateRule})(
    ::DefaultManoptProblem, ::ConjugateGradientDescentState, i
)
    return 0.0
end

@doc raw"""
    ConjugateGradientBealeRestart <: DirectionUpdateRule

An update rule might require a restart, that is using pure gradient as descent direction,
if the last two gradients are nearly orthogonal, cf. [^HagerZhang2006], page 12 (in the pdf, 46 in Journal page numbers).
This method is named after E. Beale [^Beale1972].
This method acts as a _decorator_ to any existing [`DirectionUpdateRule`](@ref) `direction_update`.

When obtain from the [`ConjugateGradientDescentState`](@ref)` cgs` the last
``p_k,X_k`` and the current ``p_{k+1},X_{k+1}`` iterate and the gradient, respectively.

Then a restart is performed, i.e. ``β_k = 0`` returned if

```math
    \frac{ ⟨X_{k+1}, P_{p_{k+1}\gets p_k}X_k⟩}{\lVert X_k \rVert_{p_k}} > ξ,
```
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``,
and ``ξ`` is the `threshold`.
The default threshold is chosen as `0.2` as recommended in [^Powell1977].

# Constructor

    ConjugateGradientBealeRestart(
        direction_update::D,
        threshold=0.2;
        manifold::AbstractManifold = DefaultManifold(),
        vector_transport_method::V=default_vector_transport_method(manifold),
        a::StoreStateAction=StoreStateAction((:Iterate, :Gradient, :δ)),
    )

[^Beale1972]:
    > E.M.L. Beale:, _A derivation of conjugate gradients_,
    > in: F.A. Lootsma, ed., Numerical methods for nonlinear optimization,
    > Academic Press, London, 1972, pp. 39-43, ISBN 9780124556508.

[^HagerZhang2006]:
    > W. W. Hager and H. Zhang, _A Survey of Nonlinear Conjugate Gradient Methods_
    > Pacific Journal of Optimization 2, 2006, pp. 35-58.
    > url: [http://www.yokohamapublishers.jp/online2/pjov2-1.html](http://www.yokohamapublishers.jp/online2/pjov2-1.html)

[^Powell1977]:
    > M.J.D. Powell, _Restart Procedures for the Conjugate Gradient Method_,
    > Methematical Programming 12, 1977, pp. 241–254
    > doi: [10.1007/BF01593790](https://doi.org/10.1007/BF01593790)
"""
mutable struct ConjugateGradientBealeRestart{
    DUR<:DirectionUpdateRule,VT<:AbstractVectorTransportMethod,F
} <: DirectionUpdateRule
    direction_update::DUR
    threshold::F
    vector_transport_method::VT
    function ConjugateGradientBealeRestart(
        direction_update::D,
        threshold=0.2;
        manifold::AbstractManifold=DefaultManifold(),
        vector_transport_method::V=default_vector_transport_method(manifold),
    ) where {D<:DirectionUpdateRule,V<:AbstractVectorTransportMethod}
        return new{D,V,typeof(threshold)}(
            direction_update, threshold, vector_transport_method
        )
    end
end

function update_rule_storage_points(dur::ConjugateGradientBealeRestart)
    return update_rule_storage_points(dur.direction_update)
end
function update_rule_storage_vectors(dur::ConjugateGradientBealeRestart)
    return update_rule_storage_vectors(dur.direction_update)
end

function (u::DirectionUpdateRuleStorage{<:ConjugateGradientBealeRestart})(
    amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i
)
    M = get_manifold(amp)
    if !has_point_storage(u.storage, :Iterate) || !has_tangent_storage(u.storage, :Gradient)
        update_storage!(u.storage, amp, cgs) # if not given store current as old
    end
    p_old = get_point_storage(u.storage, :Iterate)
    X_old = get_tangent_storage(u.storage, :Gradient)

    # call actual rule
    β = u.direction_update(amp, cgs, i)

    denom = norm(M, cgs.p, cgs.X)
    Xoldpk = vector_transport_to(
        M, p_old, X_old, cgs.p, u.coefficient.vector_transport_method
    )
    num = inner(M, cgs.p, cgs.X, Xoldpk)
    # update storage only after that in case they share
    update_storage!(u.storage, amp, cgs)
    return (num / denom) > u.threshold ? zero(β) : β
end
