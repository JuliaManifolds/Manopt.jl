@doc raw"""
    ConjugateGradientOptions <: AbstractGradientOptions

specify options for a conjugate gradient descent algorithm, that solves a
[`GradientProblem`].

# Fields
* `x` – the current iterate, a point on a manifold
* `gradient` – the current gradient
* `δ` – the current descent direction, i.e. also tangent vector
* `β` – the current update coefficient rule, see .
* `coefficient` – a [`DirectionUpdateRule`](@ref) function to determine the new `β`
* `stepsize` – a [`Stepsize`](@ref) function
* `stop` – a [`StoppingCriterion`](@ref)
* `retraction_method` – (`default_retraction_method(M)`) a type of retraction


# See also
[`conjugate_gradient_descent`](@ref), [`GradientProblem`](@ref), [`ArmijoLinesearch`](@ref)
"""
mutable struct ConjugateGradientDescentOptions{
    P,
    T,
    TCoeff<:DirectionUpdateRule,
    TStepsize<:Stepsize,
    TStop<:StoppingCriterion,
    TRetr<:AbstractRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
} <: AbstractGradientOptions
    x::P
    gradient::T
    δ::T
    β::Float64
    coefficient::TCoeff
    stepsize::TStepsize
    stop::TStop
    retraction_method::TRetr
    vector_transport_method::TVTM
    function ConjugateGradientDescentOptions{P,T}(
        M::AbstractManifold,
        x0::P,
        sC::StoppingCriterion,
        s::Stepsize,
        dC::DirectionUpdateRule,
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        initial_gradient::T=zero_vector(M, p),
    ) where {P,T}
        o = new{P,T,typeof(dC),typeof(s),typeof(sC),typeof(retr),typeof(vtr)}()
        o.x = x0
        o.gradient = initial_gradient
        o.δ = initial_gradient
        o.stop = sC
        o.retraction_method = retr
        o.stepsize = s
        o.coefficient = dC
        o.vector_transport_method = vtr
        return o
    end
end
function ConjugateGradientDescentOptions(
    M::AbstractManifold,
    x::P,
    sC::StoppingCriterion,
    s::Stepsize,
    dU::DirectionUpdateRule,
    retr::AbstractRetractionMethod=default_retraction_method(M),
    vtr::AbstractVectorTransportMethod=default_vector_transport_method(M),
    initial_gradient::T=zero_vector(M, x),
) where {P,T}
    return ConjugateGradientDescentOptions{P,T}(
        M, x, sC, s, dU, retr, vtr, initial_gradient
    )
end

@doc raw"""
    ConjugateDescentCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the
variables with prequel `Old` based on [^Flethcer1987] adapted to manifolds:

```math
β_k =
\frac{ \lVert ξ_{k+1} \rVert_{x_{k+1}}^2 }
{\langle -\delta_k,ξ_k \rangle_{x_k}}.
```

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    ConjugateDescentCoefficient(a::StoreOptionsAction=())

Construct the conjugate descent coefficient update rule, a new storage is created by default.

[^Flethcer1987]:
    > R. Fletcher, __Practical Methods of Optimization vol. 1: Unconstrained Optimization__
    > John Wiley & Sons, New York, 1987. doi [10.1137/1024028](https://doi.org/10.1137/1024028)
"""
mutable struct ConjugateDescentCoefficient <: DirectionUpdateRule
    storage::StoreOptionsAction
    function ConjugateDescentCoefficient(
        a::StoreOptionsAction=StoreOptionsAction((:x, :gradient))
    )
        return new(a)
    end
end
function (u::ConjugateDescentCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :gradient]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, gradientOld = get_storage.(Ref(u.storage), [:x, :gradient])
    update_storage!(u.storage, o)
    return inner(p.M, o.x, o.gradient, o.gradient) / inner(p.M, xOld, -o.δ, gradientOld)
end

@doc raw"""
    DaiYuanCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the
variables with prequel `Old` based on [^DaiYuan1999]

adapted to manifolds: let ``\nu_k = ξ_{k+1} - P_{x_{k+1}\gets x_k}ξ_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the coefficient reads

````math
β_k =
\frac{ \lVert ξ_{k+1} \rVert_{x_{k+1}}^2 }
{\langle P_{x_{k+1}\gets x_k}\delta_k, \nu_k \rangle_{x_{k+1}}}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    DaiYuanCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=(),
    )

Construct the Dai Yuan coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

[^DaiYuan1999]:
    > [Y. H. Dai and Y. Yuan, A nonlinear conjugate gradient method with a strong global convergence property,
    > SIAM J. Optim., 10 (1999), pp. 177–182.
    > doi: [10.1137/S1052623497318992](https://doi.org/10.1137/S1052623497318992)
"""
mutable struct DaiYuanCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function DaiYuanCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :gradient, :δ)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::DaiYuanCoefficient)(p::GradientProblem, o::ConjugateGradientDescentOptions, i)
    if !all(has_storage.(Ref(u.storage), [:x, :gradient, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, gradientOld, δOld = get_storage.(Ref(u.storage), [:x, :gradient, :δ])
    update_storage!(u.storage, o)

    gradienttr = vector_transport_to(p.M, xOld, gradientOld, o.x, u.transport_method)
    ν = o.gradient - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    return inner(p.M, o.x, o.gradient, o.gradient) / inner(p.M, xOld, δtr, ν)
end

@doc raw"""
    FletcherReevesCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the
variables with prequel `Old` based on [^FletcherReeves1964] adapted to manifolds:

````math
β_k =
\frac{\lVert ξ_{k+1}\rVert_{x_{k+1}}^2}{\lVert ξ_{k}\rVert_{x_{k}}^2}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    FletcherReevesCoefficient(a::StoreOptionsAction=())

Construct the Fletcher Reeves coefficient update rule, a new storage is created by default.

[^FletcherReeves1964]:
    > R. Fletcher and C. Reeves, Function minimization by conjugate gradients,
    > Comput. J., 7 (1964), pp. 149–154.
    > doi: [10.1093/comjnl/7.2.149](http://dx.doi.org/10.1093/comjnl/7.2.149)
"""
mutable struct FletcherReevesCoefficient <: DirectionUpdateRule
    storage::StoreOptionsAction
    function FletcherReevesCoefficient(
        a::StoreOptionsAction=StoreOptionsAction((:x, :gradient))
    )
        return new(a)
    end
end
function (u::FletcherReevesCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :gradient]))
        update_storage!(u.storage, o) # if not given store current as old
    end
    xOld, gradientOld = get_storage.(Ref(u.storage), [:x, :gradient])
    update_storage!(u.storage, o)
    return inner(p.M, o.x, o.gradient, o.gradient) /
           inner(p.M, xOld, gradientOld, gradientOld)
end

@doc raw"""
    HagerZhangCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the variables with
prequel `Old` based on [^HagerZhang2005]
adapted to manifolds: let ``\nu_k = ξ_{k+1} - P_{x_{k+1}\gets x_k}ξ_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

````math
β_k = \Bigl\langle\nu_k -
\frac{ 2\lVert \nu_k\rVert_{x_{k+1}}^2 }{ \langle P_{x_{k+1}\gets x_k}\delta_k, \nu_k \rangle_{x_{k+1}} }
P_{x_{k+1}\gets x_k}\delta_k,
\frac{ξ_{k+1}}{ \langle P_{x_{k+1}\gets x_k}\delta_k, \nu_k \rangle_{x_{k+1}} }
\Bigr\rangle_{x_{k+1}}.
````

This method includes a numerical stability proposed by those authors.

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    HagerZhangCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=(),
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
    storage::StoreOptionsAction
    function HagerZhangCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :gradient, :δ)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::HagerZhangCoefficient)(
    p::P, o::O, i
) where {P<:GradientProblem,O<:ConjugateGradientDescentOptions}
    if !all(has_storage.(Ref(u.storage), [:x, :gradient, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, gradientOld, δOld = get_storage.(Ref(u.storage), [:x, :gradient, :δ])
    update_storage!(u.storage, o)

    gradienttr = vector_transport_to(p.M, xOld, gradientOld, o.x, u.transport_method)
    ν = o.gradient - gradienttr #notation y from [HZ06]
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    denom = inner(p.M, o.x, δtr, ν)
    νknormsq = inner(p.M, o.x, ν, ν)
    β =
        inner(p.M, o.x, ν, o.gradient) / denom -
        2 * νknormsq * inner(p.M, o.x, δtr, o.gradient) / denom^2
    # Numerical stability from Manopt / Hager-Zhang paper
    ξn = norm(p.M, o.x, o.gradient)
    η = -1 / (ξn * min(0.01, norm(p.M, xOld, gradientOld)))
    return max(β, η)
end

@doc raw"""
    HeestenesStiefelCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the
variables with prequel `Old` based on [^HeestensStiefel1952]

adapted to manifolds as follows: let ``\nu_k = ξ_{k+1} - P_{x_{k+1}\gets x_k}ξ_k``.
Then the update reads

````math
β_k = \frac{\langle ξ_{k+1}, \nu_k \rangle_{x_{k+1}} }
    { \langle P_{x_{k+1}\gets x_k} \delta_k, \nu_k\rangle_{x_{k+1}} },
````
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

# Constructor
    HeestenesStiefelCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=()
    )

Construct the Heestens Stiefel coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

See also [`conjugate_gradient_descent`](@ref)

[^HeestensStiefel1952]:
    > M.R. Hestenes, E.L. Stiefel, Methods of conjugate gradients for solving linear systems,
    > J. Research Nat. Bur. Standards, 49 (1952), pp. 409–436.
    > doi: [10.6028/jres.049.044](http://dx.doi.org/10.6028/jres.049.044)
"""
mutable struct HeestenesStiefelCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function HeestenesStiefelCoefficient(
        transport_method::AbstractVectorTransportMethod=ParallelTransport(),
        storage_action::StoreOptionsAction=StoreOptionsAction((:x, :gradient, :δ)),
    )
        return new{typeof(transport_method)}(transport_method, storage_action)
    end
end
function (u::HeestenesStiefelCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :gradient, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, gradientOld, δOld = get_storage.(Ref(u.storage), [:x, :gradient, :δ])
    update_storage!(u.storage, o)
    gradienttr = vector_transport_to(p.M, xOld, gradientOld, o.x, u.transport_method)
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    ν = o.gradient - gradienttr #notation from [HZ06]
    β = inner(p.M, o.x, o.gradient, ν) / inner(p.M, o.x, δtr, ν)
    return max(0, β)
end

@doc raw"""
    LiuStoreyCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the
variables with prequel `Old` based on [^LuiStorey1991]
adapted to manifolds: let ``\nu_k = ξ_{k+1} - P_{x_{k+1}\gets x_k}ξ_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the coefficient reads

````math
β_k = -
\frac{ \langle ξ_{k+1},\nu_k \rangle_{x_{k+1}} }
{\langle \delta_k,ξ_k \rangle_{x_k}}.
````

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    LiuStoreyCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=()
    )

Construct the Lui Storey coefficient update rule, where the parallel transport is the
default vector transport and a new storage is created by default.

[^LuiStorey1991]:
    > [Y. Liu and C. Storey, Efficient generalized conjugate gradient algorithms, Part 1: Theory
    > J. Optim. Theory Appl., 69 (1991), pp. 129–137.
    > doi: [10.1007/BF00940464](https://doi.org/10.1007/BF00940464)
"""
mutable struct LiuStoreyCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function LiuStoreyCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :gradient, :δ)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::LiuStoreyCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :gradient, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
    end
    xOld, gradientOld, δOld = get_storage.(Ref(u.storage), [:x, :gradient, :δ])
    update_storage!(u.storage, o)
    gradienttr = vector_transport_to(p.M, xOld, gradientOld, o.x, u.transport_method)
    ν = o.gradient - gradienttr # notation y from [HZ06]
    return inner(p.M, o.x, o.gradient, ν) / inner(p.M, xOld, -δOld, gradientOld)
end

@doc raw"""
    PolakRibiereCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
``x_k,ξ_k``, the current iterates ``x_{k+1},ξ_{k+1}`` and the last update
direction ``\delta=\delta_k``, where the last three ones are stored in the
variables with prequel `Old` based on [^PolakRibiere1969][^Polyak1969]

adapted to manifolds: let ``\nu_k = ξ_{k+1} - P_{x_{k+1}\gets x_k}ξ_k``,
where ``P_{a\gets b}(⋅)`` denotes a vector transport from the tangent space at ``a`` to ``b``.

Then the update reads
````math
β_k =
\frac{ \langle ξ_{k+1}, \nu_k \rangle_{x_{k+1}} }
{\lVert ξ_k \rVert_{x_k}^2 }.
````

# Constructor
    PolakRibiereCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=()
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
mutable struct PolakRibiereCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function PolakRibiereCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :gradient)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::PolakRibiereCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :gradient]))
        update_storage!(u.storage, o) # if not given store current as old
    end
    xOld, gradientOld = get_storage.(Ref(u.storage), [:x, :gradient])
    update_storage!(u.storage, o)

    gradienttr = vector_transport_to(p.M, xOld, gradientOld, o.x, u.transport_method)
    ν = o.gradient - gradienttr
    β = inner(p.M, o.x, o.gradient, ν) / inner(p.M, xOld, gradientOld, gradientOld)
    return max(0, β)
end

@doc raw"""
    SteepestDirectionUpdateRule <: DirectionUpdateRule

The simplest rule to update is to have no influence of the last direction and
hence return an update ``β = 0`` for all [`ConjugateGradientDescentOptions`](@ref)` o`

See also [`conjugate_gradient_descent`](@ref)
"""
struct SteepestDirectionUpdateRule <: DirectionUpdateRule end
function (u::SteepestDirectionUpdateRule)(
    ::GradientProblem, ::ConjugateGradientDescentOptions, i
)
    return 0.0
end
