@doc raw"""
    GradientProblem <: Problem
specify a problem for gradient based algorithms.

# Fields
* `M`            – a manifold $\mathcal M$
* `cost` – a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     – the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$

# See also
[`gradient_descent`](@ref)
[`GradientDescentOptions`](@ref)

# """
struct GradientProblem{mT <: Manifold, TCost<:Base.Callable, TGradient<:Base.Callable} <: Problem
  M::mT
  cost::TCost
  gradient::TGradient
end
"""
    get_gradient(p,x)

evaluate the gradient of a [`GradientProblem`](@ref)`p` at the point `x`.
"""
function get_gradient(p::P,x) where {P <: GradientProblem{M} where M <: Manifold}
  return p.gradient(x)
end
#
# Options
#
"""
    GradientDescentOptions{P,T} <: Options

Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x0` – an a point (of type `P`) on a manifold as starting point
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`)a [`Stepsize`](@ref)
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use, defaults to
  the exponential map

# Constructor

    GradientDescentOptions(x, stop, s [, retr=ExponentialRetraction()])

construct a Gradient Descent Option with the fields and defaults as above

# See also
[`gradient_descent`](@ref), [`GradientProblem`](@ref)
"""
mutable struct GradientDescentOptions{P,TStop<:StoppingCriterion,TStepsize<:Stepsize,TRTM<:AbstractRetractionMethod} <: Options
    x::P
    stop::TStop
    stepsize::TStepsize
    ∇::P
    retraction_method::TRTM
    function GradientDescentOptions{P}(
        initialX::P,
        s::StoppingCriterion = StopAfterIteration(100),
        stepsize::Stepsize = ConstantStepsize(1.),
        retraction::AbstractRetractionMethod=ExponentialRetraction(),
    ) where {P}
        o = new{P,typeof(s),typeof(stepsize),typeof(retraction)}();
        o.x = initialX;
        o.stop = s;
        o.retraction_method = retraction;
        o.stepsize = stepsize;
        return o
    end
end
function GradientDescentOptions(
    x::P,
    stop::StoppingCriterion = StopAfterIteration(100),
    s::Stepsize = ConstantStepsize(1.),
    retraction::AbstractRetractionMethod = ExponentialRetraction(),
) where {P}
    return GradientDescentOptions{P}(x,stop,s,retraction)
end
#
# Conjugate Gradient Descent
#
"""
    DirectionUpdateRule

A general functor, that handles direction update rules. It's field(s) is usually
only a [`StoreOptionsAction`](@ref) by default initialized to the fields required
for the specific coefficient, but can also be replaced by a (common, global)
individual one that provides these values.
"""
abstract type DirectionUpdateRule end

@doc raw"""
    ConjugateGradientOptions <: Options

specify options for a conjugate gradient descent algoritm, that solves a
[`GradientProblem`].

# Fields
* `x` – the current iterate, a point on a manifold
* `∇` – the current gradient
* `δ` – the current descent direction, i.e. also tangent vector
* `β` – the current update coefficient rule, see .
* `coefficient` – a [`DirectionUpdateRule`](@ref) function to determine the new `β`
* `stepsize` – a [`Stepsize`](@ref) function
* `stop` – a [`StoppingCriterion`](@ref)
* `retraction` – a function to perform a step on the manifold

# See also
[`conjugate_gradient_descent`](@ref), [`GradientProblem`](@ref), [`ArmijoLinesearch`](@ref)
"""
mutable struct ConjugateGradientDescentOptions{T} <: Options
    x::T
    ∇::T
    δ::T
    β::Float64
    coefficient::DirectionUpdateRule
    stepsize::Stepsize
    stop::StoppingCriterion
    retraction_method::AbstractRetractionMethod
    vector_transport_method::AbstractVectorTransportMethod
    function ConjugateGradientDescentOptions{T}(
        x0::T,
        sC::StoppingCriterion,
        s::Stepsize,
        dC::DirectionUpdateRule,
        retr::AbstractRetractionMethod = ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod = ParallelTransport(),
    ) where {T}
        o = new{T}();
        o.x = x0
        o.stop = sC
        o.retraction_method = retr
        o.stepsize = s
        o.coefficient = dC
        o.vector_transport_method = vtr
        return o
    end
end
function ConjugateGradientDescentOptions(
    x::T,
    sC::StoppingCriterion,
    s::Stepsize,
    dU::DirectionUpdateRule,
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    vtr::AbstractVectorTransportMethod = ParallelTransport(),
) where {T}
    return ConjugateGradientDescentOptions{T}(x, sC, s, dU, retr, vtr)
end

@doc raw"""
    ConjugateDescentCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the
variables with prequel `Old` based on [^Flethcer1987] adapted to manifolds:

$\beta_k =
\frac{ \lVert \xi_{k+1} \rVert_{x_{k+1}}^2 }
{\langle -\delta_k,\xi_k \rangle_{x_k}}.$

See also [`conjugate_gradient_descent`](@ref)

# Constructor
    ConjugateDescentCoefficient(a::StoreOptionsAction=())

Construct the conjugate descnt coefficient update rule, a new storage is created by default.

[^Flethcer1987]:
    > R. Fletcher, __Practical Methods of Optimization vol. 1: Unconstrained Optimization__
    > John Wiley & Sons, New York, 1987. doi [10.1137/1024028](https://doi.org/10.1137/1024028)
"""
mutable struct ConjugateDescentCoefficient <: DirectionUpdateRule
    storage::StoreOptionsAction
    ConjugateDescentCoefficient(  a::StoreOptionsAction=StoreOptionsAction( (:x, :∇) )  ) = new(a)
end
function (u::ConjugateDescentCoefficient)(
    p::GradientProblem,
    o::ConjugateGradientDescentOptions,
    i
)
    if !all( has_storage.(Ref(u.storage), [:x, :∇]))
        update_storage!(u.storage,o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old = get_storage.( Ref(u.storage), [:x, :∇] )
    update_storage!(u.storage,o)
    return inner(p.M, o.x, o.∇, o.∇) / inner(p.M, xOld, -o.δ, ∇Old)
end

@doc raw"""
    DaiYuanCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the
variables with prequel `Old` based on [^DaiYuan1999]

adapted to manifolds: let $\nu_k = \xi_{k+1} - P_{x_{k+1}\gets x_k}\xi_k$,
where $P_{a\gets b}(\cdot)$ denotes a vector transport from the tangent space at $a$ to $b$.

Then the coefficient reads

````math
\beta_k =
\frac{ \lVert \xi_{k+1} \rVert_{x_{k+1}}^2 }
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
mutable struct DaiYuanCoefficient <: DirectionUpdateRule
    transport_method::AbstractVectorTransportMethod
    storage::StoreOptionsAction
    function DaiYuanCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction( (:x, :∇, :δ) ),
    )
        return new(t,a)
    end
end
function (u::DaiYuanCoefficient)(p::GradientProblem, o::ConjugateGradientDescentOptions, i)
    if !all( has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage,o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old, δOld = get_storage.( Ref(u.storage), [:x, :∇, :δ] )
    update_storage!(u.storage,o)

    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇ - ∇tr #notation y from [HZ06]
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    return inner(p.M, o.x, o.∇, o.∇)  /  inner(p.M, xOld, δtr, ν)
end

@doc raw"""
    FletcherReevesCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the
variables with prequel `Old` based on [^FletcherReeves1964] adapted to manifolds:

````math
\beta_k =
\frac{\lVert \xi_{k+1}\rVert_{x_{k+1}}^2}{\lVert \xi_{k}\rVert_{x_{k}}^2}.
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
    FletcherReevesCoefficient(  a::StoreOptionsAction=StoreOptionsAction( (:x, :∇) )  ) = new(a)
end
function (u::FletcherReevesCoefficient)(
    p::GradientProblem,
    o::ConjugateGradientDescentOptions,
    i,
)
    if !all( has_storage.(Ref(u.storage), [:x, :∇]))
        update_storage!(u.storage,o) # if not given store current as old
    end
    xOld, ∇Old = get_storage.( Ref(u.storage), [:x, :∇] )
    update_storage!(u.storage,o)
    return inner(p.M, o.x, o.∇, o.∇)/inner(p.M, xOld, ∇Old, ∇Old)
end

@doc raw"""
    HagerZhangCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the variables with
prequel `Old` based on [^HagerZhang2005]
adapted to manifolds: let $\nu_k = \xi_{k+1} - P_{x_{k+1}\gets x_k}\xi_k$,
where $P_{a\gets b}(\cdot)$ denotes a vector transport from the tangent space at $a$ to $b$.

````math
\beta_k = \Bigl\langle\nu_k -
\frac{ 2\lVert \nu_k\rVert_{x_{k+1}}^2 }{ \langle P_{x_{k+1}\gets x_k}\delta_k, \nu_k \rangle_{x_{k+1}} }
P_{x_{k+1}\gets x_k}\delta_k,
\frac{\xi_{k+1}}{ \langle P_{x_{k+1}\gets x_k}\delta_k, \nu_k \rangle_{x_{k+1}} }
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
mutable struct HagerZhangCoefficient <: DirectionUpdateRule
    transport_method::AbstractVectorTransportMethod
    storage::StoreOptionsAction
    function HagerZhangCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction( (:x, :∇, :δ) ),
    )
        return new(t,a)
    end
end
function (u::HagerZhangCoefficient)(p::P, o::O, i) where {P <: GradientProblem, O <: ConjugateGradientDescentOptions}
    if !all( has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage,o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old, δOld = get_storage.( Ref(u.storage), [:x, :∇, :δ] )
    update_storage!(u.storage,o)

    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇ - ∇tr #notation y from [HZ06]
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    denom = inner(p.M, o.x, δtr, ν)
    νknormsq = inner(p.M, o.x, ν, ν)
    β = inner(p.M, o.x, ν, o.∇)/denom - 2*νknormsq*inner(p.M, o.x, δtr, o.∇) / denom^2
    # Numerical stability from Manopt / Hager-Zhang paper
    ξn = norm(p.M, o.x, o.∇)
    η = -1 / ( ξn * min(0.01,norm(p.M, xOld, ∇Old)) )
    return max(β,η)
end

@doc raw"""
    HeestenesStiefelCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the
variables with prequel `Old` based on [^HeestensStiefel1952]

adapted to manifolds as follows: let $\nu_k = \xi_{k+1} - P_{x_{k+1}\gets x_k}\xi_k$.
Then the update reads

````math
\beta_k = \frac{\langle \xi_{k+1}, \nu_k \rangle_{x_{k+1}} }
    { \langle P_{x_{k+1}\gets x_k} \delta_k, \nu_k\rangle_{x_{k+1}} },
````
where $P_{a\gets b}(\cdot)$ denotes a vector transport from the tangent space at $a$ to $b$.

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
mutable struct HeestenesStiefelCoefficient <: DirectionUpdateRule
    transport_method::AbstractVectorTransportMethod
    storage::StoreOptionsAction
    function HeestenesStiefelCoefficient(
        transort_method::AbstractVectorTransportMethod=ParallelTransport(),
        storage_action::StoreOptionsAction=StoreOptionsAction( (:x, :∇, :δ) ),
    )
        return new(transort_method,storage_action)
    end
end
function (u::HeestenesStiefelCoefficient)(
    p::GradientProblem,
    o::ConjugateGradientDescentOptions,
    i,
)
    if !all( has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage,o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old, δOld = get_storage.( Ref(u.storage), [:x, :∇, :δ] )
    update_storage!(u.storage,o)
    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    ν = o.∇ - ∇tr #notation from [HZ06]
    β = inner(p.M, o.x, o.∇, ν) / inner(p.M, o.x, δtr, ν)
    return max(0,β)
end

@doc raw"""
    LiuStoreyCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the
variables with prequel `Old` based on [^LuiStorey1991]
adapted to manifolds: let $\nu_k = \xi_{k+1} - P_{x_{k+1}\gets x_k}\xi_k$,
where $P_{a\gets b}(\cdot)$ denotes a vector transport from the tangent space at $a$ to $b$.

Then the coefficient reads

````math
\beta_k = -
\frac{ \langle \xi_{k+1},\nu_k \rangle_{x_{k+1}} }
{\langle \delta_k,\xi_k \rangle_{x_k}}.
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
mutable struct LiuStoreyCoefficient <: DirectionUpdateRule
    transport_method::AbstractVectorTransportMethod
    storage::StoreOptionsAction
    function LiuStoreyCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction( (:x, :∇, :δ) ),
    )
        return new(t,a)
    end
end
function (u::LiuStoreyCoefficient)(
    p::GradientProblem,
    o::ConjugateGradientDescentOptions,
    i,
)
    if !all( has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage,o) # if not given store current as old
    end
    xOld, ∇Old, δOld = get_storage.( Ref(u.storage), [:x, :∇, :δ] )
    update_storage!(u.storage,o)
    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇-∇tr # notation y from [HZ06]
    return inner(p.M, o.x, o.∇, ν)  /  inner(p.M, xOld, -δOld, ∇Old)
end



@doc raw"""
    PolakRibiereCoefficient <: DirectionUpdateRule

Computes an update coefficient for the conjugate gradient method, where
the [`ConjugateGradientDescentOptions`](@ref)` o` include the last iterates
$x_k,\xi_k$, the current iterates $x_{k+1},\xi_{k+1}$ and the last update
direction $\delta=\delta_k$, where the last three ones are stored in the
variables with prequel `Old` based on [^PolakRibiere1969][^Polyak1969]

adapted to manifolds: let $\nu_k = \xi_{k+1} - P_{x_{k+1}\gets x_k}\xi_k$,
where $P_{a\gets b}(\cdot)$ denotes a vector transport from the tangent space at $a$ to $b$.

Then the update reads
````math
\beta_k =
\frac{ \langle \xi_{k+1}, \nu_k \rangle_{x_{k+1}} }
{\lVert \xi_k \rVert_{x_k}^2 }.
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
mutable struct PolakRibiereCoefficient <: DirectionUpdateRule
    transport_method::AbstractVectorTransportMethod
    storage::StoreOptionsAction
    function PolakRibiereCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction( (:x, :∇) ),
    )
        return new(t,a)
    end
end
function (u::PolakRibiereCoefficient)(p::GradientProblem, o::ConjugateGradientDescentOptions, i)
    if !all( has_storage.(Ref(u.storage), [:x, :∇]))
        update_storage!(u.storage,o) # if not given store current as old
    end
    xOld, ∇Old = get_storage.( Ref(u.storage), [:x, :∇] )
    update_storage!(u.storage,o)

    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇-∇tr
    β = inner(p.M, o.x, o.∇, ν) / inner(p.M, xOld, ∇Old, ∇Old)
    return max(0,β)
end

@doc raw"""
    steepestDirectionUpdateRule <: DirectionUpdateRule

The simplest rule to update is to have no influence of the last direction and
hence return an update $\beta = 0$ for all [`ConjugateGradientDescentOptions`](@ref)` o`

See also [`conjugate_gradient_descent`](@ref)
"""
mutable struct SteepestDirectionUpdateRule <: DirectionUpdateRule end
function (u::SteepestDirectionUpdateRule)(
    p::GradientProblem,
    o::ConjugateGradientDescentOptions,
    i,
)
    return 0.0
end


@doc raw"""
    DebugGradient <: DebugAction

debug for the gradient evaluated at the current iterate

# Constructors
    DebugGradient([long=false,p=print])

display the short (`false`) or long (`true`) default text for the gradient.

    DebugGradient(prefix[, p=print])

display the a `prefix` in front of the gradient.
"""
mutable struct DebugGradient <: DebugAction
    print::Base.Callable
    prefix::String
    DebugGradient(long::Bool=false,print::Base.Callable=print) = new(print,
        long ? "Gradient: " : "∇F(x):")
    DebugGradient(prefix::String,print::Base.Callable=print) = new(print,prefix)
end
(d::DebugGradient)(p::GradientProblem,o::GradientDescentOptions,i::Int) = d.print((i>=0) ? d.prefix*""*string(o.∇) : "")

@doc raw"""
    DebugGradientNorm <: DebugAction

debug for gradient evaluated at the current iterate.

# Constructors
    DebugGradientNorm([long=false,p=print])

display the short (`false`) or long (`true`) default text for the gradient norm.

    DebugGradientNorm(prefix[, p=print])

display the a `prefix` in front of the gradientnorm.
"""
mutable struct DebugGradientNorm <: DebugAction
    print::Base.Callable
    prefix::String
    DebugGradientNorm(long::Bool=false,print::Base.Callable=print) = new(print,
        long ? "Norm of the Gradient: " : "|∇F(x)|:")
    DebugGradientNorm(prefix::String,print::Base.Callable=print) = new(print,prefix)
end
(d::DebugGradientNorm)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: GradientDescentOptions} = d.print((i>=0) ? d.prefix*"$(norm(p.M,o.x,o.∇))" : "")

@doc raw"""
    DebugStepsize <: DebugAction

debug for the current step size.

# Constructors
    DebugStepsize([long=false,p=print])

display the short (`false`) or long (`true`) default text for the step size.

    DebugStepsize(prefix[, p=print])

display the a `prefix` in front of the step size.
"""
mutable struct DebugStepsize <: DebugAction
    print::Base.Callable
    prefix::String
    DebugStepsize(long::Bool=false,print::Base.Callable=print) = new(print,
        long ? "step size:" : "s:")
    DebugStepsize(prefix::String,print::Base.Callable=print) = new(print,prefix)
end
(d::DebugStepsize)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: GradientDescentOptions} = d.print((i>0) ? d.prefix*"$(get_last_stepsize(p,o,i))" : "")

#
# Records
#
@doc raw"""
    RecordGradient <: RecordAction

record the gradient evaluated at the current iterate

# Constructors
    RecordGradient(ξ)

initialize the [`RecordAction`](@ref) to the corresponding type of the tangent vector.
"""
mutable struct RecordGradient{T} <: RecordAction
    recordedValues::Array{T,1}
    RecordGradient{T}() where {T} = new(Array{T,1}())
end
RecordGradient(ξ::T) where {T} = RecordGradient{T}()
(r::RecordGradient{T})(p::P,o::O,i::Int) where {T, P <: GradientProblem, O <: GradientDescentOptions} = record_or_eset!(r, o.∇, i)

@doc raw"""
    RecordGradientNorm <: RecordAction

record the norm of the current gradient
"""
mutable struct RecordGradientNorm <: RecordAction
    recordedValues::Array{Float64,1}
    RecordGradientNorm() = new(Array{Float64,1}())
end
(r::RecordGradientNorm)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: GradientDescentOptions} = record_or_eset!(r, norm(p.M,o.x,o.∇), i)

@doc raw"""
    RecordStepsize <: RecordAction

record the step size
"""
mutable struct RecordStepsize <: RecordAction
    recordedValues::Array{Float64,1}
    RecordStepsize() = new(Array{Float64,1}())
end
(r::RecordStepsize)(p::P,o::O,i::Int) where {P <: GradientProblem, O <: GradientDescentOptions} = record_or_eset!(r, get_last_stepsize(p,o,i), i)
