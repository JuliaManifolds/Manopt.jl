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
struct GradientProblem{mT<:Manifold,TCost,TGradient} <: Problem
    M::mT
    cost::TCost
    gradient::TGradient
end
"""
    get_gradient(p,x)

evaluate the gradient of a [`GradientProblem`](@ref)`p` at the point `x`.
"""
function get_gradient(p::GradientProblem, x)
    return p.gradient(x)
end

"""
    DirectionUpdateRule

A general functor, that handles direction update rules. It's field(s) is usually
only a [`StoreOptionsAction`](@ref) by default initialized to the fields required
for the specific coefficient, but can also be replaced by a (common, global)
individual one that provides these values.
"""
abstract type DirectionUpdateRule end

"""
    IdentityUpdateRule <: DirectionUpdateRule

The default gradient direction update is the identity, i.e. it just evaluates the gradient.
"""
struct IdentityUpdateRule <: DirectionUpdateRule end

"""
    AbstractGradientDescentOptions <: Options

A generic [`Options`](@ref) type for gradient based options data.
"""
abstract type AbstractGradientDescentOptions <: Options end

"""
    GradientDescentOptions{P,T} <: AbstractGradientDescentOptions

Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x0` – an a point (of type `P`) on a manifold as starting point
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`)a [`Stepsize`](@ref)
* `direction` - ([`IdentityUpdateRule`](@ref)) a processor to compute the gradient
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use, defaults to
  the exponential map

# Constructor

    GradientDescentOptions(x, stop, s [, retr=ExponentialRetraction()])

construct a Gradient Descent Option with the fields and defaults as above

# See also
[`gradient_descent`](@ref), [`GradientProblem`](@ref)
"""
mutable struct GradientDescentOptions{
    P,TStop<:StoppingCriterion,TStepsize<:Stepsize,TRTM<:AbstractRetractionMethod
} <: AbstractGradientDescentOptions
    x::P
    direction::DirectionUpdateRule
    stop::TStop
    stepsize::TStepsize
    ∇::P
    retraction_method::TRTM
    function GradientDescentOptions{P}(
        initialX::P,
        s::StoppingCriterion=StopAfterIteration(100),
        stepsize::Stepsize=ConstantStepsize(1.0),
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        direction::DirectionUpdateRule=IdentityUpdateRule(),
    ) where {P}
        o = new{P,typeof(s),typeof(stepsize),typeof(retraction_method)}()
        o.x = initialX
        o.stop = s
        o.retraction_method = retraction_method
        o.stepsize = stepsize
        o.direction = direction
        return o
    end
end
function GradientDescentOptions(
    x::P;
    stopping_criterion::StoppingCriterion=StopAfterIteration(100),
    stepsize::Stepsize=ConstantStepsize(1.0),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    direction::DirectionUpdateRule=IdentityUpdateRule(),
) where {P}
    return GradientDescentOptions{P}(
        x, stopping_criterion, stepsize, retraction_method, direction
    )
end
#
# Processors
#
function (s::IdentityUpdateRule)(p::GradientProblem, o::GradientDescentOptions, i)
    return get_stepsize(p, o, i), get_gradient(p, o.x)
end

"""
    MomentumGradient <: DirectionUpdateRule

Append a momentum to a gradient processor, where the last direction and last iterate are
stored and the new is composed as ``\nabla_i = m*\nabla_{i-1}' - s d_i``,
where ``sd_i`` is the current (inner) direction and ``\nabla_{i-1}'`` is the vector transported
last direction multiplied by momentum ``m``.

# Fields
* `∇` – (`zero_tangent_vector(M,x0)`) the last gradient/direction update added as momentum
* `last_iterate` - remember the last iterate for parallel transporting the last direction
* `momentum` – (`0.2`) factor for momentum
* `direction` – internal [`DirectionUpdateRule`](@ref) to determine directions to
  add the momentum to.
* `vector_transport_method` vector transport method to use

# Constructors
    MomentumGradient(
        p::GradientProlem,
        x0,
        s::DirectionUpdateRule=Gradient();
        ∇=zero_tangent_vector(p.M, o.x), momentum=0.2
       vector_transport_method=ParallelTransport(),
    )

Add momentum to a gradient problem, where by default just a gradient evaluation is used
Equivalently you can also use a `Manifold` `M` instead of the [`GradientProblem`](@ref) `p`.

    MomentumGradient(
        p::StochasticGradientProblem
        x0
        s::DirectionUpdateRule=StochasticGradient();
        ∇=zero_tangent_vector(p.M, x0), momentum=0.2
       vector_transport_method=ParallelTransport(),
    )

Add momentum to a stochastic gradient problem, where by default just a stochastic gradient evaluation is used
"""
mutable struct MomentumGradient{P,T,R<:Real,VTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    ∇::T
    last_iterate::P
    momentum::R
    direction::DirectionUpdateRule
    vector_transport_method::VTM
end
function MomentumGradient(
    p::GradientProblem,
    x0::P,
    s::DirectionUpdateRule=IdentityUpdateRule();
    last_iterate=x0,
    vector_transport_method::VTM=ParallelTransport(),
    ∇=zero_tangent_vector(p.M, x0),
    momentum=0.2,
) where {P,VTM<:AbstractVectorTransportMethod}
    return MomentumGradient{P,typeof(∇),typeof(momentum),VTM}(
        deepcopy(x0), ∇, momentum, s, vector_transport_method
    )
end
function MomentumGradient(
    M::Manifold,
    x0::P,
    s::DirectionUpdateRule=IdentityUpdateRule();
    ∇=zero_tangent_vector(M, x0),
    last_iterate=x0,
    momentum=0.2,
    vector_transport_method::VTM=ParallelTransport(),
) where {P,VTM<:AbstractVectorTransportMethod}
    return MomentumGradient{P,typeof(∇),typeof(momentum),VTM}(
        deepcopy(x0), ∇, momentum, s, vector_transport_method
    )
end
function (m::MomentumGradient)(p::Problem, o::AbstractGradientDescentOptions, i)
    s, d = m.direction(p, o, i) #get inner direction and step size
    old_d =
        m.momentum *
        vector_transport_to(p.M, m.last_iterate, m.∇, o.x, m.vector_transport_method)
    m.∇ = old_d - s .* d
    m.last_iterate = deepcopy(o.x)
    return s, -m.∇
end

"""
    AverageGradient <: DirectionUpdateRule

Add an average of gradients to a gradient processor. A set of previous directions (from the
inner processor) and the last iterate are stored, average is taken after vector transporting
them to the current iterates tangent space.

# Fields
* `gradients` – (fill(`zero_tangent_vector(M,x0),n)`) the last `n` gradient/direction updates
* `last_iterate` – last iterate (needed to transport the gradients)
* `direction` – internal [`DirectionUpdateRule`](@ref) to determine directions to
  apply the averaging to
* `vector_transport_method` - vector transport method to use

# Constructors
    AverageGradient(
        p::GradientProlem,
        x0,
        n::Int=10
        s::DirectionUpdateRule=IdentityUpdateRule();
        gradients = fill(zero_tangent_vector(p.M, o.x),n),
        last_iterate = deepcopy(x0),
        vector_transport_method = ParallelTransport()
    )

Add average to a gradient problem, `n` determines the size of averaging and `gradients` can be prefilled with some history
Equivalently you can also use a `Manifold` `M` instead of the [`GradientProblem`](@ref) `p`.

    AverageGradient(
        p::StochasticGradientProblem
        x0
        n::Int=10
        s::DirectionUpdateRule=StochasticGradient();
        gradients = fill(zero_tangent_vector(p.M, o.x),n),
        last_iterate = deepcopy(x0),
        vector_transport_method = ParallelTransport()
    )

Add average to a stochastic gradient problem, `n` determines the size of averaging and `gradients` can be prefilled with some history
"""
mutable struct AverageGradient{P,T,VTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    gradients::AbstractVector{T}
    last_iterate::P
    direction::DirectionUpdateRule
    vector_transport_method::VTM
end
function AverageGradient(
    M::Manifold,
    x0::P,
    n::Int=10,
    s::DirectionUpdateRule=IdentityUpdateRule();
    gradients=fill(zero_tangent_vector(M, x0), n),
    vector_transport_method::VTM=ParallelTransport(),
) where {P,VTM<:AbstractVectorTransportMethod}
    return AverageGradient{P,eltype(gradients),VTM}(
        gradients, x0, s, vector_transport_method
    )
end
function AverageGradient(
    p::GradientProblem,
    x0::P,
    n::Int=10,
    s::DirectionUpdateRule=IdentityUpdateRule();
    gradients=fill(zero_tangent_vector(p.M, x0), n),
    vector_transport_method::VTM=ParallelTransport(),
) where {P,VTM}
    return AverageGradient{P,eltype(gradients),VTM}(
        gradients, deepcopy(x0), s, vector_transport_method
    )
end
function (a::AverageGradient)(p::Problem, o::AbstractGradientDescentOptions, i)
    pop!(a.gradients)
    s, d = a.direction(p, o, i) #get inner gradient and step
    a.gradients = vcat([d], a.gradients)
    for i in 1:(length(a.gradients) - 1) #transport & shift inplace
        vector_transport_to!(
            p.M,
            a.gradients[i],
            a.last_iterate,
            a.gradients[i + 1],
            o.x,
            a.vector_transport_method,
        )
    end
    a.gradients[1] = d
    a.last_iterate = deepcopy(o.x)
    return s, -1 / length(a.gradients) .* sum(a.gradients)
end

@doc raw"""
    Nesterov <: DirectionUpdateRule

## Fields
* `γ`
* `μ` the strong convexity coefficient
* `v` (=``=v_k``, ``v_0=x_0``) an interims point to compute the next gradient evaluation point `y_k`
* `shrinkage` (`= i -> 0.8`) a function to compute the shrinkage ``β_k`` per iterate.

Let's assume ``f`` is ``L``-Lipschitz and ``μ``-strongly convex.
Given
* a step size ``h_k<\frac{1}{L}`` (from the [`GradientDescentOptions`](@ref)
* a `shrinkage` parameter ``β_k``
* and a current iterate $x_k$
* as well as the interims values $γ_k`` and ``v_k`` from the previous iterate.

This compute a Nesterov type update using the following steps, see [^ZhangSra2018]

1. Copute the positive root, i.e. ``α_k∈(0,1)`` of ``α^2 = h_k\bigl((1-α_k)γ_k+α_k μ\bigr)``.
2. Set ``\bar γ_k+1 = (1-α_k)γ_k + α_kμ``
3. ``y_k = \operatorname{retr}_{x_k}\Bigl(\frac{α_kγ_k}{γ_k + α_kμ}\operatorname{retr}^{-1}_{x_k}v_k \Bigr)``
4. ``x_{k+1} = \operatorname{retr}_{y_k}(-h_k ∇f(y_k))``
5. ``v_{k+1} = `\operatorname{retr}_{y_k}\Bigl(\frac{(1-α_k)γ_k}{\barγ_k}\operatorname{retr}_{y_k}^{-1}(v_k) - \frac{α_k}{\bar γ_{k+1}}∇f(y_k) \Bigr)``
6. ``γ_{k+1} = \frac{1}{1+β_k}\bar γ_{k+1}``

Then the direction from ``x_k`` to ``x_k+1``, i.e. ``d = \operatorname{retr}^{-1}_{x_k}x_{k+1}`` is returned.

# Constructor
    Nesterov(x0::P, γ=0.001, μ=0.9, schrinkage = k -> 0.8;
        inverse_retraction_method=LogarithmicInverseRetraction())

Initialize the Nesterov acceleration, where `x0` initializes `v`.

[^ZhangSra2018]:
    > H. Zhang, S. Sra: _Towards Riemannian Accelerated Gradient Methods_,
    > Preprint, 2018, arXiv: [1806.02812](https://arxiv.org/abs/1806.02812)
"""
mutable struct Nesterov{P,T<:Real} <: DirectionUpdateRule
    γ::T
    μ::T
    v::P
    shrinkage::Function
    inverse_retraction_method::AbstractInverseRetractionMethod
end
function Nesterov(
    x0::P,
    γ::T=0.001,
    μ::T=0.9,
    shrinkage::Function=i -> 0.8;
    inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
) where {P,T}
    return Nesterov{P,T}(γ, μ, deepcopy(x0), shrinkage, inverse_retraction_method)
end
function (s::Nesterov)(p::GradientProblem, o::AbstractGradientDescentOptions, i)
    h = get_stepsize(p, o, i)
    α = (h * (s.γ - s.μ) + sqrt(h^2 * (s.γ - s.μ)^2 + 4 * h * s.γ)) / 2
    γbar = (1 - α) * s.γ + α * s.μ
    y = retract(p.M, o.x, (α * s.γ) / (s.γ + α * s.μ) .* inverse_retract(p.M, o.x, s.v))
    gradf_yk = get_gradient(p, y)
    xn = retract(p.M, y, -h * gradf_yk)
    d =
        ((1 - α) * s.γ) / γbar .*
        inverse_retract(p.M, y, s.v, s.inverse_retraction_method) - α / γbar .* gradf_yk
    s.v = retract(p.M, y, d, o.retraction_method)
    s.γ = 1 / (1 + s.shrinkage(i)) * γbar
    return h, -1 / h .* inverse_retract(p.M, o.x, xn) # outer update
end

#
# Conjugate Gradient Descent
#

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
* `retraction_method` – (`ExponentialRetraction()`) a type of retraction

# See also
[`conjugate_gradient_descent`](@ref), [`GradientProblem`](@ref), [`ArmijoLinesearch`](@ref)
"""
mutable struct ConjugateGradientDescentOptions{
    T,
    TCoeff<:DirectionUpdateRule,
    TStepsize<:Stepsize,
    TStop<:StoppingCriterion,
    TRetr<:AbstractRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
} <: Options
    x::T
    ∇::T
    δ::T
    β::Float64
    coefficient::TCoeff
    stepsize::TStepsize
    stop::TStop
    retraction_method::TRetr
    vector_transport_method::TVTM
    function ConjugateGradientDescentOptions{T}(
        x0::T,
        sC::StoppingCriterion,
        s::Stepsize,
        dC::DirectionUpdateRule,
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
    ) where {T}
        o = new{T,typeof(dC),typeof(s),typeof(sC),typeof(retr),typeof(vtr)}()
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
    vtr::AbstractVectorTransportMethod=ParallelTransport(),
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
    function ConjugateDescentCoefficient(a::StoreOptionsAction=StoreOptionsAction((:x, :∇)))
        return new(a)
    end
end
function (u::ConjugateDescentCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :∇]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old = get_storage.(Ref(u.storage), [:x, :∇])
    update_storage!(u.storage, o)
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
mutable struct DaiYuanCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function DaiYuanCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :∇, :δ)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::DaiYuanCoefficient)(p::GradientProblem, o::ConjugateGradientDescentOptions, i)
    if !all(has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old, δOld = get_storage.(Ref(u.storage), [:x, :∇, :δ])
    update_storage!(u.storage, o)

    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇ - ∇tr #notation y from [HZ06]
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    return inner(p.M, o.x, o.∇, o.∇) / inner(p.M, xOld, δtr, ν)
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
    FletcherReevesCoefficient(a::StoreOptionsAction=StoreOptionsAction((:x, :∇))) = new(a)
end
function (u::FletcherReevesCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :∇]))
        update_storage!(u.storage, o) # if not given store current as old
    end
    xOld, ∇Old = get_storage.(Ref(u.storage), [:x, :∇])
    update_storage!(u.storage, o)
    return inner(p.M, o.x, o.∇, o.∇) / inner(p.M, xOld, ∇Old, ∇Old)
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
mutable struct HagerZhangCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function HagerZhangCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :∇, :δ)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::HagerZhangCoefficient)(
    p::P, o::O, i
) where {P<:GradientProblem,O<:ConjugateGradientDescentOptions}
    if !all(has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old, δOld = get_storage.(Ref(u.storage), [:x, :∇, :δ])
    update_storage!(u.storage, o)

    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇ - ∇tr #notation y from [HZ06]
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    denom = inner(p.M, o.x, δtr, ν)
    νknormsq = inner(p.M, o.x, ν, ν)
    β = inner(p.M, o.x, ν, o.∇) / denom - 2 * νknormsq * inner(p.M, o.x, δtr, o.∇) / denom^2
    # Numerical stability from Manopt / Hager-Zhang paper
    ξn = norm(p.M, o.x, o.∇)
    η = -1 / (ξn * min(0.01, norm(p.M, xOld, ∇Old)))
    return max(β, η)
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
mutable struct HeestenesStiefelCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function HeestenesStiefelCoefficient(
        transport_method::AbstractVectorTransportMethod=ParallelTransport(),
        storage_action::StoreOptionsAction=StoreOptionsAction((:x, :∇, :δ)),
    )
        return new{typeof(transport_method)}(transport_method, storage_action)
    end
end
function (u::HeestenesStiefelCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
        return 0.0
    end
    xOld, ∇Old, δOld = get_storage.(Ref(u.storage), [:x, :∇, :δ])
    update_storage!(u.storage, o)
    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    δtr = vector_transport_to(p.M, xOld, δOld, o.x, u.transport_method)
    ν = o.∇ - ∇tr #notation from [HZ06]
    β = inner(p.M, o.x, o.∇, ν) / inner(p.M, o.x, δtr, ν)
    return max(0, β)
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
mutable struct LiuStoreyCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function LiuStoreyCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :∇, :δ)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::LiuStoreyCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :∇, :δ]))
        update_storage!(u.storage, o) # if not given store current as old
    end
    xOld, ∇Old, δOld = get_storage.(Ref(u.storage), [:x, :∇, :δ])
    update_storage!(u.storage, o)
    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇ - ∇tr # notation y from [HZ06]
    return inner(p.M, o.x, o.∇, ν) / inner(p.M, xOld, -δOld, ∇Old)
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
mutable struct PolakRibiereCoefficient{TVTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    transport_method::TVTM
    storage::StoreOptionsAction
    function PolakRibiereCoefficient(
        t::AbstractVectorTransportMethod=ParallelTransport(),
        a::StoreOptionsAction=StoreOptionsAction((:x, :∇)),
    )
        return new{typeof(t)}(t, a)
    end
end
function (u::PolakRibiereCoefficient)(
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
)
    if !all(has_storage.(Ref(u.storage), [:x, :∇]))
        update_storage!(u.storage, o) # if not given store current as old
    end
    xOld, ∇Old = get_storage.(Ref(u.storage), [:x, :∇])
    update_storage!(u.storage, o)

    ∇tr = vector_transport_to(p.M, xOld, ∇Old, o.x, u.transport_method)
    ν = o.∇ - ∇tr
    β = inner(p.M, o.x, o.∇, ν) / inner(p.M, xOld, ∇Old, ∇Old)
    return max(0, β)
end

@doc raw"""
    SteepestDirectionUpdateRule <: DirectionUpdateRule

The simplest rule to update is to have no influence of the last direction and
hence return an update $\beta = 0$ for all [`ConjugateGradientDescentOptions`](@ref)` o`

See also [`conjugate_gradient_descent`](@ref)
"""
struct SteepestDirectionUpdateRule <: DirectionUpdateRule end
function (u::SteepestDirectionUpdateRule)(
    ::GradientProblem, ::ConjugateGradientDescentOptions, i
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
    io::IO
    prefix::String
    function DebugGradient(long::Bool=false, io::IO=stdout)
        return new(io, long ? "Gradient: " : "∇F(x):")
    end
    DebugGradient(prefix::String, io::IO=stdout) = new(io, prefix)
end
function (d::DebugGradient)(::GradientProblem, o::GradientDescentOptions, i::Int)
    print(d.io, (i >= 0) ? d.prefix * "" * string(o.∇) : "")
    return nothing
end

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
    io::IO
    prefix::String
    function DebugGradientNorm(long::Bool=false, io::IO=stdout)
        return new(io, long ? "Norm of the Gradient: " : "|∇F(x)|:")
    end
    DebugGradientNorm(prefix::String, io::IO=stdout) = new(io, prefix)
end
function (d::DebugGradientNorm)(p::GradientProblem, o::Options, i::Int)
    print(d.io, (i >= 0) ? d.prefix * "$(norm(p.M,o.x,o.∇))" : "")
    return nothing
end

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
    io::IO
    prefix::String
    function DebugStepsize(long::Bool=false, io::IO=stdout)
        return new(io, long ? "step size:" : "s:")
    end
    DebugStepsize(prefix::String, io::IO=stdout) = new(io, prefix)
end
function (d::DebugStepsize)(p::GradientProblem, o::GradientDescentOptions, i::Int)
    print(d.io, (i > 0) ? d.prefix * "$(get_last_stepsize(p,o,i))" : "")
    return nothing
end

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
    recorded_values::Array{T,1}
    RecordGradient{T}() where {T} = new(Array{T,1}())
end
RecordGradient(ξ::T) where {T} = RecordGradient{T}()
function (r::RecordGradient{T})(
    ::GradientProblem, o::GradientDescentOptions, i::Int
) where {T}
    return record_or_reset!(r, o.∇, i)
end

@doc raw"""
    RecordGradientNorm <: RecordAction

record the norm of the current gradient
"""
mutable struct RecordGradientNorm <: RecordAction
    recorded_values::Array{Float64,1}
    RecordGradientNorm() = new(Array{Float64,1}())
end
function (r::RecordGradientNorm)(
    p::P, o::O, i::Int
) where {P<:GradientProblem,O<:GradientDescentOptions}
    return record_or_reset!(r, norm(p.M, o.x, o.∇), i)
end

@doc raw"""
    RecordStepsize <: RecordAction

record the step size
"""
mutable struct RecordStepsize <: RecordAction
    recorded_values::Array{Float64,1}
    RecordStepsize() = new(Array{Float64,1}())
end
function (r::RecordStepsize)(
    p::P, o::O, i::Int
) where {P<:GradientProblem,O<:GradientDescentOptions}
    return record_or_reset!(r, get_last_stepsize(p, o, i), i)
end

@doc raw"""
    AbstractQuasiNewtonDirectionUpdate

An abstract represresenation of an Quasi Newton Update rule to determine the next direction
given current [`QuasiNewtonOptions`](@ref).

All subtypes should be functors, i.e. one should be able to call them as `H(M,x,d)` to compute a new direction update.
"""
abstract type AbstractQuasiNewtonDirectionUpdate end

"""
    AbstractQuasiNewtonUpdateRule

Specify a type for the different [`AbstractQuasiNewtonDirectionUpdate`](@ref)s.
"""
abstract type AbstractQuasiNewtonUpdateRule end

@doc raw"""
    BFGS <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian BFGS update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{\mathcal{H}}_k^\mathrm{BFGS}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{H}^\mathrm{BFGS}_{k+1} = \widetilde{\mathcal{H}}^\mathrm{BFGS}_k  + \frac{y_k y^{\mathrm{T}}_k }{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{\mathcal{H}}^\mathrm{BFGS}_k s_k s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{BFGS}_k }{s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{BFGS}_k s_k}
```

where
```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```
"""
struct BFGS <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    InverseBFGS <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemanian BFGS update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{\mathcal{B}}_k^\mathrm{BFGS}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{B}^\mathrm{BFGS}_{k+1}  = \Bigl(
  \id_{T_{x_{k+1}} \mathcal{M}} - \frac{s_k y^{\mathrm{T}}_k }{s^{\mathrm{T}}_k y_k}
\Bigr)
\widetilde{\mathcal{B}}^\mathrm{BFGS}_k
\Bigl(
  \id_{T_{x_{k+1}} \mathcal{M}} - \frac{y_k s^{\mathrm{T}}_k }{s^{\mathrm{T}}_k y_k}
\Bigr) + \frac{s_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
```

where
```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```
"""
struct InverseBFGS <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    DFP <: AbstractQuasiNewtonUpdateRule

indicates in an [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian DFP update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{\mathcal{H}}_k^\mathrm{DFP}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{H}^\mathrm{DFP}_{k+1} = \Bigl(
  \id_{T_{x_{k+1}} \mathcal{M}} - \frac{y_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
\Bigr)
\widetilde{\mathcal{H}}^\mathrm{DFP}_k
\Bigl(
  \id_{T_{x_{k+1}} \mathcal{M}} - \frac{s_k y^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
\Bigr) + \frac{y_k y^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
```

where
```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```
"""
struct DFP <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    InverseDFP <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemanian DFP update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{\mathcal{B}}_k^\mathrm{DFP}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{B}^\mathrm{DFP}_{k+1} = \widetilde{\mathcal{B}}^\mathrm{DFP}_k
+ \frac{s_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
- \frac{\widetilde{\mathcal{B}}^\mathrm{DFP}_k y_k y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{DFP}_k}{y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{DFP}_k y_k}
```

where
```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```
"""
struct InverseDFP <: AbstractQuasiNewtonUpdateRule end
@doc raw"""
    SR1 <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian SR1 update is used in the Riemannian quasi-Newton method.

We denote by ``\widetilde{H}_k^\mathrm{SR1}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
H^\mathrm{SR1}_{k+1} = \widetilde{H}^\mathrm{SR1}_k
+ \frac{
  (y_k - \widetilde{H}^\mathrm{SR1}_k s_k) (y_k - \widetilde{H}^\mathrm{SR1}_k s_k)^{\mathrm{T}}
}{
(y_k - \widetilde{H}^\mathrm{SR1}_k s_k)^{\mathrm{T}} s_k
}
```

where
```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```

This method can be stabilized by only performing the update if denominator is larger than
``r\lVert s_k\rVert_{x_{k+1}}\lVert y_k - \widetilde{H}^\mathrm{SR1}_k s_k \rVert_{x_{k+1}}``
for some ``r>0``. For more details, see Section 6.2 in [^NocedalWright2006]

[^NocedalWright2006]:
    > Nocedal, J., Wright, S.: Numerical Optimization, Second Edition, Springer, 2006.
    > doi: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)

# Constructor
    SR1(r::Float64=-1.0)

Generate the `SR1` update, which by default does not include the check (since the default sets ``t<0```)

"""
struct SR1 <: AbstractQuasiNewtonUpdateRule
    r::Float64
    SR1(r::Float64=-1.0) = new(r)
end

@doc raw"""
    InverseSR1 <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the inverse Riemanian SR1 update is used in the Riemannian quasi-Newton method.


We denote by ``\widetilde{\mathcal{B}}_k^\mathrm{SR1}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{B}^\mathrm{SR1}_{k+1} = \widetilde{\mathcal{B}}^\mathrm{SR1}_k
+ \frac{
  (s_k - \widetilde{\mathcal{B}}^\mathrm{SR1}_k y_k) (s_k - \widetilde{\mathcal{B}}^\mathrm{SR1}_k y_k)^{\mathrm{T}}
}{
  (s_k - \widetilde{\mathcal{B}}^\mathrm{SR1}_k y_k)^{\mathrm{T}} y_k
}
```

where
```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```

This method can be stabilized by only performing the update if denominator is larger than
``r\lVert y_k\rVert_{x_{k+1}}\lVert s_k - \widetilde{H}^\mathrm{SR1}_k y_k \rVert_{x_{k+1}}``
for some ``r>0``. For more details, see Section 6.2 in [^NocedalWright2006].

# Constructor
    InverseSR1(r::Float64=-1.0)

Generate the `InverseSR1` update, which by default does not include the check,
since the default sets ``t<0```.
"""
struct InverseSR1 <: AbstractQuasiNewtonUpdateRule
    r::Float64
    InverseSR1(r::Float64=-1.0) = new(r)
end

@doc raw"""
    Broyden <: AbstractQuasiNewtonUpdateRule

indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian Broyden update is used in the Riemannian quasi-Newton method, which is as a convex combination of [`BFGS`](@ref) and [`DFP`](@ref).

We denote by ``\widetilde{\mathcal{H}}_k^\mathrm{Br}`` the operator concatenated with a vector transport and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{H}^\mathrm{Br}_{k+1} = \widetilde{\mathcal{H}}^\mathrm{Br}_k
  - \frac{\widetilde{\mathcal{H}}^\mathrm{Br}_k s_k s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{Br}_k}{s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{Br}_k s_k} + \frac{y_k y^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
  + φ_k s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{Br}_k s_k
  \Bigl(
        \frac{y_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{\mathcal{H}}^\mathrm{Br}_k s_k}{s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{Br}_k s_k}
  \Bigr)
  \Bigl(
        \frac{y_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{\mathcal{H}}^\mathrm{Br}_k s_k}{s^{\mathrm{T}}_k \widetilde{\mathcal{H}}^\mathrm{Br}_k s_k}
  \Bigr)^{\mathrm{T}}
```

where

```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```

and ``φ_k`` is the Broyden factor which is `:constant` by default but can also be set to `:Davidon`.

# Constructor
    Broyden(φ, update_rule::Symbol = :constant)

"""
mutable struct Broyden <: AbstractQuasiNewtonUpdateRule
    φ::Float64
    update_rule::Symbol
end
Broyden(φ::Float64) = Broyden(φ, :constant)

@doc raw"""
    InverseBroyden <: AbstractQuasiNewtonUpdateRule

Indicates in [`AbstractQuasiNewtonDirectionUpdate`](@ref) that the Riemanian Broyden update
is used in the Riemannian quasi-Newton method, which is as a convex combination
of [`InverseBFGS`](@ref) and [`InverseDFP`](@ref).

We denote by ``\widetilde{\mathcal{H}}_k^\mathrm{Br}`` the operator concatenated with a vector transport
and its inverse before and after to act on ``x_{k+1} = R_{x_k}(α_k η_k)``.
Then the update formula reads

```math
\mathcal{B}^\mathrm{Br}_{k+1} = \widetilde{\mathcal{B}}^\mathrm{Br}_k
 - \frac{\widetilde{\mathcal{B}}^\mathrm{Br}_k y_k y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{Br}_k}{y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{Br}_k y_k}
   + \frac{s_k s^{\mathrm{T}}_k}{s^{\mathrm{T}}_k y_k}
 + φ_k y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{Br}_k y_k
 \Bigl(
     \frac{s_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{\mathcal{B}}^\mathrm{Br}_k y_k}{y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{Br}_k y_k}
    \Bigr) \Bigl(
        \frac{s_k}{s^{\mathrm{T}}_k y_k} - \frac{\widetilde{\mathcal{B}}^\mathrm{Br}_k y_k}{y^{\mathrm{T}}_k \widetilde{\mathcal{B}}^\mathrm{Br}_k y_k}
 \Bigr)^{\mathrm{T}}
```

where

```math
s_k = T^{S}_{x_k, α_k η_k}(α_k η_k) \quad\text{and}\quad
y_k = ∇f(x_{k+1}) - T^{S}_{x_k, α_k η_k}(∇ f(x_k)) \in T_{x_{k+1}} \mathcal{M}.
```

and ``φ_k`` is the Broyden factor which is `:constant` by default but can also be set to `:Davidon`.

# Constructor
    InverseBroyden(φ, update_rule::Symbol = :constant)

"""
mutable struct InverseBroyden <: AbstractQuasiNewtonUpdateRule
    φ::Float64
    update_rule::Symbol
end
InverseBroyden(φ::Float64) = InverseBroyden(φ, :constant)

@doc raw"""
    QuasiNewtonOptions <: Options

These Quasi Newton [`Options`](@ref) represent any quasi-Newton based method and can be
used with any update rule for the direction.

# Fields
* `x` – the current iterate, a point on a manifold
* `∇` – the current gradient
* `sk` – the current step
* `yk` the current gradient difference
* `direction_update` - a [`AbstractQuasiNewtonDirectionUpdate`] rule.
* `retraction_method` – a function to perform a step on the manifold
* `stop` – a [`StoppingCriterion`](@ref)

# See also
[`GradientProblem`](@ref)
"""
mutable struct QuasiNewtonOptions{
    P,
    T,
    U<:AbstractQuasiNewtonDirectionUpdate,
    SC<:StoppingCriterion,
    S<:Stepsize,
    RTR<:AbstractRetractionMethod,
    VT<:AbstractVectorTransportMethod,
} <: Options
    x::P
    ∇::T
    sk::T
    yk::T
    direction_update::U
    retraction_method::RTR
    stepsize::S
    stop::SC
    vector_transport_method::VT
end
function QuasiNewtonOptions(
    x::P,
    ∇::T,
    direction_update::U,
    stop::SC,
    stepsize::S;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {P,T,U<:AbstractQuasiNewtonDirectionUpdate,SC<:StoppingCriterion,S<:Stepsize}
    return QuasiNewtonOptions{
        P,T,U,SC,S,typeof(retraction_method),typeof(vector_transport_method)
    }(
        x,
        ∇,
        deepcopy(∇),
        deepcopy(∇),
        direction_update,
        retraction_method,
        stepsize,
        stop,
        vector_transport_method,
    )
end

@doc raw"""
    QuasiNewtonDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

These [`AbstractQuasiNewtonDirectionUpdate`](@ref)s represent any quasi-Newton update rule, where the operator is stored as a matrix. A distinction is made between the update of the approximation of the Hessian, ``H_k \mapsto H_{k+1}``, and the update of the approximation of the Hessian inverse, ``B_k \mapsto B_{k+1}``. For the first case, the coordinates of the search direction ``\eta_k`` with respect to a basis ``\{b_i\}^{n}_{i=1}`` are determined by solving a linear system of equations, i.e.

```math
\text{Solve} \quad \hat{eta_k} = - H_k \widehat{\operatorname{grad} f(x_k)}
```

where ``H_k`` is the matrix representing the operator with respect to the basis ``\{b_i\}^{n}_{i=1}`` and ``\widehat{\operatorname{grad} f(x_k)}`` represents the coordinates of the gradient of the objective function ``f``` in ``x_k`` with respect to the basis ``\{b_i\}^{n}_{i=1}``.
If a method is chosen where Hessian inverse is approximated, the coordinates of the search direction ``\eta_k`` with respect to a basis ``\{b_i\}^{n}_{i=1}`` are obtained simply by matrix-vector multiplication, i.e.

```math
\hat{eta_k} = - B_k \widehat{\operatorname{grad} f(x_k)}
```

where ``B_k`` is the matrix representing the operator with respect to the basis ``\{b_i\}^{n}_{i=1}`` and ``\widehat{\operatorname{grad} f(x_k)}`` as above. In the end, the search direction ``\eta_k`` is generated from the coordinates ``\hat{eta_k}`` and the vectors of the basis ``\{b_i\}^{n}_{i=1}`` in both variants.
The [``AbstractQuasiNewtonUpdateRule``] (@ref) indicates which quasi-Newton update rule is used. In all of them, the Euclidean update formula is used to generate the matrix ``H_{k+1}`` and ``B_{k+1}``, and the basis ``\{b_i\}^{n}_{i=1}`` is transported into the upcoming tangent space ``T_{x_{k+1}} \mathcal{M}`, preferably with an isometric vector transport, or generated there. 

# Fields
* `basis` – the basis.
* `matrix` – the matrix which represents the approximating operator.
* `scale` – indicates whether the initial matrix (= identity matrix) should be scaled before the first update.
* `update` – a [`AbstractQuasiNewtonUpdateRule`](@ref).
* `vector_transport_method` – a [`AbstractVectorTransportMethod`](@ref).

# See also
[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)
[`QuasiNewtonCautiousDirectionUpdate`](@ref)
[`AbstractQuasiNewtonDirectionUpdate`](@ref)
"""
mutable struct QuasiNewtonMatrixDirectionUpdate{
    NT<:AbstractQuasiNewtonUpdateRule,
    B<:AbstractBasis,
    VT<:AbstractVectorTransportMethod,
    M<:AbstractMatrix,
} <: AbstractQuasiNewtonDirectionUpdate
    basis::B
    matrix::M
    scale::Bool
    update::NT
    vector_transport_method::VT
end
function QuasiNewtonMatrixDirectionUpdate(
    update::AbstractQuasiNewtonUpdateRule,
    basis::B,
    m::M,
    ;
    scale::Bool=true,
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {M<:AbstractMatrix,B<:AbstractBasis}
    return QuasiNewtonMatrixDirectionUpdate{
        typeof(update),B,typeof(vector_transport_method),M
    }(
        basis, m, scale, update, vector_transport_method
    )
end
function (d::QuasiNewtonMatrixDirectionUpdate{T})(
    p, o
) where {T<:Union{InverseBFGS,InverseDFP,InverseSR1,InverseBroyden}}
    return get_vector(
        p.M, o.x, -d.matrix * get_coordinates(p.M, o.x, o.∇, d.basis), d.basis
    )
end
function (d::QuasiNewtonMatrixDirectionUpdate{T})(
    p, o
) where {T<:Union{BFGS,DFP,SR1,Broyden}}
    return get_vector(
        p.M, o.x, -d.matrix \ get_coordinates(p.M, o.x, o.∇, d.basis), d.basis
    )
end

@doc raw"""
    QuasiNewtonLimitedMemoryDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

This [`AbstractQuasiNewtonDirectionUpdate`](@ref) represents the limited-memory Riemanian BFGS update, where the approximating  oprator is represented by ``m`` stored pairs of tangent vectors ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}`` in the ``k``-th iteration`. 
For the calculation of the search direction ``\eta_k``, the generalisation of the two-loop recursion is used (see [^HuangGallivanAbsil2015]), since it only requires inner products and linear combinations of tangent vectors in ``T_{x_k} \mathcal{M}``. For that the stored pairs of tangent vectors ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}``, the gradient ``\operatorname{grad} f(x_k)`` of the objective function ``f`` in ``x_k`` and the positive definite self-adjoint operator 

```math
\mathcal{B}^{(0)}_k[\cdot] = \frac{\widetilde{s}^{\flat}_{k-1} \widetilde{y}_{k-1}}{\widetilde{y}^{\flat}_{k-1} \widetilde{y}_{k-1}} \id_{T_{x_k} \mathcal{M}}[\cdot] = \frac{g_{x_k}(s_{k-1}, y_{k-1})}{g_{x_k}(y_{k-1}, y_{k-1})} \id_{T_{x_k} \mathcal{M}}[\cdot]
```

are used. The two-loop recursion can be understood as that the [`InverseBFGS`](@ref) update is executed ``m`` times in a row on ``\mathcal{B}^{(0)}_k[\cdot]`` using the tangent vectors ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}``, and in the same time the resulting operator ``\mathcal{B}^{LRBFGS}_k [\cdot]`` is directly applied on ``\operatorname{grad}f(x_k)``.
When updating there are two cases: if there is still free memory, i.e. ``k < m``, the previously stored vector pairs ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}`` have to be transported into the upcoming tangent space ``T_{x_{k+1}} \mathcal{M}``; if there is no free memory, the oldest pair ``\{ \widetilde{s}_{k−m}, \widetilde{y}_{k−m}\}`` has to be discarded and then all the remaining vector pairs ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m+1}^{k-1}`` are transported into the tangent space ``T_{x_{k+1}} \mathcal{M}``. After that we calculate and store ``s_k = \widetilde{s}_k = T^{S}_{x_k, α_k η_k}(α_k η_k)`` and ``y_k = \widetilde{y}_k``. This process ensures that new information about the objective function is always included and the old, probably no longer relevant, information is discarded.

# Fields
* `method` – the maximum number of vector pairs stored.
* `memory_s` – the set of the stored (and transported) search directions times step size ``\{ \widetilde{s}_i\}_{i=k-m}^{k-1}``. 
* `memory_y` – set of the stored gradient differences ``\{ \widetilde{y}_i\}_{i=k-m}^{k-1}``.
* `ξ` – a variable used in the two-loop recursion.
* `ρ` – a variable used in the two-loop recursion.
* `scale` – 
* `vector_transport_method` – a [`AbstractVectorTransportMethod`](@ref).

# See also
[`InverseBFGS`](@ref)
[`CautiousUpdate`](@ref)
[`AbstractQuasiNewtonDirectionUpdate`](@ref)

[^HuangGallivanAbsil2015]:
    > Huang, Wen and Gallivan, K. A. and Absil, P.-A., A Broyden Class of Quasi-Newton Methods for Riemannian Optimization,
    > SIAM J. Optim., 25 (2015), pp. 1660-1685.
    > doi: [10.1137/140955483](https://doi.org/10.1137/140955483)
"""
mutable struct QuasiNewtonLimitedMemoryDirectionUpdate{
    NT<:AbstractQuasiNewtonUpdateRule,T,VT<:AbstractVectorTransportMethod
} <: AbstractQuasiNewtonDirectionUpdate
    method::NT
    memory_s::CircularBuffer{T}
    memory_y::CircularBuffer{T}
    ξ::Vector{Float64}
    ρ::Vector{Float64}
    scale::Float64
    vector_transport_method::VT
end
function QuasiNewtonLimitedMemoryDirectionUpdate(
    method::NT,
    ::T,
    memory_size::Int;
    scale::Bool=true,
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {NT<:AbstractQuasiNewtonUpdateRule,T,VT<:AbstractVectorTransportMethod}
    return QuasiNewtonLimitedMemoryDirectionUpdate{NT,T,typeof(vector_transport_method)}(
        method,
        CircularBuffer{T}(memory_size),
        CircularBuffer{T}(memory_size),
        zeros(memory_size),
        zeros(memory_size),
        scale,
        vector_transport_method,
    )
end
function (d::QuasiNewtonLimitedMemoryDirectionUpdate{InverseBFGS})(p, o)
    r = deepcopy(o.∇)
    m = length(d.memory_s)
    m == 0 && return -r
    for i in m:-1:1
        d.ρ[i] = 1 / inner(p.M, o.x, d.memory_s[i], d.memory_y[i]) # 1 sk 2 yk
        d.ξ[i] = inner(p.M, o.x, d.memory_s[i], r) * d.ρ[i]
        r .= r .- d.ξ[i] .* d.memory_y[i]
    end
    r .= 1 / (d.ρ[m] * norm(p.M, o.x, last(d.memory_y))^2) .* r
    for i in 1:m
        r .= r .+ (d.ξ[i] - d.ρ[i] * inner(p.M, o.x, d.memory_y[i], r)) .* d.memory_s[i]
    end
    return -project(p.M, o.x, r)
end

@doc raw"""
    QuasiNewtonCautiousDirectionUpdate <: AbstractQuasiNewtonDirectionUpdate

These [`AbstractQuasiNewtonDirectionUpdate`](@ref)s represent any quasi-Newton update rule, which are based on the idea of a so-called cautious update. The search direction is calculated as given in [`QuasiNewtonDirectionUpdate`](@ref) or [`LimitedMemoryQuasiNewctionDirectionUpdate`]. But the update given in [`QuasiNewtonDirectionUpdate`](@ref) or [`LimitedMemoryQuasiNewctionDirectionUpdate`] is only executed if 

```math
\frac{g_{x_{k+1}}(y_k,s_k)}{\lVert s_k \rVert^{2}_{x_{k+1}}} \geq \theta(\lVert \operatorname{grad} f(x_k) \rVert_{x_k}),
```

is satisfied, where ``\theta`` is a monotone increasing function satisfying ``\theta(0) = 0`` and ``\theta`` is strictly increasing at ``0``. If this is not the case, the corresponding update will be skipped, which means that for [`QuasiNewtonDirectionUpdate`](@ref) the matrix ``H_k`` or ``B_k`` is not updated, but the basis ``\{b_i\}^{n}_{i=1}`` is nevertheless transported into the upcoming tangent space ``T_{x_{k+1}} \mathcal{M}``, and for [`LimitedMemoryQuasiNewctionDirectionUpdate`] neither the oldest vector pair ``\{ \widetilde{s}_{k−m}, \widetilde{y}_{k−m}\}`` is discarded nor the newest vector pair ``\{ \widetilde{s}_{k}, \widetilde{y}_{k}\}`` is added into storage, but all stored vector pairs ``\{ \widetilde{s}_i, \widetilde{y}_i\}_{i=k-m}^{k-1}`` are transported into the tangent space ``T_{x_{k+1}} \mathcal{M}``. 
If [`InverseBFGS`](@ref) or [`InverseBFGS`](@ref) is chosen as update, then the resulting method follows the method of [^HuangAbsilGallivan2018], taking into account that the corresponding step size is chosen. 


# Fields
* `update` – an [`AbstractQuasiNewtonDirectionUpdate`](@ref)
* `θ` – a monotone increasing function satisfying ``θ(0) = 0`` and ``θ`` is strictly increasing at ``0``.

# See also
[`QuasiNewtonMatrixDirectionUpdate`](@ref)
[`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref)

[^HuangAbsilGallivan2018]:
    > Huang, Wen and Absil, P.-A and Gallivan, Kyle, AA Riemannian BFGS Method Without Differentiated Retraction for Nonconvex Optimization Problems,
    > SIAM J. Optim., 28 (2018), pp. 470-495.
    > doi: [10.1137/17M1127582](https://doi.org/10.1137/17M1127582)
"""
mutable struct QuasiNewtonCautiousDirectionUpdate{U} <:
               AbstractQuasiNewtonDirectionUpdate where {
    U<:Union{
        QuasiNewtonMatrixDirectionUpdate,QuasiNewtonLimitedMemoryDirectionUpdate{T}
    },
} where {T<:AbstractQuasiNewtonUpdateRule}
    update::U
    θ::Function
end
function QuasiNewtonCautiousDirectionUpdate(
    update::U; θ::Function=x -> x
) where {
    U<:Union{
        QuasiNewtonMatrixDirectionUpdate,QuasiNewtonLimitedMemoryDirectionUpdate{T}
    },
} where {T<:AbstractQuasiNewtonUpdateRule}
    return QuasiNewtonCautiousDirectionUpdate{U}(update, θ)
end
(d::QuasiNewtonCautiousDirectionUpdate)(p, o) = d.update(p, o)

# access the inner vector transport method
function get_update_vector_transport(u::AbstractQuasiNewtonDirectionUpdate)
    return u.vector_transport_method
end
function get_update_vector_transport(u::QuasiNewtonCautiousDirectionUpdate)
    return get_update_vector_transport(u.update)
end
