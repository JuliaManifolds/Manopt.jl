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
    Gradient <: DirectionUpdateRule

The default gradient direction update is the identity, i.e. it just evaluates the gradient.
"""
struct Gradient <: DirectionUpdateRule end

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
* `direction` - ([`Gradient`](@ref)) a processor to compute the gradient
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
        direction::DirectionUpdateRule=Gradient(),
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
    direction::DirectionUpdateRule=Gradient(),
) where {P}
    return GradientDescentOptions{P}(
        x, stopping_criterion, stepsize, retraction_method, direction
    )
end
#
# Processors
#
function (s::Gradient)(p::GradientProblem, o::GradientDescentOptions, i)
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
    s::DirectionUpdateRule=Gradient();
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
    s::DirectionUpdateRule=Gradient();
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
        s::DirectionUpdateRule=Gradient();
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
    s::DirectionUpdateRule=Gradient();
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
    s::DirectionUpdateRule=Gradient();
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
    p::GradientProblem, o::ConjugateGradientDescentOptions, i
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
given current [`AbstractQuasiNewtonOptions`](@ref).

All subtypes should be functors, i.e. one should be able to call them as `H(M,x,d)` to compute a new direction update.
"""
abstract type AbstractQuasiNewtonDirectionUpdate end

abstract type AbstractQuasiNewtonType end
struct BFGS <: AbstractQuasiNewtonType end
struct InverseBFGS <: AbstractQuasiNewtonType end
struct DFP <: AbstractQuasiNewtonType end
struct InverseDFP <: AbstractQuasiNewtonType end

@doc raw"""
    QuasiNewtonOptions <: Options

Theese Quasi Newton [`Options`](@ref) represent any quasi newton based method and can be
used with any update rule for the direction.

# Fields
* `x` – the current iterate, a point on a manifold
* `∇` – the current gradient
* `sk` – the current step
* `yk` the current gradient difference
* `dirction_update` - a [`AbstractQuasiNewtonDirectionUpdate`] rule.
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

mutable struct QuasiNewtonDirectionUpdate{
    NT<:AbstractQuasiNewtonType,
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
function QuasiNewtonDirectionUpdate(
    update::AbstractQuasiNewtonType,
    basis::B,
    m::M,
    ;
    scale::Bool=true,
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {M<:AbstractMatrix,B<:AbstractBasis}
    return QuasiNewtonDirectionUpdate{typeof(update),B,typeof(vector_transport_method),M}(
        basis, m, scale, update, vector_transport_method
    )
end
function (d::QuasiNewtonDirectionUpdate{T})(p, o) where {T<:Union{InverseBFGS,InverseDFP}}
    return get_vector(
        p.M, o.x, -d.matrix * get_coordinates(p.M, o.x, o.∇, d.basis), d.basis
    )
end
function (d::QuasiNewtonDirectionUpdate{T})(p, o) where {T<:Union{BFGS,DFP}}
    return get_vector(p.M, o.x, -d.matrix \ get_coordinates(p.M, o.x, o.∇, d.basis))
end

mutable struct Broyden{
    U1<:AbstractQuasiNewtonDirectionUpdate,U2<:AbstractQuasiNewtonDirectionUpdate
} <: AbstractQuasiNewtonDirectionUpdate
    update1::U1
    update2::U2
    factor::Float64
end
function Broyden(
    u1::U1, u2::U2, factor=1.0
) where {U1<:AbstractQuasiNewtonDirectionUpdate,U2<:AbstractQuasiNewtonDirectionUpdate}
    return Broyden{U1,U2}(u1, u2, factor)
end
function (d::Broyden)(p, o)
    return (1 - d.factor) * d.update1(p, o) + d.factor * d.update2(p, o)
end

mutable struct LimitedMemoryQuasiNewctionDirectionUpdate{
    NT<:AbstractQuasiNewtonType,T,VT<:AbstractVectorTransportMethod
} <: AbstractQuasiNewtonDirectionUpdate
    method::NT
    memory_s::CircularBuffer{T}
    memory_y::CircularBuffer{T}
    scale::Float64
    vector_transport_method::VT
end
function LimitedMemoryQuasiNewctionDirectionUpdate(
    method::NT,
    ::T,
    memory_size::Int;
    scale::Bool=true,
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
) where {NT<:AbstractQuasiNewtonType,T,VT<:AbstractVectorTransportMethod}
    return LimitedMemoryQuasiNewctionDirectionUpdate{NT,T,typeof(vector_transport_method)}(
        method, CircularBuffer{T}(memory_size), CircularBuffer{T}(memory_size), scale, vector_transport_method
    )
end
function (d::LimitedMemoryQuasiNewctionDirectionUpdate{InverseBFGS})(p, o)
    r = deepcopy(o.∇)
    m = length(d.memory_s)
    m == 0 && return -r
    ξ = zeros(m)
    ρ = zeros(m)
    for i=m:-1:1
        ρ[i] = 1 / inner(p.M, o.x, d.memory_s[i], d.memory_y[i]) # 1 sk 2 yk
        ξ[i] = inner(p.M, o.x, d.memory_s[i], r) * ρ[i]
        r .= r .- ξ[i] .* d.memory_y[i]
        i -= 1
    end
    r .= 1 / (last(ρ) * norm(p.M, o.x, last(d.memory_y))^2) .* r
    for i=1:m
        ω = ρ[i] * inner(p.M, o.x, d.memory_y[i], r)
        r .= r .+ (ξ[i] - ω) .* d.memory_s[i]
    end
    return -r
end

struct CautiousUpdate{U<:AbstractQuasiNewtonDirectionUpdate} <:
       AbstractQuasiNewtonDirectionUpdate
    update::U
    θ::Function
end
function CautiousUpdate(
    update::U; θ::Function=x -> x
) where {U<:AbstractQuasiNewtonDirectionUpdate}
    return CautiousUpdate{U}(update, θ)
end
function (d::CautiousUpdate)(p, o)
    return d.update(p,o)
end
