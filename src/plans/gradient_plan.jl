"""
    AbstractGradientProblem{T} <: Problem{T}

An abstract type for all functions that provide a (full) gradient, where
`T` is a [`AbstractEvaluationType`](@ref) for the gradient function.
"""
abstract type AbstractGradientProblem{T} <: Problem{T} end

@doc raw"""
    GradientProblem{T} <: AbstractGradientProblem{T}

specify a problem for gradient based algorithms.

# Fields
* `M`        – a manifold ``\mathcal M``
* `cost`     – a function ``F: \mathcal M → ℝ`` to minimize
* `gradient!!` – the gradient ``\operatorname{grad}F:\mathcal M → \mathcal T\mathcal M`` of the cost function ``F``.

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient has to be provided

* as a function `x -> X` that allocates memory for `X` itself for an [`AllocatingEvaluation`](@ref)
* as a function `(X,x) -> X` that work in place of `X` for an [`MutatingEvaluation`](@ref)

# Constructors
    GradientProblem(M, cost, gradient; evaluation=AllocatingEvaluation())

# See also
[`gradient_descent`](@ref), [`GradientDescentOptions`](@ref)
"""
struct GradientProblem{T,mT<:Manifold,C,G} <: AbstractGradientProblem{T}
    M::mT
    cost::C
    gradient!!::G
end
function GradientProblem(
    M::mT, cost::C, gradient::G; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {mT<:Manifold,C,G}
    return GradientProblem{typeof(evaluation),mT,C,G}(M, cost, gradient)
end

"""
    get_gradient(p::AbstractGradientProblem{T},x)
    get_gradient!(p::AbstractGradientProblem{T}, X, x)

evaluate the gradient of a [`AbstractGradientProblem{T}`](@ref)`p` at the point `x`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`MutatingEvaluation`](@ref)
memory for the result is allocated.
"""
get_gradient(p::AbstractGradientProblem, x)

function get_gradient(p::AbstractGradientProblem{AllocatingEvaluation}, x)
    return p.gradient!!(p.M, x)
end
function get_gradient(p::AbstractGradientProblem{MutatingEvaluation}, x)
    X = zero_tangent_vector(p.M, x)
    return p.gradient!!(p.M, X, x)
end

function get_gradient!(p::AbstractGradientProblem{AllocatingEvaluation}, X, x)
    return copyto!(X, p.gradient!!(p.M, x))
end

function get_gradient!(p::AbstractGradientProblem{MutatingEvaluation}, X, x)
    return p.gradient!!(p.M, X, x)
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
    AbstractGradientOptions <: Options

A generic [`Options`](@ref) type for gradient based options data.
"""
abstract type AbstractGradientOptions <: Options end

@doc raw"""
    GradientDescentOptions{P,T} <: AbstractGradientOptions

Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x0` – an a point (of type `P`) on a manifold as starting point
* `gradient` – the current gradient ``\operatorname{grad}f(x)``
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`)a [`Stepsize`](@ref)
* `direction` - ([`IdentityUpdateRule`](@ref)) a processor to compute the gradient
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use, defaults to
  the exponential map

# Constructor

    GradientDescentOptions(x, stop, s [, retr=ExponentialRetraction()])
    GradientDescentOptions(M, x, stop, s [, retr=ExponentialRetraction()])
    GradientDescentOptions(x, X, stop, s [, retr=ExponentialRetraction()])

construct a Gradient Descent Option with the fields and defaults as above,
where the first can be used if points (`x`)  and tangent vectors (`gradient`) have the same type,
for exxample when they are matrices.
The second uses the `Manifold M` to set `gradient=zero_tangent_vector(M,x)`.

# See also
[`gradient_descent`](@ref), [`GradientProblem`](@ref)
"""
mutable struct GradientDescentOptions{
    P,T,TStop<:StoppingCriterion,TStepsize<:Stepsize,TRTM<:AbstractRetractionMethod
} <: AbstractGradientOptions
    x::P
    direction::DirectionUpdateRule
    stop::TStop
    stepsize::TStepsize
    gradient::T
    retraction_method::TRTM
    function GradientDescentOptions{P,T}(
        initialX::P,
        initial_gradient::T,
        s::StoppingCriterion=StopAfterIteration(100),
        stepsize::Stepsize=ConstantStepsize(1.0),
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        direction::DirectionUpdateRule=IdentityUpdateRule(),
    ) where {P,T}
        o = new{P,T,typeof(s),typeof(stepsize),typeof(retraction_method)}()
        o.x = initialX
        o.gradient = initial_gradient
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
    return GradientDescentOptions{P,P}(
        x, deepcopy(x), stopping_criterion, stepsize, retraction_method, direction
    )
end
function GradientDescentOptions(
    x::P,
    X::T;
    stopping_criterion::StoppingCriterion=StopAfterIteration(100),
    stepsize::Stepsize=ConstantStepsize(1.0),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    direction::DirectionUpdateRule=IdentityUpdateRule(),
) where {P,T}
    return GradientDescentOptions{P,T}(
        x, X, stopping_criterion, stepsize, retraction_method, direction
    )
end
function GradientDescentOptions(
    M::Manifold,
    x::P;
    stopping_criterion::StoppingCriterion=StopAfterIteration(100),
    stepsize::Stepsize=ConstantStepsize(1.0),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    direction::DirectionUpdateRule=IdentityUpdateRule(),
) where {P}
    X = zero_tangent_vector(M, x)
    return GradientDescentOptions(
        x,
        X;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        retraction_method=retraction_method,
        direction=direction,
    )
end

function (s::IdentityUpdateRule)(p::GradientProblem, o::GradientDescentOptions, i)
    return get_stepsize(p, o, i), get_gradient!(p, o.gradient, o.x)
end

"""
    MomentumGradient <: DirectionUpdateRule

Append a momentum to a gradient processor, where the last direction and last iterate are
stored and the new is composed as ``η_i = m*η_{i-1}' - s d_i``,
where ``sd_i`` is the current (inner) direction and ``η_{i-1}'`` is the vector transported
last direction multiplied by momentum ``m``.

# Fields
* `gradient` – (`zero_tangent_vector(M,x0)`) the last gradient/direction update added as momentum
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
        gradient=zero_tangent_vector(p.M, o.x), momentum=0.2
       vector_transport_method=ParallelTransport(),
    )

Add momentum to a gradient problem, where by default just a gradient evaluation is used
Equivalently you can also use a `Manifold` `M` instead of the [`GradientProblem`](@ref) `p`.

    MomentumGradient(
        p::StochasticGradientProblem
        x0
        s::DirectionUpdateRule=IdentityUpdateRule();
        gradient=zero_tangent_vector(p.M, x0), momentum=0.2
       vector_transport_method=ParallelTransport(),
    )

Add momentum to a stochastic gradient problem, where by default just a stochastic gradient evaluation is used
"""
mutable struct MomentumGradient{P,T,R<:Real,VTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    gradient::T
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
    gradient=zero_tangent_vector(p.M, x0),
    momentum=0.2,
) where {P,VTM<:AbstractVectorTransportMethod}
    return MomentumGradient{P,typeof(gradient),typeof(momentum),VTM}(
        deepcopy(x0), gradient, momentum, s, vector_transport_method
    )
end
function MomentumGradient(
    M::Manifold,
    x0::P,
    s::DirectionUpdateRule=IdentityUpdateRule();
    gradient=zero_tangent_vector(M, x0),
    last_iterate=x0,
    momentum=0.2,
    vector_transport_method::VTM=ParallelTransport(),
) where {P,VTM<:AbstractVectorTransportMethod}
    return MomentumGradient{P,typeof(gradient),typeof(momentum),VTM}(
        gradient, deepcopy(x0), momentum, s, vector_transport_method
    )
end
function (m::MomentumGradient)(p::Problem, o::AbstractGradientOptions, i)
    s, d = m.direction(p, o, i) #get inner direction and step size
    old_d =
        m.momentum *
        vector_transport_to(p.M, m.last_iterate, m.gradient, o.x, m.vector_transport_method)
    m.gradient = old_d - s .* d
    m.last_iterate = deepcopy(o.x)
    return s, -m.gradient
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
        s::DirectionUpdateRule=IdentityUpdateRule();
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
function (a::AverageGradient)(p::Problem, o::AbstractGradientOptions, i)
    pop!(a.gradients)
    s, d = a.direction(p, o, i) #get inner gradient and step
    a.gradients = vcat([deepcopy(d)], a.gradients)
    for i in 1:(length(a.gradients) - 1) #transport & shift in place
        vector_transport_to!(
            p.M,
            a.gradients[i],
            a.last_iterate,
            a.gradients[i + 1],
            o.x,
            a.vector_transport_method,
        )
    end
    a.gradients[1] = deepcopy(d)
    a.last_iterate = deepcopy(o.x)
    return s, 1 / length(a.gradients) .* sum(a.gradients)
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
* and a current iterate ``x_k``
* as well as the interims values ``γ_k`` and ``v_k`` from the previous iterate.

This compute a Nesterov type update using the following steps, see [^ZhangSra2018]

1. Copute the positive root, i.e. ``α_k∈(0,1)`` of ``α^2 = h_k\bigl((1-α_k)γ_k+α_k μ\bigr)``.
2. Set ``\bar γ_k+1 = (1-α_k)γ_k + α_kμ``
3. ``y_k = \operatorname{retr}_{x_k}\Bigl(\frac{α_kγ_k}{γ_k + α_kμ}\operatorname{retr}^{-1}_{x_k}v_k \Bigr)``
4. ``x_{k+1} = \operatorname{retr}_{y_k}(-h_k \operatorname{grad}f(y_k))``
5. ``v_{k+1} = `\operatorname{retr}_{y_k}\Bigl(\frac{(1-α_k)γ_k}{\barγ_k}\operatorname{retr}_{y_k}^{-1}(v_k) - \frac{α_k}{\bar γ_{k+1}}\operatorname{grad}f(y_k) \Bigr)``
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
function (s::Nesterov)(p::GradientProblem, o::AbstractGradientOptions, i)
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
        return new(io, long ? "Gradient: " : "gradF(x):")
    end
    DebugGradient(prefix::String, io::IO=stdout) = new(io, prefix)
end
function (d::DebugGradient)(::GradientProblem, o::GradientDescentOptions, i::Int)
    print(d.io, (i >= 0) ? d.prefix * "" * string(o.gradient) : "")
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
        return new(io, long ? "Norm of the Gradient: " : "|gradF(x)|:")
    end
    DebugGradientNorm(prefix::String, io::IO=stdout) = new(io, prefix)
end
function (d::DebugGradientNorm)(p::GradientProblem, o::Options, i::Int)
    print(d.io, (i >= 0) ? d.prefix * "$(norm(p.M,o.x,o.gradient))" : "")
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
    return record_or_reset!(r, o.gradient, i)
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
    return record_or_reset!(r, norm(p.M, o.x, o.gradient), i)
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
