"""
    ConstantStepsize <: Stepsize

A functor that always returns a fixed step size.

# Fields
* `length` – constant value for the step size.

# Constructor

    ConstantStepSize(s)

initialize the stepsie to a constant `s`
"""
mutable struct ConstantStepsize <: Stepsize
    length::Float64
    ConstantStepsize(s::Real) = new(s)
end
(s::ConstantStepsize)(p::P, o::O, i::Int, vars...) where {P<:Problem,O<:Options} = s.length
get_initial_stepsize(s::ConstantStepsize) = s.length

@doc raw"""
    DecreasingStepsize()

A functor that represents several decreasing step sizes

# Fields
* `length` – (`1`) the initial step size $l$.
* `factor` – (`1`) a value $f$ to multiply the initial step size with every iteration
* `subtrahend` – (`0`) a value $a$ that is subtracted every iteration
* `exponent` – (`1`) a value $e$ the current iteration numbers $e$th exponential
  is taken of

In total the complete formulae reads for the $i$th iterate as

$ s_i = \frac{(l-i\cdot a)f^i}{i^e}$

and hence the default simplifies to just $ s_i = \frac{l}{i} $

# Constructor

    ConstantStepSize(l,f,a,e)

initialiszes all fields above, where none of them is mandatory.
"""
mutable struct DecreasingStepsize <: Stepsize
    length::Float64
    factor::Float64
    subtrahend::Float64
    exponent::Float64
    function DecreasingStepsize(l::Real = 1.0, f::Real = 1.0, a::Real = 0.0, e::Real = 1.0)
        return new(l, f, a, e)
    end
end
function (s::DecreasingStepsize)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    return (s.length - i * s.subtrahend) * (s.factor^i) / (i^(s.exponent))
end
get_initial_stepsize(s::DecreasingStepsize) = s.length

"""
    Linesearch <: Stepsize

An abstract functor to represent line search type step size deteminations, see
[`Stepsize`](@ref) for details. One example is the [`ArmijoLinesearch`](@ref)
functor.

Compared to simple step sizes, the linesearch functors provide an interface of
the form `(p,o,i,η) -> s` with an additional (but optional) fourth parameter to
proviade a search direction; this should default to something reasonable, e.g.
the negative gradient.
"""
abstract type Linesearch <: Stepsize end

@doc raw"""
    ArmijoLineseach <: Linesearch

A functor representing Armijo line seach including the last runs state, i.e. a
last step size.

# Fields
* `initialStepsize` – (`1.0`) and initial step size
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use, defaults to
  the exponential map
* `contractionFactor` – (`0.95`) exponent for line search reduction
* `sufficientDecrease` – (`0.1`) gain within Armijo's rule
* `lastStepSize` – (`initialstepsize`) the last step size we start the search with

# Constructor

    ArmijoLineSearch()

with the Fields above in their order as optional arguments.

This method returns the functor to perform Armijo line search, where two inter
faces are available:
* based on a tuple `(p,o,i)` of a [`GradientProblem`](@ref) `p`, [`Options`](@ref) `o`
  and a current iterate `i`.
* with `(M,x,F,∇Fx[,η=-∇Fx]) -> s` where [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) `M`, a current
  point `x` a function `F`, that maps from the manifold to the reals,
  its gradient (a tangent vector) `∇F`$=\nabla F(x)$ at  `x` and an optional
  search direction tangent vector `η-∇F` are the arguments.
"""
mutable struct ArmijoLinesearch{TRM<:AbstractRetractionMethod} <: Linesearch
    initialStepsize::Float64
    retraction_method::TRM
    contractionFactor::Float64
    sufficientDecrease::Float64
    stepsizeOld::Float64
    function ArmijoLinesearch(
        s::Float64 = 1.0,
        r::AbstractRetractionMethod = ExponentialRetraction(),
        contractionFactor::Float64 = 0.95,
        sufficientDecrease::Float64 = 0.1,
    )
        return new{typeof(r)}(s, r, contractionFactor, sufficientDecrease, s)
    end
end
function (a::ArmijoLinesearch)(
    p::P,
    o::O,
    i::Int,
    η = -get_gradient(p, o.x),
) where {P<:GradientProblem{mT} where {mT<:Manifold},O<:Options}
    a.stepsizeOld = linesearch_backtrack(
        p.M,
        p.cost,
        o.x,
        get_gradient(p, o.x),
        a.stepsizeOld,
        a.sufficientDecrease,
        a.contractionFactor,
        a.retraction_method,
        η,
    )
    return a.stepsizeOld
end
get_initial_stepsize(a::ArmijoLinesearch) = a.initialStepsize

@doc raw"""
    linesearch_backtrack(M, F, x, ∇F, s, decrease, contract, retr, η = -∇F, f0 = F(x))

perform a linesearch for
* a manifold `M`
* a cost function `F`,
* an iterate `x`
* the gradient $∇F(x)``
* an initial stepsize `s` usually called $γ$
* a sufficient `decrease`
* a `contract`ion factor $σ$
* a `retr`action, which defaults to the `ExponentialRetraction()`
* a search direction $η = -∇F(x)$
* an offset, $f_0 = F(x)$
"""
function linesearch_backtrack(
    M::Manifold,
    F::TF,
    x,
    ∇F::T,
    s,
    decrease,
    contract,
    retr::AbstractRetractionMethod = ExponentialRetraction(),
    η::T = -∇F,
    f0 = F(x),
) where {TF,T}
    xNew = retract(M, x, s * η, retr)
    fNew = F(xNew)
    while fNew < f0 + decrease * s * inner(M, x, η, ∇F) # increase
        xNew = retract(M, x, s * η, retr)
        fNew = F(xNew)
        s = s / contract
    end
    s = s * contract # correct last
    while fNew > f0 + decrease * s * inner(M, x, η, ∇F) # decrease
        s = contract * s
        xNew = retract(M, x, s * η, retr)
        fNew = F(xNew)
    end
    return s
end

@doc raw"""
    get_stepsize(p::Problem, o::Options, vars...)

return the stepsize stored within [`Options`](@ref) `o` when solving [`Problem`](@ref) `p`.
This method also works for decorated options and the [`Stepsize`](@ref) function within
the options, by default stored in `o.stepsize`.
"""
function get_stepsize(p::Problem, o::Options, vars...)
    return get_stepsize(p, o, dispatch_options_decorator(o), vars...)
end
function get_stepsize(p::Problem, o::Options, ::Val{true}, vars...)
    return get_stepsize(p, o.options, vars...)
end
get_stepsize(p::Problem, o::Options, ::Val{false}, vars...) = o.stepsize(p, o, vars...)

function get_initial_stepsize(p::Problem, o::Options, vars...)
    return get_initial_stepsize(
        p::Problem,
        o::Options,
        dispatch_options_decorator(o),
        vars...,
    )
end
function get_initial_stepsize(p::Problem, o::Options, ::Val{true}, vars...)
    return get_initial_stepsize(p, o.options)
end
function get_initial_stepsize(p::Problem, o::Options, ::Val{false}, vars...)
    return get_initial_stepsize(o.stepsize)
end

function get_last_stepsize(p::Problem, o::Options, vars...)
    return get_last_stepsize(p, o, dispatch_options_decorator(o), vars...)
end
function get_last_stepsize(p::Problem, o::Options, ::Val{true}, vars...)
    return get_last_stepsize(p, o.options, vars...)
end
function get_last_stepsize(p::Problem, o::Options, ::Val{false}, vars...)
    return get_last_stepsize(p, o, o.stepsize, vars...)
end
#
# dispatch on stepsize
get_last_stepsize(p::Problem, o::Options, s::Stepsize, vars...) = s(p, o, vars...)
get_last_stepsize(p::Problem, o::Options, s::ArmijoLinesearch, vars...) = s.stepsizeOld
