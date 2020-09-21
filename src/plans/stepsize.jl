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
    return a(p.M, o.x, p.cost, get_gradient(p, o.x), η)
end
function (a::ArmijoLinesearch)(M::mT, x, F::TF, ∇F::T, η::T = -∇F) where {mT<:Manifold,TF,T}
    # for local shortness
    s = a.stepsizeOld
    f0 = F(x)
    xNew = retract(M, x, s * η, a.retraction_method)
    fNew = F(xNew)
    while fNew < f0 + a.sufficientDecrease * s * inner(M, x, η, ∇F) # increase
        xNew = retract(M, x, s * η, a.retraction_method)
        fNew = F(xNew)
        s = s / a.contractionFactor
    end
    s = s * a.contractionFactor # correct last
    while fNew > f0 + a.sufficientDecrease * s * inner(M, x, η, ∇F) # decrease
        s = a.contractionFactor * s
        xNew = retract(M, x, s * η, a.retraction_method)
        fNew = F(xNew)
    end
    a.stepsizeOld = s
    return s
end
get_initial_stepsize(a::ArmijoLinesearch) = a.initialStepsize

@doc raw"""
    NonmonotoneLineseach <: Linesearch

A functor representing a nonmonotone line seach using the Barzilai-Borwein step size[^Iannazzo2018]. Together with a gradient descent algorithm 
this line search represents the Riemannian Barzilai-Borwein with nonmonotone line-search (RBBNMLS) algorithm. However, different than in the paper 
by Iannazzo and Porcelli, our implementation of the nonmonotone line search performs first steps 4 to 6 and subsequently steps 1 to 2 of the RBBNMLS. 
Step 3 of RBBNMLS is a gradient descent step into the direction found with the nonlinear line search.

[^Iannazzo2018]:
    > B. Iannazzo, M. Porcelli, __The Riemannian Barzilai–Borwein Method with Nonmonotone Line Search and the Matrix Geometric Mean Computation__,
    > In: IMA Journal of Numerical Analysis. Volume 38, Issue 1, January 2018, Pages 495–517,
    > doi [10.1093/imanum/drx015](https://doi.org/10.1093/imanum/drx015)


# Fields
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use, defaults to
  the exponential map
* `stepsizeReduction` – (`0.5`) step size reduction factor contained in the interval (0,1)
* `sufficientDecrease` – (`1e-4`) sufficient decrease parameter contained in the interval (0,1)
* `memory` – (`10`) the number of iterations after which the cost function value has to be sufficiently decreased
* `maxStepsize` – (`1e3`) upper bound for the Barzilai-Borwein step size greater than zero
* `minStepsize` – (`1e-3`) lower bound for the Barzilai-Borwein step size between zero and maxStepsize
* `lastStepSize` – the last step size we start the search with
* `lastPoint` – the x-value of the previous iteration 
* `oldCosts` – a vector of the values of the cost function from the last `M` iterations
* `strategy` – (`direct`) defines if the new step size is computed in a direct, indirect or alternating way

# Constructor

    NonmonotoneLineSearch()

with the Fields above in their order as optional arguments.

This method returns the functor to perform nonmonotone line search.
"""
mutable struct NonmonotoneLinesearch{TRM<:AbstractRetractionMethod, T<:AbstractVector} <: Linesearch
    retraction_method::TRM
    stepsizeReduction::Float64
    sufficientDecrease::Float64
    memory_size::Int
    maxStepsize::Float64
    minStepsize::Float64
    stepsizeOld::Float64
    storage::StoreOptionsAction
    old_f::T
    strategy::Symbol
    function NonmonotoneLinesearch(
        manifold::Manifold,
        retraction_method::AbstractRetractionMethod = ExponentialRetraction(),
        stepsizeReduction::Float64 = 0.5,
        sufficientDecrease::Float64 = 1e-4,
        memory_size::Int = 10,
        a::StoreOptionsAction = StoreOptionsAction((:x, :∇)),
        maxStepsize::Float64 = 1e3, 
        minStepsize::Float64 = 1e-3, 
        stepsizeOld::Float64 = 1.0, # note in Fields
        strategy::Symbol = :direct,
    )
    if strategy ∉ [:direct, :inverse, :alternating] 
        @warn string("The strategy '", strategy,"' is not defined. The 'direct' strategy is used instead.")
        strategy = :direct
    end
    return new{typeof(retraction_method), typeof(old_f)}(
        retraction_method, 
        stepsizeReduction, 
        sufficientDecrease, 
        memory_size, 
        maxStepsize, 
        minStepsize, 
        stepsizeOld, 
        a, 
        zeros(memory_size), 
        strategy)
    end
end
function (a::NonmonotoneLinesearch)(
    p::P,
    o::O,
    i::Int,
    η = -get_gradient(p, o.x),
) where {P<:GradientProblem{mT} where {mT<:Manifold},O<:Options}
    if !all(has_storage.(Ref(abspath.storage), [:x, :∇])
        xOld = o.x
        ∇Old = get_gradient(p, o.x)
    else
        xOld, ∇Old = get_storage.(Ref(a.storage), [:x, :∇])
    end
    update_storage!(a.storage, o)
    return a(p.M, o.x, p.cost, get_gradient(p, o.x), xOld, ∇Old, i, η, o.vector_transport_method)
end
function (a::NonmonotonLinesearch)(M::mT, x, F::TF, ∇F::T, xOld, ∇Old, iter::Int, η::T = -∇F, vector_transport_method) where {mT<:Manifold,TF,T}
    s = a.stepsizeOld

    #find the difference between the current and previous gardient after the previous gradient is transported to the current tangent space 
    gradOld #save
    grad_diff = ∇F - vector_transport_to(M, a.x_old, gradOld, x, vector_transport_method)        
    #transport the previous step into the tangent space of the current manifold point
    x_diff = - s * vector_transport_to(M, a.x_old, gradOld, x, vector_transport_method)

    #compute the new Barzilai-Borwein step size
    inner(M, x, x_diff, grad_diff)
    s2 = inner(M, x, grad_diff, grad_diff)
    s2 = s2==0 ? 1.0 : s2
    inner(M, x, x_diff, x_diff)
    #indirect strategy
    if a.strategy == :inverse
        if inner(M, x, x_diff, grad_diff) > 0       
            BarzilaiBorwein_stepsize = min(maxStepsize, max(minStepsize, inner(M, x, x_diff, grad_diff)/inner(M, x, grad_diff, grad_diff)))
        else
            BarzilaiBorwein_stepsize = maxStepsize
        end
    #alternating strategy
    elseif a.strategy == :alternating
        if inner(M, x, x_diff, grad_diff) > 0
            if iter % 2 == 0        
                BarzilaiBorwein_stepsize = min(maxStepsize, max(minStepsize, inner(M, x, x_diff, grad_diff)/inner(M, x, grad_diff, grad_diff)))
            else
                BarzilaiBorwein_stepsize = min(maxStepsize, max(minStepsize, inner(M, x, x_diff, x_diff)/inner(M, x, x_diff, grad_diff)))
            end
        else
            BarzilaiBorwein_stepsize = maxStepsize
        end
    #direct strategy
    else
        if inner(M, x, x_diff, grad_diff) > 0
            BarzilaiBorwein_stepsize = min(maxStepsize, max(minStepsize, inner(M, x, grad_diff, grad_diff)/inner(M, x, x_diff, grad_diff)))
        else
            BarzilaiBorwein_stepsize = maxStepsize
        end
    end

    #compute the new step size with the help of the Barzilai-Borwein step size
    f0 = F(x)
    h = 0
    xNew = retract(M, x, a.stepsizeReduction^h * BarzilaiBorwein_stepsize * η, a.retraction_method)
    fNew = F(xNew)
    f_change = a.sufficientDecrease * a.stepsizeReduction^h * BarzilaiBorwein_stepsize * inner(M, x, ∇F, ∇F)
    fbound = max([a.old_f[iter + 1 - j] - f_change for j in 1:min(iter + 1, a.memory_size)])    #if only the last memory_size f are saved we have to call the value differently
    #find the smallest h for which fNew <= fbound    
    while fNew > fbound
        h = h + 1
        xNew = retract(M, x, a.stepsizeReduction^h * BarzilaiBorwein_stepsize * η, a.retraction_method)
        fNew = F(xNew)
        f_change = a.sufficientDecrease * a.stepsizeReduction^h * BarzilaiBorwein_stepsize * inner(M, x, ∇F, ∇F)
        fbound = max([a.old_f[iter + 1 - j] - f_change for j in 1:min(iter + 1, a.memory_size)]) 
    end 
   
    #set and return the new step size
    s = a.stepsizeReduction^h * BarzilaiBorwein_stepsize
    a.stepsizeOld = s
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
