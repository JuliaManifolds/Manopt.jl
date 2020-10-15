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
    function DecreasingStepsize(l::Real=1.0, f::Real=1.0, a::Real=0.0, e::Real=1.0)
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
        s::Float64=1.0,
        r::AbstractRetractionMethod=ExponentialRetraction(),
        contractionFactor::Float64=0.95,
        sufficientDecrease::Float64=0.1,
    )
        return new{typeof(r)}(s, r, contractionFactor, sufficientDecrease, s)
    end
end
function (a::ArmijoLinesearch)(
    p::P, o::O, i::Int, η=-get_gradient(p, o.x)
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
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    η::T=-∇F,
    f0=F(x),
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
    NonmonotoneLinesearch <: Linesearch

A functor representing a nonmonotone line seach using the Barzilai-Borwein step size[^Iannazzo2018]. Together with a gradient descent algorithm 
this line search represents the Riemannian Barzilai-Borwein with nonmonotone line-search (RBBNMLS) algorithm. We shifted the order of the algorithm steps from the paper 
by Iannazzo and Porcelli so that in each iteration we first find 

$y_{k} = \nabla F(x_{k}) - \operatorname{T}_{x_{k-1} \to x_k}(\nabla F(x_{k-1}))$

and 

$s_{k} = - \alpha_{k-1} * \operatorname{T}_{x_{k-1} \to x_k}(\nabla F(x_{k-1})),$

where $\alpha_{k-1}$ is the step size computed in the last iteration and $\operatorname{T}$ is a vector transport. 
We then find the Barzilai–Borwein step size 

$α_k^{\text{BB}} = \begin{cases}
\min(α_{\text{max}}, \max(α_{\text{min}}, τ_{k})),  & \text{if } ⟨s_{k}, y_{k}⟩_{x_k} > 0,\\
α_{\text{max}}, & \text{else,}
\end{cases}$

where 

$τ_{k} = \frac{⟨s_{k}, s_{k}⟩_{x_k}}{⟨s_{k}, y_{k}⟩_{x_k}},$

if the direct strategy is chosen,

$τ_{k} = \frac{⟨s_{k}, y_{k}⟩_{x_k}}{⟨y_{k}, y_{k}⟩_{x_k}},$

in case of the inverse strategy and an alternation between the two in case of the alternating strategy. Then we find the smallest $h = 0, 1, 2 …$ such that

$F(\operatorname{retr}_{x_k}(- σ^h α_k^{\text{BB}} \nabla F(x_k))) \leq \max_{1 ≤ j ≤ \min(k+1,m)} F(x_{k+1-j}) - γ σ^h α_k^{\text{BB}} ⟨\nabla F(x_k), \nabla F(x_k)⟩_{x_k},$

where $σ$ is a step length reduction factor $\in (0,1)$, $m$ is the number of iterations after which the function value has to be lower than the current one 
and $γ$ is the sufficient decrease parameter $\in (0,1)$. We can then find the new stepsize by

$α_k = σ^h α_k^{\text{BB}}.$

[^Iannazzo2018]:
    > B. Iannazzo, M. Porcelli, __The Riemannian Barzilai–Borwein Method with Nonmonotone Line Search and the Matrix Geometric Mean Computation__,
    > In: IMA Journal of Numerical Analysis. Volume 38, Issue 1, January 2018, Pages 495–517,
    > doi [10.1093/imanum/drx015](https://doi.org/10.1093/imanum/drx015)


# Fields
* `initial_stepsize` – (`1.0`) the step size we start the search with
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use
* `vector_transport_method` – (`ParallelTransport()`) the vector transport method to use
* `stepsize_reduction` – (`0.5`) step size reduction factor contained in the interval (0,1)
* `sufficient_decrease` – (`1e-4`) sufficient decrease parameter contained in the interval (0,1)
* `memory_size` – (`10`) number of iterations after which the cost value needs to be lower than the current one
* `min_stepsize` – (`1e-3`) lower bound for the Barzilai-Borwein step size greater than zero
* `max_stepsize` – (`1e3`) upper bound for the Barzilai-Borwein step size greater than min_stepsize
* `strategy` – (`direct`) defines if the new step size is computed using the direct, indirect or alternating strategy
* `storage` – (`x`, `∇F`) a [`StoreOptionsAction`](@ref) to store `old_x` and `old_∇`, the x-value and corresponding gradient of the previous iteration

# Constructor

    NonmonotoneLinesearch()

with the Fields above in their order as optional arguments.

This method returns the functor to perform nonmonotone line search.
"""
mutable struct NonmonotoneLinesearch{
    TRM<:AbstractRetractionMethod,VTM<:AbstractVectorTransportMethod,T<:AbstractVector
} <: Linesearch
    retraction_method::TRM
    vector_transport_method::VTM
    stepsize_reduction::Float64
    sufficient_decrease::Float64
    min_stepsize::Float64
    max_stepsize::Float64
    initial_stepsize::Float64
    old_costs::T
    strategy::Symbol
    storage::StoreOptionsAction
    function NonmonotoneLinesearch(
        initial_stepsize::Float64=1.0,
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
        stepsize_reduction::Float64=0.5,
        sufficient_decrease::Float64=1e-4,
        memory_size::Int=10,
        min_stepsize::Float64=1e-3,
        max_stepsize::Float64=1e3,
        strategy::Symbol=:direct,
        storage::StoreOptionsAction=StoreOptionsAction((:x, :∇)),
    )
        if strategy ∉ [:direct, :inverse, :alternating]
            @warn string(
                "The strategy '",
                strategy,
                "' is not defined. The 'direct' strategy is used instead.",
            )
            strategy = :direct
        end
        if min_stepsize <= 0.0
            throw(DomainError(
                min_stepsize,
                "The lower bound for the step size min_stepsize has to be greater than zero.",
            ))
        end
        if max_stepsize <= min_stepsize
            throw(DomainError(
                max_stepsize,
                "The upper bound for the step size max_stepsize has to be greater its lower bound min_stepsize.",
            ))
        end
        if memory_size <= 0
            throw(DomainError(memory_size, "The memory_size has to be greater than zero."))
        end
        return new{
            typeof(retraction_method),typeof(vector_transport_method),Vector{Float64}
        }(
            retraction_method,
            vector_transport_method,
            stepsize_reduction,
            sufficient_decrease,
            min_stepsize,
            max_stepsize,
            initial_stepsize,
            zeros(memory_size),
            strategy,
            storage,
        )
    end
end
function (a::NonmonotoneLinesearch)(
    p::P, o::O, i::Int, η=-get_gradient(p, o.x)
) where {P<:GradientProblem{mT} where {mT<:Manifold},O<:Options}
    if !all(has_storage.(Ref(a.storage), [:x, :∇]))
        old_x = o.x
        old_∇ = get_gradient(p, o.x)
    else
        old_x, old_∇ = get_storage.(Ref(a.storage), [:x, :∇])
    end
    update_storage!(a.storage, o)
    return a(p.M, o.x, p.cost, get_gradient(p, o.x), η, old_x, old_∇, i)
end
function (a::NonmonotoneLinesearch)(
    M::mT, x, F::TF, ∇F::T, η::T, old_x, old_∇, iter::Int
) where {mT<:Manifold,TF,T}
    #find the difference between the current and previous gardient after the previous gradient is transported to the current tangent space 
    grad_diff = ∇F - vector_transport_to(M, old_x, old_∇, x, a.vector_transport_method)
    #transport the previous step into the tangent space of the current manifold point
    x_diff =
        -a.initial_stepsize *
        vector_transport_to(M, old_x, old_∇, x, a.vector_transport_method)

    #compute the new Barzilai-Borwein step size
    s1 = inner(M, x, x_diff, grad_diff)
    s2 = inner(M, x, grad_diff, grad_diff)
    s2 = s2 == 0 ? 1.0 : s2
    s3 = inner(M, x, x_diff, x_diff)
    #indirect strategy
    if a.strategy == :inverse
        if s1 > 0
            BarzilaiBorwein_stepsize = min(a.max_stepsize, max(a.min_stepsize, s1 / s2))
        else
            BarzilaiBorwein_stepsize = a.max_stepsize
        end
        #alternating strategy
    elseif a.strategy == :alternating
        if s1 > 0
            if iter % 2 == 0
                BarzilaiBorwein_stepsize = min(a.max_stepsize, max(a.min_stepsize, s1 / s2))
            else
                BarzilaiBorwein_stepsize = min(a.max_stepsize, max(a.min_stepsize, s3 / s1))
            end
        else
            BarzilaiBorwein_stepsize = a.max_stepsize
        end
        #direct strategy
    else
        if s1 > 0
            BarzilaiBorwein_stepsize = min(a.max_stepsize, max(a.min_stepsize, s2 / s1))
        else
            BarzilaiBorwein_stepsize = a.max_stepsize
        end
    end

    memory_size = length(a.old_costs)
    if iter <= memory_size
        a.old_costs[iter] = F(x)
    else
        a.old_costs[1:(memory_size - 1)] = a.old_costs[2:memory_size]
        a.old_costs[memory_size] = F(x)
    end

    #compute the new step size with the help of the Barzilai-Borwein step size
    a.initial_stepsize = linesearch_backtrack(
        M,
        F,
        x,
        ∇F,
        BarzilaiBorwein_stepsize,
        a.sufficient_decrease,
        a.stepsize_reduction,
        a.retraction_method,
        η,
        maximum([a.old_costs[j] for j in 1:min(iter, memory_size)]),
    )
    return a.initial_stepsize
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
        p::Problem, o::Options, dispatch_options_decorator(o), vars...
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
