"""
    ConstantStepsize <: Stepsize

A functor that always returns a fixed step size.

# Fields
* `length` – constant value for the step size.

# Constructor

    ConstantStepSize(s)

initialize the stepsize to a constant `s`
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
* `length` – (`1`) the initial step size ``l``.
* `factor` – (`1`) a value ``f`` to multiply the initial step size with every iteration
* `subtrahend` – (`0`) a value ``a`` that is subtracted every iteration
* `exponent` – (`1`) a value ``e`` the current iteration numbers ``e``th exponential
  is taken of

In total the complete formulae reads for the ``i``th iterate as

````math
s_i = \frac{(l - i a)f^i}{i^e}
````

and hence the default simplifies to just ``s_i = \frac{l}{i}``

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
provide a search direction; this should default to something reasonable, e.g.
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
* with `(M, x, F, gradFx[,η=-gradFx]) -> s` where [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) `M`, a current
  point `x` a function `F`, that maps from the manifold to the reals,
  its gradient (a tangent vector) `gradFx```=\operatorname{grad}F(x)`` at  `x` and an optional
  search direction tangent vector `η=-gradFx` are the arguments.
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
    p::GradientProblem, o::Options, ::Int, η=-get_gradient(p, o.x)
)
    a.stepsizeOld = linesearch_backtrack(
        p.M,
        x -> p.cost(p.M, x),
        o.x,
        get_gradient!(p, o.gradient, o.x),
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
    linesearch_backtrack(M, F, x, gradFx, s, decrease, contract, retr, η = -gradFx, f0 = F(x))

perform a linesearch for
* a manifold `M`
* a cost function `F`,
* an iterate `x`
* the gradient ``\operatorname{grad}F(x)``
* an initial stepsize `s` usually called ``γ``
* a sufficient `decrease`
* a `contract`ion factor ``σ``
* a `retr`action, which defaults to the `ExponentialRetraction()`
* a search direction ``η = -\operatorname{grad}F(x)``
* an offset, ``f_0 = F(x)``
"""
function linesearch_backtrack(
    M::AbstractManifold,
    F::TF,
    x,
    gradFx::T,
    s,
    decrease,
    contract,
    retr::AbstractRetractionMethod=ExponentialRetraction(),
    η::T=-gradFx,
    f0=F(x),
) where {TF,T}
    xNew = retract(M, x, s * η, retr)
    fNew = F(xNew)
    while fNew < f0 + decrease * s * inner(M, x, η, gradFx) # increase
        s = s / contract
        retract!(M, xNew, x, s * η, retr)
        fNew = F(xNew)
    end
    s = s * contract # correct last
    while fNew > f0 + decrease * s * inner(M, x, η, gradFx) # decrease
        s = contract * s
        retract!(M, xNew, x, s * η, retr)
        fNew = F(xNew)
    end
    return s
end

@doc raw"""
    NonmonotoneLinesearch <: Linesearch

A functor representing a nonmonotone line search using the Barzilai-Borwein step size[^Iannazzo2018]. Together with a gradient descent algorithm
this line search represents the Riemannian Barzilai-Borwein with nonmonotone line-search (RBBNMLS) algorithm. We shifted the order of the algorithm steps from the paper
by Iannazzo and Porcelli so that in each iteration we first find

```math
y_{k} = \operatorname{grad}F(x_{k}) - \operatorname{T}_{x_{k-1} → x_k}(\operatorname{grad}F(x_{k-1}))
```

and

```math
s_{k} = - α_{k-1} * \operatorname{T}_{x_{k-1} → x_k}(\operatorname{grad}F(x_{k-1})),
```

where ``α_{k-1}`` is the step size computed in the last iteration and ``\operatorname{T}`` is a vector transport.
We then find the Barzilai–Borwein step size

```math
α_k^{\text{BB}} = \begin{cases}
\min(α_{\text{max}}, \max(α_{\text{min}}, τ_{k})),  & \text{if } ⟨s_{k}, y_{k}⟩_{x_k} > 0,\\
α_{\text{max}}, & \text{else,}
\end{cases}
```

where

```math
τ_{k} = \frac{⟨s_{k}, s_{k}⟩_{x_k}}{⟨s_{k}, y_{k}⟩_{x_k}},
```

if the direct strategy is chosen,

```math
τ_{k} = \frac{⟨s_{k}, y_{k}⟩_{x_k}}{⟨y_{k}, y_{k}⟩_{x_k}},
```

in case of the inverse strategy and an alternation between the two in case of the
alternating strategy. Then we find the smallest ``h = 0, 1, 2, …`` such that

```math
F(\operatorname{retr}_{x_k}(- σ^h α_k^{\text{BB}} \operatorname{grad}F(x_k)))
\leq
\max_{1 ≤ j ≤ \min(k+1,m)} F(x_{k+1-j}) - γ σ^h α_k^{\text{BB}} ⟨\operatorname{grad}F(x_k), \operatorname{grad}F(x_k)⟩_{x_k},
```

where ``σ`` is a step length reduction factor ``∈ (0,1)``, ``m`` is the number of iterations
after which the function value has to be lower than the current one
and ``γ`` is the sufficient decrease parameter ``∈(0,1)``. We can then find the new stepsize by

```math
α_k = σ^h α_k^{\text{BB}}.
```

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
* `storage` – (`x`, `gradient`) a [`StoreOptionsAction`](@ref) to store `old_x` and `old_gradient`, the x-value and corresponding gradient of the previous iteration

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
        storage::StoreOptionsAction=StoreOptionsAction((:x, :gradient)),
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
            throw(
                DomainError(
                    min_stepsize,
                    "The lower bound for the step size min_stepsize has to be greater than zero.",
                ),
            )
        end
        if max_stepsize <= min_stepsize
            throw(
                DomainError(
                    max_stepsize,
                    "The upper bound for the step size max_stepsize has to be greater its lower bound min_stepsize.",
                ),
            )
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
    p::GradientProblem, o::Options, i::Int, η=-get_gradient(p, o.x)
)
    if !all(has_storage.(Ref(a.storage), [:x, :gradient]))
        old_x = o.x
        old_gradient = get_gradient(p, o.x)
    else
        old_x, old_gradient = get_storage.(Ref(a.storage), [:x, :gradient])
    end
    update_storage!(a.storage, o)
    return a(p.M, o.x, x -> get_cost(p, x), get_gradient(p, o.x), η, old_x, old_gradient, i)
end
function (a::NonmonotoneLinesearch)(
    M::mT, x, F::TF, gradFx::T, η::T, old_x, old_gradient, iter::Int
) where {mT<:AbstractManifold,TF,T}
    #find the difference between the current and previous gardient after the previous gradient is transported to the current tangent space
    grad_diff =
        gradFx - vector_transport_to(M, old_x, old_gradient, x, a.vector_transport_method)
    #transport the previous step into the tangent space of the current manifold point
    x_diff =
        -a.initial_stepsize *
        vector_transport_to(M, old_x, old_gradient, x, a.vector_transport_method)

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
        gradFx,
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
    WolfePowellLineseach <: Linesearch

Do a backtracking linesearch to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``η`` starting from ``x``, i.e.

```math
f\bigl( \operatorname{retr}_x(αη) \bigr) ≤ f(x_k) + c_1 α_k ⟨\operatorname{grad}f(x), η⟩_x
\quad\text{and}\quad
\frac{\mathrm{d}}{\mathrm{d}t} f\bigr(\operatorname{retr}_x(tη)\bigr)
\Big\vert_{t=α}
≥ c_2 \frac{\mathrm{d}}{\mathrm{d}t} f\bigl(\operatorname{retr}_x(tη)\bigr)\Big\vert_{t=0}.
```

# Constructor

    WolfePowellLinesearch(
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        c_1::Float64=10^(-4),
        c_2::Float64=0.999
    )
"""
mutable struct WolfePowellLineseach <: Linesearch
    retraction_method::AbstractRetractionMethod
    vector_transport_method::AbstractVectorTransportMethod
    c_1::Float64
    c_2::Float64

    function WolfePowellLineseach(
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        c_1::Float64=10^(-4),
        c_2::Float64=0.999,
    )
        return new(retr, vtr, c_1, c_2)
    end
end

function (a::WolfePowellLineseach)(
    p::P, o::O, ::Int, η=-get_gradient(p, o.x)
) where {P<:GradientProblem{T,mT} where {T,mT<:AbstractManifold},O<:Options}
    s = 1.0
    s_plus = 1.0
    s_minus = 1.0
    f0 = get_cost(p, o.x)
    xNew = retract(p.M, o.x, s * η, a.retraction_method)
    fNew = get_cost(p, xNew)
    η_xNew = vector_transport_to(p.M, o.x, η, xNew, a.vector_transport_method)
    if fNew > f0 + a.c_1 * s * inner(p.M, o.x, η, o.gradient)
        while (fNew > f0 + a.c_1 * s * inner(p.M, o.x, η, o.gradient)) &&
            (s_minus > 10^(-9)) # decrease
            s_minus = s_minus * 0.5
            s = s_minus
            retract!(p.M, xNew, o.x, s * η, a.retraction_method)
            fNew = get_cost(p, xNew)
        end
        s_plus = 2.0 * s_minus
    else
        vector_transport_to!(p.M, η_xNew, o.x, η, xNew, a.vector_transport_method)
        if inner(p.M, xNew, get_gradient(p, xNew), η_xNew) <
            a.c_2 * inner(p.M, o.x, η, o.gradient)
            while fNew <= f0 + a.c_1 * s * inner(p.M, o.x, η, o.gradient) &&
                (s_plus < 10^(9))# increase
                s_plus = s_plus * 2.0
                s = s_plus
                retract!(p.M, xNew, o.x, s * η, a.retraction_method)
                fNew = get_cost(p, xNew)
            end
            s_minus = s_plus / 2.0
        end
    end
    retract!(p.M, xNew, o.x, s_minus * η, a.retraction_method)
    vector_transport_to!(p.M, η_xNew, o.x, η, xNew, a.vector_transport_method)
    while inner(p.M, xNew, get_gradient(p, xNew), η_xNew) <
          a.c_2 * inner(p.M, o.x, η, o.gradient)
        s = (s_minus + s_plus) / 2
        retract!(p.M, xNew, o.x, s * η, a.retraction_method)
        fNew = get_cost(p, xNew)
        if fNew <= f0 + a.c_1 * s * inner(p.M, o.x, η, o.gradient)
            s_minus = s
        else
            s_plus = s
        end
        if abs(s_plus - s_minus) <= 10^(-12)
            break
        end
        retract!(p.M, xNew, o.x, s_minus * η, a.retraction_method)
        vector_transport_to!(p.M, η_xNew, o.x, η, xNew, a.vector_transport_method)
    end
    s = s_minus
    return s
end

@doc raw"""
    WolfePowellBinaryLinesearch <: Linesearch

A [`Linesearch`](@ref) method that determines a step size `t` fulfilling the Wolfe conditions

based on a binary chop. Let ``η`` be a search direction and ``c_1,c_2>0`` be two constants.
Then with

```math
A(t) = f(x_+) ≤ c_1 t ⟨\operatorname{grad}f(x), η⟩_{x}
\quad\text{and}\quad 
W(t) = ⟨\operatorname{grad}f(x_+), \text{V}_{x_+\gets x}η⟩_{x_+} ≥ c_2 ⟨η, \operatorname{grad}f(x)⟩_x,
```

where ``x_+ = \operatorname{retr}_x(tη)`` is the current trial point, and ``\text{V}`` is a
vector transport, we perform the following Algorithm similar to Algorithm 7 from [^Huang2014]
1. set ``α=0``, ``β=∞`` and ``t=1``.
2. While either ``A(t)`` does not hold or ``W(t)`` does not hold do steps 3-5.
3. If ``A(t)`` fails, set ``β=t``.
4. If ``A(t)`` holds but ``W(t)`` fails, set ``α=t``.
5. If ``β<∞`` set ``t=\frac{α+β}{2}``, otherwise set ``t=2α``.

# Constructor

    WolfePowellBinaryLinesearch(
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        c_1::Float64=10^(-4),
        c_2::Float64=0.999
    )

[^Huang2014]:
    > Huang, W.: _Optimization algorithms on Riemannian manifolds with applications_,
    > Dissertation, Flordia State University, 2014.
    > [pdf](https://www.math.fsu.edu/~whuang2/pdf/Huang_W_Dissertation_2013.pdf)
"""
mutable struct WolfePowellBinaryLinesearch <: Linesearch
    retraction_method::AbstractRetractionMethod
    vector_transport_method::AbstractVectorTransportMethod

    c_1::Float64
    c_2::Float64

    function WolfePowellBinaryLinesearch(
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        c_1::Float64=10^(-4),
        c_2::Float64=0.999,
    )
        return new(retr, vtr, c_1, c_2)
    end
end

function (a::WolfePowellBinaryLinesearch)(
    p::P, o::O, ::Int, η=-get_gradient(p, o.x)
) where {P<:GradientProblem{T,mT} where {T,mT<:AbstractManifold},O<:Options}
    α = 0.0
    β = Inf
    t = 1.0
    f0 = get_cost(p, o.x)
    xNew = retract(p.M, o.x, t * η, a.retraction_method)
    fNew = get_cost(p, xNew)
    η_xNew = vector_transport_to(p.M, o.x, η, xNew, a.vector_transport_method)
    gradient_new = get_gradient(p, xNew)
    nAt = fNew > f0 + a.c_1 * t * inner(p.M, o.x, η, o.gradient)
    nWt = inner(p.M, xNew, gradient_new, η_xNew) < a.c_2 * inner(p.M, o.x, η, o.gradient)
    while (nAt || nWt) && (t > 1e-9) && ((α + β) / 2 - t > 1e-9)
        nAt && (β = t)            # A(t) fails
        (!nAt && nWt) && (α = t)  # A(t) holds but W(t) fails
        t = isinf(β) ? 2 * α : (α + β) / 2
        # Update trial point
        retract!(p.M, xNew, o.x, t * η, a.retraction_method)
        fNew = get_cost(p, xNew)
        gradient_new = get_gradient(p, xNew)
        vector_transport_to!(p.M, η_xNew, o.x, η, xNew, a.vector_transport_method)
        # Update conditions
        nAt = fNew > f0 + a.c_1 * t * inner(p.M, o.x, η, o.gradient)
        nWt =
            inner(p.M, xNew, gradient_new, η_xNew) < a.c_2 * inner(p.M, o.x, η, o.gradient)
    end
    return t
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
get_last_stepsize(::Problem, ::Options, s::ArmijoLinesearch, ::Any...) = s.stepsizeOld
