"""
    ConstantStepsize <: Stepsize

A functor that always returns a fixed step size.

# Fields
* `length` – constant value for the step size.

# Constructors

    ConstantStepsize(s)

initialize the stepsize to a constant `s` (deprecated)

    ConstantStepsize(M::AbstractManifold=DefaultManifold(2); stepsize=injectivity_radius(M)/2)

initialize the stepsize to a constant `stepsize`, which by default is half the injectivity
radius, unless the radius is infinity, then the default step size is `1`.
"""
mutable struct ConstantStepsize{T} <: Stepsize
    length::T
end
function ConstantStepsize(
    M::AbstractManifold=DefaultManifold(2);
    stepsize=isinf(injectivity_radius(M)) ? 1.0 : injectivity_radius(M) / 2,
)
    return ConstantStepsize{typeof(stepsize)}(stepsize)
end
function (cs::ConstantStepsize)(
    ::AbstractManoptProblem, ::AbstractManoptSolverState, ::Any, args...; kwargs...
)
    return cs.length
end
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
* `shift` – (`0`) shift the denominator iterator ``i`` by ``s```.

In total the complete formulae reads for the ``i``th iterate as

````math
s_i = \frac{(l - i a)f^i}{(i+s)^e}
````

and hence the default simplifies to just ``s_i = \frac{l}{i}``

# Constructor

    DecreasingStepsize(l=1,f=1,a=0,e=1,s=0)

Alternatively one can also use the following keyword.

    DecreasingStepsize(
        M::AbstractManifold=DefaultManifold(3);
        length=injectivity_radius(M)/2, multiplier=1.0, subtrahend=0.0, exponent=1.0, shift=0)

initialiszes all fields above, where none of them is mandatory and the length is set to
half and to $1$ if the injectivity radius is infinite.
"""
mutable struct DecreasingStepsize <: Stepsize
    length::Float64
    factor::Float64
    subtrahend::Float64
    exponent::Float64
    shift::Int
    function DecreasingStepsize(l::Real, f::Real=1.0, a::Real=0.0, e::Real=1.0, s::Int=0)
        return new(l, f, a, e, s)
    end
end
function DecreasingStepsize(
    M::AbstractManifold=DefaultManifold(3);
    length=isinf(manifold_dimension(M)) ? 1.0 : manifold_dimension(M) / 2,
    factor=1.0,
    subtrahend=0.0,
    exponent=1.0,
    shift=0,
)
    return DecreasingStepsize(length, factor, subtrahend, exponent, shift)
end
function (s::DecreasingStepsize)(
    ::P, ::O, i::Int, args...; kwargs...
) where {P<:AbstractManoptProblem,O<:AbstractManoptSolverState}
    return (s.length - i * s.subtrahend) * (s.factor^i) / ((i + s.shift)^(s.exponent))
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
    ArmijoLinesearch <: Linesearch

A functor representing Armijo line search including the last runs state, i.e. a
last step size.

# Fields
* `initial_stepsize` – (`1.0`) and initial step size
* `retraction_method` – (`default_retraction_method(M)`) the rectraction to use
* `contraction_factor` – (`0.95`) exponent for line search reduction
* `sufficient_decrease` – (`0.1`) gain within Armijo's rule
* `last_stepsize` – (`initialstepsize`) the last step size we start the search with
* `linesearch_stopsize` - (`0.0`) a safeguard when to stop the line search
    before the step is numerically zero. This should be combined with [`StopWhenStepsizeLess`](@ref)
* `initial_guess` (`(p,o,i,l) -> l`)  based on a [`GradientProblem`](@ref) `p`, [`AbstractManoptSolverState`](@ref) `o`
  and a current iterate `i` and a last step size `l`, this returns an initial guess. The default uses the last obtained stepsize
# Constructor

    ArmijoLineSearch(M)

with the Fields above as keyword arguments and the retraction is set to the default retraction on `M`.

The constructors return the functor to perform Armijo line search, where two interfaces are available:
* based on a tuple `(p,o,i)` of a [`GradientProblem`](@ref) `p`, [`AbstractManoptSolverState`](@ref) `o`
  and a current iterate `i`.
* with `(M, x, F, gradFx[,η=-gradFx]) -> s` where [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#Manifold) `M`, a current
  point `x` a function `F`, that maps from the manifold to the reals,
  its gradient (a tangent vector) `gradFx```=\operatorname{grad}F(x)`` at  `x` and an optional
  search direction tangent vector `η=-gradFx` are the arguments.
"""
mutable struct ArmijoLinesearch{TRM<:AbstractRetractionMethod,F} <: Linesearch
    initial_stepsize::Float64
    retraction_method::TRM
    contraction_factor::Float64
    sufficient_decrease::Float64
    last_stepsize::Float64
    linesearch_stopsize::Float64
    initial_guess::F
    function ArmijoLinesearch(
        M;
        initial_stepsize::Float64=1.0,
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        contraction_factor::Float64=0.95,
        sufficient_decrease::Float64=0.1,
        linesearch_stopsize::Float64=0.0,
        initial_guess=(p, o, i, l) -> l,
    )
        return new{typeof(retraction_method),typeof(initial_guess)}(
            initial_stepsize,
            retraction_method,
            contraction_factor,
            sufficient_decrease,
            initial_stepsize,
            linesearch_stopsize,
            initial_guess,
        )
    end
end
function (a::ArmijoLinesearch)(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    i::Int,
    η=-get_gradient(mp, get_iterate(s));
    kwargs...,
)
    a.last_stepsize = linesearch_backtrack(
        get_manifold(mp),
        p -> get_cost_function(mp)(get_manifold(mp), p),
        get_iterate(s),
        get_gradient!(mp, get_gradient(s), get_iterate(s)),
        a.initial_guess(mp, s, i, a.last_stepsize),
        a.sufficient_decrease,
        a.contraction_factor,
        a.retraction_method,
        η;
        stop_step=a.linesearch_stopsize,
    )
    return a.last_stepsize
end
get_initial_stepsize(a::ArmijoLinesearch) = a.initial_stepsize

@doc raw"""
    linesearch_backtrack(M, F, x, gradFx, s, decrease, contract, retr, η = -gradFx, f0 = F(x); stop_step=0.)

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
* a keyword `stop_step` as a minimal step size when to stop
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
    f0=F(x);
    stop_step=0.0,
) where {TF,T}
    xNew = retract(M, x, s * η, retr)
    fNew = F(xNew)
    search_dir_inner = inner(M, x, η, gradFx)
    extended = false
    while fNew < f0 + decrease * s * search_dir_inner # increase
        extended = true
        s = s / contract
        retract!(M, xNew, x, s * η, retr)
        fNew = F(xNew)
    end
    if extended
        s *= contract  # undo last increase
        retract!(M, xNew, x, s * η, retr)
        fNew = F(xNew)
    end
    while fNew > f0 + decrease * s * search_dir_inner # decrease
        s = contract * s
        retract!(M, xNew, x, s * η, retr)
        fNew = F(xNew)
        (s < stop_step) && break
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
* `linesearch_stopsize` - (`0.0`) a safeguard when to stop the line search
    before the step is numerically zero. This should be combined with [`StopWhenStepsizeLess`](@ref)
* `memory_size` – (`10`) number of iterations after which the cost value needs to be lower than the current one
* `min_stepsize` – (`1e-3`) lower bound for the Barzilai-Borwein step size greater than zero
* `max_stepsize` – (`1e3`) upper bound for the Barzilai-Borwein step size greater than min_stepsize
* `retraction_method` – (`ExponentialRetraction()`) the rectraction to use
* `strategy` – (`direct`) defines if the new step size is computed using the direct, indirect or alternating strategy
* `storage` – (`x`, `gradient`) a [`StoreStateAction`](@ref) to store `old_x` and `old_gradient`, the x-value and corresponding gradient of the previous iteration
* `stepsize_reduction` – (`0.5`) step size reduction factor contained in the interval (0,1)
* `sufficient_decrease` – (`1e-4`) sufficient decrease parameter contained in the interval (0,1)
* `vector_transport_method` – (`ParallelTransport()`) the vector transport method to use

# Constructor

    NonmonotoneLinesearch()

with the Fields above in their order as optional arguments (deprecated).

    NonmonotoneLinesearch(M)

with the Fields above in their order as keyword arguments and where the retraction
and vector transport are set to the default ones on `M`, repsectively.

The constructors return the functor to perform nonmonotone line search.
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
    storage::StoreStateAction
    linesearch_stopsize::Float64
    function NonmonotoneLinesearch(
        M::AbstractManifold=DefaultManifold(2);
        initial_stepsize::Float64=1.0,
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
            M
        ),
        stepsize_reduction::Float64=0.5,
        sufficient_decrease::Float64=1e-4,
        memory_size::Int=10,
        min_stepsize::Float64=1e-3,
        max_stepsize::Float64=1e3,
        strategy::Symbol=:direct,
        storage::StoreStateAction=StoreStateAction((:Iterate, :gradient)),
        linesearch_stopsize::Float64=0.0,
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
            linesearch_stopsize,
        )
    end
end
function (a::NonmonotoneLinesearch)(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    i::Int,
    η=-get_gradient(mp, get_iterate(s));
    kwargs...,
)
    if !all(has_storage.(Ref(a.storage), [:Iterate, :gradient]))
        old_x = get_iterate(s)
        old_gradient = get_gradient(mp, get_iterate(s))
    else
        old_x, old_gradient = get_storage.(Ref(a.storage), [:Iterate, :gradient])
    end
    update_storage!(a.storage, s)
    return a(
        get_manifold(mp),
        get_iterate(s),
        x -> get_cost(mp, x),
        get_gradient(mp, get_iterate(s)),
        η,
        old_x,
        old_gradient,
        i,
    )
end
function (a::NonmonotoneLinesearch)(
    M::mT, x, F::TF, gradFx::T, η::T, old_x, old_gradient, iter::Int; kwargs...
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
        maximum([a.old_costs[j] for j in 1:min(iter, memory_size)]);
        stop_step=a.linesearch_stopsize,
    )
    return a.initial_stepsize
end

@doc raw"""
    WolfePowellLinesearch <: Linesearch

Do a backtracking linesearch to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``η`` starting from ``x``, i.e.

```math
f\bigl( \operatorname{retr}_x(αη) \bigr) ≤ f(x_k) + c_1 α_k ⟨\operatorname{grad}f(x), η⟩_x
\quad\text{and}\quad
\frac{\mathrm{d}}{\mathrm{d}t} f\bigr(\operatorname{retr}_x(tη)\bigr)
\Big\vert_{t=α}
≥ c_2 \frac{\mathrm{d}}{\mathrm{d}t} f\bigl(\operatorname{retr}_x(tη)\bigr)\Big\vert_{t=0}.
```

# Constructors

There exist two constructors, where, when prodivind the manifold `M` as a first (optional)
parameter, its default retraction and vector transport are the default.
In this case the retraction and the vector transport are also keyword arguments for ease of use.
The other constructor is kept for backward compatibility.
Note that the `linesearch_stopsize` to stop for too small stepsizes is only available in the
new signature including `M`.
For the old (deprecated) signature the `linesearch_stopsize` is set to the old hard-coded default of  `1e-12`

    WolfePowellLinesearch(
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        c1::Float64=10^(-4),
        c2::Float64=0.999
    )

    WolfePowellLinesearch(
        M,
        c1::Float64=10^(-4),
        c2::Float64=0.999;
        retraction_method = default_retraction_method(M),
        vector_transport_method = default_vector_transport(M),
        linesearch_stopsize = 0.0
    )
"""
mutable struct WolfePowellLinesearch{
    TRM<:AbstractRetractionMethod,VTM<:AbstractVectorTransportMethod
} <: Linesearch
    retraction_method::TRM
    vector_transport_method::VTM
    c1::Float64
    c2::Float64
    last_stepsize::Float64
    linesearch_stopsize::Float64

    function WolfePowellLinesearch(
        M::AbstractManifold,
        c1::Float64=10^(-4),
        c2::Float64=0.999;
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
            M
        ),
        linesearch_stopsize::Float64=0.0,
    )
        return new{typeof(retraction_method),typeof(vector_transport_method)}(
            retraction_method, vector_transport_method, c1, c2, 0.0, linesearch_stopsize
        )
    end
end

function (a::WolfePowellLinesearch)(
    p::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    ::Int,
    η=-get_gradient(p, get_iterate(s));
    kwargs...,
)
    step = 1.0
    s_plus = 1.0
    s_minus = 1.0
    f0 = get_cost(p, get_iterate(s))
    xNew = retract(p.M, get_iterate(s), step * η, a.retraction_method)
    fNew = get_cost(p, xNew)
    η_xNew = vector_transport_to(p.M, get_iterate(s), η, xNew, a.vector_transport_method)
    if fNew > f0 + a.c1 * step * inner(p.M, get_iterate(s), η, get_gradient(s))
        while (fNew > f0 + a.c1 * step * inner(p.M, get_iterate(s), η, get_gradient(s))) &&
            (s_minus > 10^(-9)) # decrease
            s_minus = s_minus * 0.5
            s = s_minus
            retract!(p.M, xNew, get_iterate(s), s * η, a.retraction_method)
            fNew = get_cost(p, xNew)
        end
        s_plus = 2.0 * s_minus
    else
        vector_transport_to!(
            p.M, η_xNew, get_iterate(s), η, xNew, a.vector_transport_method
        )
        if inner(p.M, xNew, get_gradient(p, xNew), η_xNew) <
            a.c2 * inner(p.M, get_iterate(s), η, get_gradient(s))
            while fNew <= f0 + a.c1 * s * inner(p.M, get_iterate(s), η, get_gradient(s)) &&
                (s_plus < 10^(9))# increase
                s_plus = s_plus * 2.0
                s = s_plus
                retract!(p.M, xNew, get_iterate(s), s * η, a.retraction_method)
                fNew = get_cost(p, xNew)
            end
            s_minus = s_plus / 2.0
        end
    end
    retract!(p.M, xNew, get_iterate(s), s_minus * η, a.retraction_method)
    vector_transport_to!(p.M, η_xNew, get_iterate(s), η, xNew, a.vector_transport_method)
    while inner(p.M, xNew, get_gradient(p, xNew), η_xNew) <
          a.c2 * inner(p.M, get_iterate(s), η, get_gradient(s))
        s = (s_minus + s_plus) / 2
        retract!(p.M, xNew, get_iterate(s), s * η, a.retraction_method)
        fNew = get_cost(p, xNew)
        if fNew <= f0 + a.c1 * s * inner(p.M, get_iterate(s), η, get_gradient(s))
            s_minus = s
        else
            s_plus = s
        end
        abs(s_plus - s_minus) <= a.linesearch_stopsize && break
        retract!(p.M, xNew, get_iterate(s), s_minus * η, a.retraction_method)
        vector_transport_to!(
            p.M, η_xNew, get_iterate(s), η, xNew, a.vector_transport_method
        )
    end
    s = s_minus
    a.last_stepsize = s
    return s
end

@doc raw"""
    WolfePowellBinaryLinesearch <: Linesearch

A [`Linesearch`](@ref) method that determines a step size `t` fulfilling the Wolfe conditions

based on a binary chop. Let ``η`` be a search direction and ``c1,c_2>0`` be two constants.
Then with

```math
A(t) = f(x_+) ≤ c1 t ⟨\operatorname{grad}f(x), η⟩_{x}
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

# Constructors

There exist two constructors, where, when prodivind the manifold `M` as a first (optional)
parameter, its default retraction and vector transport are the default.
In this case the retraction and the vector transport are also keyword arguments for ease of use.
The other constructor is kept for backward compatibility.
Note that the `linesearch_stopsize` to stop for too small stepsizes is only available in the
new signature including `M`, for the first it is set to the old default of `1e-9`.

    WolfePowellBinaryLinesearch(
        retr::AbstractRetractionMethod=ExponentialRetraction(),
        vtr::AbstractVectorTransportMethod=ParallelTransport(),
        c1::Float64=10^(-4),
        c2::Float64=0.999
    )

    WolfePowellLinesearch(
        M,
        c1::Float64=10^(-4),
        c2::Float64=0.999;
        retraction_method = default_retraction_method(M),
        vector_transport_method = default_vector_transport(M),
        linesearch_stopsize = 0.0
    )

[^Huang2014]:
    > Huang, W.: _Optimization algorithms on Riemannian manifolds with applications_,
    > Dissertation, Flordia State University, 2014.
    > [pdf](https://www.math.fsu.edu/~whuang2/pdf/Huang_W_Dissertation_2013.pdf)
"""
mutable struct WolfePowellBinaryLinesearch{
    TRM<:AbstractRetractionMethod,VTM<:AbstractVectorTransportMethod
} <: Linesearch
    retraction_method::TRM
    vector_transport_method::VTM
    c1::Float64
    c2::Float64
    last_stepsize::Float64
    linesearch_stopsize::Float64

    function WolfePowellBinaryLinesearch(
        M::AbstractManifold,
        c1::Float64=10^(-4),
        c2::Float64=0.999;
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
            M
        ),
        linesearch_stopsize::Float64=0.0,
    )
        return new{typeof(retraction_method),typeof(vector_transport_method)}(
            retraction_method, vector_transport_method, c1, c2, 0.0, linesearch_stopsize
        )
    end
end

function (a::WolfePowellBinaryLinesearch)(
    p::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    ::Int,
    η=-get_gradient(p, get_iterate(s));
    kwargs...,
)
    α = 0.0
    β = Inf
    t = 1.0
    f0 = get_cost(p, get_iterate(s))
    xNew = retract(p.M, get_iterate(s), t * η, a.retraction_method)
    fNew = get_cost(p, xNew)
    η_xNew = vector_transport_to(p.M, get_iterate(s), η, xNew, a.vector_transport_method)
    gradient_new = get_gradient(p, xNew)
    nAt = fNew > f0 + a.c1 * t * inner(p.M, get_iterate(s), η, get_gradient(s))
    nWt =
        inner(p.M, xNew, gradient_new, η_xNew) <
        a.c2 * inner(p.M, get_iterate(s), η, get_gradient(s))
    while (nAt || nWt) &&
              (t > a.linesearch_stopsize) &&
              ((α + β) / 2 - 1 > a.linesearch_stopsize)
        nAt && (β = t)            # A(t) fails
        (!nAt && nWt) && (α = t)  # A(t) holds but W(t) fails
        t = isinf(β) ? 2 * α : (α + β) / 2
        # Update trial point
        retract!(p.M, xNew, get_iterate(s), t * η, a.retraction_method)
        fNew = get_cost(p, xNew)
        gradient_new = get_gradient(p, xNew)
        vector_transport_to!(
            p.M, η_xNew, get_iterate(s), η, xNew, a.vector_transport_method
        )
        # Update conditions
        nAt = fNew > f0 + a.c1 * t * inner(p.M, get_iterate(s), η, get_gradient(s))
        nWt =
            inner(p.M, xNew, gradient_new, η_xNew) <
            a.c2 * inner(p.M, get_iterate(s), η, get_gradient(s))
    end
    a.last_stepsize = t
    return t
end

@doc raw"""
    get_stepsize(p::AbstractManoptProblem, s::AbstractManoptSolverState, vars...)

return the stepsize stored within [`AbstractManoptSolverState`](@ref) `o` when solving [`Problem`](@ref) `p`.
This method also works for decorated options and the [`Stepsize`](@ref) function within
the options, by default stored in `o.stepsize`.
"""
function get_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, vars...; kwargs...
)
    return _get_stepsize(p, s, dispatch_state_decorator(s), vars...; kwargs...)
end
function _get_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{true}, vars...; kwargs...
)
    return get_stepsize(p, s.state, vars...; kwargs...)
end
function _get_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{false}, vars...; kwargs...
)
    return s.stepsize(p, s, vars...; kwargs...)
end

function get_initial_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, vars...; kwargs...
)
    return get_initial_stepsize(
        p::AbstractManoptProblem,
        s::AbstractManoptSolverState,
        dispatch_state_decorator(s),
        vars...;
        kwargs...,
    )
end
function _get_initial_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{true}, vars...
)
    return get_initial_stepsize(p, s.state)
end
function _get_initial_stepsize(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{false}, vars...
)
    return get_initial_stepsize(s.stepsize)
end

function get_last_stepsize(p::AbstractManoptProblem, s::AbstractManoptSolverState, vars...)
    return _get_last_stepsize(p, s, dispatch_state_decorator(s), vars...)
end
function _get_last_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{true}, vars...
)
    return get_last_stepsize(p, s.state, vars...)
end
function _get_last_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{false}, vars...
)
    return get_last_stepsize(p, s, s.stepsize, vars...)
end
#
# dispatch on stepsize
function get_last_stepsize(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, step::Stepsize, vars...
)
    return step(p, s, vars...)
end
function get_last_stepsize(
    ::AbstractManoptProblem, ::AbstractManoptSolverState, step::ArmijoLinesearch, ::Any...
)
    return step.last_stepsize
end
function get_last_stepsize(
    ::AbstractManoptProblem,
    ::AbstractManoptSolverState,
    step::WolfePowellLinesearch,
    ::Any...,
)
    return step.last_stepsize
end
function get_last_stepsize(
    ::AbstractManoptProblem,
    ::AbstractManoptSolverState,
    step::WolfePowellBinaryLinesearch,
    ::Any...,
)
    return step.last_stepsize
end
