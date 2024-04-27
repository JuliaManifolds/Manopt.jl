"""
    Stepsize

An abstract type for the functors representing step sizes. These are callable
structures. The naming scheme is `TypeOfStepSize`, for example `ConstantStepsize`.

Every Stepsize has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`AbstractManoptProblem`](@ref) as well as [`AbstractManoptSolverState`](@ref)
and the current number of iterations are the arguments
and returns a number, namely the stepsize to use.

# See also

[`Linesearch`](@ref)
"""
abstract type Stepsize end

get_message(::S) where {S<:Stepsize} = ""

"""
    default_stepsize(M::AbstractManifold, ams::AbstractManoptSolverState)

Returns the default [`Stepsize`](@ref) functor used when running the solver specified by the
[`AbstractManoptSolverState`](@ref) `ams` running with an objective on the `AbstractManifold M`.
"""
default_stepsize(M::AbstractManifold, sT::Type{<:AbstractManoptSolverState})

"""
    max_stepsize(M::AbstractManifold, p)
    max_stepsize(M::AbstractManifold)

Get the maximum stepsize (at point `p`) on manifold `M`. It should be used to limit the
distance an algorithm is trying to move in a single step.
"""
function max_stepsize(M::AbstractManifold, p)
    return max_stepsize(M)
end
function max_stepsize(M::AbstractManifold)
    return injectivity_radius(M)
end

"""
    ConstantStepsize <: Stepsize

A functor that always returns a fixed step size.

# Fields

* `length`: constant value for the step size
* `type`:   a symbol that indicates whether the stepsize is relatively (:relative),
    with respect to the gradient norm, or absolutely (:absolute) constant.

# Constructors

    ConstantStepsize(s::Real, t::Symbol=:relative)

initialize the stepsize to a constant `s` of type `t`.

    ConstantStepsize(M::AbstractManifold=DefaultManifold(2);
        stepsize=injectivity_radius(M)/2, type::Symbol=:relative
    )

initialize the stepsize to a constant `stepsize`, which by default is half the injectivity
radius, unless the radius is infinity, then the default step size is `1`.
"""
mutable struct ConstantStepsize{T} <: Stepsize
    length::T
    type::Symbol
end
function ConstantStepsize(
    M::AbstractManifold=DefaultManifold(2);
    stepsize=isinf(injectivity_radius(M)) ? 1.0 : injectivity_radius(M) / 2,
    type=:relative,
)
    return ConstantStepsize{typeof(stepsize)}(stepsize, type)
end
function ConstantStepsize(stepsize::T) where {T<:Number}
    return ConstantStepsize{T}(stepsize, :relative)
end
function (cs::ConstantStepsize)(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Any, args...; kwargs...
)
    s = cs.length
    if cs.type == :absolute
        ns = norm(get_manifold(amp), get_iterate(ams), get_gradient(ams))
        if ns > eps(eltype(s))
            s /= ns
        end
    end
    return s
end
get_initial_stepsize(s::ConstantStepsize) = s.length
show(io::IO, cs::ConstantStepsize) = print(io, "ConstantStepsize($(cs.length), $(cs.type))")

@doc raw"""
    DecreasingStepsize()

A functor that represents several decreasing step sizes

# Fields

* `exponent`:   (`1`) a value ``e`` the current iteration numbers ``e``th exponential is
  taken of
* `factor`:     (`1`) a value ``f`` to multiply the initial step size with every iteration
* `length`:     (`1`) the initial step size ``l``.
* `subtrahend`: (`0`) a value ``a`` that is subtracted every iteration
* `shift`:      (`0`) shift the denominator iterator ``i`` by ``s```.
* `type`:       a symbol that indicates whether the stepsize is relatively (:relative),
    with respect to the gradient norm, or absolutely (:absolute) constant.

In total the complete formulae reads for the ``i``th iterate as

````math
s_i = \frac{(l - i a)f^i}{(i+s)^e}
````

and hence the default simplifies to just ``s_i = \frac{l}{i}``

# Constructor

    DecreasingStepsize(l=1,f=1,a=0,e=1,s=0,type=:relative)

Alternatively one can also use the following keyword.

    DecreasingStepsize(
        M::AbstractManifold=DefaultManifold(3);
        length=injectivity_radius(M)/2, multiplier=1.0, subtrahend=0.0,
        exponent=1.0, shift=0, type=:relative
    )

initializes all fields, where none of them is mandatory and the length is set to
half and to ``1`` if the injectivity radius is infinite.
"""
mutable struct DecreasingStepsize <: Stepsize
    length::Float64
    factor::Float64
    subtrahend::Float64
    exponent::Float64
    shift::Int
    type::Symbol
    function DecreasingStepsize(
        l::Real, f::Real=1.0, a::Real=0.0, e::Real=1.0, s::Int=0, type::Symbol=:relative
    )
        return new(l, f, a, e, s, type)
    end
end
function DecreasingStepsize(
    M::AbstractManifold=DefaultManifold(3);
    length=isinf(manifold_dimension(M)) ? 1.0 : manifold_dimension(M) / 2,
    factor=1.0,
    subtrahend=0.0,
    exponent=1.0,
    shift=0,
    type::Symbol=:relative,
)
    return DecreasingStepsize(length, factor, subtrahend, exponent, shift, type)
end
function (s::DecreasingStepsize)(
    amp::P, ams::O, i::Int, args...; kwargs...
) where {P<:AbstractManoptProblem,O<:AbstractManoptSolverState}
    ds = (s.length - i * s.subtrahend) * (s.factor^i) / ((i + s.shift)^(s.exponent))
    if s.type == :absolute
        ns = norm(get_manifold(amp), get_iterate(ams), get_gradient(ams))
        if ns > eps(eltype(ds))
            ds /= ns
        end
    end
    return ds
end
get_initial_stepsize(s::DecreasingStepsize) = s.length
function show(io::IO, s::DecreasingStepsize)
    return print(
        io,
        "DecreasingStepsize(; length=$(s.length),  factor=$(s.factor),  subtrahend=$(s.subtrahend),  shift=$(s.shift))",
    )
end
"""
    Linesearch <: Stepsize

An abstract functor to represent line search type step size determinations, see
[`Stepsize`](@ref) for details. One example is the [`ArmijoLinesearch`](@ref)
functor.

Compared to simple step sizes, the line search functors provide an interface of
the form `(p,o,i,η) -> s` with an additional (but optional) fourth parameter to
provide a search direction; this should default to something reasonable,
most prominently the negative gradient.
"""
abstract type Linesearch <: Stepsize end

function armijo_initial_guess(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, ::Int, l::Real
)
    M = get_manifold(mp)
    X = get_gradient(s)
    p = get_iterate(s)
    grad_norm = norm(M, p, X)
    max_step = max_stepsize(M, p)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

@doc raw"""
    ArmijoLinesearch <: Linesearch

A functor representing Armijo line search including the last runs state string the last stepsize.

# Fields

* `initial_stepsize`:          (`1.0`) and initial step size
* `retraction_method`:         (`default_retraction_method(M)`) the retraction to use
* `contraction_factor`:        (`0.95`) exponent for line search reduction
* `sufficient_decrease`:       (`0.1`) gain within Armijo's rule
* `last_stepsize`:             (`initialstepsize`) the last step size to start the search with
* `initial_guess`:             (`(p,s,i,l) -> l`)  based on a [`AbstractManoptProblem`](@ref) `p`,
  [`AbstractManoptSolverState`](@ref) `s` and a current iterate `i` and a last step size `l`,
  this returns an initial guess. The default uses the last obtained stepsize

as well as for internal use

* `candidate_point`:           (`allocate_result(M, rand)`) to store an interim result

Furthermore the following fields act as safeguards

* `stop_when_stepsize_less`:    (`0.0`) smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds`: ([`max_stepsize`](@ref)`(M, p)`) largest stepsize when to stop.
* `stop_increasing_at_step`:    (`100`) last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step`:    (`1000`) last step size to decrease the stepsize (phase 2),

Pass `:Messages` to a `debug=` to see `@info`s when these happen.

# Constructor

    ArmijoLinesearch(M=DefaultManifold())

with the fields keyword arguments and the retraction is set to the default retraction on `M`.

The constructors return the functor to perform Armijo line search, where

    (a::ArmijoLinesearch)(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i)

of a [`AbstractManoptProblem`](@ref) `amp`, [`AbstractManoptSolverState`](@ref) `ams` and a current iterate `i`
with keywords.

## Keyword arguments

  * `candidate_point`: (`allocate_result(M, rand)`) to pass memory for the candidate point
  * `η`:               (`-get_gradient(mp, get_iterate(s));`) the search direction to use,
  by default the steepest descent direction.
"""
mutable struct ArmijoLinesearch{TRM<:AbstractRetractionMethod,P,F} <: Linesearch
    candidate_point::P
    contraction_factor::Float64
    initial_guess::F
    initial_stepsize::Float64
    last_stepsize::Float64
    message::String
    retraction_method::TRM
    sufficient_decrease::Float64
    stop_when_stepsize_less::Float64
    stop_when_stepsize_exceeds::Float64
    stop_increasing_at_step::Int
    stop_decreasing_at_step::Int
    function ArmijoLinesearch(
        M::AbstractManifold=DefaultManifold();
        candidate_point::P=allocate_result(M, rand),
        contraction_factor::Real=0.95,
        initial_stepsize::Real=1.0,
        initial_guess=armijo_initial_guess,
        retraction_method::TRM=default_retraction_method(M),
        stop_when_stepsize_less::Real=0.0,
        stop_when_stepsize_exceeds::Real=max_stepsize(M),
        stop_increasing_at_step::Int=100,
        stop_decreasing_at_step::Int=1000,
        sufficient_decrease=0.1,
    ) where {TRM,P}
        return new{TRM,P,typeof(initial_guess)}(
            candidate_point,
            contraction_factor,
            initial_guess,
            initial_stepsize,
            initial_stepsize,
            "", # initialize an empty message
            retraction_method,
            sufficient_decrease,
            stop_when_stepsize_less,
            stop_when_stepsize_exceeds,
            stop_increasing_at_step,
            stop_decreasing_at_step,
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
    p = get_iterate(s)
    X = get_gradient!(mp, get_gradient(s), p)
    (a.last_stepsize, a.message) = linesearch_backtrack!(
        get_manifold(mp),
        a.candidate_point,
        (M, p) -> get_cost_function(get_objective(mp))(M, p),
        p,
        X,
        a.initial_guess(mp, s, i, a.last_stepsize),
        a.sufficient_decrease,
        a.contraction_factor,
        η;
        retraction_method=a.retraction_method,
        stop_when_stepsize_less=a.stop_when_stepsize_less / norm(get_manifold(mp), p, η),
        stop_when_stepsize_exceeds=a.stop_when_stepsize_exceeds /
                                   norm(get_manifold(mp), p, η),
        stop_increasing_at_step=a.stop_increasing_at_step,
        stop_decreasing_at_step=a.stop_decreasing_at_step,
    )
    return a.last_stepsize
end
get_initial_stepsize(a::ArmijoLinesearch) = a.initial_stepsize
function show(io::IO, als::ArmijoLinesearch)
    return print(
        io,
        """
        ArmijoLinesearch() with keyword parameters
          * initial_stepsize    = $(als.initial_stepsize)
          * retraction_method   = $(als.retraction_method)
          * contraction_factor  = $(als.contraction_factor)
          * sufficient_decrease = $(als.sufficient_decrease)""",
    )
end
function status_summary(als::ArmijoLinesearch)
    return "$(als)\nand a computed last stepsize of $(als.last_stepsize)"
end
get_message(a::ArmijoLinesearch) = a.message

@doc raw"""
    (s, msg) = linesearch_backtrack(M, F, p, X, s, decrease, contract η = -X, f0 = f(p))
    (s, msg) = linesearch_backtrack!(M, q, F, p, X, s, decrease, contract η = -X, f0 = f(p))

perform a line search

* on manifold `M`
* for the cost function `f`,
* at the current point `p`
* with current gradient provided in `X`
* an initial stepsize `s`
* a sufficient `decrease`
* a `contract`ion factor ``σ``
* a `retr`action, which defaults to the `default_retraction_method(M)`
* a search direction ``η = -X``
* an offset, ``f_0 = F(x)``

the method can also be performed in-place of `q`, that is the resulting best point one reaches
with the step size `s` as second argument.

## Keywords

* `retraction_method`:          (`default_retraction_method(M)`) the retraction to use.
* `stop_when_stepsize_less`:    (`0.0`) to avoid numerical underflow
* `stop_when_stepsize_exceeds`: ([`max_stepsize`](@ref)`(M, p) / norm(M, p, η)`) to avoid leaving the injectivity radius on a manifold
* `stop_increasing_at_step`:    (`100`) stop the initial increase of step size after these many steps
* `stop_decreasing_at_step`:    (`1000`) stop the decreasing search after these many steps

These keywords are used as safeguards, where only the max stepsize is a very manifold specific one.

# Return value

A stepsize `s` and a message `msg` (in case any of the 4 criteria hit)
"""
function linesearch_backtrack(
    M::AbstractManifold, f, p, X::T, s, decrease, contract, η::T=-X, f0=f(M, p); kwargs...
) where {T}
    q = allocate(M, p)
    return linesearch_backtrack!(M, q, f, p, X, s, decrease, contract, η, f0; kwargs...)
end

"""
    (s, msg) = linesearch_backtrack!(M, q, F, p, X, s, decrease, contract η = -X, f0 = f(p))

Perform a line search backtrack in-place of `q`.
For all details and options, see [`linesearch_backtrack`](@ref)
"""
function linesearch_backtrack!(
    M::AbstractManifold,
    q,
    f::TF,
    p,
    X::T,
    s,
    decrease,
    contract,
    η::T=-X,
    f0=f(M, p);
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stop_when_stepsize_less=0.0,
    stop_when_stepsize_exceeds=max_stepsize(M, p) / norm(M, p, η),
    stop_increasing_at_step=100,
    stop_decreasing_at_step=1000,
) where {TF,T}
    msg = ""
    retract!(M, q, p, η, s, retraction_method)
    f_q = f(M, q)
    search_dir_inner = real(inner(M, p, η, X))
    if search_dir_inner >= 0
        msg = "The search direction η might not be a descent direction, since ⟨η, grad_f(p)⟩ ≥ 0."
    end
    i = 0
    while f_q < f0 + decrease * s * search_dir_inner # increase
        i = i + 1
        s = s / contract
        retract!(M, q, p, η, s, retraction_method)
        f_q = f(M, q)
        if i == stop_increasing_at_step
            (length(msg) > 0) && (msg = "$msg\n")
            msg = "$(msg)Max increase steps ($(stop_increasing_at_step)) reached"
            break
        end
        if s > stop_when_stepsize_exceeds
            (length(msg) > 0) && (msg = "$msg\n")
            s = s * contract
            msg = "$(msg)Max step size ($(stop_when_stepsize_exceeds)) reached, reducing to $s"
            break
        end
    end
    i = 0
    while f_q > f0 + decrease * s * search_dir_inner # decrease
        i = i + 1
        s = contract * s
        retract!(M, q, p, η, s, retraction_method)
        f_q = f(M, q)
        if i == stop_decreasing_at_step
            (length(msg) > 0) && (msg = "$msg\n")
            msg = "$(msg)Max decrease steps ($(stop_decreasing_at_step)) reached"
            break
        end
        if s < stop_when_stepsize_less
            (length(msg) > 0) && (msg = "$msg\n")
            s = s / contract
            msg = "$(msg)Min step size ($(stop_when_stepsize_less)) exceeded, increasing back to $s"
            break
        end
    end
    return (s, msg)
end

@doc raw"""
    NonmonotoneLinesearch <: Linesearch

A functor representing a nonmonotone line search using the Barzilai-Borwein step size [IannazzoPorcelli:2017](@cite).
Together with a gradient descent algorithm this line search represents the Riemannian Barzilai-Borwein with nonmonotone line-search (RBBNMLS) algorithm.
The order is shifted in comparison of the algorithm steps from the paper
by Iannazzo and Porcelli so that in each iteration this line search first finds

```math
y_{k} = \operatorname{grad}F(x_{k}) - \operatorname{T}_{x_{k-1} → x_k}(\operatorname{grad}F(x_{k-1}))
```

and

```math
s_{k} = - α_{k-1} * \operatorname{T}_{x_{k-1} → x_k}(\operatorname{grad}F(x_{k-1})),
```

where ``α_{k-1}`` is the step size computed in the last iteration and ``\operatorname{T}`` is a vector transport.
Then the Barzilai—Borwein step size is

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
alternating strategy. Then find the smallest ``h = 0, 1, 2, …`` such that

```math
F(\operatorname{retr}_{x_k}(- σ^h α_k^{\text{BB}} \operatorname{grad}F(x_k)))
\leq
\max_{1 ≤ j ≤ \min(k+1,m)} F(x_{k+1-j}) - γ σ^h α_k^{\text{BB}} ⟨\operatorname{grad}F(x_k), \operatorname{grad}F(x_k)⟩_{x_k},
```

where ``σ`` is a step length reduction factor ``∈ (0,1)``, ``m`` is the number of iterations
after which the function value has to be lower than the current one
and ``γ`` is the sufficient decrease parameter ``∈(0,1)``.
Then find the new stepsize by

```math
α_k = σ^h α_k^{\text{BB}}.
```

# Fields

* `initial_stepsize`:        (`1.0`) the step size to start the search with
* `memory_size`:             (`10`) number of iterations after which the cost value needs to be lower than the current one
* `bb_min_stepsize`:         (`1e-3`) lower bound for the Barzilai-Borwein step size greater than zero
* `bb_max_stepsize`:         (`1e3`) upper bound for the Barzilai-Borwein step size greater than min_stepsize
* `retraction_method`:       (`ExponentialRetraction()`) the retraction to use
* `strategy`:                (`direct`) defines if the new step size is computed using the direct, indirect or alternating strategy
* `storage`:                 (for `:Iterate` and `:Gradient`) a [`StoreStateAction`](@ref)
* `stepsize_reduction`:      (`0.5`) step size reduction factor contained in the interval (0,1)
* `sufficient_decrease`:     (`1e-4`) sufficient decrease parameter contained in the interval (0,1)
* `vector_transport_method`: (`ParallelTransport()`) the vector transport method to use

as well as for internal use

* `candidate_point`:           (`allocate_result(M, rand)`) to store an interim result

Furthermore the following fields act as safeguards

* `stop_when_stepsize_less:     (`0.0`) smallest stepsize when to stop (the last one before is taken)
* `stop_when_stepsize_exceeds`: ([`max_stepsize`](@ref)`(M, p)`) largest stepsize when to stop.
* `stop_increasing_at_step`:    (^100`) last step to increase the stepsize (phase 1),
* `stop_decreasing_at_step`:    (`1000`) last step size to decrease the stepsize (phase 2),

Pass `:Messages` to a `debug=` to see `@info`s when these happen.

# Constructor

    NonmonotoneLinesearch()

with the fields their order as optional arguments (deprecated).
THis is deprecated, since both defaults and the memory allocation for the candidate do
not take into account which manifold the line search operates on.

    NonmonotoneLinesearch(M)

with the fields as keyword arguments and where the retraction
and vector transport are set to the default ones on `M`, respectively.

The constructors return the functor to perform nonmonotone line search.
"""
mutable struct NonmonotoneLinesearch{
    TRM<:AbstractRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
    T<:AbstractVector,
    TSSA<:StoreStateAction,
    P,
} <: Linesearch
    bb_min_stepsize::Float64
    bb_max_stepsize::Float64
    candiate_point::P
    initial_stepsize::Float64
    message::String
    old_costs::T
    retraction_method::TRM
    stepsize_reduction::Float64
    stop_decreasing_at_step::Int
    stop_increasing_at_step::Int
    stop_when_stepsize_exceeds::Float64
    stop_when_stepsize_less::Float64
    storage::TSSA
    strategy::Symbol
    sufficient_decrease::Float64
    vector_transport_method::VTM
    function NonmonotoneLinesearch(
        M::AbstractManifold=DefaultManifold();
        bb_min_stepsize::Float64=1e-3,
        bb_max_stepsize::Float64=1e3,
        candidate_point::P=allocate_result(M, rand),
        initial_stepsize::Float64=1.0,
        memory_size::Int=10,
        retraction_method::TRM=default_retraction_method(M),
        stepsize_reduction::Float64=0.5,
        stop_when_stepsize_less::Float64=0.0,
        stop_when_stepsize_exceeds::Float64=max_stepsize(M),
        stop_increasing_at_step::Int=100,
        stop_decreasing_at_step::Int=1000,
        storage::Union{Nothing,StoreStateAction}=StoreStateAction(
            M; store_fields=[:Iterate, :Gradient]
        ),
        strategy::Symbol=:direct,
        sufficient_decrease::Float64=1e-4,
        vector_transport_method::VTM=default_vector_transport_method(M),
    ) where {TRM,VTM,P}
        if strategy ∉ [:direct, :inverse, :alternating]
            @warn string(
                "The strategy '",
                strategy,
                "' is not defined. The 'direct' strategy is used instead.",
            )
            strategy = :direct
        end
        if bb_min_stepsize <= 0.0
            throw(
                DomainError(
                    bb_min_stepsize,
                    "The lower bound for the step size min_stepsize has to be greater than zero.",
                ),
            )
        end
        if bb_max_stepsize <= bb_min_stepsize
            throw(
                DomainError(
                    bb_max_stepsize,
                    "The upper bound for the step size max_stepsize has to be greater its lower bound min_stepsize.",
                ),
            )
        end
        if memory_size <= 0
            throw(DomainError(memory_size, "The memory_size has to be greater than zero."))
        end
        return new{TRM,VTM,Vector{Float64},typeof(storage),P}(
            bb_min_stepsize,
            bb_max_stepsize,
            candidate_point,
            initial_stepsize,
            "",
            zeros(memory_size),
            retraction_method,
            stepsize_reduction,
            stop_decreasing_at_step,
            stop_increasing_at_step,
            stop_when_stepsize_exceeds,
            stop_when_stepsize_less,
            storage,
            strategy,
            sufficient_decrease,
            vector_transport_method,
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
    if !has_storage(a.storage, PointStorageKey(:Iterate)) ||
        !has_storage(a.storage, VectorStorageKey(:Gradient))
        p_old = get_iterate(s)
        X_old = get_gradient(mp, p_old)
    else
        #fetch
        p_old = get_storage(a.storage, PointStorageKey(:Iterate))
        X_old = get_storage(a.storage, VectorStorageKey(:Gradient))
    end
    update_storage!(a.storage, mp, s)
    return a(
        get_manifold(mp),
        get_iterate(s),
        (M, p) -> get_cost(M, get_objective(mp), p),
        get_gradient(mp, get_iterate(s)),
        η,
        p_old,
        X_old,
        i,
    )
end
function (a::NonmonotoneLinesearch)(
    M::mT, p, f::TF, X::T, η::T, old_p, old_X, iter::Int; kwargs...
) where {mT<:AbstractManifold,TF,T}
    #find the difference between the current and previous gradient after the previous gradient is transported to the current tangent space
    grad_diff = X - vector_transport_to(M, old_p, old_X, p, a.vector_transport_method)
    #transport the previous step into the tangent space of the current manifold point
    x_diff =
        -a.initial_stepsize *
        vector_transport_to(M, old_p, old_X, p, a.vector_transport_method)

    #compute the new Barzilai-Borwein step size
    s1 = real(inner(M, p, x_diff, grad_diff))
    s2 = real(inner(M, p, grad_diff, grad_diff))
    s2 = s2 == 0 ? 1.0 : s2
    s3 = real(inner(M, p, x_diff, x_diff))
    #indirect strategy
    if a.strategy == :inverse
        if s1 > 0
            BarzilaiBorwein_stepsize = min(
                a.bb_max_stepsize, max(a.bb_min_stepsize, s1 / s2)
            )
        else
            BarzilaiBorwein_stepsize = a.bb_max_stepsize
        end
        #alternating strategy
    elseif a.strategy == :alternating
        if s1 > 0
            if iter % 2 == 0
                BarzilaiBorwein_stepsize = min(
                    a.bb_max_stepsize, max(a.bb_min_stepsize, s1 / s2)
                )
            else
                BarzilaiBorwein_stepsize = min(
                    a.bb_max_stepsize, max(a.bb_min_stepsize, s3 / s1)
                )
            end
        else
            BarzilaiBorwein_stepsize = a.bb_max_stepsize
        end
        #direct strategy
    else
        if s1 > 0
            BarzilaiBorwein_stepsize = min(
                a.bb_max_stepsize, max(a.bb_min_stepsize, s2 / s1)
            )
        else
            BarzilaiBorwein_stepsize = a.bb_max_stepsize
        end
    end

    memory_size = length(a.old_costs)
    if iter <= memory_size
        a.old_costs[iter] = f(M, p)
    else
        a.old_costs[1:(memory_size - 1)] = a.old_costs[2:memory_size]
        a.old_costs[memory_size] = f(M, p)
    end

    #compute the new step size with the help of the Barzilai-Borwein step size
    (a.initial_stepsize, a.message) = linesearch_backtrack!(
        M,
        a.candiate_point,
        f,
        p,
        X,
        BarzilaiBorwein_stepsize,
        a.sufficient_decrease,
        a.stepsize_reduction,
        η,
        maximum([a.old_costs[j] for j in 1:min(iter, memory_size)]);
        retraction_method=a.retraction_method,
        stop_when_stepsize_less=a.stop_when_stepsize_less / norm(M, p, η),
        stop_when_stepsize_exceeds=a.stop_when_stepsize_exceeds / norm(M, p, η),
        stop_increasing_at_step=a.stop_increasing_at_step,
        stop_decreasing_at_step=a.stop_decreasing_at_step,
    )
    return a.initial_stepsize
end
function show(io::IO, a::NonmonotoneLinesearch)
    return print(
        io,
        """
        NonmonotoneLinesearch() with keyword arguments
          * initial_stepsize = $(a.initial_stepsize)
          * bb_max_stepsize = $(a.bb_max_stepsize)
          * bb_min_stepsize = $(a.bb_min_stepsize),
          * memory_size = $(length(a.old_costs))
          * stepsize_reduction = $(a.stepsize_reduction)
          * strategy = :$(a.strategy)
          * sufficient_decrease = $(a.sufficient_decrease)
          * retraction_method = $(a.retraction_method)
          * vector_transport_method = $(a.vector_transport_method)""",
    )
end
get_message(a::NonmonotoneLinesearch) = a.message

@doc raw"""
    PolyakStepsize <: Linesearch


Compute a step size

````math
α_i = \frac{f(p^{(i)}) - f_{\text{best}}+γ_i}{\lVert{ ∂f(p^{(i)})} \rVert^2},
````

where ``f_{\text{best}}`` is the best cost value seen until the current iteration,
and ``γ_i`` is a sequence of numbers that is square summable, but not summable (the sum must diverge to infinity).
Furthermore ``∂f`` denotes a subgradient of ``f`` at the current iterate ``p^{(i)}``.

The step size is an adaption from the Dynamic step size discussed in Section 3.2 of [Bertsekas:2015](@cite),
both to the Riemannian case and to approximate the minimum cost value.

# Fields

* `γ`               : a function `i -> ...` representing the sequence.
* `best_cost_value` : storing the value `f_{\text{best}}`

# Constructor

    PolyakStepsize(;
        γ = i -> 1/i,
        initial_cost_estimate=0.0
    )

initialize the Polyak stepsize to a certain sequence and an initial estimate of ``f_{\text{best}}``
"""
mutable struct PolyakStepsize{F,R} <: Stepsize
    γ::F
    best_cost_value::R
end
function PolyakStepsize(; γ::F=(i) -> 1 / i, initial_cost_estimate::R=0.0) where {F,R}
    return PolyakStepsize{F,R}(γ, initial_cost_estimate)
end
function (ps::PolyakStepsize)(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i::Int, args...; kwargs...
)
    # We get these by reference, so that should not allocate in general
    M = get_manifold(amp)
    p = get_iterate(ams)
    X = get_subgradient(amp, p)
    # Evaluate the cost
    c = get_cost(M, get_objective(amp), p)
    (c < ps.best_cost_value) && (ps.best_cost_value = c)
    α = (c - ps.best_cost_value + ps.γ(i)) / (norm(M, p, X)^2)
    return α
end
function show(io::IO, ps::PolyakStepsize)
    return print(
        io,
        "PolyakStepsize() with keyword parameters\n  * initial_cost_estimate = $(ps.best_cost_value)",
    )
end

@doc raw"""
    WolfePowellLinesearch <: Linesearch

Do a backtracking line search to find a step size ``α`` that fulfils the
Wolfe conditions along a search direction ``η`` starting from ``x`` by

```math
f\bigl( \operatorname{retr}_x(αη) \bigr) ≤ f(x_k) + c_1 α_k ⟨\operatorname{grad}f(x), η⟩_x
\quad\text{and}\quad
\frac{\mathrm{d}}{\mathrm{d}t} f\bigr(\operatorname{retr}_x(tη)\bigr)
\Big\vert_{t=α}
≥ c_2 \frac{\mathrm{d}}{\mathrm{d}t} f\bigl(\operatorname{retr}_x(tη)\bigr)\Big\vert_{t=0}.
```

# Constructors

There exist two constructors, where, when provided the manifold `M` as a first (optional)
parameter, its default retraction and vector transport are the default.
In this case the retraction and the vector transport are also keyword arguments for ease of use.
The other constructor is kept for backward compatibility.
Note that the `stop_when_stepsize_less` to stop for too small stepsizes is only available in the
new signature including `M`.

    WolfePowellLinesearch(M, c1::Float64=10^(-4), c2::Float64=0.999; kwargs...

Generate a Wolfe-Powell line search

## Keyword arguments

* `candidate_point`:         (`allocate_result(M, rand)`) memory for a candidate
* `candidate_tangent`:       (`allocate_result(M, zero_vector, candidate_point)`) memory for a gradient
* `candidate_direcntion`:    (`allocate_result(M, zero_vector, candidate_point)`) memory for a direction
* `max_stepsize`:            ([`max_stepsize`](@ref)`(M, p)`) largest stepsize allowed here.
* `retraction_method`:       (`ExponentialRetraction()`) the retraction to use
* `stop_when_stepsize_less`: (`0.0`) smallest stepsize when to stop (the last one before is taken)
* `vector_transport_method`: (`ParallelTransport()`) the vector transport method to use
"""
mutable struct WolfePowellLinesearch{
    TRM<:AbstractRetractionMethod,VTM<:AbstractVectorTransportMethod,P,T
} <: Linesearch
    c1::Float64
    c2::Float64
    candidate_direction::T
    candidate_point::P
    candidate_tangent::T
    last_stepsize::Float64
    max_stepsize::Float64
    retraction_method::TRM
    stop_when_stepsize_less::Float64
    vector_transport_method::VTM

    function WolfePowellLinesearch(
        M::AbstractManifold=DefaultManifold(),
        c1::Float64=10^(-4),
        c2::Float64=0.999;
        candidate_point::P=allocate_result(M, rand),
        candidate_tangent::T=allocate_result(M, zero_vector, candidate_point),
        candidate_direction::T=allocate_result(M, zero_vector, candidate_point),
        max_stepsize::Real=max_stepsize(M, candidate_point),
        retraction_method::TRM=default_retraction_method(M),
        vector_transport_method::VTM=default_vector_transport_method(M),
        linesearch_stopsize::Float64=0.0,            # deprecated remove on next breaking change
        stop_when_stepsize_less::Float64=linesearch_stopsize, #
    ) where {TRM,VTM,P,T}
        (linesearch_stopsize > 0.0) && Base.depwarn(
            WolfePowellLinesearch,
            "`linesearch_backtrack` is deprecated – use `stop_when_stepsize_less` instead´.",
        )
        return new{TRM,VTM,P,T}(
            c1,
            c2,
            candidate_direction,
            candidate_point,
            candidate_tangent,
            0.0,
            max_stepsize,
            retraction_method,
            stop_when_stepsize_less,
            vector_transport_method,
        )
    end
end
function (a::WolfePowellLinesearch)(
    mp::AbstractManoptProblem,
    ams::AbstractManoptSolverState,
    ::Int,
    η=-get_gradient(mp, get_iterate(ams));
    kwargs...,
)
    # For readability extract a few variables
    M = get_manifold(mp)
    p = get_iterate(ams)
    X = get_gradient(ams)
    l = real(inner(M, p, η, X))
    grad_norm = norm(M, p, η)
    max_step_increase = ifelse(
        isfinite(a.max_stepsize), min(1e9, a.max_stepsize / grad_norm), 1e9
    )
    step = ifelse(isfinite(a.max_stepsize), min(1.0, a.max_stepsize / grad_norm), 1.0)
    s_plus = step
    s_minus = step

    f0 = get_cost(mp, p)
    retract!(M, a.candidate_point, p, η, step, a.retraction_method)
    fNew = get_cost(mp, a.candidate_point)
    vector_transport_to!(
        M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method
    )
    if fNew > f0 + a.c1 * step * l
        while (fNew > f0 + a.c1 * step * l) && (s_minus > 10^(-9)) # decrease
            s_minus = s_minus * 0.5
            step = s_minus
            retract!(M, a.candidate_point, p, η, step, a.retraction_method)
            fNew = get_cost(mp, a.candidate_point)
        end
        s_plus = 2.0 * s_minus
    else
        vector_transport_to!(
            M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method
        )
        get_gradient!(mp, a.candidate_tangent, a.candidate_point)
        if real(inner(M, a.candidate_point, a.candidate_tangent, a.candidate_direction)) <
            a.c2 * l
            while fNew <= f0 + a.c1 * step * l && (s_plus < max_step_increase)# increase
                s_plus = s_plus * 2.0
                step = s_plus
                retract!(M, a.candidate_point, p, η, step, a.retraction_method)
                fNew = get_cost(mp, a.candidate_point)
            end
            s_minus = s_plus / 2.0
        end
    end
    retract!(M, a.candidate_point, p, η, s_minus, a.retraction_method)
    vector_transport_to!(
        M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method
    )
    get_gradient!(mp, a.candidate_tangent, a.candidate_point)
    while real(inner(M, a.candidate_point, a.candidate_tangent, a.candidate_direction)) <
          a.c2 * l
        step = (s_minus + s_plus) / 2
        retract!(M, a.candidate_point, p, η, step, a.retraction_method)
        fNew = get_cost(mp, a.candidate_point)
        if fNew <= f0 + a.c1 * step * l
            s_minus = step
        else
            s_plus = step
        end
        abs(s_plus - s_minus) <= a.stop_when_stepsize_less && break
        retract!(M, a.candidate_point, p, η, s_minus, a.retraction_method)
        vector_transport_to!(
            M, a.candidate_direction, p, η, a.candidate_point, a.vector_transport_method
        )
        get_gradient!(mp, a.candidate_tangent, a.candidate_point)
    end
    step = s_minus
    a.last_stepsize = step
    return step
end
function show(io::IO, a::WolfePowellLinesearch)
    return print(
        io,
        """
        WolfePowellLinesearch(DefaultManifold(), $(a.c1), $(a.c2)) with keyword arguments
          * retraction_method = $(a.retraction_method)
          * vector_transport_method = $(a.vector_transport_method)""",
    )
end
function status_summary(a::WolfePowellLinesearch)
    s = (a.last_stepsize > 0) ? "\nand the last stepsize used was $(a.last_stepsize)." : ""
    return "$a$s"
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
vector transport.
Then the following Algorithm is performed similar to Algorithm 7 from [Huang:2014](@cite)

1. set ``α=0``, ``β=∞`` and ``t=1``.
2. While either ``A(t)`` does not hold or ``W(t)`` does not hold do steps 3-5.
3. If ``A(t)`` fails, set ``β=t``.
4. If ``A(t)`` holds but ``W(t)`` fails, set ``α=t``.
5. If ``β<∞`` set ``t=\frac{α+β}{2}``, otherwise set ``t=2α``.

# Constructors

There exist two constructors, where, when provided the manifold `M` as a first (optional)
parameter, its default retraction and vector transport are the default.
In this case the retraction and the vector transport are also keyword arguments for ease of use.
The other constructor is kept for backward compatibility.

    WolfePowellLinesearch(
        M=DefaultManifold(),
        c1::Float64=10^(-4),
        c2::Float64=0.999;
        retraction_method = default_retraction_method(M),
        vector_transport_method = default_vector_transport(M),
        linesearch_stopsize = 0.0
    )
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
        M::AbstractManifold=DefaultManifold(),
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
    amp::AbstractManoptProblem,
    ams::AbstractManoptSolverState,
    ::Int,
    η=-get_gradient(amp, get_iterate(ams));
    kwargs...,
)
    M = get_manifold(amp)
    α = 0.0
    β = Inf
    t = 1.0
    f0 = get_cost(amp, get_iterate(ams))
    xNew = retract(M, get_iterate(ams), η, t, a.retraction_method)
    fNew = get_cost(amp, xNew)
    η_xNew = vector_transport_to(M, get_iterate(ams), η, xNew, a.vector_transport_method)
    gradient_new = get_gradient(amp, xNew)
    nAt = fNew > f0 + a.c1 * t * real(inner(M, get_iterate(ams), η, get_gradient(ams)))
    nWt =
        real(inner(M, xNew, gradient_new, η_xNew)) <
        a.c2 * real(inner(M, get_iterate(ams), η, get_gradient(ams)))
    while (nAt || nWt) &&
              (t > a.linesearch_stopsize) &&
              ((α + β) / 2 - 1 > a.linesearch_stopsize)
        nAt && (β = t)            # A(t) fails
        (!nAt && nWt) && (α = t)  # A(t) holds but W(t) fails
        t = isinf(β) ? 2 * α : (α + β) / 2
        # Update trial point
        retract!(M, xNew, get_iterate(ams), η, t, a.retraction_method)
        fNew = get_cost(amp, xNew)
        gradient_new = get_gradient(amp, xNew)
        vector_transport_to!(
            M, η_xNew, get_iterate(ams), η, xNew, a.vector_transport_method
        )
        # Update conditions
        nAt = fNew > f0 + a.c1 * t * real(inner(M, get_iterate(ams), η, get_gradient(ams)))
        nWt =
            real(inner(M, xNew, gradient_new, η_xNew)) <
            a.c2 * real(inner(M, get_iterate(ams), η, get_gradient(ams)))
    end
    a.last_stepsize = t
    return t
end
function show(io::IO, a::WolfePowellBinaryLinesearch)
    return print(
        io,
        """
        WolfePowellBinaryLinesearch(DefaultManifold(), $(a.c1), $(a.c2)) with keyword arguments
          * retraction_method = $(a.retraction_method)
          * vector_transport_method = $(a.vector_transport_method)
          * linesearch_stopsize = $(a.linesearch_stopsize)""",
    )
end
function status_summary(a::WolfePowellBinaryLinesearch)
    s = (a.last_stepsize > 0) ? "\nand the last stepsize used was $(a.last_stepsize)." : ""
    return "$a$s"
end

@doc raw"""
    AdaptiveWNGradient <: DirectionUpdateRule

Represent an adaptive gradient method introduced by [GrapigliaStella:2023](@cite).

Given a positive threshold ``\hat c \mathbb N``,
an minimal bound ``b_{\mathrm{min}} > 0``,
an initial ``b_0 ≥ b_{\mathrm{min}}``, and a
gradient reduction factor threshold ``\alpha ∈ [0,1)``.

Set ``c_0=0`` and use ``\omega_0 = \lVert \operatorname{grad} f(p_0) \rvert_{p_0}``.

For the first iterate use the initial step size ``s_0 = \frac{1}{b_0}``.

Then, given the last gradient ``X_{k-1} = \operatorname{grad} f(x_{k-1})``,
and a previous ``\omega_{k-1}``, the values ``(b_k, \omega_k, c_k)`` are computed
using ``X_k = \operatorname{grad} f(p_k)`` and the following cases

If ``\lVert X_k \rVert_{p_k} \leq \alpha\omega_{k-1}``, then let
``\hat b_{k-1} ∈ [b_\mathrm{min},b_{k-1}]`` and set

```math
(b_k, \omega_k, c_k) = \begin{cases}
\bigl(\hat b_{k-1}, \lVert X_k\rVert_{p_k}, 0 \bigr) & \text{ if } c_{k-1}+1 = \hat c\\
\Bigl(b_{k-1} + \frac{\lVert X_k\rVert_{p_k}^2}{b_{k-1}}, \omega_{k-1}, c_{k-1}+1 \Bigr) & \text{ if } c_{k-1}+1<\hat c
\end{cases}
```

If ``\lVert X_k \rVert_{p_k} > \alpha\omega_{k-1}``, the set

```math
(b_k, \omega_k, c_k) =
\Bigl( b_{k-1} + \frac{\lVert X_k\rVert_{p_k}^2}{b_{k-1}}, \omega_{k-1}, 0)
```

and return the step size ``s_k = \frac{1}{b_k}``.

Note that for ``α=0`` this is the Riemannian variant of `WNGRad`.

# Fields

* `count_threshold::Int`: (`4`) an `Integer` for ``\hat c``
* `minimal_bound::Float64`: (`1e-4`) for ``b_{\mathrm{min}}``
* `alternate_bound::Function`: (`(bk, hat_c) -> min(gradient_bound, max(gradient_bound, bk/(3*hat_c)`)
  how to determine ``\hat b_k`` as a function of `(bmin, bk, hat_c) -> hat_bk`
* `gradient_reduction::Float64`: (`0.9`)
* `gradient_bound` `norm(M, p0, grad_f(M,p0))` the bound ``b_k``.

as well as the internal fields

* `weight` for ``ω_k`` initialised to ``ω_0 = ```norm(M, p0, grad_f(M,p0))` if this is not zero, `1.0` otherwise.
* `count` for the ``c_k``, initialised to ``c_0 = 0``.

# Constructor

    AdaptiveWNGrad(M=DefaultManifold, grad_f=(M, p) -> zero_vector(M, rand(M)), p=rand(M); kwargs...)

Where all fields with defaults are keyword arguments and additional keyword arguments are

* `adaptive`:   (`true`) switches the `gradient_reduction ``α`` to `0`.
* `evaluation`: (`AllocatingEvaluation()`) specifies whether the gradient (that is used for initialisation only) is mutating or allocating
"""
mutable struct AdaptiveWNGradient{I<:Integer,R<:Real,F<:Function} <: Stepsize
    count_threshold::I
    minimal_bound::R
    alternate_bound::F
    gradient_reduction::R
    gradient_bound::R
    weight::R
    count::I
end
function AdaptiveWNGradient(
    M::AbstractManifold=DefaultManifold(),
    (grad_f!!)=(M, p) -> zero_vector(M, rand(M)),
    p=rand(M);
    evaluation::E=AllocatingEvaluation(),
    adaptive::Bool=true,
    count_threshold::I=4,
    minimal_bound::R=1e-4,
    gradient_reduction::R=adaptive ? 0.9 : 0.0,
    gradient_bound::R=norm(
        M,
        p,
        if evaluation == AllocatingEvaluation()
            grad_f!!(M, p)
        else
            grad_f!!(M, zero_vector(M, p), p)
        end,
    ),
    alternate_bound::F=(bk, hat_c) -> min(
        gradient_bound == 0 ? 1.0 : gradient_bound, max(minimal_bound, bk / (3 * hat_c))
    ),
    kwargs...,
) where {I<:Integer,R<:Real,F<:Function,E<:AbstractEvaluationType}
    if gradient_bound == 0
        # If the gradient bound defaults to zero, set it to 1
        gradient_bound = 1.0
    end
    return AdaptiveWNGradient{I,R,F}(
        count_threshold,
        minimal_bound,
        alternate_bound,
        gradient_reduction,
        gradient_bound,
        gradient_bound,
        0,
    )
end
function (awng::AdaptiveWNGradient)(
    mp::AbstractManoptProblem, s::AbstractGradientSolverState, i, args...; kwargs...
)
    M = get_manifold(mp)
    p = get_iterate(s)
    X = get_gradient(mp, p)
    isnan(awng.weight) || (awng.weight = norm(M, p, X)) # init ω_0
    if i == 0 # init fields
        awng.weight = norm(M, p, X) # init ω_0
        (awng.weight == 0) && (awng.weight = 1.0)
        awng.count = 0
        return 1 / awng.gradient_bound
    end
    grad_norm = norm(M, p, X)
    if grad_norm < awng.gradient_reduction * awng.weight # grad norm < αω_{k-1}
        if awng.count + 1 == awng.count_threshold
            awng.gradient_bound = awng.alternate_bound(
                awng.gradient_bound, awng.count_threshold
            )
            awng.weight = grad_norm
            awng.count = 0
        else
            awng.gradient_bound = awng.gradient_bound + grad_norm^2 / awng.gradient_bound
            #weight stays unchanged
            awng.count += 1
        end
    else
        awng.gradient_bound = awng.gradient_bound + grad_norm^2 / awng.gradient_bound
        #weight stays unchanged
        awng.count = 0
    end
    return 1 / awng.gradient_bound
end
get_initial_stepsize(awng::AdaptiveWNGradient) = 1 / awng.gradient_bound
get_last_stepsize(awng::AdaptiveWNGradient) = 1 / awng.gradient_bound
function show(io::IO, awng::AdaptiveWNGradient)
    s = """
    AdaptiveWNGradient(;
      count_threshold=$(awng.count_threshold),
      minimal_bound=$(awng.minimal_bound),
      alternate_bound=$(awng.alternate_bound),
      gradient_reduction=$(awng.gradient_reduction),
      gradient_bound=$(awng.gradient_bound)
    )

    as well as internally the weight ω_k = $(awng.weight) and current count c_k = $(awng.count).
    """
    return print(io, s)
end

@doc raw"""
    get_stepsize(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...)

return the stepsize stored within [`AbstractManoptSolverState`](@ref) `ams` when solving the
[`AbstractManoptProblem`](@ref) `amp`.
This method also works for decorated options and the [`Stepsize`](@ref) function within
the options, by default stored in `ams.stepsize`.
"""
function get_stepsize(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...; kwargs...
)
    return _get_stepsize(amp, ams, dispatch_state_decorator(ams), vars...; kwargs...)
end
function _get_stepsize(
    amp::AbstractManoptProblem,
    ams::AbstractManoptSolverState,
    ::Val{true},
    vars...;
    kwargs...,
)
    return get_stepsize(amp, ams.state, vars...; kwargs...)
end
function _get_stepsize(
    amp::AbstractManoptProblem,
    ams::AbstractManoptSolverState,
    ::Val{false},
    vars...;
    kwargs...,
)
    return ams.stepsize(amp, ams, vars...; kwargs...)
end

function get_initial_stepsize(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...; kwargs...
)
    return _get_initial_stepsize(
        amp::AbstractManoptProblem,
        ams::AbstractManoptSolverState,
        dispatch_state_decorator(ams),
        vars...;
        kwargs...,
    )
end
function _get_initial_stepsize(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{true}, vars...
)
    return get_initial_stepsize(amp, ams.state)
end
function _get_initial_stepsize(
    ::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{false}, vars...
)
    return get_initial_stepsize(ams.stepsize)
end

@doc raw"""
    get_last_stepsize(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...)

return the last computed stepsize stored within [`AbstractManoptSolverState`](@ref) `ams`
when solving the [`AbstractManoptProblem`](@ref) `amp`.

This method takes into account that `ams` might be decorated,
then calls [`get_last_stepsize`](@ref get_last_stepsize(::Stepsize, ::Any...)),
where the stepsize is assumed to be in `ams.stepsize`.
In case this returns `NaN`, a concrete call to the stored stepsize is performed.
For this, usually, the first of the `vars...` should be the current iterate.
"""
function get_last_stepsize(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, vars...
)
    return _get_last_stepsize(amp, ams, dispatch_state_decorator(ams), vars...)
end
function _get_last_stepsize(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{true}, vars...
)
    return get_last_stepsize(amp, ams.state, vars...)
end
function _get_last_stepsize(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, ::Val{false}, vars...
)
    s = get_last_stepsize(ams.stepsize) # if it stores the stepsize itself -> return
    !isnan(s) && return s
    # if not -> call step.
    return ams.stepsize(amp, ams, vars...)
end
@doc raw"""
    get_last_stepsize(::Stepsize, vars...)

return the last computed stepsize from within the stepsize.
If no last step size is stored, this returns `NaN`.
"""
get_last_stepsize(::Stepsize, ::Any...) = NaN
function get_last_stepsize(step::ArmijoLinesearch, ::Any...)
    return step.last_stepsize
end
function get_last_stepsize(step::WolfePowellLinesearch, ::Any...)
    return step.last_stepsize
end
function get_last_stepsize(step::WolfePowellBinaryLinesearch, ::Any...)
    return step.last_stepsize
end
