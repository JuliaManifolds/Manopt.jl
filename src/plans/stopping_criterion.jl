@doc raw"""
    StoppingCriterion

An abstract type for the functors representing stopping criteria, so they are
callable structures. The naming Scheme follows functions, see for
example [`StopAfterIteration`](@ref).

Every StoppingCriterion has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`AbstractManoptProblem`](@ref) as well as [`AbstractManoptSolverState`](@ref)
and the current number of iterations are the arguments and returns a boolean whether
to stop or not.

By default each `StoppingCriterion` should provide a fields `reason` to provide
details when a criterion is met (and that is empty otherwise).
"""
abstract type StoppingCriterion end

"""
    indicates_convergence(c::StoppingCriterion)

Return whether (true) or not (false) a [`StoppingCriterion`](@ref) does _always_
mean that, when it indicates to stop, the solver has converged to a
minimizer or critical point.

Note that this is independent of the actual state of the stopping criterion,
whether some of them indicate to stop, but a purely type-based, static
decision.

# Examples

With `s1=StopAfterIteration(20)` and `s2=StopWhenGradientNormLess(1e-7)` the indicator yields

* `indicates_convergence(s1)` is `false`
* `indicates_convergence(s2)` is `true`
* `indicates_convergence(s1 | s2)` is `false`, since this might also stop after 20 iterations
* `indicates_convergence(s1 & s2)` is `true`, since `s2` is fulfilled if this stops.
"""
indicates_convergence(c::StoppingCriterion) = false

function get_count(c::StoppingCriterion, ::Val{:Iterations})
    if hasfield(typeof(c), :at_iteration)
        return getfield(c, :at_iteration)
    else
        return 0
    end
end
@doc raw"""
    StoppingCriterionGroup <: StoppingCriterion

An abstract type for a Stopping Criterion that itself consists of a set of
Stopping criteria. In total it acts as a stopping criterion itself. Examples
are [`StopWhenAny`](@ref) and [`StopWhenAll`](@ref) that can be used to
combine stopping criteria.
"""
abstract type StoppingCriterionSet <: StoppingCriterion end

"""
    StopAfter <: StoppingCriterion

store a threshold when to stop looking at the complete runtime. It uses
`time_ns()` to measure the time and you provide a `Period` as a time limit,
for example `Minute(15)`.

# Fields

* `threshold` stores the `Period` after which to stop
* `start` stores the starting time when the algorithm is started, that is a call with `i=0`.
* `time` stores the elapsed time
* `at_iteration` indicates at which iteration (including `i=0`) the stopping criterion
  was fulfilled and is `-1` while it is not fulfilled.

# Constructor

    StopAfter(t)

initialize the stopping criterion to a `Period t` to stop after.
"""
mutable struct StopAfter <: StoppingCriterion
    threshold::Period
    start::Nanosecond
    time::Nanosecond
    at_iteration::Int
    function StopAfter(t::Period)
        return if value(t) < 0
            error("You must provide a positive time period")
        else
            new(t, Nanosecond(0), Nanosecond(0), -1)
        end
    end
end
function (c::StopAfter)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    if value(c.start) == 0 || k <= 0 # (re)start timer
        c.at_iteration = -1
        c.start = Nanosecond(time_ns())
        c.time = Nanosecond(0)
    else
        c.time = Nanosecond(time_ns()) - c.start
        if k > 0 && (c.time > Nanosecond(c.threshold))
            c.at_iteration = k
            return true
        end
    end
    return false
end
function get_reason(c::StopAfter)
    if (c.at_iteration >= 0)
        return "The algorithm ran for $(floor(c.time, typeof(c.threshold))) (threshold: $(c.threshold)).\n"
    end
    return ""
end
function status_summary(c::StopAfter)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "stopped after $(c.threshold):\t$s"
end
indicates_convergence(c::StopAfter) = false
function show(io::IO, c::StopAfter)
    return print(io, "StopAfter($(repr(c.threshold)))\n    $(status_summary(c))")
end

"""
    set_parameter!(c::StopAfter, :MaxTime, v::Period)

Update the time period after which an algorithm shall stop.
"""
function set_parameter!(c::StopAfter, ::Val{:MaxTime}, v::Period)
    (value(v) < 0) && error("You must provide a positive time period")
    c.threshold = v
    return c
end

@doc raw"""
    StopAfterIteration <: StoppingCriterion

A functor for a stopping criterion to stop after a maximal number of iterations.

# Fields

* `max_iterations`  stores the maximal iteration number where to stop at
* `at_iteration` indicates at which iteration (including `i=0`) the stopping criterion
  was fulfilled and is `-1` while it is not fulfilled.

# Constructor

    StopAfterIteration(maxIter)

initialize the functor to indicate to stop after `maxIter` iterations.
"""
mutable struct StopAfterIteration <: StoppingCriterion
    max_iterations::Int
    at_iteration::Int
    StopAfterIteration(k::Int) = new(k, -1)
end
function (c::StopAfterIteration)(
    ::P, ::S, k::Int
) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if k >= c.max_iterations
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopAfterIteration)
    if c.at_iteration >= c.max_iterations
        return "The algorithm reached its maximal number of iterations ($(c.max_iterations)).\n"
    end
    return ""
end
function status_summary(c::StopAfterIteration)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Max Iteration $(c.max_iterations):\t$s"
end
function show(io::IO, c::StopAfterIteration)
    return print(io, "StopAfterIteration($(c.max_iterations))\n    $(status_summary(c))")
end

"""
    set_parameter!(c::StopAfterIteration, :;MaxIteration, v::Int)

Update the number of iterations after which the algorithm should stop.
"""
function set_parameter!(c::StopAfterIteration, ::Val{:MaxIteration}, v::Int)
    c.max_iterations = v
    return c
end

"""
    StopWhenChangeLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`AbstractManoptSolverState`](@ref) `s`.
That ism by accessing `get_iterate(s)` and comparing successive iterates.
For the storage a [`StoreStateAction`](@ref) is used.

# Fields


$(_var(:Field, :at_iteration))
$(_var(:Field, :last_change))
$(_var(:Field, :inverse_retraction_method))
$(_var(:Field, :storage))
* `at_iteration::Int`: indicate at which iteration this stopping criterion was last active.
* `inverse_retraction`: An [`AbstractInverseRetractionMethod`](@ref) that can be passed
  to approximate the distance by this inverse retraction and a norm on the tangent space.
  This can be used if neither the distance nor the logarithmic map are availannle on `M`.
* `last_change`: store the last change
* `storage`: A [`StoreStateAction`](@ref) to access the previous iterate.
* `threshold`: the threshold for the change to check (run under to stop)
* `outer_norm`: if `M` is a manifold with components, this can be used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.
  You can deactivate this, but setting this value to `missing`.

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:M)) = $(_math(:M; M="N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:M))`` is a vector of length ``n`` with of points ``p_i ∈ $(_math(:M; M="N"))``.
Then, denoting the `outer_norm` by ``r``, the distance of two points ``p,q ∈ $(_math(:M))``
is given by

```
$(_math(:distance))(p,q) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_math(:distance))(p_k,q_k)^r $(_tex(:Bigr)))^{$(_tex(:frac, "1","r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifolds that do not consist of components.


If the manifold does not have components, the outer norm is ignored.


# Constructor

    StopWhenChangeLess(
        M::AbstractManifold,
        threshold::Float64;
        storage::StoreStateAction=StoreStateAction([:Iterate]),
        inverse_retraction_method::IRT=default_inverse_retraction_method(M)
        outer_norm::Union{Missing,Real}=missing
    )

initialize the stopping criterion to a threshold `ε` using the
[`StoreStateAction`](@ref) `a`, which is initialized to just store `:Iterate` by
default. You can also provide an inverse_retraction_method for the `distance` or a manifold
to use its default inverse retraction.
"""
mutable struct StopWhenChangeLess{
    F,IRT<:AbstractInverseRetractionMethod,TSSA<:StoreStateAction,N<:Union{Missing,Real}
} <: StoppingCriterion
    threshold::F
    last_change::F
    storage::TSSA
    inverse_retraction_method::IRT
    at_iteration::Int
    outer_norm::N
end
function StopWhenChangeLess(
    M::AbstractManifold,
    ε::F;
    storage::StoreStateAction=StoreStateAction(M; store_points=Tuple{:Iterate}),
    inverse_retraction_method::IRT=default_inverse_retraction_method(M),
    outer_norm::N=missing,
) where {F,N<:Union{Missing,Real},IRT<:AbstractInverseRetractionMethod}
    return StopWhenChangeLess{F,IRT,typeof(storage),N}(
        ε, zero(ε), storage, inverse_retraction_method, -1, outer_norm
    )
end
function StopWhenChangeLess(ε::R; kwargs...) where {R<:Real}
    return StopWhenChangeLess(DefaultManifold(), ε; kwargs...)
end
function (c::StopWhenChangeLess)(mp::AbstractManoptProblem, s::AbstractManoptSolverState, k)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if has_storage(c.storage, PointStorageKey(:Iterate))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        r = (has_components(M) && !ismissing(c.outer_norm)) ? (c.outer_norm,) : ()
        c.last_change = distance(
            M, get_iterate(s), p_old, c.inverse_retraction_method, r...
        )
        if c.last_change < c.threshold && k > 0
            c.at_iteration = k
            c.storage(mp, s, k)
            return true
        end
    end
    c.storage(mp, s, k)
    return false
end
function get_reason(c::StopWhenChangeLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "At iteration $(c.at_iteration) the algorithm performed a step with a change ($(c.last_change)) less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenChangeLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|Δp| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenChangeLess) = true
function show(io::IO, c::StopWhenChangeLess)
    s = ismissing(c.outer_norm) ? "" : "and outer norm $(c.outer_norm)"
    return print(
        io,
        "StopWhenChangeLess with threshold $(c.threshold)$(s).\n    $(status_summary(c))",
    )
end

"""
    set_parameter!(c::StopWhenChangeLess, :MinIterateChange, v::Int)

Update the minimal change below which an algorithm shall stop.
"""
function set_parameter!(c::StopWhenChangeLess, ::Val{:MinIterateChange}, v)
    c.threshold = v
    return c
end

"""
    StopWhenCostLess <: StoppingCriterion

store a threshold when to stop looking at the cost function of the
optimization problem from within a [`AbstractManoptProblem`](@ref), i.e `get_cost(p,get_iterate(o))`.

# Constructor

    StopWhenCostLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenCostLess{F} <: StoppingCriterion
    threshold::F
    last_cost::F
    at_iteration::Int
    function StopWhenCostLess(ε::F) where {F<:Real}
        return new{F}(ε, zero(ε), -1)
    end
end
function (c::StopWhenCostLess)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    c.last_cost = get_cost(p, get_iterate(s))
    if c.last_cost < c.threshold
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenCostLess)
    if (c.last_cost < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached a cost function value ($(c.last_cost)) less than the threshold ($(c.threshold)).\n"
    end
    return ""
end
function status_summary(c::StopWhenCostLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "f(x) < $(c.threshold):\t$s"
end
function show(io::IO, c::StopWhenCostLess)
    return print(io, "StopWhenCostLess($(c.threshold))\n    $(status_summary(c))")
end

"""
    set_parameter!(c::StopWhenCostLess, :MinCost, v)

Update the minimal cost below which the algorithm shall stop
"""
function set_parameter!(c::StopWhenCostLess, ::Val{:MinCost}, v)
    c.threshold = v
    return c
end

@doc raw"""
    StopWhenEntryChangeLess

Evaluate whether a certain fields change is less than a certain threshold

## Fields

* `field`:     a symbol addressing the corresponding field in a certain subtype of [`AbstractManoptSolverState`](@ref) to track
* `distance`:  a function `(problem, state, v1, v2) -> R` that computes the distance between two possible values of the `field`
* `storage`:   a [`StoreStateAction`](@ref) to store the previous value of the `field`
* `threshold`: the threshold to indicate to stop when the distance is below this value

# Internal fields

* `at_iteration`: store the iteration at which the stop indication happened

stores a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`AbstractManoptSolverState`](@ref), i.e `get_iterate(o)`.
For the storage a [`StoreStateAction`](@ref) is used

# Constructor

    StopWhenEntryChangeLess(
        field::Symbol
        distance,
        threshold;
        storage::StoreStateAction=StoreStateAction([field]),
    )

"""
mutable struct StopWhenEntryChangeLess{F,TF,TSSA<:StoreStateAction} <: StoppingCriterion
    at_iteration::Int
    distance::F
    field::Symbol
    storage::TSSA
    threshold::TF
    last_change::TF
end
function StopWhenEntryChangeLess(
    field::Symbol, distance::F, threshold::TF; storage::TSSA=StoreStateAction([field])
) where {F,TF,TSSA<:StoreStateAction}
    return StopWhenEntryChangeLess{F,TF,TSSA}(
        -1, distance, field, storage, threshold, zero(threshold)
    )
end

function (sc::StopWhenEntryChangeLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, k
)
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    if has_storage(sc.storage, sc.field)
        old_field_value = get_storage(sc.storage, sc.field)
        sc.last_change = sc.distance(mp, s, old_field_value, getproperty(s, sc.field))
        if (k > 0) && (sc.last_change < sc.threshold)
            sc.at_iteration = k
            sc.storage(mp, s, k)
            return true
        end
    end
    sc.storage(mp, s, k)
    return false
end
function get_reason(sc::StopWhenEntryChangeLess)
    if (sc.last_change < sc.threshold) && (sc.at_iteration >= 0)
        return "At iteation $(sc.at_iteration) the algorithm performed a step with a change ($(sc.last_change)) in $(sc.field) less than $(sc.threshold).\n"
    end
    return ""
end
function status_summary(sc::StopWhenEntryChangeLess)
    has_stopped = (sc.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|Δ:$(sc.field)| < $(sc.threshold): $s"
end

"""
    set_parameter!(c::StopWhenEntryChangeLess, :Threshold, v)

Update the minimal cost below which the algorithm shall stop
"""
function set_parameter!(c::StopWhenEntryChangeLess, ::Val{:Threshold}, v)
    c.threshold = v
    return c
end
function show(io::IO, c::StopWhenEntryChangeLess)
    return print(io, "StopWhenEntryChangeLess\n    $(status_summary(c))")
end

@doc """
    StopWhenGradientChangeLess <: StoppingCriterion

A stopping criterion based on the change of the gradient.

# Fields

$(_var(:Field, :at_iteration))
$(_var(:Field, :last_change))
$(_var(:Field, :vector_transport_method))
$(_var(:Field, :storage))
* `threshold`: the threshold for the change to check (run under to stop)
* `outer_norm`: if `M` is a manifold with components, this can be used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.
  You can deactivate this, but setting this value to `missing`.

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:M)) = $(_math(:M; M="N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:M))`` is a vector of length ``n`` with of points ``p_i ∈ $(_math(:M; M="N"))``.
Then, denoting the `outer_norm` by ``r``, the norm of the difference of tangent vectors like the last and current gradien ``X,Y ∈ $(_math(:M))``
is given by

```
$(_tex(:norm, "X-Y"; index="p")) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_tex(:norm, "X_k-Y_k"; index="p_k"))^r $(_tex(:Bigr)))^{$(_tex(:frac, "1","r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifols, that do not consist of components.

# Constructor

    StopWhenGradientChangeLess(
        M::AbstractManifold,
        ε::Float64;
        storage::StoreStateAction=StoreStateAction([:Iterate]),
        vector_transport_method::IRT=default_vector_transport_method(M),
        outer_norm::N=missing
    )

Create a stopping criterion with threshold `ε` for the change gradient, that is, this criterion
indicates to stop when [`get_gradient`](@ref) is in (norm of) its change less than `ε`, where
`vector_transport_method` denotes the vector transport ``$(_tex(:Cal,"T"))`` used.
"""
mutable struct StopWhenGradientChangeLess{
    F,VTM<:AbstractVectorTransportMethod,TSSA<:StoreStateAction,N<:Union{Missing,Real}
} <: StoppingCriterion
    threshold::F
    last_change::F
    storage::TSSA
    vector_transport_method::VTM
    at_iteration::Int
    outer_norm::N
end
function StopWhenGradientChangeLess(
    M::AbstractManifold,
    ε::F;
    storage::StoreStateAction=StoreStateAction(
        M; store_points=Tuple{:Iterate}, store_vectors=Tuple{:Gradient}
    ),
    vector_transport_method::VTM=default_vector_transport_method(M),
    outer_norm::N=missing,
) where {F,N<:Union{Missing,Real},VTM<:AbstractVectorTransportMethod}
    return StopWhenGradientChangeLess{F,VTM,typeof(storage),N}(
        ε, zero(ε), storage, vector_transport_method, -1, outer_norm
    )
end
function StopWhenGradientChangeLess(
    ε::Float64; storage::StoreStateAction=StoreStateAction([:Iterate, :Gradient]), kwargs...
)
    return StopWhenGradientChangeLess(DefaultManifold(1), ε; storage=storage, kwargs...)
end
function (c::StopWhenGradientChangeLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    M = get_manifold(mp)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if has_storage(c.storage, PointStorageKey(:Iterate)) &&
        has_storage(c.storage, VectorStorageKey(:Gradient))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        X_old = get_storage(c.storage, VectorStorageKey(:Gradient))
        p = get_iterate(s)
        Xt = vector_transport_to(M, p_old, X_old, p, c.vector_transport_method)
        r = (has_components(M) && !ismissing(c.outer_norm)) ? (c.outer_norm,) : ()
        c.last_change = norm(M, p, Xt - get_gradient(s), r...)
        if c.last_change < c.threshold && k > 0
            c.at_iteration = k
            c.storage(mp, s, k)
            return true
        end
    end
    c.storage(mp, s, k)
    return false
end
function get_reason(c::StopWhenGradientChangeLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "At iteration $(c.at_iteration) the change of the gradient ($(c.last_change)) was less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenGradientChangeLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|Δgrad f| < $(c.threshold): $s"
end
function show(io::IO, c::StopWhenGradientChangeLess)
    s = ismissing(c.outer_norm) ? "" : "outer_norm=$(c.outer_norm), "
    return print(
        io,
        "StopWhenGradientChangeLess with threshold $(c.threshold); $(s)vector_transport_method=$(c.vector_transport_method))\n    $(status_summary(c))",
    )
end

"""
    set_parameter!(c::StopWhenGradientChangeLess, :MinGradientChange, v)

Update the minimal change below which an algorithm shall stop.
"""
function set_parameter!(c::StopWhenGradientChangeLess, ::Val{:MinGradientChange}, v)
    c.threshold = v
    return c
end

"""
    StopWhenGradientNormLess <: StoppingCriterion

A stopping criterion based on the current gradient norm.

# Fields

* `norm`:      a function `(M::AbstractManifold, p, X) -> ℝ` that computes a norm
  of the gradient `X` in the tangent space at `p` on `M``.
  For manifolds with components provide `(M::AbstractManifold, p, X, r) -> ℝ`.
* `threshold`: the threshold to indicate to stop when the distance is below this value
* `outer_norm`: if `M` is a manifold with components, this can be used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.

# Internal fields

* `last_change` store the last change
* `at_iteration` store the iteration at which the stop indication happened

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:M)) = $(_math(:M; M="N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:M))`` is a vector of length ``n`` with of points ``p_i ∈ $(_math(:M; M="N"))``.
Then, denoting the `outer_norm` by ``r``, the norm of a tangent vector like the current gradient ``X ∈ $(_math(:M))``
is given by

```
$(_tex(:norm, "X"; index="p")) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_tex(:norm, "X_k"; index="p_k"))^r $(_tex(:Bigr)))^{$(_tex(:frac, "1","r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifolds that do not consist of components.

If you pass in your individual norm, this can be deactivated on such manifolds
by passing `missing` to `outer_norm`.

# Constructor

    StopWhenGradientNormLess(ε; norm=ManifoldsBase.norm, outer_norm=missing)

Create a stopping criterion with threshold `ε` for the gradient, that is, this criterion
indicates to stop when [`get_gradient`](@ref) returns a gradient vector of norm less than `ε`,
where the norm to use can be specified in the `norm=` keyword.
"""
mutable struct StopWhenGradientNormLess{F,TF,N<:Union{Missing,Real}} <: StoppingCriterion
    norm::F
    threshold::TF
    last_change::TF
    at_iteration::Int
    outer_norm::N
    function StopWhenGradientNormLess(
        ε::TF; norm::F=norm, outer_norm::N=missing
    ) where {F,TF,N<:Union{Missing,Real}}
        return new{F,TF,N}(norm, ε, zero(ε), -1, outer_norm)
    end
end

function (sc::StopWhenGradientNormLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    M = get_manifold(mp)
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    if (k > 0)
        r = (has_components(M) && !ismissing(sc.outer_norm)) ? (sc.outer_norm,) : ()
        sc.last_change = sc.norm(M, get_iterate(s), get_gradient(s), r...)
        if sc.last_change < sc.threshold
            sc.at_iteration = k
            return true
        end
    end
    return false
end
function get_reason(c::StopWhenGradientNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the gradient norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenGradientNormLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|grad f| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenGradientNormLess) = true
function show(io::IO, c::StopWhenGradientNormLess)
    return print(io, "StopWhenGradientNormLess($(c.threshold))\n    $(status_summary(c))")
end

"""
    set_parameter!(c::StopWhenGradientNormLess, :MinGradNorm, v::Float64)

Update the minimal gradient norm when an algorithm shall stop
"""
function set_parameter!(c::StopWhenGradientNormLess, ::Val{:MinGradNorm}, v::Float64)
    c.threshold = v
    return c
end

"""
    StopWhenStepsizeLess <: StoppingCriterion

stores a threshold when to stop looking at the last step size determined or found
during the last iteration from within a [`AbstractManoptSolverState`](@ref).

# Constructor

    StopWhenStepsizeLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenStepsizeLess{F} <: StoppingCriterion
    threshold::F
    last_stepsize::F
    at_iteration::Int
    function StopWhenStepsizeLess(ε::F) where {F<:Real}
        return new{F}(ε, zero(ε), -1)
    end
end
function (c::StopWhenStepsizeLess)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    c.last_stepsize = get_last_stepsize(p, s, k)
    if c.last_stepsize < c.threshold && k > 0
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenStepsizeLess)
    if (c.last_stepsize < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm computed a step size ($(c.last_stepsize)) less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenStepsizeLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Stepsize s < $(c.threshold):\t$s"
end
function show(io::IO, c::StopWhenStepsizeLess)
    return print(io, "StopWhenStepsizeLess($(c.threshold))\n    $(status_summary(c))")
end
"""
    set_parameter!(c::StopWhenStepsizeLess, :MinStepsize, v)

Update the minimal step size below which the algorithm shall stop
"""
function set_parameter!(c::StopWhenStepsizeLess, ::Val{:MinStepsize}, v)
    c.threshold = v
    return c
end

"""
    StopWhenCostNaN <: StoppingCriterion

stop looking at the cost function of the optimization problem from within a [`AbstractManoptProblem`](@ref), i.e `get_cost(p,get_iterate(o))`.

# Constructor

    StopWhenCostNaN()

initialize the stopping criterion to NaN.
"""
mutable struct StopWhenCostNaN <: StoppingCriterion
    at_iteration::Int
    StopWhenCostNaN() = new(-1)
end
function (c::StopWhenCostNaN)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    # but still verify whether it yields NaN
    if isnan(get_cost(p, get_iterate(s)))
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenCostNaN)
    if c.at_iteration >= 0
        return "The algorithm reached a cost function value of NaN.\n"
    end
    return ""
end
function status_summary(c::StopWhenCostNaN)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "f(x) is NaN:\t$s"
end
function show(io::IO, c::StopWhenCostNaN)
    return print(io, "StopWhenCostNaN()\n    $(status_summary(c))")
end

"""
    StopWhenIterateNaN <: StoppingCriterion

stop looking at the cost function of the optimization problem from within a [`AbstractManoptProblem`](@ref), i.e `get_cost(p,get_iterate(o))`.

# Constructor

    StopWhenIterateNaN()

initialize the stopping criterion to NaN.
"""
mutable struct StopWhenIterateNaN <: StoppingCriterion
    at_iteration::Int
    StopWhenIterateNaN() = new(-1)
end
function (c::StopWhenIterateNaN)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if (k >= 0) && any(isnan.(get_iterate(s)))
        c.at_iteration = 0
        return true
    end
    return false
end
function get_reason(c::StopWhenIterateNaN)
    if (c.at_iteration >= 0)
        return "The algorithm reached an iterate containing NaNs iterate.\n"
    end
    return ""
end
function status_summary(c::StopWhenIterateNaN)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "f(x) is NaN:\t$s"
end
function show(io::IO, c::StopWhenIterateNaN)
    return print(io, "StopWhenIterateNaN()\n    $(status_summary(c))")
end

@doc raw"""
    StopWhenSmallerOrEqual <: StoppingCriterion

A functor for an stopping criterion, where the algorithm if stopped when a variable is smaller than or equal to its minimum value.

# Fields

* `value`    stores the variable which has to fall under a threshold for the algorithm to stop
* `minValue` stores the threshold where, if the value is smaller or equal to this threshold, the algorithm stops

# Constructor

    StopWhenSmallerOrEqual(value, minValue)

initialize the functor to indicate to stop after `value` is smaller than or equal to `minValue`.
"""
mutable struct StopWhenSmallerOrEqual{R} <: StoppingCriterion
    value::Symbol
    minValue::R
    at_iteration::Int
    function StopWhenSmallerOrEqual(value::Symbol, mValue::R) where {R<:Real}
        return new{R}(value, mValue, -1)
    end
end
function (c::StopWhenSmallerOrEqual)(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    if getfield(s, c.value) <= c.minValue
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenSmallerOrEqual)
    if (c.at_iteration >= 0)
        return "The value of the variable ($(string(c.value))) is smaller than or equal to its threshold ($(c.minValue)).\n"
    end
    return ""
end
function status_summary(c::StopWhenSmallerOrEqual)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Field :$(c.value) ≤ $(c.minValue):\t$s"
end
function show(io::IO, c::StopWhenSmallerOrEqual)
    return print(
        io, "StopWhenSmallerOrEqual(:$(c.value), $(c.minValue))\n    $(status_summary(c))"
    )
end

"""
    StopWhenSubgradientNormLess <: StoppingCriterion

A stopping criterion based on the current subgradient norm.

# Constructor

    StopWhenSubgradientNormLess(ε::Float64)

Create a stopping criterion with threshold `ε` for the subgradient, that is, this criterion
indicates to stop when [`get_subgradient`](@ref) returns a subgradient vector of norm less than `ε`.
"""
mutable struct StopWhenSubgradientNormLess{R} <: StoppingCriterion
    at_iteration::Int
    threshold::R
    value::R
    StopWhenSubgradientNormLess(ε::R) where {R<:Real} = new{R}(-1, ε, zero(ε))
end
function (c::StopWhenSubgradientNormLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    M = get_manifold(mp)
    if (k == 0) # reset on init
        c.at_iteration = -1
    end
    c.value = norm(M, get_iterate(s), get_subgradient(s))
    if (c.value < c.threshold) && (k > 0)
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenSubgradientNormLess)
    if (c.value < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the subgradient norm ($(c.value)) is less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenSubgradientNormLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "|∂f| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenSubgradientNormLess) = true
function show(io::IO, c::StopWhenSubgradientNormLess)
    return print(
        io, "StopWhenSubgradientNormLess($(c.threshold))\n    $(status_summary(c))"
    )
end
"""
    set_parameter!(c::StopWhenSubgradientNormLess, :MinSubgradNorm, v::Float64)

Update the minimal subgradient norm when an algorithm shall stop
"""
function set_parameter!(c::StopWhenSubgradientNormLess, ::Val{:MinSubgradNorm}, v::Float64)
    c.threshold = v
    return c
end

#
# Meta Criteria
#

@doc raw"""
    StopWhenAll <: StoppingCriterionSet

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _all_ indicate to stop. The `reason` is given by the concatenation of all
reasons.

# Constructor

    StopWhenAll(c::NTuple{N,StoppingCriterion} where N)
    StopWhenAll(c::StoppingCriterion,...)
"""
mutable struct StopWhenAll{TCriteria<:Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    at_iteration::Int
    StopWhenAll(c::Vector{StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), -1)
    StopWhenAll(c...) = new{typeof(c)}(c, -1)
end
function (c::StopWhenAll)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    (k == 0) && (c.at_iteration = -1) # reset on init
    if all(subC -> subC(p, s, k), c.criteria)
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenAll)
    if c.at_iteration >= 0
        return string([get_reason(subC) for subC in c.criteria]...)
    end
    return ""
end
function status_summary(c::StopWhenAll)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    r = "Stop When _all_ of the following are fulfilled:\n"
    for cs in c.criteria
        r = "$r    $(status_summary(cs))\n"
    end
    return "$(r)Overall: $s"
end
function indicates_convergence(c::StopWhenAll)
    return any(indicates_convergence(ci) for ci in c.criteria)
end
function get_count(c::StopWhenAll, v::Val{:Iterations})
    return maximum(get_count(ci, v) for ci in c.criteria)
end
function show(io::IO, c::StopWhenAll)
    s = replace(status_summary(c), "\n" => "\n    ") #increase indent
    return print(io, "StopWhenAll with the Stopping Criteria\n    $(s)")
end

"""
    &(s1,s2)
    s1 & s2

Combine two [`StoppingCriterion`](@ref) within an [`StopWhenAll`](@ref).
If either `s1` (or `s2`) is already an [`StopWhenAll`](@ref), then `s2` (or `s1`) is
appended to the list of [`StoppingCriterion`](@ref) within `s1` (or `s2`).

# Example
    a = StopAfterIteration(200) & StopWhenChangeLess(M, 1e-6)
    b = a & StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6))
    b = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6), StopWhenGradientNormLess(1e-6))
"""
function Base.:&(s1::S, s2::T) where {S<:StoppingCriterion,T<:StoppingCriterion}
    return StopWhenAll(s1, s2)
end
function Base.:&(s1::S, s2::StopWhenAll) where {S<:StoppingCriterion}
    return StopWhenAll(s1, s2.criteria...)
end
function Base.:&(s1::StopWhenAll, s2::T) where {T<:StoppingCriterion}
    return StopWhenAll(s1.criteria..., s2)
end
function Base.:&(s1::StopWhenAll, s2::StopWhenAll)
    return StopWhenAll(s1.criteria..., s2.criteria...)
end

@doc raw"""
    StopWhenAny <: StoppingCriterionSet

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _any_ single one indicates to stop. The `reason` is given by the
concatenation of all reasons (assuming that all non-indicating return `""`).

# Constructor
    StopWhenAny(c::NTuple{N,StoppingCriterion} where N)
    StopWhenAny(c::StoppingCriterion...)
"""
mutable struct StopWhenAny{TCriteria<:Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    at_iteration::Int
    StopWhenAny(c::Vector{<:StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), -1)
    StopWhenAny(c::StoppingCriterion...) = new{typeof(c)}(c, -1)
end

# `_fast_any(f, tup::Tuple)`` is functionally equivalent to `any(f, tup)`` but on Julia 1.10
# this implementation is faster on heterogeneous tuples
@inline _fast_any(f, tup::Tuple{}) = true
@inline _fast_any(f, tup::Tuple{T}) where {T} = f(tup[1])
@inline function _fast_any(f, tup::Tuple)
    if f(tup[1])
        return true
    else
        return _fast_any(f, tup[2:end])
    end
end

function (c::StopWhenAny)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    (k == 0) && (c.at_iteration = -1) # reset on init
    if _fast_any(subC -> subC(p, s, k), c.criteria)
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenAny)
    if (c.at_iteration >= 0)
        return string((get_reason(subC) for subC in c.criteria)...)
    end
    return ""
end
function status_summary(c::StopWhenAny)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    r = "Stop When _one_ of the following are fulfilled:\n"
    for cs in c.criteria
        r = "$r    $(status_summary(cs))\n"
    end
    return "$(r)Overall: $s"
end
function indicates_convergence(c::StopWhenAny)
    return any(indicates_convergence(ci) for ci in get_active_stopping_criteria(c))
end
function get_count(c::StopWhenAny, v::Val{:Iterations})
    iters = filter(x -> x > 0, [get_count(ci, v) for ci in c.criteria])
    (length(iters) == 0) && (return 0)
    return minimum(iters)
end
function show(io::IO, c::StopWhenAny)
    s = replace(status_summary(c), "\n" => "\n    ") #increase indent
    return print(io, "StopWhenAny with the Stopping Criteria\n    $(s)")
end
"""
    |(s1,s2)
    s1 | s2

Combine two [`StoppingCriterion`](@ref) within an [`StopWhenAny`](@ref).
If either `s1` (or `s2`) is already an [`StopWhenAny`](@ref), then `s2` (or `s1`) is
appended to the list of [`StoppingCriterion`](@ref) within `s1` (or `s2`)

# Example
    a = StopAfterIteration(200) | StopWhenChangeLess(M, 1e-6)
    b = a | StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6))
    b = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(M, 1e-6), StopWhenGradientNormLess(1e-6))
"""
function Base.:|(s1::S, s2::T) where {S<:StoppingCriterion,T<:StoppingCriterion}
    return StopWhenAny(s1, s2)
end
function Base.:|(s1::S, s2::StopWhenAny) where {S<:StoppingCriterion}
    return StopWhenAny(s1, s2.criteria...)
end
function Base.:|(s1::StopWhenAny, s2::T) where {T<:StoppingCriterion}
    return StopWhenAny(s1.criteria..., s2)
end
function Base.:|(s1::StopWhenAny, s2::StopWhenAny)
    return StopWhenAny(s1.criteria..., s2.criteria...)
end

is_active_stopping_criterion(c::StoppingCriterion) = (c.at_iteration >= 0)

@doc raw"""
    get_active_stopping_criteria(c)

returns all active stopping criteria, if any, that are within a
[`StoppingCriterion`](@ref) `c`, and indicated a stop, that is their reason is nonempty.
To be precise for a simple stopping criterion, this returns either an empty
array if no stop is indicated or the stopping criterion as the only element of
an array. For a [`StoppingCriterionSet`](@ref) all internal (even nested)
criteria that indicate to stop are returned.
"""
function get_active_stopping_criteria(c::sCS) where {sCS<:StoppingCriterionSet}
    c = get_active_stopping_criteria.(get_stopping_criteria(c))
    return vcat(c...)
end
# for non-array containing stopping criteria, the recursion ends in either
# returning nothing or an 1-element array containing itself
function get_active_stopping_criteria(c::sC) where {sC<:StoppingCriterion}
    if is_active_stopping_criterion(c)
        return [c] # recursion top
    else
        return []
    end
end

@doc raw"""
    get_stopping_criteria(c)

return the array of internally stored [`StoppingCriterion`](@ref)s for a
[`StoppingCriterionSet`](@ref) `c`.
"""
function get_stopping_criteria(c::S) where {S<:StoppingCriterionSet}
    return error("get_stopping_criteria() not defined for a $(typeof(c)).")
end
get_stopping_criteria(c::StopWhenAll) = c.criteria
get_stopping_criteria(c::StopWhenAny) = c.criteria

function set_parameter!(s::AbstractManoptSolverState, ::Val{:StoppingCriterion}, args...)
    set_parameter!(s.stop, args...)
    return s
end
function set_parameter!(c::StopWhenAll, s::Symbol, v)
    for d in c.criteria
        set_parameter!(d, s, v)
    end
    return c
end
function set_parameter!(c::StopWhenAny, s::Symbol, v)
    for d in c.criteria
        set_parameter!(d, s, v)
    end
    return c
end

@doc raw"""
    get_reason(s::AbstractManoptSolverState)

return the current reason stored within the [`StoppingCriterion`](@ref) from
within the [`AbstractManoptSolverState`](@ref).
This reason is empty (`""`) if the criterion has never been met.
"""
get_reason(s::AbstractManoptSolverState) = get_reason(get_state(s).stop)
