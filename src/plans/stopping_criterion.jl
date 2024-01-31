@doc raw"""
    StoppingCriterion

An abstract type for the functors representing stopping criteria, i.e. they are
callable structures. The naming Scheme follows functions, see for
example [`StopAfterIteration`](@ref).

Every StoppingCriterion has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`AbstractManoptProblem`](@ref) as well as [`AbstractManoptSolverState`](@ref)
and the current number of iterations are the arguments and returns a Bool whether
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
i.e. whether some of them indicate to stop, but a purely type-based, static
decision

# Examples

With `s1=StopAfterIteration(20)` and `s2=StopWhenGradientNormLess(1e-7)` we have

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
i.e. `Minute(15)`

# Constructor

    StopAfter(t)

initialize the stopping criterion to a `Period t` to stop after.
"""
mutable struct StopAfter <: StoppingCriterion
    threshold::Period
    reason::String
    start::Nanosecond
    at_iteration::Int
    function StopAfter(t::Period)
        return if value(t) < 0
            error("You must provide a positive time period")
        else
            new(t, "", Nanosecond(0), 0)
        end
    end
end
function (c::StopAfter)(::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int)
    if value(c.start) == 0 || i <= 0 # (re)start timer
        c.reason = ""
        c.at_iteration = 0
        c.start = Nanosecond(time_ns())
    else
        cTime = Nanosecond(time_ns()) - c.start
        if i > 0 && (cTime > Nanosecond(c.threshold))
            c.reason = "The algorithm ran for about $(floor(cTime, typeof(c.threshold))) and has hence reached the threshold of $(c.threshold).\n"
            c.at_iteration = i
            return true
        end
    end
    return false
end
function status_summary(c::StopAfter)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "stopped after $(c.threshold):\t$s"
end
indicates_convergence(c::StopAfter) = false
function show(io::IO, c::StopAfter)
    return print(io, "StopAfter($(repr(c.threshold)))\n    $(status_summary(c))")
end

"""
    update_stopping_criterion!(c::StopAfter, :MaxTime, v::Period)

Update the time period after which an algorithm shall stop.
"""
function update_stopping_criterion!(c::StopAfter, ::Val{:MaxTime}, v::Period)
    (value(v) < 0) && error("You must provide a positive time period")
    c.threshold = v
    return c
end

@doc raw"""
    StopAfterIteration <: StoppingCriterion

A functor for an easy stopping criterion, i.e. to stop after a maximal number
of iterations.

# Fields
* `maxIter` – stores the maximal iteration number where to stop at
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`get_reason`](@ref).

# Constructor

    StopAfterIteration(maxIter)

initialize the stopafterIteration functor to indicate to stop after `maxIter`
iterations.
"""
mutable struct StopAfterIteration <: StoppingCriterion
    maxIter::Int
    reason::String
    at_iteration::Int
    StopAfterIteration(mIter::Int) = new(mIter, "", 0)
end
function (c::StopAfterIteration)(
    ::P, ::S, i::Int
) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if i >= c.maxIter
        c.at_iteration = i
        c.reason = "The algorithm reached its maximal number of iterations ($(c.maxIter)).\n"
        return true
    end
    return false
end
function status_summary(c::StopAfterIteration)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Max Iteration $(c.maxIter):\t$s"
end
function show(io::IO, c::StopAfterIteration)
    return print(io, "StopAfterIteration($(c.maxIter))\n    $(status_summary(c))")
end

"""
    update_stopping_criterion!(c::StopAfterIteration, :;MaxIteration, v::Int)

Update the number of iterations after which the algorithm should stop.
"""
function update_stopping_criterion!(c::StopAfterIteration, ::Val{:MaxIteration}, v::Int)
    c.maxIter = v
    return c
end

"""
    StopWhenSubgradientNormLess <: StoppingCriterion

A stopping criterion based on the current subgradient norm.

# Constructor

    StopWhenSubgradientNormLess(ε::Float64)

Create a stopping criterion with threshold `ε` for the subgradient, that is, this criterion
indicates to stop when [`get_subgradient`](@ref) returns a subgradient vector of norm less than `ε`.
"""
mutable struct StopWhenSubgradientNormLess <: StoppingCriterion
    threshold::Float64
    reason::String
    StopWhenSubgradientNormLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenSubgradientNormLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    M = get_manifold(mp)
    (i == 0) && (c.reason = "") # reset on init
    if (norm(M, get_iterate(s), get_subgradient(s)) < c.threshold) && (i > 0)
        c.reason = "The algorithm reached approximately critical point after $i iterations; the subgradient norm ($(norm(M,get_iterate(s),get_subgradient(s)))) is less than $(c.threshold).\n"
        return true
    end
    return false
end
function status_summary(c::StopWhenSubgradientNormLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "|subgrad f| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenSubgradientNormLess) = true
function show(io::IO, c::StopWhenSubgradientNormLess)
    return print(
        io, "StopWhenSubgradientNormLess($(c.threshold))\n    $(status_summary(c))"
    )
end
"""
    update_stopping_criterion!(c::StopWhenSubgradientNormLess, :MinSubgradNorm, v::Float64)

Update the minimal subgradient norm when an algorithm shall stop
"""
function update_stopping_criterion!(
    c::StopWhenSubgradientNormLess, ::Val{:MinSubgradNorm}, v::Float64
)
    c.threshold = v
    return c
end

"""
    StopWhenChangeLess <: StoppingCriterion

stores a threshold when to stop looking at the norm of the change of the
optimization variable from within a [`AbstractManoptSolverState`](@ref), i.e `get_iterate(o)`.
For the storage a [`StoreStateAction`](@ref) is used

# Constructor

    StopWhenChangeLess(
        M::AbstractManifold,
        ε::Float64;
        storage::StoreStateAction=StoreStateAction([:Iterate]),
        inverse_retraction_method::IRT=default_inverse_retraction_method(manifold)
    )

initialize the stopping criterion to a threshold `ε` using the
[`StoreStateAction`](@ref) `a`, which is initialized to just store `:Iterate` by
default. You can also provide an inverse_retraction_method for the `distance` or a manifold
to use its default inverse retraction.
"""
mutable struct StopWhenChangeLess{
    IRT<:AbstractInverseRetractionMethod,TSSA<:StoreStateAction
} <: StoppingCriterion
    threshold::Float64
    reason::String
    storage::TSSA
    inverse_retraction::IRT
    at_iteration::Int
end
function StopWhenChangeLess(
    M::AbstractManifold,
    ε::Float64;
    storage::StoreStateAction=StoreStateAction(M; store_points=Tuple{:Iterate,:Population}),
    inverse_retraction_method::IRT=default_inverse_retraction_method(M),
) where {IRT<:AbstractInverseRetractionMethod}
    return StopWhenChangeLess{IRT,typeof(storage)}(
        ε, "", storage, inverse_retraction_method, 0
    )
end
function StopWhenChangeLess(
    ε::Float64;
    storage::StoreStateAction=StoreStateAction([:Iterate, :Population]),
    manifold::AbstractManifold=DefaultManifold(),
    inverse_retraction_method::IRT=default_inverse_retraction_method(manifold),
) where {IRT<:AbstractInverseRetractionMethod}
    if !(manifold isa DefaultManifold)
        @warn "The `manifold` keyword is deprecated, use the first positional argument `M` instead."
    end
    return StopWhenChangeLess{IRT,typeof(storage)}(
        ε, "", storage, inverse_retraction_method, 0
    )
end
function (c::StopWhenChangeLess)(mp::AbstractManoptProblem, s::AbstractManoptSolverState, i)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if has_storage(c.storage, PointStorageKey(:Iterate))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        d = distance(M, get_iterate(s), p_old, c.inverse_retraction)
        if d < c.threshold && i > 0
            c.reason = "The algorithm performed a step with a change ($d) less than $(c.threshold).\n"
            c.at_iteration = i
            c.storage(mp, s, i)
            return true
        end
    end
    c.storage(mp, s, i)
    return false
end
function status_summary(c::StopWhenChangeLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "|Δp| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenChangeLess) = true
function show(io::IO, c::StopWhenChangeLess)
    return print(io, "StopWhenChangeLess($(c.threshold))\n    $(status_summary(c))")
end

"""
    update_stopping_criterion!(c::StopWhenChangeLess, :MinIterateChange, v::Int)

Update the minimal change below which an algorithm shall stop.
"""
function update_stopping_criterion!(c::StopWhenChangeLess, ::Val{:MinIterateChange}, v)
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
mutable struct StopWhenCostLess <: StoppingCriterion
    threshold::Float64
    reason::String
    at_iteration::Int
    StopWhenCostLess(ε::Float64) = new(ε, "", 0)
end
function (c::StopWhenCostLess)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if i > 0 && get_cost(p, get_iterate(s)) < c.threshold
        c.reason = "The algorithm reached a cost function value ($(get_cost(p,get_iterate(s)))) less than the threshold ($(c.threshold)).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenCostLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "f(x) < $(c.threshold):\t$s"
end
function show(io::IO, c::StopWhenCostLess)
    return print(io, "StopWhenCostLess($(c.threshold))\n    $(status_summary(c))")
end

"""
    update_stopping_criterion!(c::StopWhenCostLess, :MinCost, v)

Update the minimal cost below which the algorithm shall stop
"""
function update_stopping_criterion!(c::StopWhenCostLess, ::Val{:MinCost}, v)
    c.threshold = v
    return c
end

@doc raw"""
    StopWhenEntryChangeLess

Evaluate whether a certain fields change is less than a certain threshold

## Fields

* `field`     – a symbol adressing the corresponding field in a certain subtype of [`AbstractManoptSolverState`](@ref)
  to track
* `distance`  – a function `(problem, state, v1, v2) -> R` that computes the distance between two possible values of the `field`
* `storage`   – a [`StoreStateAction`](@ref) to store the previous value of the `field`
* `threshold` – the threshold to indicate to stop when the distance is below this value

# Internal fields

* `reason`    – store a string reason when the stop was indicated
* `at_iteration` – store the iteration at which the stop indication happened

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
    reason::String
    storage::TSSA
    threshold::TF
end
function StopWhenEntryChangeLess(
    field::Symbol, distance::F, threshold::TF; storage::TSSA=StoreStateAction([field])
) where {F,TF,TSSA<:StoreStateAction}
    return StopWhenEntryChangeLess{F,TF,TSSA}(0, distance, field, "", storage, threshold)
end

function (sc::StopWhenEntryChangeLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, i
)
    if i == 0 # reset on init
        sc.reason = ""
        sc.at_iteration = 0
    end
    if has_storage(sc.storage, sc.field)
        old_field_value = get_storage(sc.storage, sc.field)
        ε = sc.distance(mp, s, old_field_value, getproperty(s, sc.field))
        if (i > 0) && (ε < sc.threshold)
            sc.reason = "The algorithm performed a step with a change ($ε) in $(sc.field) less than $(sc.threshold).\n"
            sc.at_iteration = i
            sc.storage(mp, s, i)
            return true
        end
    end
    sc.storage(mp, s, i)
    return false
end
function status_summary(sc::StopWhenEntryChangeLess)
    has_stopped = length(sc.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "|Δ:$(sc.field)| < $(sc.threshold): $s"
end

"""
    update_stopping_criterion!(c::StopWhenEntryChangeLess, :Threshold, v)

Update the minimal cost below which the algorithm shall stop
"""
function update_stopping_criterion!(c::StopWhenEntryChangeLess, ::Val{:Threshold}, v)
    c.threshold = v
    return c
end
function show(io::IO, c::StopWhenEntryChangeLess)
    return print(io, "StopWhenEntryChangeLess\n    $(status_summary(c))")
end

@doc raw"""
    StopWhenGradientChangeLess <: StoppingCriterion

A stopping criterion based on the change of the gradient

```
\lVert \mathcal T_{p^{(k)}\gets p^{(k-1)} \operatorname{grad} f(p^{(k-1)}) -  \operatorname{grad} f(p^{(k-1)}) \rVert < ε
```

# Constructor

    StopWhenGradientChangeLess(
        M::AbstractManifold,
        ε::Float64;
        storage::StoreStateAction=StoreStateAction([:Iterate]),
        vector_transport_method::IRT=default_vector_transport_method(M),
    )

Create a stopping criterion with threshold `ε` for the change gradient, that is, this criterion
indicates to stop when [`get_gradient`](@ref) is in (norm of) its change less than `ε`, where
`vector_transport_method` denotes the vector transport ``\mathcal T`` used.
"""
mutable struct StopWhenGradientChangeLess{
    VTM<:AbstractVectorTransportMethod,TSSA<:StoreStateAction
} <: StoppingCriterion
    threshold::Float64
    reason::String
    storage::TSSA
    vector_transport_method::VTM
    at_iteration::Int
end
function StopWhenGradientChangeLess(
    M::AbstractManifold,
    ε::Float64;
    storage::StoreStateAction=StoreStateAction(
        M; store_points=Tuple{:Iterate}, store_vectors=Tuple{:Gradient}
    ),
    vector_transport_method::VTM=default_vector_transport_method(M),
) where {VTM<:AbstractVectorTransportMethod}
    return StopWhenGradientChangeLess{VTM,typeof(storage)}(
        ε, "", storage, vector_transport_method, 0
    )
end
function StopWhenGradientChangeLess(
    ε::Float64; storage::StoreStateAction=StoreStateAction([:Iterate, :Gradient]), kwargs...
)
    return StopWhenGradientChangeLess(DefaultManifold(1), ε; storage=storage, kwargs...)
end
function (c::StopWhenGradientChangeLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    M = get_manifold(mp)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if has_storage(c.storage, PointStorageKey(:Iterate)) &&
        has_storage(c.storage, VectorStorageKey(:Gradient))
        M = get_manifold(mp)
        p_old = get_storage(c.storage, PointStorageKey(:Iterate))
        X_old = get_storage(c.storage, VectorStorageKey(:Gradient))
        p = get_iterate(s)
        Xt = vector_transport_to(M, p_old, X_old, p, c.vector_transport_method)
        d = norm(M, p, Xt - get_gradient(s))
        if d < c.threshold && i > 0
            c.reason = "At iteration $i the change of the gradient ($d) was less than $(c.threshold).\n"
            c.at_iteration = i
            c.storage(mp, s, i)
            return true
        end
    end
    c.storage(mp, s, i)
    return false
end
function status_summary(c::StopWhenGradientChangeLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "|Δgrad f| < $(c.threshold): $s"
end
function show(io::IO, c::StopWhenGradientChangeLess)
    return print(
        io,
        "StopWhenGradientChangeLess($(c.threshold); vector_transport_method=$(c.vector_transport_method))\n    $(status_summary(c))",
    )
end

"""
    update_stopping_criterion!(c::StopWhenGradientChangeLess, :MinGradientChange, v)

Update the minimal change below which an algorithm shall stop.
"""
function update_stopping_criterion!(
    c::StopWhenGradientChangeLess, ::Val{:MinGradientChange}, v
)
    c.threshold = v
    return c
end

"""
    StopWhenGradientNormLess <: StoppingCriterion

A stopping criterion based on the current gradient norm.

# Fields

* `norm`  – a function `(M::AbstractManifold, p, X) -> ℝ` that computes a norm of the gradient `X` in the tangent space at `p` on `M``
* `threshold` – the threshold to indicate to stop when the distance is below this value

# Internal fields

* `reason`    – store a string reason when the stop was indicated
* `at_iteration` – store the iteration at which the stop indication happened


# Constructor

    StopWhenGradientNormLess(ε; norm=(M,p,X) -> norm(M,p,X))

Create a stopping criterion with threshold `ε` for the gradient, that is, this criterion
indicates to stop when [`get_gradient`](@ref) returns a gradient vector of norm less than `ε`,
where the norm to use can be specified in the `norm=` keyword.
"""
mutable struct StopWhenGradientNormLess{F,TF} <: StoppingCriterion
    norm::F
    threshold::Float64
    reason::String
    at_iteration::Int
    function StopWhenGradientNormLess(ε::TF; norm::F=norm) where {F,TF}
        return new{F,TF}(norm, ε, "", 0)
    end
end

function (sc::StopWhenGradientNormLess)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    M = get_manifold(mp)
    if i == 0 # reset on init
        sc.reason = ""
        sc.at_iteration = 0
    end
    if (i > 0)
        grad_norm = sc.norm(M, get_iterate(s), get_gradient(s))
        if grad_norm < sc.threshold
            sc.reason = "The algorithm reached approximately critical point after $i iterations; the gradient norm ($(grad_norm)) is less than $(sc.threshold).\n"
            sc.at_iteration = i
            return true
        end
    end
    return false
end
function status_summary(c::StopWhenGradientNormLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "|grad f| < $(c.threshold): $s"
end
indicates_convergence(c::StopWhenGradientNormLess) = true
function show(io::IO, c::StopWhenGradientNormLess)
    return print(io, "StopWhenGradientNormLess($(c.threshold))\n    $(status_summary(c))")
end

"""
    update_stopping_criterion!(c::StopWhenGradientNormLess, :MinGradNorm, v::Float64)

Update the minimal gradient norm when an algorithm shall stop
"""
function update_stopping_criterion!(
    c::StopWhenGradientNormLess, ::Val{:MinGradNorm}, v::Float64
)
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
mutable struct StopWhenStepsizeLess <: StoppingCriterion
    threshold::Float64
    reason::String
    at_iteration::Int
    function StopWhenStepsizeLess(ε::Float64)
        return new(ε, "", 0)
    end
end
function (c::StopWhenStepsizeLess)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    step = get_last_stepsize(p, s, i)
    if step < c.threshold && i > 0
        c.reason = "The algorithm computed a step size ($step) less than $(c.threshold).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenStepsizeLess)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Stepsize s < $(c.threshold):\t$s"
end
function show(io::IO, c::StopWhenStepsizeLess)
    return print(io, "StopWhenStepsizeLess($(c.threshold))\n    $(status_summary(c))")
end
"""
    update_stopping_criterion!(c::StopWhenStepsizeLess, :MinStepsize, v)

Update the minimal step size below which the algorithm shall stop
"""
function update_stopping_criterion!(c::StopWhenStepsizeLess, ::Val{:MinStepsize}, v)
    c.threshold = v
    return c
end

"""
    StopWhenCostNan <: StoppingCriterion

stop looking at the cost function of the optimization problem from within a [`AbstractManoptProblem`](@ref), i.e `get_cost(p,get_iterate(o))`.

# Constructor

    StopWhenCostNan()

initialize the stopping criterion to NaN.
"""
mutable struct StopWhenCostNan <: StoppingCriterion
    reason::String
    at_iteration::Int
    StopWhenCostNan() = new("", 0)
end
function (c::StopWhenCostNan)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if i > 0 && isnan(get_cost(p, get_iterate(s)))
        c.reason = "The algorithm reached a cost function value ($(get_cost(p,get_iterate(s)))).\n"
        c.at_iteration = 0
        return true
    end
    return false
end
function status_summary(c::StopWhenCostNan)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "f(x) is NaN:\t$s"
end
function show(io::IO, c::StopWhenCostNan)
    return print(io, "StopWhenCostNan()\n    $(status_summary(c))")
end

"""
    StopWhenIterNan <: StoppingCriterion

stop looking at the cost function of the optimization problem from within a [`AbstractManoptProblem`](@ref), i.e `get_cost(p,get_iterate(o))`.

# Constructor

    StopWhenIterNan()

initialize the stopping criterion to NaN.
"""
mutable struct StopWhenIterNan <: StoppingCriterion
    reason::String
    at_iteration::Int
    StopWhenIterNan() = new("", 0)
end
function (c::StopWhenIterNan)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if i > 0 && any(isnan.(get_iterate(s)))
        c.reason = "The algorithm reached a $(get_iterate(s)) iterate.\n"
        c.at_iteration = 0
        return true
    end
    return false
end
function status_summary(c::StopWhenIterNan)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "f(x) is NaN:\t$s"
end
function show(io::IO, c::StopWhenIterNan)
    return print(io, "StopWhenIterNan()\n    $(status_summary(c))")
end

@doc raw"""
    StopWhenSmallerOrEqual <: StoppingCriterion

A functor for an stopping criterion, where the algorithm if stopped when a variable is smaller than or equal to its minimum value.

# Fields
* `value` – stores the variable which has to fall under a threshold for the algorithm to stop
* `minValue` – stores the threshold where, if the value is smaller or equal to this threshold, the algorithm stops
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`get_reason`](@ref).

# Constructor

    StopWhenSmallerOrEqual(value, minValue)

initialize the stopifsmallerorequal functor to indicate to stop after `value` is smaller than or equal to `minValue`.
"""
mutable struct StopWhenSmallerOrEqual <: StoppingCriterion
    value::Symbol
    minValue::Real
    reason::String
    at_iteration::Int
    StopWhenSmallerOrEqual(value::Symbol, mValue::Real) = new(value, mValue, "", 0)
end
function (c::StopWhenSmallerOrEqual)(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if getfield(s, c.value) <= c.minValue
        c.reason = "The value of the variable ($(string(c.value))) is smaller than or equal to its threshold ($(c.minValue)).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenSmallerOrEqual)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Field :$(c.value) ≤ $(c.minValue):\t$s"
end
function show(io::IO, c::StopWhenSmallerOrEqual)
    return print(
        io, "StopWhenSmallerOrEqual(:$(c.value), $(c.minValue))\n    $(status_summary(c))"
    )
end

#
# Meta Criteria
#

@doc raw"""
    StopWhenAll <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _all_ indicate to stop. The `reason` is given by the concatenation of all
reasons.

# Constructor
    StopWhenAll(c::NTuple{N,StoppingCriterion} where N)
    StopWhenAll(c::StoppingCriterion,...)
"""
mutable struct StopWhenAll{TCriteria<:Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    reason::String
    StopWhenAll(c::Vector{StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), "")
    StopWhenAll(c...) = new{typeof(c)}(c, "")
end
function (c::StopWhenAll)(p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    (i == 0) && (c.reason = "") # reset on init
    if all(subC -> subC(p, s, i), c.criteria)
        c.reason = string([get_reason(subC) for subC in c.criteria]...)
        return true
    end
    return false
end
function status_summary(c::StopWhenAll)
    has_stopped = length(c.reason) > 0
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
    a = StopAfterIteration(200) & StopWhenChangeLess(1e-6)
    b = a & StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(1e-6))
    b = StopWhenAll(StopAfterIteration(200), StopWhenChangeLess(1e-6), StopWhenGradientNormLess(1e-6))
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

@doc raw"""
    StopWhenAny <: StoppingCriterion

store an array of [`StoppingCriterion`](@ref) elements and indicates to stop,
when _any_ single one indicates to stop. The `reason` is given by the
concatenation of all reasons (assuming that all non-indicating return `""`).

# Constructor
    StopWhenAny(c::NTuple{N,StoppingCriterion} where N)
    StopWhenAny(c::StoppingCriterion...)
"""
mutable struct StopWhenAny{TCriteria<:Tuple} <: StoppingCriterionSet
    criteria::TCriteria
    reason::String
    StopWhenAny(c::Vector{StoppingCriterion}) = new{typeof(tuple(c...))}(tuple(c...), "")
    StopWhenAny(c::StoppingCriterion...) = new{typeof(c)}(c, "")
end
function (c::StopWhenAny)(p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    (i == 0) && (c.reason = "") # reset on init
    if any(subC -> subC(p, s, i), c.criteria)
        c.reason = string((get_reason(subC) for subC in c.criteria)...)
        return true
    end
    return false
end
function status_summary(c::StopWhenAny)
    has_stopped = length(c.reason) > 0
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
    a = StopAfterIteration(200) | StopWhenChangeLess(1e-6)
    b = a | StopWhenGradientNormLess(1e-6)

Is the same as

    a = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(1e-6))
    b = StopWhenAny(StopAfterIteration(200), StopWhenChangeLess(1e-6), StopWhenGradientNormLess(1e-6))
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

@doc raw"""
    get_active_stopping_criteria(c)

returns all active stopping criteria, if any, that are within a
[`StoppingCriterion`](@ref) `c`, and indicated a stop, i.e. their reason is
nonempty.
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
    if c.reason != ""
        return [c] # recursion top
    else
        return []
    end
end

@doc raw"""
    get_reason(c)

return the current reason stored within a [`StoppingCriterion`](@ref) `c`.
This reason is empty if the criterion has never been met.
"""
get_reason(c::sC) where {sC<:StoppingCriterion} = c.reason

@doc raw"""
    get_reason(o)

return the current reason stored within the [`StoppingCriterion`](@ref) from
within the [`AbstractManoptSolverState`](@ref) This reason is empty if the criterion has never
been met.
"""
get_reason(s::AbstractManoptSolverState) = get_reason(get_state(s).stop)

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

@doc raw"""
    update_stopping_criterion!(c::Stoppingcriterion, s::Symbol, v::value)
    update_stopping_criterion!(s::AbstractManoptSolverState, symbol::Symbol, v::value)
    update_stopping_criterion!(c::Stoppingcriterion, ::Val{Symbol}, v::value)

Update a value within a stopping criterion, specified by the symbol `s`, to `v`.
If a criterion does not have a value assigned that corresponds to `s`, the update is ignored.

For the second signature, the stopping criterion within the [`AbstractManoptSolverState`](@ref) `o` is updated.

To see which symbol updates which value, see the specific stopping criteria. They should
use dispatch per symbol value (the third signature).
"""
update_stopping_criterion!(c, s, v)

function update_stopping_criterion!(s::AbstractManoptSolverState, symbol::Symbol, v)
    update_stopping_criterion!(s.stop, symbol, v)
    return s
end
function update_stopping_criterion!(c::StopWhenAll, s::Symbol, v)
    for d in c.criteria
        update_stopping_criterion!(d, s, v)
    end
    return c
end
function update_stopping_criterion!(c::StopWhenAny, s::Symbol, v)
    for d in c.criteria
        update_stopping_criterion!(d, s, v)
    end
    return c
end
function update_stopping_criterion!(c::StoppingCriterion, s::Symbol, v::Any)
    update_stopping_criterion!(c, Val(s), v)
    return c
end
# fallback: do nothing
function update_stopping_criterion!(c::StoppingCriterion, ::Val, v)
    return c
end
