"""
    Stepsize

An abstract type for the functors representing step sizes. These are callable
structures. The naming scheme is `TypeOfStepSize`, for example `ConstantStepsize`.

Every Stepsize has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`AbstractManoptProblem`](@ref) as well as [`AbstractManoptSolverState`](@ref)
and the current number of iterations are the arguments
and returns a number, namely the stepsize to use.

The functor usually should accept arbitrary keyword arguments. Common ones used are
* `gradient=nothing`: to pass a pre-calculated gradient, otherwise it is computed.

For most it is advisable to employ a [`ManifoldDefaultsFactory`](@ref). Then
the function creating the factory should either be called `TypeOf` or if that is confusing or too generic, `TypeOfLength`

# See also

[`Linesearch`](@ref)
"""
abstract type Stepsize end

get_message(::S) where {S <: Stepsize} = ""

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

By default, this returns $(_link(:injectivity_radius))`(M)`, if this exists.
If this is not available on the the method returns `Inf`.
"""
function max_stepsize(M::AbstractManifold, p)
    s = try
        injectivity_radius(M, p)
    catch
        is_tutorial_mode() &&
            @warn "`max_stepsize was called, but there seems to not be an `injectivity_raidus` available on $M."
        Inf
    end
    return s
end
function max_stepsize(M::AbstractManifold)
    s = try
        injectivity_radius(M)
    catch
        is_tutorial_mode() &&
            @warn "`max_stepsize was called, but there seems to not be an `injectivity_raidus` available on $M."
        Inf
    end
    return s
end

"""
    Linesearch <: Stepsize

An abstract functor to represent line search type step size determinations, see
[`Stepsize`](@ref) for details. One example is the [`ArmijoLinesearchStepsize`](@ref)
functor.

Compared to simple step sizes, the line search functors provide an interface of
the form `(p,o,i,X) -> s` with an additional (but optional) fourth parameter to
provide a search direction; this should default to something reasonable,
most prominently the negative gradient.
"""
abstract type Linesearch <: Stepsize end

@doc """
    s = linesearch_backtrack(M, F, p, X, s, decrease, contract η = -X, f0 = f(p); kwargs...)
    s = linesearch_backtrack!(M, q, F, p, X, s, decrease, contract η = -X, f0 = f(p); kwargs...)

perform a line search

* on manifold `M`
* for the cost function `f`,
* at the current point `p`
* with current gradient provided in `X`
* an initial stepsize `s`
* a sufficient `decrease`
* a `contract`ion factor ``σ``
* a search direction ``η = -X``
* an offset, ``f_0 = F(x)``

## Keyword arguments

$(_var(:Keyword, :retraction_method))
* `stop_when_stepsize_less=0.0`: to avoid numerical underflow
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M, p) / norm(M, p, η)`) to avoid leaving the injectivity radius on a manifold
* `stop_increasing_at_step=100`: stop the initial increase of step size after these many steps
* `stop_decreasing_at_step=`1000`: stop the decreasing search after these many steps
* `report_messages_in::NamedTuple = (; )`: a named tuple of [`StepsizeMessage`](@ref)s to report messages in.
  currently supported keywords are `:non_descent_direction`, `:stepsize_exceeds`, `:stepsize_less`, `:stop_increasing`, `:stop_decreasing`
* `additional_increase_condition=(M,p) -> true`: impose an additional condition for an increased step size to be accepted
* `additional_decrease_condition=(M,p) -> true`: impose an additional condition for an decreased step size to be accepted

  These keywords are used as safeguards, where only the max stepsize is a very manifold specific one.

# Return value

A stepsize `s` and a message `msg` (in case any of the 4 criteria hit)
"""
function linesearch_backtrack(
        M::AbstractManifold, f, p, X::T, s, decrease, contract, η::T = (-X), f0 = f(M, p); kwargs...
    ) where {T}
    q = allocate(M, p)
    return linesearch_backtrack!(M, q, f, p, X, s, decrease, contract, η, f0; kwargs...)
end

"""
    s = linesearch_backtrack!(M, q, F, p, X, s, decrease, contract η = -X, f0 = f(p))

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
        η::T = (-X),
        f0 = f(M, p);
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        additional_increase_condition = (M, p) -> true,
        additional_decrease_condition = (M, p) -> true,
        stop_when_stepsize_less = 0.0,
        stop_when_stepsize_exceeds = max_stepsize(M, p) / norm(M, p, η),
        stop_increasing_at_step = 100,
        stop_decreasing_at_step = 1000,
        report_messages_in::NamedTuple = (;),
    ) where {TF, T}
    ManifoldsBase.retract_fused!(M, q, p, η, s, retraction_method)
    f_q = f(M, q)
    search_dir_inner = real(inner(M, p, η, X))
    if search_dir_inner >= 0
        set_message!(report_messages_in, :non_descent_direction, at = 0, value = search_dir_inner)
    end

    i = 0
    # Ensure that both the original condition and the additional one are fulfilled afterwards
    while f_q < f0 + decrease * s * search_dir_inner || !additional_increase_condition(M, q)
        (stop_increasing_at_step == 0) && break
        i = i + 1
        s = s / contract
        ManifoldsBase.retract_fused!(M, q, p, η, s, retraction_method)
        f_q = f(M, q)
        if i == stop_increasing_at_step
            set_message!(report_messages_in, :stop_increasing, at = i, bound = stop_increasing_at_step, value = s)
            break
        end
        if s > stop_when_stepsize_exceeds
            set_message!(report_messages_in, :stepsize_exceeds, at = i, bound = stop_when_stepsize_exceeds, value = s)
            break
        end
    end
    i = 0
    # Ensure that both the original condition and the additional one are fulfilled afterwards
    while (f_q > f0 + decrease * s * search_dir_inner) ||
            (!additional_decrease_condition(M, q))
        i = i + 1
        s = contract * s
        ManifoldsBase.retract_fused!(M, q, p, η, s, retraction_method)
        f_q = f(M, q)
        if i == stop_decreasing_at_step
            set_message!(report_messages_in, :stop_decreasing, at = i, bound = stop_decreasing_at_step, value = s)
            break
        end
        if s < stop_when_stepsize_less
            set_message!(report_messages_in, :stepsize_less, at = i, bound = stop_when_stepsize_less, value = s)
            break
        end
    end
    return s
end
