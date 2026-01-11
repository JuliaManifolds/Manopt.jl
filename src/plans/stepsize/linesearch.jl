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

_doc_linesearch_backtrack = """
    s = linesearch_backtrack(M, F, p, s, decrease, contract, η; kwargs...)
    s = linesearch_backtrack!(M, q, F, p, s, decrease, contract, η; kwargs...)

perform a line search along ``\ell_f(s) = f($(_tex(:retr))_p(sη)`` to find a stepsize `s`.
See [NocedalWright:2006; Section 3](@cite) for details.

The linesearch starts with a first phase where the stepsize is increased as ``s ↦ s / σ``
until

```math
f($(_tex(:retr))_p(sη)) ≥ f(p) + a * s * Df(p)[η]
````

where ``a`` is the `decrease` parameter, and ``Df(p)[η]`` is the directional derivative.

Then the actual backtracking phase starts, where the stepsize is decreased as ``s ↦ σ s``
until
```math
f($(_tex(:retr))_p(sη)) ≤ f(p) + b * s * Df(p)[η]
```

where ``b`` is the `decrease` parameter.

This can be done in-place, where `q` is the point to store the point reached in.

Both phases have a safeguard on the maximal number of steps to perform as well as an
upper and lower bound for the stepsize, respectively.
The upper bound is a special case on manifolds to avoid exceeding the injectivity radius.
Furthermore, both phases can be equipped with additional conditions to be fulfilled in order to
accept the current stepsize.

## Arguments

* on manifold `M`
* for the cost function `f`,
* at the current point `p`
* an initial stepsize `s`
* a sufficient `decrease`
* a `contract`ion factor ``σ``
* a search direction ``η``

## Keyword arguments

$(_kwargs(:retraction_method))
* `additional_increase_condition=(M,p) -> true`: impose an additional condition for an increased step size to be accepted
* `additional_decrease_condition=(M,p) -> true`: impose an additional condition for an decreased step size to be accepted
* `Dlf0`: precomputed directional derivative at point `p` in direction `η`
  if the `gradient` is specified, this is computed as the real part of `inner(M, p, gradient, η)`, otherwise it it nothing
* `lf0 = f(M, p)`: the function value at the initial point `p`
* `gradient = nothing`: precomputed gradient at point `p`
* `report_messages_in::NamedTuple = (; )`: a named tuple of [`StepsizeMessage`](@ref)s to report messages in.
  currently supported keywords are `:non_descent_direction`, `:stepsize_exceeds`, `:stepsize_less`, `:stop_increasing`, `:stop_decreasing`
* `stop_when_stepsize_less=0.0`: to avoid numerical underflow
* `stop_when_stepsize_exceeds=`[`max_stepsize`](@ref)`(M, p) / norm(M, p, η)`) to avoid leaving the injectivity radius on a manifold
* `stop_increasing_at_step=100`: stop the initial increase of step size after these many steps
* `stop_decreasing_at_step=`1000`: stop the decreasing search after these many steps

  These keywords are used as safeguards, where only the max stepsize is a very manifold specific one.

# Return value

A stepsize `s` and a message `msg` (in case any of the 4 criteria hit)
"""

@doc "$_doc_linesearch_backtrack"
function linesearch_backtrack(
        M::AbstractManifold, f, p, s, decrease, contract, η; kwargs...
    )
    q = allocate(M, p)
    return linesearch_backtrack!(M, q, f, p, s, decrease, contract, η; kwargs...)
end

@doc "$_doc_linesearch_backtrack"
function linesearch_backtrack!(
        M::AbstractManifold,
        q,
        f::TF,
        p,
        s,
        decrease,
        contract,
        η::T;
        lf0 = f(M, p),
        gradient = nothing,
        Dlf0 = isnothing(gradient) ? nothing : real(inner(M, p, gradient, η)),
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
    if Dlf0 >= 0
        set_message!(report_messages_in, :non_descent_direction, at = 0, value = Dlf0)
    end

    i = 0
    # Ensure that both the original condition and the additional one are fulfilled afterwards
    while f_q < lf0 + decrease * s * Dlf0 || !additional_increase_condition(M, q)
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
    while (f_q > lf0 + decrease * s * Dlf0) ||
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
