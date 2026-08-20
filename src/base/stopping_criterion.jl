@doc """
    StoppingCriterion <: AbstractManifoldFunction

An abstract type for the functions representing stopping criteria, so they are callable structures.
The naming scheme follows functions, see for example [`StopAfterIteration`](@ref).

Every `StoppingCriterion` has to provide a constructor and its function has to have
the interface `(amp, ams, k)` where an [`AbstractManoptProblem`](@ref) `amp` as well as an [`AbstractManoptSolverState`](@ref) `ams`
and the current iteration `k` are the arguments.
The function always returns a boolean indicating whether to stop or not.

By default each `StoppingCriterion` should provide a field `at_iteration` to provide
the iteration it (last) indicated to stop. Further fields should provide enough information,
such that a human-readable reason can be assembled, see [`get_reason`](@ref).
"""
abstract type StoppingCriterion <: AbstractManifoldFunction end

@doc """
    StoppingCriterionSet <: StoppingCriterion

An abstract type for a stopping criterion that itself consists of a set of
stopping criteria. In total it acts as a stopping criterion itself. Examples
are [`StopWhenAny`](@ref) and [`StopWhenAll`](@ref) that can be used to
combine stopping criteria.
"""
abstract type StoppingCriterionSet <: StoppingCriterion end

"""
    has_converged(c::StoppingCriterion)

Return whether a [`StoppingCriterion`](@ref) `c` has indicated to stop _and_ is a stopping criterion
that allows one to conclude that the corresponding solver has converged.

By default this is given by the static [`indicates_convergence`](@ref)`(c)` as well as
the test whether the stopping criterion has stopped.
For some stopping criteria, for example [`StopWhenAny`](@ref) a more advanced test can be done,
that is more precise.

# Examples

With `s1=StopAfterIteration(20)` and `s2=StopWhenGradientNormLess(1e-7)` we obtain

* `has_converged(s1)` is always `false` (even if it has stopped)
* `has_converged(s2)` is always `true` as soon as it has stopped
* `has_converged(s1 | s2)` is `true` if it has stopped _and_ `s2` is the reason for that.
* `has_converged(s1 & s2)` is `true` as soon as the algorithm stopped, since here `s2` always
  has to be fulfilled as well.
"""
has_converged(c::StoppingCriterion) = indicates_convergence(c) && (get_count(c, Val(:Iterations)) >= 0)

function get_count(c::StoppingCriterion, ::Val{:Iterations})
    if hasfield(typeof(c), :at_iteration)
        return getfield(c, :at_iteration)
    else
        return 0
    end
end

@doc """
    get_active_stopping_criteria(c)

Return all active stopping criteria, if any, that are within a
[`StoppingCriterion`](@ref) `c`, and indicated a stop, that is their reason is nonempty.
To be precise for a simple stopping criterion, this returns either an empty
array if no stop is indicated or the stopping criterion as the only element of
an array. For a [`StoppingCriterionSet`](@ref) all internal (even nested)
criteria that indicate to stop are returned.
"""
function get_active_stopping_criteria(c::sCS) where {sCS <: StoppingCriterionSet}
    c = get_active_stopping_criteria.(get_stopping_criteria(c))
    return vcat(c...)
end
# for non-array containing stopping criteria, the recursion ends in either
# returning nothing or an 1-element array containing itself
function get_active_stopping_criteria(c::sC) where {sC <: StoppingCriterion}
    if is_active_stopping_criterion(c)
        return [c] # recursion top
    else
        return []
    end
end


function get_reason end

@doc """
    get_reason(s::StoppingCriterion)

Return the reason why a [`StoppingCriterion`](@ref) has indicated to stop.
This reason is empty (`""`) if the criterion has never been met.
"""
get_reason(s::StoppingCriterion)

@doc """
    get_reason(s::AbstractManoptSolverState)

Return the current reason stored within the [`StoppingCriterion`](@ref) from
within the [`AbstractManoptSolverState`](@ref).
This reason is empty (`""`) if the criterion has never been met.
"""
get_reason(s::AbstractManoptSolverState) = get_reason(get_state(s).stop)


function get_stopping_criteria end
@doc """
    get_stopping_criteria(c::StoppingCriterionSet)

Return the array of internally stored [stopping criteria](@ref StoppingCriterion)
for a [`StoppingCriterionSet`](@ref) `c`.
"""
get_stopping_criteria(c::StoppingCriterionSet)

"""
    indicates_convergence(c::StoppingCriterion)

Return whether a [`StoppingCriterion`](@ref) does _always_
mean that, when it indicates to stop, the solver has converged to a
minimizer or critical point.

Note that this is independent of the actual state of the stopping criterion,
that is, of whether it currently indicates to stop; it is a purely type-based,
static decision.

# Examples

With `s1=StopAfterIteration(20)` and `s2=StopWhenGradientNormLess(1e-7)` the indicator yields

* `indicates_convergence(s1)` is `false`
* `indicates_convergence(s2)` is `true`
* `indicates_convergence(s1 | s2)` is `false`, since this might also stop after 20 iterations,
  or in other words, for [`StopWhenAny`](@ref) _all_ its criteria have to indicate convergence for this to return `true`.
* `indicates_convergence(s1 & s2)` is `true`, since `s2` is fulfilled if this stops.
"""
indicates_convergence(c::StoppingCriterion) = false

"""
    is_active_stopping_criterion(c::StoppingCriterion)

Return whether a [`StoppingCriterion`](@ref) is active, i.e. it has been called and indicates to stop.
"""
is_active_stopping_criterion(c::StoppingCriterion) = (c.at_iteration >= 0)

# pass down from state to stopping criterion
function set_parameter!(s::AbstractManoptSolverState, ::Val{:StoppingCriterion}, args...)
    set_parameter!(s.stop, args...)
    return s
end

function Base.show(io::IO, ::MIME"text/plain", asc::StoppingCriterion)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, asc) : show(io, asc)
end
