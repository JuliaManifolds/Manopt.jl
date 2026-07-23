@doc """
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

function Base.show(io::IO, ::MIME"text/plain", asc::StoppingCriterion)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, asc) : show(io, asc)
end


"""
    indicates_convergence(c::StoppingCriterion)

Return whether a [`StoppingCriterion`](@ref) does _always_
mean that, when it indicates to stop, the solver has converged to a
minimizer or critical point.

Note that this is independent of the actual state of the stopping criterion,
whether some of them indicate to stop, but a purely type-based, static
decision.

# Examples

With `s1=StopAfterIteration(20)` and `s2=StopWhenGradientNormLess(1e-7)` the indicator yields

* `indicates_convergence(s1)` is `false`
* `indicates_convergence(s2)` is `true`
* `indicates_convergence(s1 | s2)` is `false`, since this might also stop after 20 iterations,
  or in other words, for [`StopWhenAny`](@ref) _all_ its criteria have to indicate convergence, for this to return true.
* `indicates_convergence(s1 & s2)` is `true`, since `s2` is fulfilled if this stops.
"""
indicates_convergence(c::StoppingCriterion) = false

"""
    has_converged(c::StoppingCriterion)

Return whether a [`StoppingCriterion`](@ref) that has indicated to stop _and_ is a stopping criterion
that allows to conclude that the corresponding solver has converged.

By default this is given by the static [`indicates_convergence`](@ref)`(c)` as well as
the test whether the stopping criterion has stopped.
For some stopping criteria, for example [`StopWhenAny`](@ref) a more advanced test can be done,
that is more precise.

# Examples
With `s1=StopAfterIteration(20)` and `s2=StopWhenGradientNormLess(1e-7)` we obtain

* `has_converged(s1)` is always `false` (even if it has stopped)
* `has_converged(s2)` is always `true` as soon as it has stopped
* `has_converged(s1 | s2)` is always `true` if it has stopped _and_ `s2` is the reason for that.
* `has_converged(s1 & s2)` is `true` as soon as the algorithm stopped, since here `s2` always
"""
has_converged(c::StoppingCriterion) = indicates_convergence(c) && (get_count(c, Val(:Iterations)) >= 0)

function get_count(c::StoppingCriterion, ::Val{:Iterations})
    if hasfield(typeof(c), :at_iteration)
        return getfield(c, :at_iteration)
    else
        return 0
    end
end
