@doc """
    AbstractManoptSolverState

A general super type for all solver states.

# Fields

The following fields are assumed to be available by default.
If you use different ones, adapt the access functions
[`get_iterate`](@ref), [`get_stopping_criterion`](@ref),
and [`get_callbacks`](@ref) accordingly.

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:callbacks; add_properties = [:as_dict]))
$(_fields(:stopping_criterion; name = "stop"))
"""
abstract type AbstractManoptSolverState end

@doc """
    AbstractGradientSolverState <: AbstractManoptSolverState

A generic [`AbstractManoptSolverState`](@ref) type for gradient based solver states.

It assumes that

* the iterate is stored in the field `p`
* the gradient at `p` is stored in `X`.

# See also

[`GradientDescentState`](@ref), [`StochasticGradientDescentState`](@ref), [`SubGradientMethodState`](@ref), [`QuasiNewtonState`](@ref).
"""
abstract type AbstractGradientSolverState <: AbstractManoptSolverState end

@doc """
    AbstractHessianSolverState <: AbstractGradientSolverState

An [`AbstractManoptSolverState`](@ref) type to represent algorithms that employ the Hessian.
These states are assumed to have a field `X` to store the current gradient ``$(_tex(:grad))f(p)``.
"""
abstract type AbstractHessianSolverState <: AbstractGradientSolverState end

"""
    AbstractRestartCondition

A general struct that indicates when to restart.
It is used within the [`ConjugateGradientDescentState`](@ref).

It is implemented to work as a functor `(problem, state, k) -> true|false`
and what is done in the restart case (`true`) is decided by the single solver.
"""
abstract type AbstractRestartCondition end

"""
    get_count(ams::AbstractManoptSolverState, ::Symbol)

Obtain the count for a certain countable size, for example the `:Iterations`.
This function returns 0 if there was nothing to count.

Available symbols from within the solver state:

* `:Iterations` is passed on to the `stop` field to obtain the
  iteration at which the solver stopped.
"""
function get_count(ams::AbstractManoptSolverState, s::Symbol)
    return get_count(ams, Val(s))
end

function get_count(ams::AbstractManoptSolverState, v::Val{:Iterations})
    return get_count(get_stopping_criterion(ams), v)
end
@doc """
    get_gradient(agst::AbstractGradientSolverState)

Return the gradient stored within the gradient solver state.
The default returns `agst.X`.
"""
get_gradient(agst::AbstractGradientSolverState) = agst.X

"""
    get_gradient(s::AbstractManoptSolverState)

Return the (last stored) gradient within the [`AbstractManoptSolverState`](@ref) `s`.
By default this also undecorates the state beforehand.
"""
get_gradient(s::AbstractManoptSolverState) = _get_gradient(s, dispatch_state_decorator(s))
function _get_gradient(s::AbstractManoptSolverState, ::Val{false})
    return error("It seems that $s does not provide access to a gradient")
end
_get_gradient(s::AbstractManoptSolverState, ::Val{true}) = get_gradient(s.state)

"""
    get_iterate(state::AbstractManoptSolverState)

Return the (last stored) iterate within the [`AbstractManoptSolverState`](@ref) `state`.
This should usually refer to a single point on the manifold the solver is working on.

By default this also removes all decorators of the state beforehand.
"""
get_iterate(s::AbstractManoptSolverState) = _get_iterate(s, dispatch_state_decorator(s))
function _get_iterate(s::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s does not provide access to an iterate.
        If it has the iterate stored internally, please implement `get_iterate(s::$(typeof(s)))`.
        ",
    )
end
_get_iterate(s::AbstractManoptSolverState, ::Val{true}) = get_iterate(s.state)

@doc """
    get_iterate(agst::AbstractGradientSolverState)

Return the iterate stored within the gradient solver state.
The default returns `agst.p`.
"""
get_iterate(agst::AbstractGradientSolverState) = agst.p

@doc """
    get_message(du::AbstractManoptSolverState)

Get a message (`String`) from internal functors, in a summary.
This should return any message a sub-step might have issued as well.
"""
function get_message(s::AbstractManoptSolverState)
    return _get_message(s, dispatch_state_decorator(s))
end
_get_message(s::AbstractManoptSolverState, ::Val{true}) = get_message(s.state)
#INtroduce a default that there is no message
_get_message(s::AbstractManoptSolverState, ::Val{false}) = ""

"""
    get_solver_return(s::AbstractManoptSolverState)

Determine the result value of a call to a solver.

By default this returns the same as [`get_solver_result`](@ref).
"""
function get_solver_return(s::AbstractManoptSolverState)
    return _get_solver_return(s, dispatch_state_decorator(s))
end
_get_solver_return(s::AbstractManoptSolverState, ::Val{false}) = get_solver_result(s)
_get_solver_return(s::AbstractManoptSolverState, ::Val{true}) = get_solver_return(s.state)

# also work in combination with the objective
"""
    get_solver_return(o::AbstractManifoldObjective, s::AbstractManoptSolverState)

Determine the result value of a call to a solver.

By default this returns the same as [`get_solver_result`](@ref).
"""
function get_solver_return(o::AbstractManifoldObjective, s::AbstractManoptSolverState)
    #resolve objective first
    return _get_solver_return(o, s, dispatch_objective_decorator(o))
end
# remove decorator
function _get_solver_return(o::AbstractManifoldObjective, s, ::Val{true})
    return get_solver_return(get_objective(o, false), s)
end
_get_solver_return(::AbstractManifoldObjective, s, ::Val{false}) = get_solver_return(s)
"""
    get_solver_return(o::ReturnManifoldObjective, s::AbstractManoptSolverState)

Return both the objective and the state as a tuple.
"""
function get_solver_return(o::ReturnManifoldObjective, s::AbstractManoptSolverState)
    return o.objective, get_solver_return(s)
end

"""
    get_solver_result(state::AbstractManoptSolverState)
    get_solver_result(tos::Tuple{AbstractManifoldObjective,AbstractManoptSolverState})
    get_solver_result(objective::AbstractManifoldObjective, state::AbstractManoptSolverState)
    get_solver_result(problem::AbstractManoptProblem, state::AbstractManoptSolverState)

Return the final result after all iterations that is stored within
the [`AbstractManoptSolverState`](@ref) `state`, which was modified during the iterations.

For the case that an [`AbstractManifoldObjective`](@ref) `objective` is passed as well
– either as a tuple or as two parameters –, by default the objective is ignored,
and the solver result for the state is returned; this is for display reasons in the REPL
related to statistics, where such a tuple might appear.

For the case that an [`AbstractManoptProblem`](@ref) `problem` is passed as
a first optional parameter, by default the problem is ignored.
This can be used to change the representation of a result stored in a state, for example
when a tangent vector is (part of) the result, changing between representations in
coefficients and different tangent vector representations could be performed as a final step,
depending on which problem was aimed to be solved.

Note that the returned value or point might still be aliased to the original `state`.
"""
function get_solver_result(state::AbstractManoptSolverState)
    return _get_solver_result(state, dispatch_state_decorator(state))
end
function get_solver_result(
        tos::Tuple{<:AbstractManifoldObjective, <:AbstractManoptSolverState}
    )
    return get_solver_result(tos...)
end
function get_solver_result(::AbstractManifoldObjective, state::AbstractManoptSolverState)
    return get_solver_result(state)
end
#A problem or – hence untyped – a closed form solution / function
function get_solver_result(pf, state::AbstractManoptSolverState)
    return get_solver_result(state)
end
function get_solver_result(tos::Tuple{<:AbstractManifoldObjective, S}) where {S}
    return tos[2]
end
# if the second one is anything else, assume it is a point/result -> return that
function get_solver_result(::AbstractManifoldObjective, p)
    return p
end
_get_solver_result(state::AbstractManoptSolverState, ::Val{false}) = get_iterate(state)
_get_solver_result(state::AbstractManoptSolverState, ::Val{true}) = get_solver_result(state.state)


@doc """
    get_state(s::AbstractManoptSolverState, recursive::Bool=true)

Return the undecorated [`AbstractManoptSolverState`](@ref) of the (possibly) decorated `s`.
As long as your decorated state stores the state within `s.state` and
[`dispatch_state_decorator`](@ref) is set to `Val{true}`,
the internal state is extracted automatically.

By default the state that is stored within a decorated state is assumed to be at
`s.state`. Overwrite `_get_state(s, ::Val{true}, recursive)` to change this behaviour for your state `s`
for both the recursive and the direct case.

If `recursive` is set to `false`, only the most outer decorator is taken away instead of all.
"""
function get_state(s::AbstractManoptSolverState, recursive::Bool = true)
    return _get_state(s, dispatch_state_decorator(s), recursive)
end
_get_state(s::AbstractManoptSolverState, ::Val{false}, rec = true) = s
function _get_state(s::AbstractManoptSolverState, ::Val{true}, rec = true)
    return rec ? get_state(s.state) : s.state
end

@doc """
    get_stopping_criterion(ams::AbstractManoptSolverState)

Return the [`StoppingCriterion`](@ref) stored within the [`AbstractManoptSolverState`](@ref) `ams`.

For an undecorated state, this is assumed to be in `ams.stop`.
Overwrite `_get_stopping_criterion(yms::YMS)`
to change this for your `Manopt.jl` solver `yms`, assuming it has type `YMS`.
"""
function get_stopping_criterion(ams::AbstractManoptSolverState)
    return _get_stopping_criterion(get_state(ams, true))
end
_get_stopping_criterion(ams::AbstractManoptSolverState) = ams.stop

"""
    has_converged(ams::AbstractManoptSolverState)

Return whether the solver has converged, based on the internal [`StoppingCriterion`](@ref).
"""
has_converged(ams::AbstractManoptSolverState) = has_converged(get_stopping_criterion(ams))

@doc """
    set_gradient!(state::AbstractGradientSolverState, M, p, X)

Set the (current) gradient stored within an [`AbstractGradientSolverState`](@ref) to `X`.
The default function modifies `state.X`.
"""
function set_gradient!(state::AbstractGradientSolverState, M, p, X)
    copyto!(M, state.X, p, X)
    return state
end

"""
    set_gradient!(s::AbstractManoptSolverState, M::AbstractManifold, p, X)

Set the gradient within a (possibly decorated) [`AbstractManoptSolverState`](@ref)
to some (start) value `X` in the tangent space at `p`.
"""
function set_gradient!(s::AbstractManoptSolverState, M, p, X)
    return _set_gradient!(s, M, p, X, dispatch_state_decorator(s))
end
function _set_gradient!(s::AbstractManoptSolverState, ::Any, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s does not provide (write) access to a gradient",
    )
end
function _set_gradient!(s::AbstractManoptSolverState, M, p, X, ::Val{true})
    return set_gradient!(s.state, M, p, X)
end

"""
    set_iterate!(s::AbstractManoptSolverState, M::AbstractManifold, p)

Set the iterate within an [`AbstractManoptSolverState`](@ref) to some (start) value `p`.
"""
function set_iterate!(s::AbstractManoptSolverState, M, p)
    return _set_iterate!(s, M, p, dispatch_state_decorator(s))
end
function _set_iterate!(s::AbstractManoptSolverState, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s does not provide (write) access to an iterate",
    )
end
_set_iterate!(s::AbstractManoptSolverState, M, p, ::Val{true}) = set_iterate!(s.state, M, p)

"""
    set_parameter!(ams::AbstractManoptSolverState, element::Symbol, args...)

Set a certain field or semantic element from the [`AbstractManoptSolverState`](@ref) `ams` to a value specified by `args...`.
This function passes to `Val(element)` and specific setters should dispatch on `Val{element}`.

By default, this function just does nothing.
"""
set_parameter!(ams::AbstractManoptSolverState, e::Symbol, args...)

# Default: do nothing
function set_parameter!(ams::AbstractManoptSolverState, ::Val, args...)
    return ams
end

@doc """
    stopped_at(state::AbstractManoptSolverState)

Return the number of iterations the solver represented by the `state` took to stop.
If the solver has not yet stopped, this function returns `-1`.

By default, this function calls the `get_count` function on the state's stopping criterion to access its `:Iterations` count.
"""
function stopped_at(state::AbstractManoptSolverState)
    return get_count(get_stopping_criterion(state), Val(:Iterations))
end

function Base.show(io::IO, ::MIME"text/plain", ams::AbstractManoptSolverState)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, ams) : show(io, ams)
end

#
#
# ---
@doc """
    AbstractPrimalDualSolverState

A general type for all primal dual based states to be used within primal dual
based algorithms.
"""
abstract type AbstractPrimalDualSolverState <: AbstractManoptSolverState end

function dual_residual(
        M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective,
        apds::AbstractPrimalDualSolverState, p_old, X_old, n_old,
    )
    if apds.variant === :linearized
        return norm(
            N,
            apds.n,
            1 / apds.dual_stepsize * (
                vector_transport_to(N, n_old, X_old, apds.n, apds.vector_transport_method_dual) - apds.X
            ) - linearized_forward_operator(
                M, N, apdmo, apds.m,
                vector_transport_to(
                    M, apds.p,
                    inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method),
                    apds.m, apds.vector_transport_method,
                ),
                apds.n,
            ),
        )
    elseif apds.variant === :exact
        return norm(
            N,
            apds.n,
            1 / apds.dual_stepsize * (
                vector_transport_to(N, n_old, X_old, apds.n, apds.vector_transport_method_dual) - apds.X
            ) - inverse_retract(
                N, apds.n,
                forward_operator(
                    M, N, apdmo,
                    retract(
                        M, apds.m,
                        vector_transport_to(
                            M, apds.p,
                            inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method),
                            apds.m, apds.vector_transport_method,
                        ),
                        apds.retraction_method,
                    ),
                ),
                apds.inverse_retraction_method_dual,
            ),
        )
    else
        throw(
            DomainError(
                apds.variant, "Unknown Chambolle-Pock variant, allowed are `:exact` or `:linearized`.",
            ),
        )
    end
end
function primal_residual(
        M::AbstractManifold, N::AbstractManifold, apdmo::AbstractPrimalDualManifoldObjective,
        apds::AbstractPrimalDualSolverState, p_old, X_old, n_old,
    )
    return norm(
        M,
        apds.p,
        1 / apds.primal_stepsize *
            inverse_retract(M, apds.p, p_old, apds.inverse_retraction_method) -
            vector_transport_to(
            M, apds.m,
            adjoint_linearized_operator(
                M, N, apdmo, apds.m, apds.n,
                vector_transport_to(N, n_old, X_old, apds.n, apds.vector_transport_method_dual) - apds.X,
            ),
            apds.p, apds.vector_transport_method,
        ),
    )
end
