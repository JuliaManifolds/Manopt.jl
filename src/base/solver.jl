"""
    ClosedFormSubSolverState <: AbstractManoptSolverState

Subsolver state indicating that a closed-form solution is available.

# Constructor

    ClosedFormSubSolverState()
"""
struct ClosedFormSubSolverState{} <: AbstractManoptSolverState end
Base.show(io::IO, ::ClosedFormSubSolverState) = print(io, "ClosedFormSubSolverState()")
status_summary(cfss::ClosedFormSubSolverState; context::Symbol = :default) = repr(cfss)

@doc """
    ReturnSolverState{O<:AbstractManoptSolverState} <: AbstractManoptSolverState

This internal type is used to indicate that the contained [`AbstractManoptSolverState`](@ref) `state`
should be returned at the end of a solver instead of the usual minimizer.

# See also

[`get_solver_result`](@ref)
"""
struct ReturnSolverState{S <: AbstractManoptSolverState} <: AbstractManoptSolverState
    state::S
end
status_summary(rst::ReturnSolverState; context::Symbol = :default) = status_summary(rst.state; context = context)
show(io::IO, rst::ReturnSolverState) = print(io, "ReturnSolverState(", rst.state, ")")
dispatch_state_decorator(::ReturnSolverState) = Val(true)

doc_get_solver_return = """
    get_solver_return(s::ReturnSolverState)
    get_solver_return(o::AbstractManifoldObjective, s::ReturnSolverState)

Determine the solver return.

Return the internally stored state of the [`ReturnSolverState`](@ref) instead of the minimizer.

Since a solver might return both a state and an objective in a tuple, this then re-iterates on the second argument.
"""

@doc "$(doc_get_solver_return)"
get_solver_return(s::ReturnSolverState) = s.state

function decorate_state! end

@doc """
    decorate_state!(s::AbstractManoptSolverState; kwargs...)

Decorate the [`AbstractManoptSolverState`](@ref) `s` with specific decorators.

# Optional arguments

The optional arguments provide necessary details on the decorators.

* `callback=missing`: (deprecated) add an arbitrary (simple) callback function `cb()` to be called every iteration.
* `debug=Array{Union{Symbol,DebugAction,String,Int, Function},1}()`: a set of symbols
  representing [`DebugAction`](@ref)s, `Strings` used as dividers and a sub-sampling
  integer. These are passed as a [`DebugGroup`](@ref) within `:Iteration` to the
  [`DebugSolverState`](@ref) decorator dictionary. A function is added as a (non-simple) callback within a [`DebugCallback`](@ref).
  Only exception is `:Stop` that is passed to `:Stop`.
* `record=Array{Union{Symbol,RecordAction,Int},1}()`: specify recordings
  by using `Symbol`s or [`RecordAction`](@ref)s directly.
  An integer can again be used for only recording every ``k``-th iteration.
* `return_state=false`: indicate whether to wrap the state in a [`ReturnSolverState`](@ref),
  indicating that the solver should return the state and not (only) the minimizer.

All other keywords are ignored.

# See also

[`DebugSolverState`](@ref), [`RecordSolverState`](@ref), [`ReturnSolverState`](@ref)
"""
decorate_state!(s::AbstractManoptSolverState; kwargs...)

function decorate_state!(
        s::S;
        debug::Union{
            Missing, # none
            Function, # a function to indicate a (non-simple) callback
            DebugAction, # single one -> to :Iteration
            Array{DebugAction, 1}, # a group -> to :Iteration
            Dict{Symbol, DebugAction}, # the most elaborate, a dictionary
            Array{<:Any, 1}, # short hand for Factory.
        } = missing,
        record::Union{
            Missing, # none
            Symbol, # single action shortcut by symbol
            RecordAction, # single action -> to :Iteration
            Array{RecordAction, 1}, # a group -> to :Iteration
            Dict{Symbol, RecordAction}, # a dictionary for precise settings
            Array{<:Any, 1}, # a formatted string with symbols or AbstractStateActions
        } = missing,
        callback = missing, # a (simple) callback function – deprecated
        return_state = false,
        kwargs..., # ignore all others
    ) where {S <: AbstractManoptSolverState}
    deco_s = s
    # Add callback to debug parameter
    if !ismissing(callback) # we got a simple callback
        @warn """
            the `callback =` keyword/decorator step is deprecated, use
            `callbacks = [:Step => [...]]` to add your callback to the (end of)
            an iteration step
        """
        if ismissing(debug)
            debug = DebugCallback(callback; simple = true)
        else
            # From complex to simple, first array, since the other ones create an array
            (debug isa Array) && (debug = [debug..., DebugCallback(callback; simple = true)])
            if ((debug isa Function) || (debug isa DebugAction))
                debug = [debug, DebugCallback(callback; simple = true)]
            end
            (debug isa Dict) && @warn(
                "Adding callback to decorator too complicated; Callback ignored. Please add it to your Dictionary at :Iteration as a `DebugCallback` manually",
            )
        end
    end
    if !ismissing(debug) && !(debug isa AbstractArray && length(debug) == 0)
        deco_s = DebugSolverState(s, debug)
    end
    if !ismissing(record) && !(record isa AbstractArray && length(record) == 0)
        deco_s = RecordSolverState(deco_s, record)
    end
    deco_s = (return_state) ? ReturnSolverState(deco_s) : deco_s
    return deco_s
end

function decorate_objective! end
@doc """
    decorate_objective!(M, o::AbstractManifoldObjective; kwargs...)

Decorate the [`AbstractManifoldObjective`](@ref) `o` with specific decorators.

# Optional arguments

The optional arguments provide necessary details on the decorators;
providing one of them activates the corresponding decorator.

* `cache=missing`: specify a cache. Currently `:Simple` is supported and `:LRU` if you
  load [`LRUCache.jl`](https://github.com/JuliaCollections/LRUCache.jl).
  For this case a tuple specifying what to cache and how many entries to keep has to be provided.
  For example `(:LRU, [:Cost, :Gradient], 10)` states that the last 10 used cost function
  evaluations and gradient evaluations should be stored. See [`objective_cache_factory`](@ref) for details.
* `count=missing`: specify which calls to the objective should be counted, see [`ManifoldCountObjective`](@ref) for the full list.
* `objective_type=:Riemannian`: specify that an objective is `:Riemannian` or `:Euclidean`.
  The `:Euclidean` symbol is equivalent to specifying it as `:Embedding`, since in the end,
  both refer to converting an objective from the embedding (whether it is Euclidean or not)
  to the Riemannian one.
* `return_objective=false`: indicate whether to wrap the objective in a [`ReturnManifoldObjective`](@ref),
  indicating that the solver should return the objective as well.

# See also

[`objective_cache_factory`](@ref)
"""
decorate_objective!(M::AbstractManifold, o::AbstractManifoldObjective; kwargs...)

function decorate_objective!(
        M::AbstractManifold, o::O;
        cache::Union{
            Missing, Symbol, Tuple{Symbol, <:AbstractArray}, Tuple{Symbol, <:AbstractArray, <:Any},
        } = missing,
        count::Union{Missing, AbstractVector{<:Symbol}} = missing,
        objective_type::Symbol = :Riemannian,
        p = objective_type == :Riemannian ? missing : rand(M),
        _embedded_p = objective_type == :Riemannian ? missing : embed(M, p),
        _embedded_X = objective_type == :Riemannian ? missing : embed(M, p, zero_vector(M, p)),
        return_objective = false,
        kwargs...,
    ) where {O <: AbstractManifoldObjective}
    # Order:
    # 1) wrap embedding,
    # 2) _then_ count
    # 3) _then_ cache,
    # count should not be affected by 1) but cache should be on manifold not embedding
    # => only count _after_ cache misses
    # and always last wrapper: `ReturnManifoldObjective`.
    deco_o = o
    if objective_type ∈ [:Embedding, :Euclidean]
        deco_o = EmbeddedManifoldObjective(o, _embedded_p, _embedded_X)
    end
    (!ismissing(count)) && (deco_o = objective_count_factory(M, deco_o, count))
    (!ismissing(cache)) && (deco_o = objective_cache_factory(M, deco_o, cache))
    (return_objective) && (deco_o = ReturnManifoldObjective(deco_o))
    return deco_o
end

"""
    initialize_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState)

Initialize the solver to the optimization [`AbstractManoptProblem`](@ref) `amp` by
initializing the necessary values in the [`AbstractManoptSolverState`](@ref) `ams`.
"""
initialize_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState)

function initialize_solver!(p::AbstractManoptProblem, s::ReturnSolverState)
    return initialize_solver!(p, s.state)
end

"""
    solve!(problem::AbstractManoptProblem, state::AbstractManoptSolverState)

Run the solver implemented for the [`AbstractManoptProblem`](@ref) `problem` and the
[`AbstractManoptSolverState`](@ref) `state` employing [`initialize_solver!`](@ref), [`step_solver!`](@ref),
as well as the [`stop_solver!`](@ref) of the solver.

This includes the callbacks `:BeforeInit`, `:Init`, `:BeforeStep`, `:Step`, and `:Stop`.
"""
function solve!(problem::AbstractManoptProblem, state::AbstractManoptSolverState)
    k = 0
    callback(:BeforeInit, problem, state, 0)
    initialize_solver!(problem, state)
    callback(:Init, problem, state, 0)
    while !stop_solver!(problem, state, k)
        k = k + 1
        callback(:BeforeStep, problem, state, k)
        step_solver!(problem, state, k)
        callback(:Step, problem, state, k)
    end
    callback(:Stop, problem, state, k)
    return state
end

"""
    step_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k)

Do one iteration step (the `k`-th) for an [`AbstractManoptProblem`](@ref) `amp` by modifying
the values in the [`AbstractManoptSolverState`](@ref) `ams`.
"""
step_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k)
function step_solver!(p::AbstractManoptProblem, s::ReturnSolverState, k)
    return step_solver!(p, s.state, k)
end

"""
    stop_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k)

Determine whether the solver should stop.

Depending on the current [`AbstractManoptProblem`](@ref) `amp`, the current state of the solver
stored in [`AbstractManoptSolverState`](@ref) `ams` and the current iteration `k`, this by default
calls the internal [`StoppingCriterion`](@ref) `ams.stop`.

This includes a callback `:BeforeStop`.
"""
function stop_solver!(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k)
    callback(:BeforeStop, amp, ams, k)
    return ams.stop(amp, ams, k)
end
function stop_solver!(p::AbstractManoptProblem, s::ReturnSolverState, k)
    return stop_solver!(p, s.state, k)
end

"""
    get_cost(p::AbstractManoptProblem, s::AbstractManoptSolverState)

Get cost at the current iterate of the solver state `s` for the problem `p`.
The method may be implemented by particular solvers if they store the cost at the current
iterate in the state, but by default it is obtained by calling `get_cost(p, get_iterate(s))`.
"""
function get_cost(p::AbstractManoptProblem, s::AbstractManoptSolverState)
    return _get_cost(p, s, dispatch_state_decorator(s))
end
function _get_cost(p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{false})
    return get_cost(p, get_iterate(s))
end
function _get_cost(p::AbstractManoptProblem, s::AbstractManoptSolverState, ::Val{true})
    return get_cost(p, s.state)
end
