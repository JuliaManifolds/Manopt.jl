#
#
# State
"""
    ProximalPointState{P} <: AbstractGradientSolverState

# Fields

$(_var(:Field, :p; add=[:as_Iterate]))
$(_var(:Field, :stopping_criterion, "stop"))

# Constructor

    ProximalPointState(M::AbstractManifold; kwargs...)

Initialize the proximal point method solver state, where

## Input

$(_var(:Argument, :M; type=true))

## Keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(100)`"))

# See also

[`proximal point`](@ref)
"""
mutable struct ProximalPointState{
    P,
    TStop<:StoppingCriterion,
} <: AbstractGradientSolverState
    p::P
    stop::TStop
end
function ProximalPointState(
    M::AbstractManifold;
    p::P=rand(M),
    stopping_criterion::SC=StopAfterIteration(200),
) where {
    P,
    SC<:StoppingCriterion,
}
    return ProximalPointState{P,SC}(p, stopping_criterion)
end
function get_message(pps::ProximalPointState)
    return get_message(pps.stepsize)
end
function show(io::IO, gds::ProximalPointState)
    i = get_count(gds, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(gds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Proximal POint Method
    $Iter

    ## Stopping criterion

    $(status_summary(gds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
#
#
# solver interface
