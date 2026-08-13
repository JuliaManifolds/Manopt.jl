@doc """
    AbstractManifoldCostObjective{F} <: AbstractManifoldObjective

Representing objectives on manifolds with a cost function implemented.
The parameter `F` represents the type of the costy function
"""
abstract type AbstractManifoldCostObjective{F} <: AbstractManifoldObjective end

@doc """
    get_cost(M::AbstractManifold, mco::AbstractManifoldCostObjective, p)

Evaluate the cost function from within the [`AbstractManifoldCostObjective`](@ref) on `M`
at `p`.

## See also
[`get_cost_function`](@ref) is used here internally to access the cost function.
"""
function get_cost(M::AbstractManifold, mco::AbstractManifoldCostObjective, p)
    return get_cost_function(mco)(M, p)
end
function get_cost(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_cost(M, get_objective(admo, false), p)
end

@doc """
    get_cost_function(amco::AbstractManifoldCostObjective, recursive=false)

return the function to evaluate (just) the cost ``f(p)=c`` as a function `(M,p) -> c`.
If `amco` has more than one decorator, `recursive` determines whether just one (`false`)
or all wrappers (`true`) should be “unwrapped” at once.

In the non-recursive case, this implementation assumes that the cost function is stored
in `amco.cost`. This way you only have to implement this function if your cost function
is stored in a different field.
"""
get_cost_function(mco::AbstractManifoldCostObjective, recursive = false) = mco.cost
function get_cost_function(admo::AbstractDecoratedManifoldObjective, recursive = false)
    return get_cost_function(get_objective(admo, recursive))
end

function set_parameter!(amo::AbstractManifoldCostObjective, ::Val{:Cost}, args...)
    set_parameter!(get_cost_function(amo, true), args...)
    return amo
end
