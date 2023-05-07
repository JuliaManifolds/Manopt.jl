"""
     CountObjective{E,O<:AbstractManifoldObjective,I<:Integer} <: AbstractManifoldObjective{E}

A wrapper for any [`AbstractManifoldObjective`](@ref) to count different calls to parts of
the objective.

# Fields

* `counts` a dictionary of symbols mapping to integers keeping the counted values
* `objective` the wrapped objective

# Supported Symbols

| Symbol          | Counts calls to             |
| --------------- | --------------------------- |
| `:Cost`         | [`get_cost`](@ref)          |
| `:Gradient`     | [`get_gradient`](@ref)      |

# Constructors

    CountObjective(objective::AbstractManifoldObjective, counts::Dict{Symbol, <:Integer})

Initialise the `CountObjective` to wrap `objective` initializing the set of counts

    CountObjective(objective::AbstractManifoldObjective, count::AbstractVecor{Symbol}, init=0)

Count function calls on `objective` using the symbols in `count` initialising all entries to `init`.
"""
struct CountObjective{E,O<:AbstractManifoldObjective,I<:Integer} <:
       AbstractManifoldObjective{E}
    counts::Dict{Symbol,I}
    objective::O
end
function CountObjective(
    objective::O, counts::Dict{Symbol,I}
) where {E<:AbstractEvaluationType,I<:Integer,O<:AbstractManifoldCostObjective{E}}
    return CountObjective{E,O,I}(objective, counts)
end
function CountObjective(
    objective::O, count::AbstractVector{Symbol}, init::I=0
) where {E<:AbstractEvaluationType,I<:Integer,O<:AbstractManifoldCostObjective{E}}
    return CountObjective(objective, Dict([symbol => init for symbol in count]))
end

function _count_if_exists(co::CountObjective, s::symbol)
    return haskey(co.counts, s) && (co.counts[s] += 1)
end

get_count(co::CountObjective) = haskey(co.counts, s) && (return co.counts[s])

function get_cost(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :Cost)
    return get_cost(M, co.objective, p)
end

function get_gradient(M::AbstractManifold, co::CountObjective, p, k)
    _count_if_exists(co, :Gradient)
    return get_gradient(M, co.objective, p, X)
end
function get_gradient!(M::AbstractManifold, X, co::CountObjective, p)
    _count_if_exists(co, :Gradient)
    get_gradient!(M, X, co.objective, p)
    return X
end
