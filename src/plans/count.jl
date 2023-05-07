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
    return CountObjective{E,O,I}(counts, objective)
end
function CountObjective(
    objective::O, count::AbstractVector{Symbol}, init::I=0
) where {E<:AbstractEvaluationType,I<:Integer,O<:AbstractManifoldCostObjective{E}}
    return CountObjective(objective, Dict([symbol => init for symbol in count]))
end

function _count_if_exists(co::CountObjective, s::Symbol)
    return haskey(co.counts, s) && (co.counts[s] += 1)
end

"""
    get_count(co::CountObjective, s::Symbol, mode::Symbol=:None)

Get the number of counts for a certain symbel `s`.

Depending on the `mode` different results appear if the symbol does not exist in the dictionary

* `:None` – (default) silent mode, returns `-1` for non-existing entries
* `:warn` – issues a warning if a field does not exist
* `:error` – issues an error if a field does not exist
"""
function get_count(co::CountObjective, s::Symbol, mode::Symbol=:None)
    if !haskey(co.counts, s)
        msg = "There is no recorded count for $s."
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msh))
        return -1
    end
    return co.counts[s]
end

#
# Overwrite accessors
#
function get_cost(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :Cost)
    return get_cost(M, co.objective, p)
end

function get_gradient(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :Gradient)
    return get_gradient(M, co.objective, p)
end
function get_gradient!(M::AbstractManifold, X, co::CountObjective, p)
    _count_if_exists(co, :Gradient)
    get_gradient!(M, X, co.objective, p)
    return X
end

#
# State
#
