"""
    CountObjective{E,P,O<:AbstractManifoldObjective,I<:Integer} <: AbstractDecoratedManifoldObjective{E,P}

A wrapper for any [`AbstractManifoldObjective`](@ref) of type `O` to count different calls
to parts of the objective.

# Fields

* `counts` a dictionary of symbols mapping to integers keeping the counted values
* `objective` the wrapped objective

# Supported Symbols

| Symbol                    | Counts calls to                      | Comment                      |
| ------------------------- | -----------------------------        | ------------------           |
| `:Cost`                   | [`get_cost`](@ref)                   |                              |
| `:Gradient`               | [`get_gradient`](@ref)               |                              |
| `:Hessian`                | [`get_hessian`](@ref)                |                              |
| `:Preconditioner`         | [`get_preconditioner`](@ref)         |                              |
| `:Constraints`            | [`get_constraints`](@ref)            |                              |
| `:EqualityConstraint`     | [`get_equality_constraint`](@ref)    | requires vector of counters  |
| `:EqualityConstraints`    | [`get_equality_constraints`](@ref)   | does not count single access |
| `:InequalityConstraint`   | [`get_inequality_constraint`](@ref)  | requires vector of counters  |
| `:InequalityConstraints`  | [`get_inequality_constraints`](@ref) | does not count single access |

# Constructors

    CountObjective(objective::AbstractManifoldObjective, counts::Dict{Symbol, <:Integer})

Initialise the `CountObjective` to wrap `objective` initializing the set of counts

    CountObjective(M::AbtractManifold, objective::AbstractManifoldObjective, count::AbstractVecor{Symbol}, init=0)

Count function calls on `objective` using the symbols in `count` initialising all entries to `init`.
"""
struct CountObjective{E,P,O<:AbstractManifoldObjective,I<:Integer} <:
       AbstractDecoratedManifoldObjective{E,P}
    counts::Dict{Symbol,I}
    objective::O
end
function CountObjective(
    ::AbstractManifold, objective::O, counts::Dict{Symbol,I}
) where {E<:AbstractEvaluationType,I<:Integer,O<:AbstractManifoldObjective{E}}
    return CountObjective{E,O,O,I}(counts, objective)
end
# Store the undecorated type of the input is decorated
function CountObjective(
    ::AbstractManifold, objective::O, counts::Dict{Symbol,I}
) where {
    E<:AbstractEvaluationType,
    I<:Integer,
    P<:AbstractManifoldObjective,
    O<:AbstractDecoratedManifoldObjective{E,P},
}
    return CountObjective{E,P,O,I}(counts, objective)
end
function CountObjective(
    M::AbstractManifold, objective::O, count::AbstractVector{Symbol}, init::I=0
) where {E<:AbstractEvaluationType,I<:Integer,O<:AbstractManifoldObjective{E}}
    return CountObjective(M, objective, Dict([symbol => init for symbol in count]))
end

function _count_if_exists(co::CountObjective, s::Symbol)
    return haskey(co.counts, s) && (co.counts[s] += 1)
end
function _count_if_exists(co::CountObjective, s::Symbol, i)
    return haskey(co.counts, s) &&
           length(i) == ndims(co.counts[s]) &&
           all(i .<= size(co.counts[s])) &&
           (co.counts[s][i] += 1)
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
function get_count(o::AbstractManifoldObjective, s::Symbol, mode::Symbol=:None)
    return _get_count(o, dispatch_objective_decorator(o), s, mode)
end
function _get_count(o::AbstractManifoldObjective, ::Val{false}, s, m)
    return error("It seems $o does not provide access to a `CountObjective`.")
end
_get_count(o::AbstractManifoldObjective, ::Val{true}, s, m) = get_count(o.objective, s, m)

function get_count(co::CountObjective, s::Symbol, i, mode::Symbol=:None)
    if !haskey(co.counts, s)
        msg = "There is no recorded count for $s."
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msh))
        return -1
    end
    if length(i) != ndims(co.counts[s])
        msg = "The entru for $s has $(ndims(co.counts[s])) dimensions but the index you provided has $(length(i))"
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msh))
        return -1
    end
    if any(i .> size(co.counts[s]))
        msg = "The index $i is out of range for the stored counts in $s ($(size(co.counts[s])))."
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msh))
        return -1
    end
    return co.counts[s]
end
function get_count(o::AbstractManifoldObjective, s::Symbol, i, mode::Symbol=:None)
    return _get_count(o, dispatch_objective_decorator(o), s, i, mode)
end
function _get_count(o::AbstractManifoldObjective, ::Val{false}, s, i, m)
    return error("It seems $o does not provide access to a `CountObjective`.")
end
function _get_count(o::AbstractManifoldObjective, ::Val{true}, s, i, m)
    return get_count(o.objective, s, i, m)
end

#
# Overwrite accessors
#
function get_cost(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :Cost)
    return get_cost(M, co.objective, p)
end

function get_cost_and_gradient(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :Cost)
    _count_if_exists(co, :Gradient)
    return get_cost_and_gradient(M, co.objective, p)
end

function get_cost_and_gradient(M::AbstractManifold, X, co::CountObjective, p)
    _count_if_exists(co, :Cost)
    _count_if_exists(co, :Gradient)
    return get_cost_and_gradient!(M, X, co.objective, p)
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

function get_hessian(M::AbstractManifold, co::CountObjective, p, X)
    _count_if_exists(co, :Hessian)
    return get_hessian(M, co.objective, p, X)
end
function get_hessian!(M::AbstractManifold, Y, co::CountObjective, p, X)
    _count_if_exists(co, :Hessian)
    get_hessian!(M, Y, co.objective, p, X)
    return Y
end

function get_preconditioner(M::AbstractManifold, co::CountObjective, p, X)
    _count_if_exists(co, :Preconditioner)
    return get_preconditioner(M, co.objective, p, X)
end
function get_preconditioner!(M::AbstractManifold, Y, co::CountObjective, p, X)
    _count_if_exists(co, :Preconditioner)
    get_preconditioner!(M, Y, co.objective, p, X)
    return Y
end

#
# Constraint
function get_constraints(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :Constraints)
    return get_constraints(M, co.objective, p)
end
function get_equatlity_constraints(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :EqualityConstraints)
    return get_equatlity_constraints(M, co.objective, p)
end
function get_equatlity_constraint(M::AbstractManifold, co::CountObjective, p, i)
    _count_if_exists(co, :EqualityConstraint, i)
    return get_equatlity_constraint(M, co.objective, p, i)
end
function get_inequality_constraints(M::AbstractManifold, co::CountObjective, p)
    _count_if_exists(co, :InequalityConstraints)
    get_inequality_constraints(M, co.objective, p)
    return Y
end
function get_inequality_constraint(M::AbstractManifold, co::CountObjective, p, i)
    _count_if_exists(co, :InequalityConstraint, i)
    get_inequality_constraint(M, co.objective, p, i)
    return Y
end

# A small todo list for further objective accessors.
#
# get_grad_constraints
# get_grad_equality_constraint(s)
# get_grad_inequality_constraint(s)
# get_proximal_map (single)
# get_proximal_map (indexed)
# Subgradient
# Stochastic
#

function objective_count_factory(
    M::AbstractManifold, o::AbstractManifoldCostObjective, counts::Vector{<:Symbol}
)
    return CountObjective(M, o, counts)
end
