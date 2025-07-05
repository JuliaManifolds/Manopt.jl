module ManoptJuMPExt

using Manopt
using LinearAlgebra
using JuMP: JuMP
using ManifoldsBase
using ManifoldDiff
const MOI = JuMP.MOI

function __init__()
    setglobal!(Manopt, :JuMP_Optimizer, Optimizer)
    setglobal!(Manopt, :JuMP_ManifoldSet, ManifoldSet)
    setglobal!(Manopt, :JuMP_ArrayShape, ArrayShape)
    return nothing
end

"""
    ManifoldSet{M<:ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet

Model a manifold from [`ManifoldsBase`](@ref) as a vectorial set in the
[`MathOptInterface`](@ref MOI).

"""
struct ManifoldSet{M<:ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet
    manifold::M
end

"""
    MOI.dimension(set::ManifoldSet)

Return the representation side of points on the (vectorized in representation) manifold.
As the MOI variables are real, this means if the [`representation_size`](@extref `ManifoldsBase.representation_size-Tuple{AbstractManifold}`)
yields (in product) `n`, this refers to the vectorized point / tangent vector  from (a subset of ``ℝ^n``).
"""
function MOI.dimension(set::ManifoldSet)
    return prod(ManifoldsBase.representation_size(set.manifold))
end

struct RiemannianFunction{MO<:Manopt.AbstractManifoldObjective} <:
       MOI.AbstractScalarFunction
    func::MO
end

function JuMP.jump_function_type(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})
    return F
end

JuMP.jump_function(::JuMP.AbstractModel, f::RiemannianFunction) = f

JuMP.function_string(mime::MIME, f::RiemannianFunction) = string(f.func)

MOI.Utilities.map_indices(::Function, func::RiemannianFunction) = func

# We we don't support `MOI.modify` and `RiemannianFunction` is not mutable, no need to copy anything
Base.copy(func::RiemannianFunction) = func

# This is called for instance when the user does `@objective(model, Min, func)`.
# JuMP only accepts subtypes of `MOI.AbstractFunction` as objective so we wrap `func`.
# It will then be allowed to go through all the MOI layers because it is of the right type
# We will then receive it in `MOI.set(::Optimizer, ::MOI.ObjectiveFunction, RiemannianFunction)`
# where we will unwrap it and recover `func`.
function JuMP.set_objective_function(
    model::JuMP.Model, func::Manopt.AbstractManifoldObjective
)
    return JuMP.set_objective_function(model, RiemannianFunction(func))
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    # Manifold in which all the decision variables leave
    manifold::Union{Nothing,ManifoldsBase.AbstractManifold}
    # Description of the problem in Manopt
    problem::Union{Nothing,Manopt.AbstractManoptProblem}
    # State of the optimizer
    state::Union{Nothing,Manopt.AbstractManoptSolverState}
    # Starting value for each variable
    variable_primal_start::Vector{Union{Nothing,Float64}}
    # Sense of the optimization, that is whether it is for example min, max or no objective
    sense::MOI.OptimizationSense
    # Objective function of the optimization
    objective::Union{Nothing,Manopt.AbstractManifoldObjective}
    # Solver parameters set with `MOI.RawOptimizerAttribute`
    options::Dict{String,Any}
    function Optimizer()
        return new(
            nothing,
            nothing,
            nothing,
            Union{Nothing,Float64}[],
            MOI.FEASIBILITY_SENSE,
            nothing,
            Dict{String,Any}(DESCENT_STATE_TYPE => Manopt.GradientDescentState),
        )
    end
end

"""
    MOI.get(::Optimizer, ::MOI.SolverVersion)

Return the version of the Manopt solver, it corresponds to the version of
Manopt.jl.
"""
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.4.37"

function MOI.is_empty(model::Optimizer)
    return isnothing(model.manifold) &&
           isempty(model.variable_primal_start) &&
           isnothing(model.objective) &&
           model.sense == MOI.FEASIBILITY_SENSE
end

"""
    MOI.empty!(model::ManoptJuMPExt.Optimizer)

Clear all model data from `model` but keep the `options` set.
"""
function MOI.empty!(model::Optimizer)
    model.manifold = nothing
    model.problem = nothing
    model.state = nothing
    empty!(model.variable_primal_start)
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    return nothing
end

"""
    MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)

Return a `Bool` indicating whether `attr.name` is a valid option name
for `Manopt`.
"""
function MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute)
    # FIXME Ideally, this should only return `true` if it is a valid keyword argument for
    #       one of the `...DescentState()` constructors. Is there an easy way to check this ?
    #       Does it depend on the different solvers ?
    return true
end

"""
    MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)

Return last `value` set by `MOI.set(model, attr, value)`.
"""
function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    return model.options[attr.name]
end

"""
    MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)

Set the value for the keyword argument `attr.name` to give for the constructor
`model.options[DESCENT_STATE_TYPE]`.
"""
function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    model.options[attr.name] = value
    return nothing
end

"""
    MOI.get(::Optimizer, ::MOI.SolverName)

Return the name of the `Optimizer` with the value of
the `descent_state_type` option.
"""
function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "Manopt with $(model.options[DESCENT_STATE_TYPE])"
end

"""
    MOI.supports_incremental_interface(::JuMP_Optimizer)

Return `true` indicating that `Manopt.JuMP_Optimizer` implements
`MOI.add_constrained_variables` and `MOI.set` for
`MOI.ObjectiveFunction` so it can be used with [`JuMP.direct_model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.direct_model)
and does not require a `MOI.Utilities.CachingOptimizer`.
See [`MOI.supports_incremental_interface`](https://jump.dev/JuMP.jl/stable/moi/reference/models/#MathOptInterface.supports_incremental_interface).
"""
MOI.supports_incremental_interface(::Optimizer) = true

"""
    MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)

Because `supports_incremental_interface(dest)` is `true`, this simply
uses `MOI.Utilities.default_copy_to` and copies the variables with
`MOI.add_constrained_variables` and the objective sense with `MOI.set`.
"""
function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

"""
    MOI.supports_add_constrained_variables(::JuMP_Optimizer, ::Type{<:ManifoldSet})

Return `true` indicating that `Manopt.JuMP_Optimizer` support optimization on
variables constrained to belong in a vectorized manifold [`Manopt.JuMP_ManifoldSet`](@ref).
"""
function MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:ManifoldSet})
    return true
end

"""
    MOI.add_constrained_variables(model::Optimizer, set::ManifoldSet)

Add `MOI.dimension(set)` variables constrained in `set` and return the list
of variable indices that can be used to reference them as well a constraint
index for the constraint enforcing the membership of the variables in the
[`Manopt.JuMP_ManifoldSet`](@ref) `set`.
"""
function MOI.add_constrained_variables(model::Optimizer, set::ManifoldSet)
    F = MOI.VectorOfVariables
    if !isnothing(model.manifold)
        throw(
            MOI.AddConstraintNotAllowed{F,typeof(set)}(
                "Only one manifold allowed, variables in `$(model.manifold)` have already been added.",
            ),
        )
    end
    model.manifold = set.manifold
    model.problem = nothing
    model.state = nothing
    n = MOI.dimension(set)
    v = MOI.VariableIndex.(1:n)
    for _ in 1:n
        push!(model.variable_primal_start, nothing)
    end
    return v, MOI.ConstraintIndex{F,typeof(set)}(1)
end

"""
    MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)

Return whether `vi` is a valid variable index.
"""
function MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)
    return !isnothing(model.manifold) &&
           1 <= vi.value <= MOI.dimension(ManifoldSet(model.manifold))
end

"""
    MOI.get(model::Optimizer, ::MOI.NumberOfVariables)

Return the number of variables added in the model, this corresponds
to the [`MOI.dimension`](@ref) of the [`Manopt.JuMP_ManifoldSet`](@ref).
"""
function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    if isnothing(model.manifold)
        return 0
    else
        return MOI.dimension(ManifoldSet(model.manifold))
    end
end

"""
    MOI.supports(::Manopt.JuMP_Optimizer, attr::MOI.RawOptimizerAttribute)

Return `true` indicating that `Manopt.JuMP_Optimizer` supports starting values
for the variables.
"""
function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

"""
    function MOI.set(
        model::Optimizer,
        ::MOI.VariablePrimalStart,
        vi::MOI.VariableIndex,
        value::Union{Real,Nothing},
    )

Set the starting value of the variable of index `vi` to `value`. Note that if
`value` is `nothing` then it essentially unset any previous starting values set
and hence `MOI.optimize!` unless another starting value is set.
"""
function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[vi.value] = value
    model.state = nothing
    return nothing
end

"""
    MOI.supports(::Optimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction})

Return `true` indicating that `Optimizer` supports being set the objective
sense (that is, min, max or feasibility) and the objective function.
"""
function MOI.supports(::Optimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction})
    return true
end

"""
    MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)

Modify the objective sense to either `MOI.MAX_SENSE`, `MOI.MIN_SENSE` or
`MOI.FEASIBILITY_SENSE`.
"""
function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return nothing
end

"""
    MOI.get(model::Optimizer, ::MOI.ObjectiveSense)

Return the objective sense, defaults to `MOI.FEASIBILITY_SENSE` if no sense has
already been set.
"""
MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

# We could have it be a subtype of `AbstractManifoldGradientObjective{E,TC,TG}`
# but I wouldn't know what to do with `TC` and `TG` in this case.
# But we still implement an API similar to `get_cost` and `get_gradient!`
# so for consistency.
struct _EmbeddingObjective{E<:MOI.AbstractNLPEvaluator,T}
    evaluator::E
    # Used to store the vectorized point
    vectorized_point::Vector{Float64}
    # Used to store the vectorized tangent
    vectorized_tangent::Vector{Float64}
    # Used to store the tangent in the embedding space
    embedding_tangent::T
end

function _get_cost(M, objective::_EmbeddingObjective, p)
    _vectorize!(objective.vectorized_point, p, _shape(M))
    return MOI.eval_objective(objective.evaluator, objective.vectorized_point)
end

# We put all arguments
function _get_gradient!(M, gradient, objective::_EmbeddingObjective, p)
    _vectorize!(objective.vectorized_point, p, _shape(M))
    MOI.eval_objective_gradient(
        objective.evaluator, objective.vectorized_tangent, objective.vectorized_point
    )
    _reshape_vector!(objective.embedding_tangent, objective.vectorized_tangent, _shape(M))
    return ManifoldDiff.riemannian_gradient!(M, gradient, p, objective.embedding_tangent)
end

"""
    MOI.set(model::Optimizer, ::MOI.ObjectiveFunction{F}, func::F) where {F}

Set the objective function as `func` for `model`.
"""
function MOI.set(
    model::Optimizer, ::MOI.ObjectiveFunction, func::MOI.AbstractScalarFunction
)
    backend = MOI.Nonlinear.SparseReverseMode()
    vars = [MOI.VariableIndex(i) for i in eachindex(model.variable_primal_start)]
    nlp_model = MOI.Nonlinear.Model()
    nl = convert(MOI.ScalarNonlinearFunction, func)
    MOI.Nonlinear.set_objective(nlp_model, nl)
    evaluator = MOI.Nonlinear.Evaluator(nlp_model, backend, vars)
    MOI.initialize(evaluator, [:Grad])
    objective = let
        # To avoid creating a closure capturing the `embedding_obj` object,
        # we use the `let` block trick detailed in:
        # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
        embedding_obj = _EmbeddingObjective(
            evaluator,
            zeros(length(_shape(model.manifold))),
            zeros(length(_shape(model.manifold))),
            _zero(_shape(model.manifold)),
        )
        RiemannianFunction(
            Manopt.ManifoldGradientObjective(
                (M, x) -> _get_cost(M, embedding_obj, x),
                (M, g, x) -> _get_gradient!(M, g, embedding_obj, x);
                evaluation=Manopt.InplaceEvaluation(),
            ),
        )
    end
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    return nothing
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction, func::RiemannianFunction)
    model.objective = func.func
    model.problem = nothing
    model.state = nothing
    return nothing
end

# Name of the attribute for the type of the descent state to be used as follows:
# ```julia
# set_attribute(model, "descent_state_type", Manopt.TrustRegionsState)
# ```
const DESCENT_STATE_TYPE = "descent_state_type"

function MOI.optimize!(model::Optimizer)
    start = Float64[
        if isnothing(model.variable_primal_start[i])
            error("No starting value specified for `$i`th variable.")
        else
            model.variable_primal_start[i]
        end for i in eachindex(model.variable_primal_start)
    ]
    objective = model.objective
    if model.sense == MOI.FEASIBILITY_SENSE
        objective = Manopt.ManifoldGradientObjective(
            (_, _) -> 0.0, ManifoldsBase.zero_vector
        )
    elseif model.sense == MOI.MAX_SENSE
        objective = -objective
    end
    dmgo = decorate_objective!(model.manifold, objective)
    model.problem = DefaultManoptProblem(model.manifold, dmgo)
    reshaped_start = JuMP.reshape_vector(start, _shape(model.manifold))
    descent_state_type = model.options[DESCENT_STATE_TYPE]
    kws = Dict{Symbol,Any}(
        Symbol(key) => value for (key, value) in model.options if key != DESCENT_STATE_TYPE
    )
    s = descent_state_type(model.manifold; p=reshaped_start, kws...)
    model.state = decorate_state!(s)
    solve!(model.problem, model.state)
    return nothing
end

struct ArrayShape{N} <: JuMP.AbstractShape
    size::NTuple{N,Int}
end

"""
    length(shape::ArrayShape)

Return the length of the vectors in the vectorized representation.
"""
Base.length(shape::ArrayShape) = prod(shape.size)

"""
    _vectorize!(res::Vector{T}, array::Array{T,N}, shape::ArrayShape{N}) where {T,N}

Inplace version of `res = JuMP.vectorize(array, shape)`.
"""
function _vectorize!(res::Vector{T}, array::Array{T,N}, ::ArrayShape{N}) where {T,N,M}
    return copyto!(res, array)
end

"""
    _reshape_vector!(res::Array{T,N}, vec::Vector{T}, ::ArrayShape{N}) where {T,N}

Inplace version of `res = JuMP.reshape_vector(vec, shape)`.
"""
function _reshape_vector!(res::Array{T,N}, vec::Vector{T}, ::ArrayShape{N}) where {T,N}
    return copyto!(res, vec)
end

"""
    _zero(shape::ArrayShape)

Return a zero element of the shape `shape`.
"""
_zero(shape::ArrayShape{N}) where {N} = zeros(shape.size)

function JuMP.vectorize(array::Array{T,N}, ::ArrayShape{N}) where {T,N}
    return vec(array)
end

function JuMP.reshape_vector(vector::Vector, shape::ArrayShape)
    return reshape(vector, shape.size)
end

function JuMP.reshape_set(set::ManifoldSet, shape::ArrayShape)
    return set.manifold
end

function _shape(m::ManifoldsBase.AbstractManifold)
    return ArrayShape(ManifoldsBase.representation_size(m))
end

_in(mime::MIME"text/plain") = "in"
_in(mime::MIME"text/latex") = "\\in"

function JuMP.in_set_string(mime, set::ManifoldsBase.AbstractManifold)
    return _in(mime) * " " * string(set)
end

"""
    JuMP.build_variable(::Function, func, m::ManifoldsBase.AbstractManifold)

Build a `JuMP.VariablesConstrainedOnCreation` object containing variables
and the [`Manopt.JuMP_ManifoldSet`](@ref) in which they should belong as well as the
`shape` that can be used to go from the vectorized MOI representation to the
shape of the manifold, that is, [`Manopt.JuMP_ArrayShape`](@ref).
"""
function JuMP.build_variable(::Function, func, m::ManifoldsBase.AbstractManifold)
    shape = _shape(m)
    return JuMP.VariablesConstrainedOnCreation(
        JuMP.vectorize(func, shape), ManifoldSet(m), shape
    )
end

"""
    MOI.get(model::Optimizer, ::MOI.ResultCount)

Return `MOI.OPTIMIZE_NOT_CALLED` if `optimize!` hasn't been called yet and
`MOI.LOCALLY_SOLVED` otherwise indicating that the solver has solved the
problem to local optimality the value of `MOI.RawStatusString` for more
details on why the solver stopped.
"""
function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if isnothing(model.state)
        return MOI.OPTIMIZE_NOT_CALLED
    else
        return MOI.LOCALLY_SOLVED
    end
end

"""
    MOI.get(model::Optimizer, ::MOI.ResultCount)

Return `0` if `optimize!` hasn't been called yet and
`1` otherwise indicating that one solution is available.
"""
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if isnothing(model.state)
        return 0
    else
        return 1
    end
end

"""
    MOI.get(model::Optimizer, ::MOI.PrimalStatus)

Return `MOI.NO_SOLUTION` if `optimize!` hasn't been called yet and
`MOI.FEASIBLE_POINT` otherwise indicating that a solution is available
to query with `MOI.VariablePrimalStart`.
"""
function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    if isnothing(model.state)
        return MOI.NO_SOLUTION
    else
        return MOI.FEASIBLE_POINT
    end
end

"""
    MOI.get(::Optimizer, ::MOI.DualStatus)

Returns `MOI.NO_SOLUTION` indicating that there is no dual solution
available.
"""
MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

"""
    MOI.get(model::Optimizer, ::MOI.RawStatusString)

Return a `String` containing `Manopt.get_reason` without the ending newline
character.
"""
function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    # `strip` removes the `\n` at the end and returns an `AbstractString`
    # Since MOI wants a `String`, pass it through `string`
    return string(strip(get_reason(model.state)))
end

"""
    MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)

Return the value of the objective function evaluated at the solution.
"""
function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    solution = Manopt.get_solver_return(model.state)
    value = get_cost(model.problem, solution)
    if model.sense == MOI.MAX_SENSE
        value = -value
    end
    return value
end

"""
    MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)

Return the value of the solution for the variable of index `vi`.
"""
function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    solution = Manopt.get_solver_return(get_objective(model.problem), model.state)
    return solution[vi.value]
end

end # module
