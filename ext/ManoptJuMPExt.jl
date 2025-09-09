module ManoptJuMPExt

using Manopt
using LinearAlgebra
using JuMP: JuMP
using ManifoldsBase
using ManifoldDiff
const MOI = JuMP.MOI

"""
    ManoptOptimizer <: MOI.AbstractOptimizer

Represent a solver from `Manopt.jl` within the [`MathOptInterface` (MOI)](@extref JuMP :std:label:`The-MOI-interface`) framework of [`JuMP.jl`](@extref JuMP :std:doc:`index`)

# Fields
* `problem::`[`AbstractManoptProblem`](@ref) a problem in manopt, especially
    containing the manifold and the objective function. It can be constructed as soon as
    the manifold and the objective are present.
* `manifold::`[`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) the manifold on which the optimization is performed.
* `objective::`[`AbstractManifoldObjective`](@ref) the objective function to be optimized.
* `state::`[`AbstractManoptSolverState`](@ref) the state specifying the solver to use.
* `variable_primal_start::Vector{Union{Nothing,Float64}}` starting value for the solver,
    in a vectorized form that [`JuMP.jl`](@extref JuMP :std:doc:`index`) requires.
* `sense::`[`MOI.OptimizationSense`](@extref JuMP :jl:type:`MathOptInterface.OptimizationSense`) the sense of optimization,
  currently only minimization and maximization are supported.
* `options::Dict{String,Any}`: parameters specifying a solver before the `state`
  is initialized, so especially which [`AbstractManoptSolverState`](@ref) to use,
  when setting up the `state.
All types in brackets can also be `Nothing`, indicating they were not yet initialized.
"""
mutable struct ManoptOptimizer <: MOI.AbstractOptimizer
    problem::Union{Nothing, Manopt.AbstractManoptProblem}
    manifold::Union{Nothing, ManifoldsBase.AbstractManifold}
    objective::Union{Nothing, Manopt.AbstractManifoldObjective}
    state::Union{Nothing, Manopt.AbstractManoptSolverState}
    # Does this make sense to be elementwise Nothing? On a manifold a partial init is not possible
    variable_primal_start::Vector{Union{Nothing, Float64}}
    sense::MOI.OptimizationSense
    # Not sure what these are for? All parameters set should be reflected in the `state` parameter.
    options::Dict{String, Any}
    function ManoptOptimizer()
        return new(
            nothing,
            nothing,
            nothing,
            nothing,
            Union{Nothing, Float64}[],
            MOI.FEASIBILITY_SENSE,
            Dict{String, Any}(DESCENT_STATE_TYPE => Manopt.GradientDescentState),
        )
    end
end
"""
    Manopt.JuMP_Optimizer()

Represent a solver from `Manopt.jl` within the [`MathOptInterface` (MOI)](@extref JuMP :std:label:`The-MOI-interface`) framework.
See [`ManoptOptimizer`](@ref) for the fields and their meaning.
"""
function Manopt.JuMP_Optimizer(args...)
    return ManoptOptimizer(args...)
end

"""
    ManifoldSet{M<:ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet

Model a manifold from [`ManifoldsBase.jl`](@extref) as a vectorial set in the
[`MathOptInterface` (MOI)](@extref JuMP :std:label:`The-MOI-interface`).
This is a slight misuse of notation, since the manifold itself might not be embedded,
but just be parametrized in a certain way.

# Fields

* `manifold::M`: The manifold in which the variables are constrained to lie.
  This is a [`ManifoldsBase.AbstractManifold`](@extref) object.
"""
struct ManifoldSet{M <: ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet
    manifold::M
end

"""
    MOI.dimension(set::ManifoldSet)

Return the representation size of points on the (vectorized in representation) manifold.
As the MOI variables are real, this means if the [`representation_size`](@extref `ManifoldsBase.representation_size-Tuple{AbstractManifold}`)
yields (in product) `n`, this refers to the vectorized point / tangent vector  from (a subset of ``ℝ^n``).

Note that this is not the dimension of the manifold itself, but the
vector length of the vectorized representation of the manifold.
"""
function MOI.dimension(set::ManifoldSet)
    return length(_shape(set.manifold))
end

@doc """
    RiemannianFunction{MO<:Manopt.AbstractManifoldObjective} <: MOI.AbstractScalarFunction
A wrapper for a [`AbstractManifoldObjective`](@ref) that can be used
as a [`MOI.AbstractScalarFunction`](@extref JuMP :jl:type:`MathOptInterface.AbstractScalarFunction`).



# Fields
* `func::MO`: The [`AbstractManifoldObjective`](@ref) function to be wrapped.
"""
struct RiemannianFunction{MO <: Manopt.AbstractManifoldObjective} <:
    MOI.AbstractScalarFunction
    func::MO
end

@doc """
    JuMP.jump_function_type(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})

The [`JuMP.jl`](@extref JuMP :std:doc:`index`) function type of a function of type [`RiemannianFunction`](@ref) for any [`AbstractModel`](@extref JuMP.AbstractModel)
is that function type itself
"""
function JuMP.jump_function_type(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})
    return F
end

@doc """
    JuMP.jump_function(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})

The [`JuMP.jl`](@extref JuMP :std:doc:`index`) function of a [`RiemannianFunction`](@ref) for any [`AbstractModel`](@extref JuMP.AbstractModel)
is that function itself.
"""
JuMP.jump_function(::JuMP.AbstractModel, f::RiemannianFunction) = f

#
# The string representation
# maybe not document this since it seems to be mainly for display reasons
JuMP.function_string(mime::MIME, f::RiemannianFunction) = string(f.func)

"""
    MOI.Utilities.map_indices(index_map::Function, func::RiemannianFunction)

The original docstring states something about substituting some variable indices
by their index map variants.
On a [`RiemannianFunction`](@ref) there is nothing to substitute,
"""
MOI.Utilities.map_indices(::Function, func::RiemannianFunction) = func

# We we don't support `MOI.modify` and `RiemannianFunction` is not mutable, no need to copy anything
Base.copy(func::RiemannianFunction) = func

# This is called for instance when the user does `@objective(model, Min, func)`.
# JuMP only accepts subtypes of `MOI.AbstractFunction` as objective so we wrap `func`.
# It will then be allowed to go through all the MOI layers because it is of the right type
# We will then receive it in `MOI.set(::ManoptOptimizer, ::MOI.ObjectiveFunction, RiemannianFunction)`
# where we will unwrap it and recover `func`.
@doc """
    JuMP.set_objective_function(model::JuMP.Model, obj::Manopt.AbstractManifoldObjective)

Set the objective function of a [`JuMP.Model`](@extref) `model` to an [`AbstractManifoldObjective`](@ref) `obj`.
This allows to use `@objective` with an objective from `Manopt.jl`.
"""
function JuMP.set_objective_function(
        model::JuMP.Model, func::Manopt.AbstractManifoldObjective
    )
    return JuMP.set_objective_function(model, RiemannianFunction(func))
end

"""
    MOI.get(::ManoptOptimizer, ::MOI.SolverVersion)

Return the version of the Manopt solver, it corresponds to the version of
Manopt.jl.
"""
MOI.get(::ManoptOptimizer, ::MOI.SolverVersion) = "Manopt.jl $(pkgversion(Manopt))"

function MOI.is_empty(model::ManoptOptimizer)
    return isnothing(model.manifold) &&
        isempty(model.variable_primal_start) &&
        isnothing(model.objective) &&
        model.sense == MOI.FEASIBILITY_SENSE
end

"""
    MOI.empty!(model::ManoptOptimizer)

Clear all model data from `model` but keep the `options` set.
"""
function MOI.empty!(model::ManoptOptimizer)
    model.manifold = nothing
    model.problem = nothing
    model.state = nothing
    empty!(model.variable_primal_start)
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    return nothing
end

"""
    MOI.supports(::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)

Return a `Bool` indicating whether `attr.name` is a valid option name
for `Manopt`.
"""
function MOI.supports(::ManoptOptimizer, ::MOI.RawOptimizerAttribute)
    @show @__LINE__
    # FIXME Ideally, this should only return `true` if it is a valid keyword argument for
    #       one of the `...DescentState()` constructors. Is there an easy way to check this ?
    #       Does it depend on the different solvers ?
    return true
end

"""
    MOI.get(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)

Return last `value` set by [`set`](@extref `MathOptInterface.set`)`(model, attr, value)`.
"""
function MOI.get(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)
    @show @__LINE__
    return model.options[attr.name]
end

"""
    MOI.get(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)

Set the value for the keyword argument `attr.name` to give for the constructor
`model.options[DESCENT_STATE_TYPE]`.
"""
function MOI.set(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute, value)
    @show @__LINE__
    model.options[attr.name] = value
    return nothing
end

"""
    MOI.get(::ManoptOptimizer, ::MOI.SolverName)

Return the name of the [`ManoptOptimizer`](@ref) with the value of
the `descent_state_type` option.
"""
function MOI.get(model::ManoptOptimizer, ::MOI.SolverName)
    return "A Manopt.jl solver, namely $(model.options[DESCENT_STATE_TYPE])"
end

"""
    MOI.supports_incremental_interface(::ManoptOptimizer)

Return `true` indicating that [`ManoptOptimizer`](@ref) implements
[`add_constrained_variables`](@extref `MathOptInterface.add_constrained_variables`) and [`set`](@extref `MathOptInterface.set`) for
[`ObjectiveFunction`](@extref `MathOptInterface.ObjectiveFunction`) so it can be used with [`direct_model`](@extref `JuMP.direct_model`)
and does not require a [`CachingOptimizer`](@extref `MathOptInterface.Utilities.CachingOptimizer`).
See See [`supports_incremental_interface`](@extref `MathOptInterface.supports_incremental_interface`).
"""
MOI.supports_incremental_interface(::ManoptOptimizer) = true

"""
    MOI.copy_to(dest::ManoptOptimizer, src::MOI.ModelLike)

Because [`supports_incremental_interface`](@extref `MathOptInterface.supports_incremental_interface`)`(dest)` is `true`, this simply
uses [`default_copy_to`](@extref `MathOptInterface.Utilities.default_copy_to`) and copies the variables with
[`add_constrained_variables`](@extref `MathOptInterface.add_constrained_variables`) and the objective sense with [`set`](@extref `MathOptInterface.set`).
"""
function MOI.copy_to(dest::ManoptOptimizer, src::MOI.ModelLike)
    @show @__LINE__
    return MOI.Utilities.default_copy_to(dest, src)
end

"""
    MOI.supports_add_constrained_variables(::ManoptOptimizer, ::Type{<:ManifoldSet})

Return `true` indicating that [`ManoptOptimizer`](@ref) support optimization on
variables constrained to belong in a vectorized manifold.
"""
function MOI.supports_add_constrained_variables(::ManoptOptimizer, ::Type{<:ManifoldSet})
    return true
end

"""
    MOI.add_constrained_variables(model::ManoptOptimizer, set::ManifoldSet)

Add [`dimension`](@extref `MathOptInterface.dimension`)`(set)` variables constrained in `set` and return the list
of variable indices that can be used to reference them as well a constraint
index for the constraint enforcing the membership of the variables manifold as a set.
"""
function MOI.add_constrained_variables(model::ManoptOptimizer, set::ManifoldSet)
    F = MOI.VectorOfVariables
    if !isnothing(model.manifold)
        throw(
            MOI.AddConstraintNotAllowed{F, typeof(set)}(
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
    return v, MOI.ConstraintIndex{F, typeof(set)}(1)
end

"""
    MOI.is_valid(model::ManoptOptimizer, vi::MOI.VariableIndex)

Return whether `vi` is a valid variable index.
"""
function MOI.is_valid(model::ManoptOptimizer, vi::MOI.VariableIndex)
    return !isnothing(model.manifold) &&
        1 <= vi.value <= MOI.dimension(ManifoldSet(model.manifold))
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.NumberOfVariables)

Return the number of variables added in the model, this corresponds
to the [`dimension`](@extref JuMP :jl:function:`MathOptInterface.dimension`) of the [`ManifoldSet`](@ref).
"""
function MOI.get(model::ManoptOptimizer, ::MOI.NumberOfVariables)
    if isnothing(model.manifold)
        return 0
    else
        return MOI.dimension(ManifoldSet(model.manifold))
    end
end

"""
    MOI.supports(::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)

Return `true` indicating that [`ManoptOptimizer`](@ref) supports starting values
for the variables.
"""
function MOI.supports(
        ::ManoptOptimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}
    )
    return true
end

"""
    function MOI.set(
        model::ManoptOptimizer,
        ::MOI.VariablePrimalStart,
        vi::MOI.VariableIndex,
        value::Union{Real,Nothing},
    )

Set the starting value of the variable of index `vi` to `value`. Note that if
`value` is `nothing` then it essentially unset any previous starting values set
and hence `MOI.optimize!` unless another starting value is set.
"""
function MOI.set(
        model::ManoptOptimizer,
        ::MOI.VariablePrimalStart,
        vi::MOI.VariableIndex,
        value::Union{Real, Nothing},
    )
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[vi.value] = value
    model.state = nothing
    return nothing
end

"""
    MOI.supports(::ManoptOptimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction})

Return `true` indicating that `Optimizer` supports being set the objective
sense (that is, min, max or feasibility) and the objective function.
"""
function MOI.supports(::ManoptOptimizer, ::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction})
    return true
end

"""
    MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)

Modify the objective sense to either [`MAX_SENSE`](@extref), [`MIN_SENSE`](@extref) or
[`FEASIBILITY_SENSE`](@extref).
"""
function MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return nothing
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.ObjectiveSense)

Return the objective sense, defaults to [`FEASIBILITY_SENSE`](@extref) if no sense has
already been set.
"""
MOI.get(model::ManoptOptimizer, ::MOI.ObjectiveSense) = model.sense

"""
    _EmbeddingObjective{E<:MOI.AbstractNLPEvaluator,T}

Objective where `evaluator` is a MathOptInterface evaluator for the objective
in the embedding. The fields `vectorized_point`, `vectorized_tangent`
and `embedding_tangent` are used as preallocated buffer so that the conversion
to Euclidean objective is allocation-free.
"""
struct _EmbeddingObjective{E<:MOI.AbstractNLPEvaluator,T}
    evaluator::E
    # Used to store the vectorized point
    vectorized_point::Vector{Float64}
    # Used to store the vectorized tangent
    vectorized_tangent::Vector{Float64}
    # Used to store the tangent in the embedding space
    embedding_tangent::T
end

"""
    _get_cost(M, objective::_EmbeddingObjective, p)

Convert the point `p` to its vectorization and then evaluate the objective
using `objective.evaluator`.
"""
function _get_cost(M, objective::_EmbeddingObjective, p)
    _vectorize!(objective.vectorized_point, p, _shape(M))
    return MOI.eval_objective(objective.evaluator, objective.vectorized_point)
end

"""
    _get_cost(M, objective::_EmbeddingObjective, p)

Convert the point `p` to its vectorization and then evaluate the gradient
using `objective.evaluator` to get the vectorized gradient. Then reshape the
gradient and convert it to the Riemannian gradient.
"""
function _get_gradient!(M, gradient, objective::_EmbeddingObjective, p)
    _vectorize!(objective.vectorized_point, p, _shape(M))
    MOI.eval_objective_gradient(
        objective.evaluator, objective.vectorized_tangent, objective.vectorized_point
    )
    _reshape_vector!(objective.embedding_tangent, objective.vectorized_tangent, _shape(M))
    return ManifoldDiff.riemannian_gradient!(M, gradient, p, objective.embedding_tangent)
end

"""
    MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveFunction{F}, func::F) where {F}

Set the objective function as `func` for `model`.
"""
function MOI.set(
    model::ManoptOptimizer, ::MOI.ObjectiveFunction, func::MOI.AbstractScalarFunction
)
    backend = MOI.Nonlinear.SparseReverseMode()
    vars = [MOI.VariableIndex(i) for i in eachindex(model.variable_primal_start)]
    nlp_model = MOI.Nonlinear.Model()
    nl = convert(MOI.ScalarNonlinearFunction, func)
    MOI.Nonlinear.set_objective(nlp_model, nl)
    evaluator = MOI.Nonlinear.Evaluator(nlp_model, backend, vars)
    MOI.initialize(evaluator, [:Grad])
    objective = let                                             # COV_EXCL_LINE
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

function MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveFunction, func::RiemannianFunction)
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

function MOI.optimize!(model::ManoptOptimizer)
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
    kws = Dict{Symbol, Any}(
        Symbol(key) => value for (key, value) in model.options if key != DESCENT_STATE_TYPE
    )
    s = descent_state_type(model.manifold; p = reshaped_start, kws...)
    model.state = decorate_state!(s)
    solve!(model.problem, model.state)
    return nothing
end

@doc """
    ManifoldPointArrayShape{N} <: JuMP.AbstractShape

Represent some generic `AbstractArray` of a certain size representing an point
on a manifold

# Fields

* `size::NTuple{N,Int}`: The size of the array
"""
struct ManifoldPointArrayShape{N} <: JuMP.AbstractShape
    size::NTuple{N, Int}
end

"""
    length(shape::ManifoldPointArrayShape)

Return the length of the vectors in the vectorized representation.
"""
Base.length(shape::ManifoldPointArrayShape) = prod(shape.size)

"""
    _vectorize!(res::Vector{T}, array::Array{T,N}, shape::ManifoldPointArrayShape{N}) where {T,N}

Inplace version of `res = JuMP.vectorize(array, shape)`.
"""
function _vectorize!(
    res::Vector{T}, array::Array{T,N}, ::ManifoldPointArrayShape{N}
) where {T,N}
    return copyto!(res, array)
end

"""
    _reshape_vector!(res::Array{T,N}, vec::Vector{T}, ::ManifoldPointArrayShape{N}) where {T,N}

Inplace version of `res = JuMP.reshape_vector(vec, shape)`.
"""
function _reshape_vector!(
    res::Array{T,N}, vec::Vector{T}, ::ManifoldPointArrayShape{N}
) where {T,N}
    return copyto!(res, vec)
end

"""
    _zero(shape::ManifoldPointArrayShape)

Return a zero element of the shape `shape`.
"""
_zero(shape::ManifoldPointArrayShape{N}) where {N} = zeros(shape.size)

"""
    JuMP.vectorize(p::Array{T,N}, shape::ManifoldPointArrayShape{N}) where {T,N}

Given a point `p` as an ``N``-dimensional array representing a point on a certain
manifold, reshape it to a vector, which is necessary within [`JuMP`](@extref JuMP :std:doc:`index`).
For the inverse see [`JuMP.reshape_vector`](@ref JuMP.reshape_vector(::Vector, ::ManifoldPointArrayShape)).
"""
function JuMP.vectorize(array::Array{T,N}, ::ManifoldPointArrayShape{N}) where {T,N}
    return vec(array)
end

"""
    JuMP.reshape_vector(vector::Vector, shape::ManifoldPointArrayShape)

Given some vector representation `vector` used within [`JuMP`](@extref JuMP :std:doc:`index`) of a point on a manifold represents points
by arrays, use the information from the `shape` to reshape it back into such an array.
For the inverse see [`JuMP.vectorize`](@ref JuMP.vectorize(::Array, ::ManifoldPointArrayShape)).
"""
function JuMP.reshape_vector(vector::Vector, shape::ManifoldPointArrayShape)
    return reshape(vector, shape.size)
end

function JuMP.reshape_set(set::ManifoldSet, shape::ManifoldPointArrayShape)
    return set.manifold
end

"""
    _shape(m::ManifoldsBase.AbstractManifold)

Return the shape of points of the manifold `m`.
At the moment, we only support manifolds for which the shape is a `Array`.
"""
function _shape(m::ManifoldsBase.AbstractManifold)
    return ManifoldPointArrayShape(ManifoldsBase.representation_size(m))
end

_in(mime::MIME"text/plain") = "in"
_in(mime::MIME"text/latex") = "\\in"

function JuMP.in_set_string(mime, set::ManifoldsBase.AbstractManifold)
    return _in(mime) * " " * string(set)
end

"""
    JuMP.build_variable(::Function, func, m::ManifoldsBase.AbstractManifold)

Build a `JuMP.VariablesConstrainedOnCreation` object containing variables
and the [`ManifoldSet`](@ref) in which they should belong as well as the
`shape` that can be used to go from the vectorized MOI representation to the
shape of the manifold, that is, [`ManifoldPointArrayShape`](@ref).
"""
function JuMP.build_variable(::Function, func, m::ManifoldsBase.AbstractManifold)
    shape = _shape(m)
    return JuMP.VariablesConstrainedOnCreation(
        JuMP.vectorize(func, shape), ManifoldSet(m), shape
    )
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.ResultCount)

Return [`OPTIMIZE_NOT_CALLED`](@extref `MathOptInterface.OPTIMIZE_NOT_CALLED`) if [`optimize!`](@extref `JuMP.optimize!`) hasn't been called yet and
[`LOCALLY_SOLVED`](@extref `MathOptInterface.LOCALLY_SOLVED`) otherwise indicating that the solver has solved the
problem to local optimality the value of [`RawStatusString`](@extref `MathOptInterface.RawStatusString`) for more
details on why the solver stopped.
"""
function MOI.get(model::ManoptOptimizer, ::MOI.TerminationStatus)
    if isnothing(model.state)
        return MOI.OPTIMIZE_NOT_CALLED
    else
        return MOI.LOCALLY_SOLVED
    end
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.ResultCount)

Return `0` if [`optimize!`](@extref `JuMP.optimize!`) hasn't been called yet and
`1` otherwise indicating that one solution is available.
"""
function MOI.get(model::ManoptOptimizer, ::MOI.ResultCount)
    if isnothing(model.state)
        return 0
    else
        return 1
    end
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.PrimalStatus)

Return [`MOI.NO_SOLUTION`](@extref JuMP :jl:constant:`MathOptInterface.NO_SOLUTION`) if `optimize!` hasn't been called yet and
[`MOI.FEASIBLE_POINT`](@extref `MathOptInterface.FEASIBLE_POINT`) if it is otherwise indicating that a solution is available
to query with [`VariablePrimalStart`](@extref `MathOptInterface.VariablePrimalStart`).
"""
function MOI.get(model::ManoptOptimizer, ::MOI.PrimalStatus)
    if isnothing(model.state)
        return MOI.NO_SOLUTION
    else
        return MOI.FEASIBLE_POINT
    end
end

"""
    MOI.get(::ManoptOptimizer, ::MOI.DualStatus)

Returns [`MOI.NO_SOLUTION`](@extref `MathOptInterface.NO_SOLUTION`) indicating that there is no dual solution
available.
"""
MOI.get(::ManoptOptimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

"""
    MOI.get(model::ManoptOptimizer, ::MOI.RawStatusString)

Return a `String` containing [`get_reason`](@ref) without the ending newline
character.
"""
function MOI.get(model::ManoptOptimizer, ::MOI.RawStatusString)
    # `strip` removes the `\n` at the end and returns an `AbstractString`
    # Since MOI wants a `String`, pass it through `string`
    return string(strip(get_reason(model.state)))
end

"""
    MOI.get(model::ManoptOptimizer, attr::MOI.ObjectiveValue)

Return the value of the objective function evaluated at the solution.
"""
function MOI.get(model::ManoptOptimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    solution = Manopt.get_solver_return(model.state)
    value = get_cost(model.problem, solution)
    if model.sense == MOI.MAX_SENSE
        value = -value
    end
    return value
end

"""
    MOI.get(model::ManoptOptimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)

Return the value of the solution for the variable of index `vi`.
"""
function MOI.get(model::ManoptOptimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    solution = Manopt.get_solver_return(get_objective(model.problem), model.state)
    return solution[vi.value]
end

end # module
