module ManoptJuMPExt

using Manopt
using LinearAlgebra
using JuMP: JuMP
using ManifoldsBase
using ManifoldDiff
const MOI = JuMP.MOI

"""
    ManifoldSet{M<:ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet

Model a manifold from [`ManifoldsBase.jl`](@extref) as a vectorial set in the
`MathOptInterface` (`MOI`).
This is a slight misuse of notation, since the manifold itself might not be embedded,
but just be paremetrized in a certain way.

# Fields
* `manifold::M`: The manifold in which the variables are constrained to lie.
  This is a [`ManifoldsBase.AbstractManifold`](@extref) object.

"""
struct ManifoldSet{M<:ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet
    manifold::M
end
@doc """
    JuMP_ManifoldSet(M::ManifoldsBase.AbstractManifold) = ManifoldSet(M)

Create a [`ManifoldSet`](@ref).
"""
Manopt.JuMP_ManifoldSet(M::ManifoldsBase.AbstractManifold) = ManifoldSet(M)

@doc """
    MOI.dimension(set::ManifoldSet)

Return the representation side of points on the (vectorized in representation) manifold.
As the MOI variables are real, this means if the [`representation_size`](@extref `ManifoldsBase.representation_size-Tuple{AbstractManifold}`)
yields (in product) `n`, this refers to the vectorized point / tangent vector  from (a subset of ``ℝ^n``).
"""
function MOI.dimension(set::ManifoldSet)
    return prod(ManifoldsBase.representation_size(set.manifold))
end

@doc """
    RiemannianFunction{MO<:Manopt.AbstractManifoldObjective} <: MOI.AbstractScalarFunction

A wrapper for a [`AbstractManifoldObjective`](@ref) that can be used
as a [`MOI.AbstractScalarFunction`](@extref JuMP :jl:type:`MathOptInterface.AbstractScalarFunction`).

# Fields

* `func::MO`: The [`AbstractManifoldObjective`](@ref) function to be wrapped.
"""
struct RiemannianFunction{MO<:Manopt.AbstractManifoldObjective} <:
       MOI.AbstractScalarFunction
    func::MO
end

@doc """
    JuMP.jump_function_type(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})

The JuMP function type of a function of type [`RiemannianFunction`](@ref) for any [`AbstractModel`](@extref JuMP.AbstractModel)
is that function type itself
"""
function JuMP.jump_function_type(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})
    return F
end

@doc """
    JuMP.jump_function(::JuMP.AbstractModel, F::Type{<:RiemannianFunction})

The JuMP function of a [`RiemannianFunction`](@ref) for any [`AbstractModel`](@extref JuMP.AbstractModel)
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
MOI.Utilities.map_indices(::Function, f::RiemannianFunction) = f

# We we don't support `MOI.modify` and `RiemannianFunction` is not mutable, no need to copy anything
Base.copy(f::RiemannianFunction) = f

@doc """
    JuMP.set_objective_function(model::JuMP.Model, obj::Manopt.AbstractManifoldObjective)

Set the objective function of a [`JuMP.Model`](@ref) `model` to an [`AbstractManifoldObjective`](@ref) `obj`.
This allows to use `@objective` with an objective from `Manopt.jl`.
"""
function JuMP.set_objective_function(
    model::JuMP.Model, func::Manopt.AbstractManifoldObjective
)
    return JuMP.set_objective_function(model, RiemannianFunction(func))
end

"""
    ManoptOptimizer <: MOI.AbstractOptimizer

Represent a solver from `Manopt.jl` within the `MathOptInterface` (`MOI`) framework.

# Fields
* `problem::([`AbstractManoptProblem`](@ref) a problem in manopt, especially
    containing the manifold and the objective function. It can be constructed as soon as
    the manifold and the objective are present.
* `manifold::`([`AbstractManifold`](@ref) the manifold on which the optimization is performed.
* `objective::`([`AbstractManifoldObjective`](@ref) the objective function to be optimized.
* `state::`([`AbstractManoptSolverState`](@ref) the state specifying the solver to use.
* `variable_primal_start::Vector{Union{Nothing,Float64}}` starting value for the solver,
    in a vectorized form that [`JuMP.jl`](@extref JuMP :std:doc:`index`) requires.
* `sense::`[`MOI.OptimizationSense`](@extref JuMP :jl:type:`MathOptInterface.OptimizationSense`) the sense of optimization,
  currently only minimization and maximization are supported.
* `options::Dict{String,Any}`: parameters specifying a solver before the `state`
  is initialized, so especially which [`AbstractManoptSolverState`](@ref) to use,
  when setting up the `state.

All types in brackets can also be `Nothing`, indicating they were not yet initialized.

# TODO: We might have to store the point and vector types `P` and `T` here maybe?
at least for the case where we are not on ManifoldArrayShape
"""
mutable struct ManoptOptimizer <: MOI.AbstractOptimizer
    problem::Union{Nothing,Manopt.AbstractManoptProblem}
    manifold::Union{Nothing,ManifoldsBase.AbstractManifold}
    objective::Union{Nothing,Manopt.AbstractManifoldObjective}
    state::Union{Nothing,Manopt.AbstractManoptSolverState}
    # Does this make sense to be elementwise Nothing? On a manifold a partial init is not possible
    variable_primal_start::Vector{Union{Nothing,Float64}}
    sense::MOI.OptimizationSense
    # Not sure what these are for? All parameters set should be reflected in the `state` parameter.
    options::Dict{String,Any}
    function ManoptOptimizer()
        return new(
            nothing,
            nothing,
            nothing,
            nothing,
            Union{Nothing,Float64}[],
            MOI.FEASIBILITY_SENSE,
            Dict{String,Any}(DESCENT_STATE_TYPE => Manopt.GradientDescentState),
        )
    end
end
"""
    Manopt.JuMP_Optimizer()

Represent a solver from `Manopt.jl` within the `MathOptInterface` (`MOI`) framework.

See [`ManoptOptimizer`](@ref) for the fields and their meaning.
"""
function Manopt.JuMP_Optimizer(args...)
    return ManoptOptimizer(args...)
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
           isnothing(model.problem) &&
           model.sense == MOI.FEASIBILITY_SENSE
end

"""
    MOI.empty!(model::ManoptJuMPExt.Optimizer)

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
    # FIXME Ideally, this should only return `true` if it is a valid keyword argument for
    #       one of the `...DescentState()` constructors. Is there an easy way to check this ?
    #       Does it depend on the different solvers ?
    return true
end

"""
    MOI.get(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)

Return last `value` set by `MOI.set(model, attr, value)`.
"""
function MOI.get(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)
    return model.options[attr.name]
end

"""
    MOI.get(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute)

Set the value for the keyword argument `attr.name` to give for the constructor
`model.options[DESCENT_STATE_TYPE]`.
"""
function MOI.set(model::ManoptOptimizer, attr::MOI.RawOptimizerAttribute, value)
    model.options[attr.name] = value
    return nothing
end

"""
    MOI.get(::ManoptOptimizer, ::MOI.SolverName)

Return the name of the `Optimizer` with the value of
the `descent_state_type` option.
"""
function MOI.get(model::ManoptOptimizer, ::MOI.SolverName)
    return "A Manopt.jl solver, namely $(model.options[DESCENT_STATE_TYPE])"
end

"""
    MOI.supports_incremental_interface(::JuMP_Optimizer)

Return `true` indicating that `Manopt.JuMP_Optimizer` implements
`MOI.add_constrained_variables` and `MOI.set` for
`MOI.ObjectiveFunction` so it can be used with [`JuMP.direct_model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.direct_model)
and does not require a `MOI.Utilities.CachingOptimizer`.
See [`MOI.supports_incremental_interface`](@extref JuMP :jl:function:`MathOptInterface.supports_incremental_interface`).
"""
MOI.supports_incremental_interface(::ManoptOptimizer) = true

"""
    MOI.copy_to(dest::ManoptOptimizer, src::MOI.ModelLike)

Because `supports_incremental_interface(dest)` is `true`, this simply
uses `MOI.Utilities.default_copy_to` and copies the variables with
`MOI.add_constrained_variables` and the objective sense with `MOI.set`.
"""
function MOI.copy_to(dest::ManoptOptimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

"""
    MOI.supports_add_constrained_variables(::JuMP_Optimizer, ::Type{<:ManifoldSet})

Return `true` indicating that `Manopt.JuMP_Optimizer` support optimization on
variables constrained to belong in a vectorized manifold [`ManifoldSet`](@ref).
"""
function MOI.supports_add_constrained_variables(::ManoptOptimizer, ::Type{<:ManifoldSet})
    return true
end

"""
    MOI.add_constrained_variables(model::ManoptOptimizer, set::ManifoldSet)

Add `MOI.dimension(set)` variables constrained in `set` and return the list
of variable indices that can be used to reference them as well a constraint
index for the constraint enforcing the membership of the variables in the
[`ManifoldSet`](@ref) `set`.
"""
function MOI.add_constrained_variables(model::ManoptOptimizer, set::ManifoldSet)
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
to the [`MOI.dimension`](@ref) of the [`ManifoldSet`](@ref).
"""
function MOI.get(model::ManoptOptimizer, ::MOI.NumberOfVariables)
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
    value::Union{Real,Nothing},
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
function MOI.supports(::ManoptOptimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction})
    return true
end

"""
    MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)

Modify the objective sense to either `MOI.MAX_SENSE`, `MOI.MIN_SENSE` or
`MOI.FEASIBILITY_SENSE`.
"""
function MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return nothing
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.ObjectiveSense)

Return the objective sense, defaults to `MOI.FEASIBILITY_SENSE` if no sense has
already been set.
"""
MOI.get(model::ManoptOptimizer, ::MOI.ObjectiveSense) = model.sense

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

function MOI.set(model::ManoptOptimizer, ::MOI.ObjectiveFunction, rf::RiemannianFunction)
    model.objective = rf.func
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
    kws = Dict{Symbol,Any}(
        Symbol(key) => value for (key, value) in model.options if key != DESCENT_STATE_TYPE
    )
    s = descent_state_type(model.manifold; p=reshaped_start, kws...)
    model.state = decorate_state!(s)
    solve!(model.problem, model.state)
    return nothing
end

@doc """
    ManifoldArrayShape{N} <: JuMP.AbstractShape

Represent some generic `AbstractArray` of a certain size representing an element
on a manifold, that is either a point or a tangent vector.

# Fields
* `size::NTuple{N,Int}`: The size of the array
"""
struct ManifoldArrayShape{N} <: JuMP.AbstractShape
    size::NTuple{N,Int}
end

@doc """
    ManifoldPointShape{M<:ManifoldsBase.AbstractManifold,P}

Given a concrete manifold and a point type [`AbstractManifoldPoint`](@extref `ManifoldsBase.AbstractManifoldPoint`),
this type can represent all information necessary to “reshape” or “vectorize”
such a point or to transform a vector back into a point of type `P` on `M`.

# Fields
* `manifold::M`: The manifold on which the point resides

# Constructor
    ManifoldPointShape(M::TM, ::Type{P<:ManifoldsBase.AbstractManifoldPoint})

Create a shape of a point on the manifold `M` of type `P`
"""
struct ManifoldPointShape{M<:ManifoldsBase.AbstractManifold,P} <: JuMP.AbstractShape
    manifold::M
end
function ManifoldPointShape(
    M::TM, ::Type{P}
) where {TM<:ManifoldsBase.AbstractManifold,P<:ManifoldsBase.AbstractManifoldPoint}
    return ManifoldPointShape{TM,P}(M)
end
@doc """
    TangentVectorShape{M<:ManifoldsBase.AbstractManifold,T}

Represent a tangect vector on the tangent bundle of a manifold `M`.

# Fields
* `manifold::M`: The manifold on which the tangent vector resides
* `p::P` (optioal) the base point P the tangent space is at where the vector of type `T`
    is defined.

# Constructor
    TangentVectorShape(M::TM, ::Type{T<:ManifoldsBase.AbstractTangentVector}, p::P=nothing)

Create a shape of a tangent vector on the manifold `M` of type `T`, where optionally a base
point `p` can be specified.
"""
struct TangentVectorShape{M<:ManifoldsBase.AbstractManifold,P,T} <: JuMP.AbstractShape
    manifold::M
    p::P
end
function TangentVectorShape(
    M::TM, ::Type{T}, p::P=nothing
) where {
    TM<:ManifoldsBase.AbstractManifold,
    P<:Union{Nothing,ManifoldsBase.AbstractManifoldPoint},
    T<:ManifoldsBase.AbstractTangentVector,
}
    return TangentVectorShape{TM,P,T}(M, p)
end

"""
    length(shape::ManifoldArrayShape)

Return the length of the vectors in the vectorized representation.
"""
Base.length(shape::ManifoldArrayShape) = prod(shape.size)

"""
    _vectorize!(res::Vector{T}, array::Array{T,N}, shape::ManifoldArrayShape{N}) where {T,N}

Inplace version of `res = JuMP.vectorize(array, shape)`.
"""
function _vectorize!(res::Vector{T}, array::Array{T,N}, ::ManifoldArrayShape{N}) where {T,N}
    return copyto!(res, array)
end

"""
    _reshape_vector!(res::Array{T,N}, vec::Vector{T}, ::ManifoldArrayShape{N}) where {T,N}

Inplace version of `res = JuMP.reshape_vector(vec, shape)`.
"""
function _reshape_vector!(
    res::Array{T,N}, vec::Vector{T}, ::ManifoldArrayShape{N}
) where {T,N}
    return copyto!(res, vec)
end

"""
    _zero(shape::ManifoldArrayShape)

Return a zero element of the shape `shape`.
"""
_zero(shape::ManifoldArrayShape{N}) where {N} = zeros(shape.size)

function JuMP.vectorize(array::Array{T,N}, ::ManifoldArrayShape{N}) where {T,N}
    return vec(array)
end

function JuMP.reshape_vector(vector::Vector, shape::ManifoldArrayShape)
    return reshape(vector, shape.size)
end

JuMP.reshape_set(set::ManifoldSet, shape::ManifoldArrayShape) = set.manifold
JuMP.reshape_set(set::ManifoldSet, shape::ManifoldPointShape) = set.manifold

function _shape(m::ManifoldsBase.AbstractManifold)
    return ManifoldArrayShape(ManifoldsBase.representation_size(m))
end

_in(mime::MIME"text/plain") = "in"
_in(mime::MIME"text/latex") = "\\in"

function JuMP.in_set_string(mime, set::ManifoldsBase.AbstractManifold)
    return _in(mime) * " " * string(set)
end

"""
    JuMP.build_variable(::Function, func, manifold::ManifoldsBase.AbstractManifold)

Build a `JuMP.VariablesConstrainedOnCreation` object containing variables
and the [`ManifoldSet`](@ref) in which they should belong as well as the
`shape` that can be used to go from the vectorized MOI representation to the
shape of the manifold, that is, a [`ManifoldArrayShape`](@ref).
"""
function JuMP.build_variable(::Function, array, M::ManifoldsBase.AbstractManifold)
    shape = _shape(M)
    return JuMP.VariablesConstrainedOnCreation(
        JuMP.vectorize(array, shape), ManifoldSet(M), shape
    )
end

#
#
# NonArrayPoints define own variable
# TODO: Document
struct ManifoldVariable{P<:ManifoldsBase.AbstractManifoldPoint} <: JuMP.AbstractVariable
    p::P
end
# Taken / adapted from https://github.com/JuliaManifolds/Manopt.jl/pull/466#issuecomment-2862071520
# TODO: I think I would prefer PoincareHalfPlanePoint(p) in Hyperbolic(3),
# but it seems the in and such where not present for polynomials?
function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    p::ManifoldsBase.AbstractManifoldPoint;
    extra_kwargs...,
)
    # cvarchecks(_error, info; extra_kwargs...) # What does this do? Necessary?
    # _warnbounds(_error, p, info)              # What does this do?
    return ManifoldVariable(p)
end
#
#
# TODO: Understand parameters here and document them
function JuMP.add_variable(model::JuMP.AbstractModel, v::ManifoldVariable, name::String="")
    return nothing
end

"""
    MOI.get(model::ManoptOptimizer, ::MOI.ResultCount)

Return `MOI.OPTIMIZE_NOT_CALLED` if `optimize!` hasn't been called yet and
`MOI.LOCALLY_SOLVED` otherwise indicating that the solver has solved the
problem to local optimality the value of `MOI.RawStatusString` for more
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

Return `0` if `optimize!` hasn't been called yet and
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

Return `MOI.NO_SOLUTION` if `optimize!` hasn't been called yet and
`MOI.FEASIBLE_POINT` otherwise indicating that a solution is available
to query with `MOI.VariablePrimalStart`.
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

Returns `MOI.NO_SOLUTION` indicating that there is no dual solution
available.
"""
MOI.get(::ManoptOptimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

"""
    MOI.get(model::ManoptOptimizer, ::MOI.RawStatusString)

Return a `String` containing `Manopt.get_reason` without the ending newline
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

end # module ManoptJuMPExt
