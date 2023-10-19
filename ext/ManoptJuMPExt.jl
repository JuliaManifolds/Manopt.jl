module ManoptJuMPExt

using Manopt
using LinearAlgebra
if isdefined(Base, :get_extension)
    using JuMP: JuMP
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..JuMP: JuMP
end
const MOI = JuMP.MOI
using ManifoldsBase
using ManifoldDiff

function __init__()
    # So that the user can use the convenient `Manopt.Optimizer`
    if isdefined(Base, :setglobal!)
        setglobal!(Manopt, :JuMP_Optimizer, Optimizer)
        setglobal!(Manopt, :JuMP_VectorizedManifold, VectorizedManifold)
        setglobal!(Manopt, :JuMP_ArrayShape, ArrayShape)
    else
        Manopt.eval(:(const JuMP_Optimizer = $Optimizer))
        Manopt.eval(:(const JuMP_VectorizedManifold = $VectorizedManifold))
        Manopt.eval(:(const JuMP_ArrayShape = $ArrayShape))
    end
    return nothing
end

#     struct VectorizedManifold{M} <: MOI.AbstractVectorSet
#         manifold::M
#     end
#
# Representation of points of `manifold` as a vector of `R^n` where `n` is
# `MOI.dimension(VectorizedManifold(manifold))`.
struct VectorizedManifold{M<:ManifoldsBase.AbstractManifold} <: MOI.AbstractVectorSet
    manifold::M
end

"""
    MOI.dimension(set::VectorizedManifold)

Return the representation side of points on the (vectorized in representation) manifold.
As the MOI variables are real, this means if the [`representation_size`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.representation_size-Tuple{AbstractManifold}) yields (in product) `n`, this refers to the vectorized point / tangent vector  from (a subset of ``\\mathbb R^n``).
"""
function MOI.dimension(set::VectorizedManifold)
    return prod(ManifoldsBase.representation_size(set.manifold))
end

"""
    Manopt.ManoptJuMPExt.Optimizer()

Creates a new optimizer object for the [MathOptInterface](https://jump.dev/MathOptInterface.jl/) (MOI).
An alias `Manopt.JuMP_Optimizer` is defined for convenience.

The minimization of a function `f(X)` of an an array `X[1:n1,1:n2,...]`
over a manifold `M` starting at `X0`, can be modeled as follows:
```julia
using JuMP
model = Model(Manopt.JuMP_Optimizer)
@variable(model, X[i1=1:n1,i2=1:n2,...] in M, start = X0[i1,i2,...])
@objective(model, Min, f(X))
```
The optimizer assumes that `M` has a `Array` shape described
by `ManifoldsBase.representation_size`.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    # Manifold in which all the decision variables leave
    manifold::Union{Nothing,ManifoldsBase.AbstractManifold}
    # Description of the problem in Manopt
    problem::Union{Nothing,Manopt.AbstractManoptProblem}
    # State of the optimizer
    state::Union{Nothing,Manopt.AbstractManoptSolverState}
    # Starting value for each variable
    variable_primal_start::Vector{Union{Nothing,Float64}}
    # Sense of the optimization, e.g., min, max or no objective
    sense::MOI.OptimizationSense
    # Model used to compute gradient of the objective function with AD
    nlp_model::MOI.Nonlinear.Model
    # Solver parameters set with `MOI.RawOptimizerAttribute`
    options::Dict{String,Any}
    function Optimizer()
        return new(
            nothing,
            nothing,
            nothing,
            Union{Nothing,Float64}[],
            MOI.FEASIBILITY_SENSE,
            MOI.Nonlinear.Model(),
            Dict{String,Any}(DESCENT_STATE_TYPE => Manopt.GradientDescentState),
        )
    end
end

MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.4.37"

function MOI.is_empty(model::Optimizer)
    # TODO replace `isnothing(model.nlp_model.objective)`
    #      by `MOI.is_empty` once the following is fixed
    #      https://github.com/jump-dev/MathOptInterface.jl/issues/2302
    return isnothing(model.manifold) &&
           isempty(model.variable_primal_start) &&
           isnothing(model.nlp_model.objective) &&
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
    # TODO replace by `MOI.empty!` once the following is fixed
    #      https://github.com/jump-dev/MathOptInterface.jl/issues/2302
    model.nlp_model.objective = nothing
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
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    return model.options[attr.name]
end

"""
    MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)

Set the value for the keyword argument `attr.name` to give for the constructor
`model.options[DESCENT_STATE_TYPE]`.
"""
function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    model.options[attr.name] = value
    return nothing
end

function MOI.get(model::Optimizer, ::MOI.SolverName)
    return "Manopt with $(model.options[DESCENT_STATE_TYPE])"
end

"""
    MOI.supports_incremental_interface(::Optimizer)

Return `true` indicating that `Manopt.Optimizer` implements
`MOI.add_constrained_variables` and `MOI.set` for
`MOI.ObjectiveFunction` so it can be used with [`JuMP.direct_model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.direct_model)
and does not require a `MOI.Utilities.CachingOptimizer`.
See [`MOI.supports_incremental_interface`](https://jump.dev/JuMP.jl/stable/moi/reference/models/#MathOptInterface.supports_incremental_interface).
"""
MOI.supports_incremental_interface(::Optimizer) = true

"""
    MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)

Because `supports_incremental_interface(dest)` is `true` we can simply
use `MOI.Utilities.default_copy_to`. This copies the variables with
`MOI.add_constrained_variables` and the objective sense with `MOI.set`.
`
"""
function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

"""
    MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:VectorizedManifold})

Return `true` indicating that `Manopt.Optimizer` support optimization on
variables constrained to belong in a vectorized manifold `VectorizedManifold`.
"""
function MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:VectorizedManifold})
    return true
end

"""
    MOI.add_constrained_variables(model::Optimizer, set::VectorizedManifold)

Add `MOI.dimension(set)` variables constrained in `set` and return the list
of variable indices that can be used to reference them as well a constraint
index for the constraint enforcing the membership of the variables in the
`VectorizedManifold` `set`.
"""
function MOI.add_constrained_variables(model::Optimizer, set::VectorizedManifold)
    F = MOI.VectorOfVariables
    if !isnothing(model.manifold)
        throw(
            AddConstraintNotAllowed{F,typeof(set)}(
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
           1 <= vi.value <= MOI.dimension(VectorizedManifold(model.manifold))
end

"""
    MOI.get(model::Optimizer, ::MOI.NumberOfVariables)

Return the number of variables added in the model, this corresponds
to the [`MOI.dimension`](@ref) of the `VectorizedManifold`.
"""
function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    if isnothing(model.manifold)
        return 0
    else
        return MOI.dimension(VectorizedManifold(model.manifold))
    end
end

"""
    MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)

Return `true` indicating that `Manopt.Optimizer` supports starting values
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

function MOI.supports(::Optimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction})
    return true
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return nothing
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.get(
    model::Optimizer, attr::Union{MOI.ObjectiveFunctionType,MOI.ObjectiveFunction}
)
    return MOI.get(model.nlp_model, attr)
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction{F}, func::F) where {F}
    nl = convert(MOI.ScalarNonlinearFunction, func)
    MOI.Nonlinear.set_objective(model.nlp_model, nl)
    model.problem = nothing
    model.state = nothing
    return nothing
end

const DESCENT_STATE_TYPE = "descent_state_type"

function MOI.optimize!(model::Optimizer)
    start = Float64[
        if isnothing(model.variable_primal_start[i])
            error("No starting value specified for `$i`th variable.")
        else
            model.variable_primal_start[i]
        end for i in eachindex(model.variable_primal_start)
    ]
    backend = MOI.Nonlinear.SparseReverseMode()
    vars = [MOI.VariableIndex(i) for i in eachindex(model.variable_primal_start)]
    evaluator = MOI.Nonlinear.Evaluator(model.nlp_model, backend, vars)
    MOI.initialize(evaluator, [:Grad])
    function eval_f_cb(M, x)
        if model.sense == MOI.FEASIBILITY_SENSE
            return 0.0
        end
        obj = MOI.eval_objective(evaluator, JuMP.vectorize(x, _shape(model.manifold)))
        if model.sense == MOI.MAX_SENSE
            obj = -obj
        end
        return obj
    end
    function eval_grad_f_cb(M, X)
        x = JuMP.vectorize(X, _shape(model.manifold))
        grad_f = zeros(length(x))
        if model.sense == MOI.FEASIBILITY_SENSE
            grad_f .= zero(eltype(grad))
        else
            MOI.eval_objective_gradient(evaluator, grad_f, x)
        end
        if model.sense == MOI.MAX_SENSE
            LinearAlgebra.rmul!(grad_f, -1)
        end
        reshaped_grad_f = JuMP.reshape_vector(grad_f, _shape(model.manifold))
        return ManifoldDiff.riemannian_gradient(model.manifold, X, reshaped_grad_f)
    end
    mgo = Manopt.ManifoldGradientObjective(eval_f_cb, eval_grad_f_cb)
    dmgo = decorate_objective!(model.manifold, mgo)
    model.problem = DefaultManoptProblem(model.manifold, dmgo)
    reshaped_start = JuMP.reshape_vector(start, _shape(model.manifold))
    descent_state_type = model.options[DESCENT_STATE_TYPE]
    kws = Dict{Symbol,Any}(
        Symbol(key) => value for (key, value) in model.options if key != DESCENT_STATE_TYPE
    )
    s = descent_state_type(model.manifold, reshaped_start; kws...)
    model.state = decorate_state!(s)
    solve!(model.problem, model.state)
    return nothing
end

#     struct ArrayShape{N} <: JuMP.AbstractShape
#
# Shape of an `Array{T,N}` of size `size`.
struct ArrayShape{N} <: JuMP.AbstractShape
    size::NTuple{N,Int}
end

function JuMP.vectorize(array::Array{T,N}, ::ArrayShape{M}) where {T,N,M}
    return vec(array)
end

function JuMP.reshape_vector(vector::Vector, shape::ArrayShape)
    return reshape(vector, shape.size)
end

function _shape(m::ManifoldsBase.AbstractManifold)
    return ArrayShape(ManifoldsBase.representation_size(m))
end

function JuMP.build_variable(::Function, func, m::ManifoldsBase.AbstractManifold)
    shape = _shape(m)
    return JuMP.VariablesConstrainedOnCreation(
        JuMP.vectorize(func, shape), VectorizedManifold(m), shape
    )
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if isnothing(model.state)
        return MOI.OPTIMIZE_NOT_CALLED
    else
        return MOI.LOCALLY_SOLVED
    end
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if isnothing(model.state)
        return 0
    else
        return 1
    end
end

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    if isnothing(model.state)
        return MOI.NO_SOLUTION
    else
        return MOI.FEASIBLE_POINT
    end
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    # `strip` removes the `\n` at the end and returns an `AbstractString`
    # Since MOI wants a `String`, we pass it through `string`
    return string(strip(get_reason(model.state)))
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    solution = Manopt.get_solver_return(model.state)
    return get_cost(model.problem, solution)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    solution = Manopt.get_solver_return(get_objective(model.problem), model.state)
    return solution[vi.value]
end

end # module
