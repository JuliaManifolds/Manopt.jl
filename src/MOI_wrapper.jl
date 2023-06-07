using LinearAlgebra
import MathOptInterface as MOI
using ManifoldsBase
using ManifoldDiff

include("qp_block_data.jl")

const _FUNCTIONS = Union{
    MOI.VariableIndex,MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}
}

struct VectorizedManifold{M} <: MOI.AbstractVectorSet
    manifold::M
end
function MOI.dimension(set::VectorizedManifold)
    return prod(ManifoldsBase.representation_size(set.manifold))
end

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing

mutable struct Optimizer <: MOI.AbstractOptimizer
    manifold::Union{Nothing,ManifoldsBase.AbstractManifold}
    problem::Union{Nothing,AbstractManoptProblem}
    state::Union{Nothing,AbstractManoptSolverState}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    sense::MOI.OptimizationSense
    nlp_data::MOI.NLPBlockData
    qp_data::QPBlockData{Float64}
    function Optimizer()
        return new(
            nothing,
            nothing,
            nothing,
            Union{Nothing,Float64}[],
            MOI.FEASIBILITY_SENSE,
            MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
            QPBlockData{Float64}(),
        )
    end
end

function MOI.is_empty(model::Optimizer)
    return isnothing(model.manifold) &&
           isempty(model.variable_primal_start) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE
end

function MOI.empty!(model::Optimizer)
    model.manifold = nothing
    model.problem = nothing
    model.state = nothing
    empty!(model.variable_primal_start)
    model.sense = MOI.FEASIBILITY_SENSE
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.qp_data = QPBlockData{Float64}()
    return nothing
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Manopt"

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.default_copy_to(dest, src)
end

function MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:VectorizedManifold})
    return true
end

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

function MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)
    return !isnothing(model.manifold) &&
           1 <= vi.value <= MOI.dimension(VectorizedManifold(model.manifold))
end

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

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

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    model.problem = nothing
    model.state = nothing
    return nothing
end

function MOI.supports(
    ::Optimizer, ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction{F}}
) where {F<:_FUNCTIONS}
    return true
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    model.sense = sense
    return nothing
end

function MOI.get(
    model::Optimizer, attr::Union{MOI.ObjectiveFunctionType,MOI.ObjectiveFunction}
)
    return MOI.get(model.qp_data, attr)
end

function MOI.set(
    model::Optimizer, attr::MOI.ObjectiveFunction{F}, func::F
) where {F<:_FUNCTIONS}
    MOI.set(model.qp_data, attr, func)
    model.problem = nothing
    model.state = nothing
    return nothing
end

function MOI.eval_objective(model::Optimizer, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    end
    return MOI.eval_objective(model.qp_data, x)
end

function MOI.eval_objective_gradient(model::Optimizer, grad, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        grad .= zero(eltype(grad))
    elseif model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    else
        MOI.eval_objective_gradient(model.qp_data, grad, x)
    end
    return nothing
end

function MOI.optimize!(model::Optimizer)
    start = Float64[
        if isnothing(model.variable_primal_start[i])
            error("No starting value specified for `$i`th variable.")
        else
            model.variable_primal_start[i]
        end for i in eachindex(model.variable_primal_start)
    ]
    function eval_f_cb(M, x)
        obj = MOI.eval_objective(model, x)
        if model.sense == MOI.MAX_SENSE
            obj = -obj
        end
        return obj
    end
    function eval_grad_f_cb(M, x)
        grad_f = zeros(length(x))
        MOI.eval_objective_gradient(model, grad_f, x)
        if model.sense == MOI.MAX_SENSE
            LinearAlgebra.rmul!(grad_f, -1)
        end
        return ManifoldDiff.riemannian_gradient(model.manifold, x, grad_f)
    end
    MOI.initialize(model.nlp_data.evaluator, [:Grad])
    mgo = Manopt.ManifoldGradientObjective(eval_f_cb, eval_grad_f_cb)
    dmgo = decorate_objective!(model.manifold, mgo)
    model.problem = DefaultManoptProblem(model.manifold, dmgo)
    s = GradientDescentState(model.manifold, start)
    model.state = decorate_state!(s)
    solve!(model.problem, model.state)
    return nothing
end

using JuMP: JuMP

struct ArrayShape{N} <: JuMP.AbstractShape
    size::NTuple{N,Int}
end

function JuMP.vectorize(array::Array{T,N}, ::ArrayShape{M}) where {T,N,M}
    return vec(array)
end

function JuMP.reshape_vector(vector::Vector, shape::ArrayShape)
    return reshape(vector, shape.size)
end

function JuMP.build_variable(::Function, func, m::ManifoldsBase.AbstractManifold)
    size = ManifoldsBase.representation_size(m)
    return JuMP.VariablesConstrainedOnCreation(
        func, VectorizedManifold(m), ArrayShape(size)
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
    return status_summary(model.state)
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    solution = get_solver_return(get_objective(model.problem), model.state)
    return solution[vi.value]
end
