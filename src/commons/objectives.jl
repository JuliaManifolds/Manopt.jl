#
#
# ---
"""
    ConstrainedManifoldObjective{C<:ConstraintType} <: AbstractManifoldObjective

Describes a constrained objective

$(_problem(:Constrained))

# Fields

* `objective`: an [`AbstractManifoldObjective`](@ref) representing the unconstrained
  objective, that is containing cost ``f``, the gradient of the cost ``f`` and maybe the Hessian.
* `equality_constraints`: an [`AbstractManifoldObjective`](@ref) representing the equality constraints
  ``h: $(_math(:Manifold)) → ℝ^n`` also possibly containing its gradient and/or Hessian
* `inequality_constraints`: an [`AbstractManifoldObjective`](@ref) representing the inequality constraints
  ``g: $(_math(:Manifold)) → ℝ^m`` also possibly containing its gradient and/or Hessian

# Constructors
    ConstrainedManifoldObjective(f, grad_f;
        equality_constraints=nothing,
        inequality_constraints=nothing,
        g=missing, grad_g=missing,
        h=missing, grad_h=missing,
        hess_f=missing, hess_g=missing, hess_h=missing,
        evaluation=AllocatingEvaluation(),
        M = missing,
        p = ismissing(M) ? missing : rand(M),
        atol = 0,
    )

Generate the constrained objective based on all involved single functions `f`, `grad_f`, `g`,
`grad_g`, `h`, `grad_h`, and optionally a Hessian for each of these.
With `equality_constraints` and `inequality_constraints` you have to provide the dimension
of the ranges of `h` and `g`, respectively.
You can also provide a manifold `M` and a point `p` to use one evaluation of the constraints
to automatically try to determine these sizes.

    ConstrainedManifoldObjective(mho::AbstractManifoldObjective;
        equality_constraints = nothing,
        inequality_constraints = nothing
    )

Generate the constrained objective either with explicit constraints ``g`` and ``h``, and
their gradients, or in the form where these are already encapsulated in [`VectorGradientFunction`](@ref)s.

Both variants require that at least one of the constraints (and its gradient) is provided.
If any of the three parts provides a Hessian, the corresponding object, that is a
[`ManifoldHessianObjective`](@ref) for `f` or a [`VectorHessianFunction`](@ref) for `g` or `h`,
respectively, is created.

Feasibility of points with respect to the constraints is determined up to the tolerance `atol`.
"""
struct ConstrainedManifoldObjective{
        MO <: AbstractManifoldObjective,
        EMO <: Union{AbstractVectorGradientFunction, Nothing}, IMO <: Union{AbstractVectorGradientFunction, Nothing},
    } <: AbstractManifoldObjective
    objective::MO
    equality_constraints::EMO
    inequality_constraints::IMO
    atol::Float64
end
function _vector_function_type_hint(f)
    (!ismissing(f) && isa(f, AbstractVector)) && return ComponentVectorialType()
    return FunctionVectorialType()
end

function _val_to_ncons(val)
    sv = size(val)
    if sv === ()
        return 1
    else
        return sv[end]
    end
end

# Try to infer the number of constraints
function _number_of_constraints(
        g, grad_g;
        function_type::Union{AbstractVectorialType, Nothing} = nothing,
        jacobian_type::Union{AbstractVectorialType, Nothing} = nothing,
        M::Union{AbstractManifold, Missing} = missing,
        p = ismissing(M) ? missing : rand(M),
    )
    if !ismissing(g)
        if isa(function_type, ComponentVectorialType) || isa(g, AbstractVector)
            return length(g)
        end
    end
    if !ismissing(grad_g)
        if isa(jacobian_type, ComponentVectorialType) || isa(grad_g, AbstractVector)
            return length(grad_g)
        end
    end
    # These are more expensive, since they evaluate and hence allocate
    if !ismissing(M) && !ismissing(p)
        # For functions on vector representations, the last size is equal to length
        # on array power manifolds, this also yields the number of elements
        (!ismissing(g)) && (return _val_to_ncons(g(M, p)))
        (!ismissing(grad_g)) && (return _val_to_ncons(grad_g(M, p)))
    end
    return -1
end

function ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h;
        hess_f = missing, hess_g = missing, hess_h = missing,
        equality_type::AbstractVectorialType = _vector_function_type_hint(h),
        equality_gradient_type::AbstractVectorialType = _vector_function_type_hint(grad_h),
        equality_hessian_type::AbstractVectorialType = _vector_function_type_hint(hess_h),
        inequality_type::AbstractVectorialType = _vector_function_type_hint(g),
        inequality_gradient_type::AbstractVectorialType = _vector_function_type_hint(grad_g),
        inequality_hessian_type::AbstractVectorialType = _vector_function_type_hint(hess_g),
        equality_constraints::Union{Integer, Nothing} = nothing,
        inequality_constraints::Union{Integer, Nothing} = nothing,
        M::Union{AbstractManifold, Missing} = missing, p = ismissing(M) ? missing : rand(M), atol = 0,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    )
    if ismissing(hess_f)
        objective = ManifoldGradientObjective(f, grad_f; evaluation = evaluation, p = p)
    else
        objective = ManifoldHessianObjective(f, grad_f, hess_f; evaluation = evaluation, p = p)
    end
    num_eq = isnothing(equality_constraints) ? -1 : equality_constraints
    if ismissing(h) || ismissing(grad_h)
        eq = nothing
    else
        if isnothing(equality_constraints)
            # try to guess
            num_eq = _number_of_constraints(
                h, grad_h;
                function_type = equality_type, jacobian_type = equality_gradient_type,
                M = M, p = p,
            )
        end
        # if it is still < 0, this can not be used
        (num_eq < 0) && error("Please specify a positive number of `equality_constraints` (provided $(equality_constraints))")
        if ismissing(hess_h)
            eq = VectorGradientFunction(
                h, grad_h, num_eq; evaluation = evaluation, p = p,
                function_type = equality_type, jacobian_type = equality_gradient_type,
            )
        else
            eq = VectorHessianFunction(
                h, grad_h, hess_h, num_eq; evaluation = evaluation, p = p,
                function_type = equality_type, jacobian_type = equality_gradient_type, hessian_type = equality_hessian_type,
            )
        end
    end
    num_ineq = isnothing(inequality_constraints) ? -1 : inequality_constraints
    if ismissing(g) || ismissing(grad_g)
        ineq = nothing
    else
        if isnothing(inequality_constraints)
            # try to guess
            num_ineq = _number_of_constraints(
                g, grad_g;
                function_type = inequality_type, jacobian_type = inequality_gradient_type,
                M = M, p = p,
            )
        end
        # if it is still < 0, this can not be used
        (num_ineq < 0) && error("Please specify a positive number of `inequality_constraints` (provided $(inequality_constraints))")
        if ismissing(hess_g)
            ineq = VectorGradientFunction(
                g, grad_g, num_ineq; evaluation = evaluation, p = p,
                function_type = inequality_type, jacobian_type = inequality_gradient_type,
            )
        else
            ineq = VectorHessianFunction(
                g, grad_g, hess_g, num_ineq; evaluation = evaluation, p = p,
                function_type = inequality_type, jacobian_type = inequality_gradient_type, hessian_type = inequality_hessian_type,
            )
        end
    end
    return ConstrainedManifoldObjective(
        objective; equality_constraints = eq, inequality_constraints = ineq, atol = atol
    )
end
function ConstrainedManifoldObjective(
        objective::MO; atol = 0,
        equality_constraints::EMO = nothing, inequality_constraints::IMO = nothing, kwargs...,
    ) where {MO <: AbstractManifoldObjective, IMO, EMO}
    if isnothing(equality_constraints) && isnothing(inequality_constraints)
        throw(
            ErrorException(
                """
                Neither the inequality and the equality constraints are provided.
                You can not generate a `ConstrainedManifoldObjective` without actual
                constraints.

                If you do not have any constraints, you could also take the `objective`
                (probably `f` and `grad_f`) and work with an unconstrained solver.
                """
            )
        )
    end
    return ConstrainedManifoldObjective{MO, EMO, IMO}(
        objective, equality_constraints, inequality_constraints, atol
    )
end
function ConstrainedManifoldObjective(
        f, grad_f; g = missing, grad_g = missing, h = missing, grad_h = missing, kwargs...
    )
    return ConstrainedManifoldObjective(f, grad_f, g, grad_g, h, grad_h; kwargs...)
end
function status_summary(cmo::ConstrainedManifoldObjective; context::Symbol = :default)
    el = isnothing(cmo.equality_constraints) ? 0 : length(cmo.equality_constraints)
    il = isnothing(cmo.inequality_constraints) ? 0 : length(cmo.inequality_constraints)
    _is_inline(context) && (return "A constrained objective based on $(status_summary(cmo.objective; context = context)) with $(el == 0 ? "no" : el) equality and $(il == 0 ? "no" : il) inequality constraints.")
    s = status_summary(cmo.objective; context = context)
    return """
    A constrained objective with $(el == 0 ? "no" : el) equality and $(il == 0 ? "no" : il) inequality constraints.
    For verifications, the inequalities are checked with an absolute tolerance of `atol = $(cmo.atol)`

    ## Unconstrained Objective
    $(_in_str(s; indent = 1, headers = 1))


    ## Equality constraints
    $(el == 0 ? "$(_MANOPT_INDENT)none" : _in_str(status_summary(cmo.equality_constraints; context = context); indent = 1, headers = 1))

    ## Inequality constraints
    $(il == 0 ? "$(_MANOPT_INDENT)none" : _in_str(status_summary(cmo.inequality_constraints; context = context); indent = 1, headers = 1))
    """
end
function show(io::IO, cmo::ConstrainedManifoldObjective)
    print(io, "ConstrainedManifoldObjective("); print(io, cmo.objective)
    print(io, "; atol = ")
    print(io, cmo.atol)
    if !isnothing(cmo.equality_constraints)
        print(io, "; equality_constraints = "); print(io, cmo.equality_constraints)
    end
    if !isnothing(cmo.inequality_constraints)
        print(io, "; inequality_constraints = "); print(io, cmo.inequality_constraints)
    end
    return print(io, ")")
end
@doc """
    equality_constraints_length(co::ConstrainedManifoldObjective)

Return the number of equality constraints of an [`ConstrainedManifoldObjective`](@ref).
This acts transparently through [`AbstractDecoratedManifoldObjective`](@ref)s
"""
function equality_constraints_length(co::ConstrainedManifoldObjective)
    return isnothing(co.equality_constraints) ? 0 : length(co.equality_constraints)
end
function equality_constraints_length(co::AbstractDecoratedManifoldObjective)
    return equality_constraints_length(get_objective(co, false))
end

@doc """
    get_unconstrained_objective(co::ConstrainedManifoldObjective)

Returns the internally stored unconstrained [`AbstractManifoldObjective`](@ref)
within the [`ConstrainedManifoldObjective`](@ref).
"""
get_unconstrained_objective(co::ConstrainedManifoldObjective) = co.objective

function get_cost(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return get_cost(M, co.objective, p)
end
function get_cost_function(co::ConstrainedManifoldObjective, recursive = false)
    return get_cost_function(co.objective, recursive)
end

@doc """
    get_equality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_equality_constraint(M::AbstractManifold, objective, p, j=:)

Evaluate equality constraints of a [`ConstrainedManifoldObjective`](@ref) `objective`
at point `p` and indices `j` (by default `:` which corresponds to all indices).
"""
function get_equality_constraint end

function get_equality_constraint(mp::AbstractManoptProblem, p, j = :)
    return get_equality_constraint(get_manifold(mp), get_objective(mp), p, j)
end

function get_equality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, j = :
    )
    return get_equality_constraint(M, get_objective(admo, false), p, j)
end

function get_equality_constraint(
        M::AbstractManifold, co::ConstrainedManifoldObjective, p, j = :
    )
    if isnothing(co.equality_constraints)
        return number_eltype(p)[]
    else
        return get_value(M, co.equality_constraints, p, j)
    end
end

function get_gradient(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return get_gradient(M, co.objective, p)
end
function get_gradient!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p)
    return get_gradient!(M, X, co.objective, p)
end
function get_gradient_function(co::ConstrainedManifoldObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    return get_gradient_function(co.objective, recursive; evaluation = evaluation)
end

@doc """
    get_inequality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:)

Evaluate inequality constraints of a [`ConstrainedManifoldObjective`](@ref) `objective`
at point `p` and indices `j` (by default `:` which corresponds to all indices).
"""
function get_inequality_constraint end

function get_inequality_constraint(mp::AbstractManoptProblem, p, j = :)
    return get_inequality_constraint(get_manifold(mp), get_objective(mp), p, j)
end
function get_inequality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, j = :
    )
    return get_inequality_constraint(M, get_objective(admo, false), p, j)
end
function get_inequality_constraint(
        M::AbstractManifold, co::ConstrainedManifoldObjective, p, j = :
    )
    if isnothing(co.inequality_constraints)
        return number_eltype(p)[]
    else
        return get_value(M, co.inequality_constraints, p, j)
    end
end

_doc_get_grad_equality_constraint = """
    get_grad_equality_constraint(amp::AbstractManoptProblem, p, j)
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())
    get_grad_equality_constraint!(amp::AbstractManoptProblem, X, p, j)
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())

Evaluate the gradient or gradients  of the equality constraint ``($(_tex(:grad)) h(p))_j`` or ``$(_tex(:grad)) h_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""

@doc "$(_doc_get_grad_equality_constraint)"
function get_grad_equality_constraint end

function get_grad_equality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_grad_equality_constraint(M, get_objective(admo, false), args...)
end
function get_grad_equality_constraint(
        M::AbstractManifold, co::ConstrainedManifoldObjective, p, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    if isnothing(co.equality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_gradient(M, co.equality_constraints, p, j, range)
end

@doc "$(_doc_get_grad_equality_constraint)"
function get_grad_equality_constraint!(
        amp::AbstractManoptProblem, X, p, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    return get_grad_equality_constraint!(
        get_manifold(amp), X, get_objective(amp), p, j, range
    )
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_grad_equality_constraint!(M, X, get_objective(admo, false), args...)
end

function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    isnothing(co.equality_constraints) && (return X)
    return get_gradient!(M, X, co.equality_constraints, p, j, range)
end

_doc_get_grad_inequality_constraint = """
    get_grad_inequality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())
    get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, j=:)
    get_grad_inequality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())

Evaluate the gradient or gradients of the inequality constraint ``($(_tex(:grad)) g(p))_j`` or ``$(_tex(:grad)) g_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""

@doc "$(_doc_get_grad_inequality_constraint)"
function get_grad_inequality_constraint end
function get_grad_inequality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_grad_inequality_constraint(M, get_objective(admo, false), args...)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, co::ConstrainedManifoldObjective, p, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    if isnothing(co.inequality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_gradient(M, co.inequality_constraints, p, j, range)
end
@doc "$(_doc_get_grad_inequality_constraint)"
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_grad_inequality_constraint!(M, X, get_objective(admo, false), args...)
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    isnothing(co.inequality_constraints) && (return X)
    return get_gradient!(M, X, co.inequality_constraints, p, j, range)
end
function get_hessian(M::AbstractManifold, co::ConstrainedManifoldObjective, p, X)
    return get_hessian(M, co.objective, p, X)
end
function get_hessian!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X)
    return get_hessian!(M, Y, co.objective, p, X)
end
function get_hessian_function(co::ConstrainedManifoldObjective, recursive = false; kwargs...)
    return get_hessian_function(co.objective, recursive; kwargs...)
end

_doc_get_hess_equality_constraint = """
    get_hess_equality_constraint(amp::AbstractManoptProblem, p, X, j=:)
    get_hess_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, X, j=:, range=NestedPowerRepresentation())
    get_hess_equality_constraint!(amp::AbstractManoptProblem, Y, p, X, j=:)
    get_hess_equality_constraint!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j=:, range=NestedPowerRepresentation())

Evaluate the Hessian or Hessians of the equality constraint ``($(_tex(:Hess)) h(p))_j`` or ``$(_tex(:Hess)) h_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the Hessian.
"""

@doc "$(_doc_get_hess_equality_constraint)"
function get_hess_equality_constraint end

function get_hess_equality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_hess_equality_constraint(M, get_objective(admo, false), args...)
end
function get_hess_equality_constraint(
        M::AbstractManifold, co::ConstrainedManifoldObjective, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    if isnothing(co.equality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_hessian(M, co.equality_constraints, p, X, j, range)
end
@doc "$(_doc_get_hess_equality_constraint)"
function get_hess_equality_constraint!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_hess_equality_constraint!(M, Y, get_objective(admo, false), args...)
end
function get_hess_equality_constraint!(
        M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    isnothing(co.equality_constraints) && (return Y)
    return get_hessian!(M, Y, co.equality_constraints, p, X, j, range)
end

_doc_get_hess_inequality_constraint = """
    get_hess_inequality_constraint(amp::AbstractManoptProblem, p, X, j=:)
    get_hess_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, X, j=:, range=NestedPowerRepresentation())
    get_hess_inequality_constraint!(amp::AbstractManoptProblem, Y, p, X, j=:)
    get_hess_inequality_constraint!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j=:, range=NestedPowerRepresentation())

Evaluate the Hessian or Hessians of the inequality constraint ``($(_tex(:Hess)) g(p)[X])_j`` or ``$(_tex(:Hess)) g_j(p)[X]``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the Hessian.
"""

@doc "$(_doc_get_hess_inequality_constraint)"
function get_hess_inequality_constraint end

function get_hess_inequality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_hess_inequality_constraint(M, get_objective(admo, false), args...)
end

function get_hess_inequality_constraint(
        M::AbstractManifold, co::ConstrainedManifoldObjective, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    if isnothing(co.inequality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_hessian(M, co.inequality_constraints, p, X, j, range)
end
@doc "$(_doc_get_hess_inequality_constraint)"
function get_hess_inequality_constraint!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_hess_inequality_constraint!(M, Y, get_objective(admo, false), args...)
end
function get_hess_inequality_constraint!(
        M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    isnothing(co.inequality_constraints) && (return Y)
    return get_hessian!(M, Y, co.inequality_constraints, p, X, j, range)
end

@doc """
    inequality_constraints_length(cmo::ConstrainedManifoldObjective)

Return the number of inequality constraints of an [`ConstrainedManifoldObjective`](@ref) `cmo`.
This acts transparently through [`AbstractDecoratedManifoldObjective`](@ref)s
"""
function inequality_constraints_length(cmo::ConstrainedManifoldObjective)
    return isnothing(cmo.inequality_constraints) ? 0 : length(cmo.inequality_constraints)
end
function inequality_constraints_length(admo::AbstractDecoratedManifoldObjective)
    return inequality_constraints_length(get_objective(admo, false))
end

@doc """
    is_feasible(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p; kwargs...)
    is_feasible(M::AbstractManifold, o::AbstractDecoratedManifoldObjective, p; kwargs...)

Evaluate whether a point `p` on `M` is feasible with respect to the [`ConstrainedManifoldObjective`](@ref) `cmo`.
That is for the provided inequality constraints ``g: $(_math(:Manifold)) → ℝ^m`` and equality constraints ``h: $(_math(:Manifold)) → ℝ^n``
from within `cmo`, the point ``p ∈ $(_math(:Manifold))`` is feasible if

```math
g_i(p) ≤ 0, $(_tex(:text, " for all ")) i=1,…,m$(_tex(:quad))$(_tex(:text, " and "))$(_tex(:quad)) h_j(p) = 0, $(_tex(:text, " for all ")) j=1,…,n.
```

# Keyword arguments
* `check_point::Bool=true`: whether to also verify that ``p∈$(_math(:Manifold))`` holds, using [`is_point`](@extref ManifoldsBase :jl:method:`ManifoldsBase.is_point-Tuple{AbstractManifold, Any, Bool}`)
* `error::Symbol=:none`: if the point is not feasible, this symbol determines how to report the error.
    * `:error`: throws an error
    * `:info`: displays the error message as an @info
    * `:none`: (default) the function just returns true/false
    * `:warn`: displays the error message as a @warning.

The keyword `error=` and all other `kwargs...` are passed on to [`is_point`](@extref ManifoldsBase :jl:method:`ManifoldsBase.is_point-Tuple{AbstractManifold, Any, Bool}`)
if the point is verified (see `check_point`).
"""
function is_feasible(M, o, p; check_point::Bool = true, error::Symbol = :none, kwargs...)
    cmo = get_objective(o)
    v = !check_point || is_point(M, p; error = error, kwargs...)
    g = get_inequality_constraint(M, cmo, p, :)
    h = get_equality_constraint(M, cmo, p, :)
    feasible = v && all(g .<= cmo.atol) && isapprox.(h, 0; atol = cmo.atol) |> all
    # if we are feasible or no error shall be generated
    ((error === :none) || feasible) && return feasible
    # collect information about infeasibily
    if (error === :info) || (error === :warn) || (error === :error)
        s = get_feasibility_status(M, cmo, p; g = g, h = h)
        (error === :error) && throw(ErrorException(s))
        (error === :info) && @info s
        (error === :warn) && @warn s
    end
    return feasible
end

@doc """
    get_feasibility_status(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        g = get_inequality_constraint(M, cmo, p, :),
        h = get_equality_constraint(M, cmo, p, :),
    )

Generate a message about the feasibiliy of `p` with respect to the [`ConstrainedManifoldObjective`](@ref).
You can also provide the evaluated vectors for the values of `g` and `h` as keyword arguments,
in case you had them evaluated before.
"""
function get_feasibility_status(
        M, cmo, p;
        g = get_inequality_constraint(M, cmo, p, :), h = get_equality_constraint(M, cmo, p, :),
    )
    g_violated = sum(g .> 0)
    h_violated = sum(h .!= 0)
    return """
    The point $p on $M is not feasible for the provided constraints.

    * There are $(g_violated) of $(length(g)) inequality constraints violated. $(
        g_violated > 0 ? "The sum of violation is $(sum(max.(g, Ref(0))))." : ""
    )
    * There are $(h_violated) of $(length(h)) equality constraints violated. $(
        h_violated > 0 ? "The sum of violation is $(sum(abs.(h)))." : ""
    )
    """
end

#
#
# ---
@doc """
    ManifoldAlternatingGradientObjective{F,G} <: AbstractManifoldFirstOrderObjective{F, G}

An alternating gradient objective consists of

* a cost function ``F(p)``
* a gradient ``$(_tex(:grad))F`` that is either
  * given as one function ``$(_tex(:grad))F`` returning a tangent vector `X` on `M` or
  * an array of gradient functions ``$(_tex(:grad))F_i``, `i=1,…,n`, each returning a component of the gradient
  which might be allocating or mutating variants, but not a mix of both.

!!! note

    This Objective is usually defined using the `ProductManifold` from `Manifolds.jl`, so `Manifolds.jl` has to be loaded.

# Constructors

    ManifoldAlternatingGradientObjective(F, gradF::Function; evaluation=AllocatingEvaluation(), p = missing)
    ManifoldAlternatingGradientObjective(F, gradF::AbstractVector{<:Function}; evaluation=AllocatingEvaluation(), p = missing)

Create an alternating gradient problem with an optional `cost` and the gradient either as one
function (returning an array) or a vector of functions.

## Keyword Arguments

$(_kwargs(:evaluation))
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.
"""
struct ManifoldAlternatingGradientObjective{F, G} <: AbstractManifoldFirstOrderObjective{F, G}
    cost::F
    gradient!::G
    function ManifoldAlternatingGradientObjective(f::F, grad_f::G; evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing) where {F, G}
        f_ = maybe_wrap_function(f, p; result = :Number)
        grad_f_ = maybe_wrap_function(grad_f, p, evaluation; result = :TangentVector)
        return new{typeof(f_), typeof(grad_f_)}(f_, grad_f_)
    end
    function ManifoldAlternatingGradientObjective(f::F, grad_f::AbstractVector{<:G}; evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing) where {F, G}
        f_ = maybe_wrap_function(f, p; result = :Number)
        grad_f_ = [ maybe_wrap_function(gf, p, evaluation; result = :TangentVector) for gf in grad_f]
        return new{typeof(f_), typeof(grad_f_)}(f_, grad_f_)
    end
end
function get_gradient(M::AbstractManifold, mago::ManifoldAlternatingGradientObjective{C, <:Function}, p) where {C}
    X = zero_vector(M, p)
    mago.gradient!(M, X, p)
    return X
end
function get_gradient(
        M::AbstractManifold, mago::ManifoldAlternatingGradientObjective{C, <:AbstractVector}, p,
    ) where {C}
    X = zero_vector(M, p)
    get_gradient!(M, X, mago, p)
    return X
end
function get_gradient!(
        M::AbstractManifold, X, mago::ManifoldAlternatingGradientObjective{TC}, p,
    ) where {TC}
    mago.gradient!(M, X, p)
    return X
end
function get_gradient(
        M::AbstractManifold, mago::ManifoldAlternatingGradientObjective{C}, p, i,
    ) where {C}
    X = zero_vector(M[i], p[M, i])
    get_gradient!(M, X, mago, p, i)
    return X
end
function get_gradient!(
        M::AbstractManifold, X, mago::ManifoldAlternatingGradientObjective{C}, p, k
    ) where {C}
    # this takes a lot more allocations than other methods, but the gradient can only be evaluated in full
    Xf = zero_vector(M, p)
    get_gradient!(M, Xf, mago, p)
    copyto!(M[k], X, p[M, k], Xf[M, k])
    return X
end
function get_gradient!(
        M::AbstractManifold, X, mago::ManifoldAlternatingGradientObjective{C, <:AbstractVector}, p, k,
    ) where {C}
    mago.gradient![k](M, X, p)
    return X
end

function Base.show(io::IO, mago::ManifoldAlternatingGradientObjective)
    print(io, "ManifoldAlternatingGradientObjective(")
    print(io, mago.cost); print(io, ", "); print(io, mago.gradient!)
    return print(io, ")")
end
function status_summary(mago::ManifoldAlternatingGradientObjective; context::Symbol = :default)
    (context === :short) && (return repr(mago))
    (context === :inline) && (return "An alternating gradient objective on a manifold.")
    return """
    An alternating gradient objective providing the gradient as components with which to alternate

    ## Functions
    * cost:    $(_MANOPT_INDENT)$(mago.cost)
    * gradient:$(_MANOPT_INDENT)$(mago.gradient!)"""
end


#
#
# ---
"""
    ManifoldConstrainedSetObjective{MO, PF, IF} <: AbstractManifoldObjective

Model a constrained objective restricted to a set

```math
$(_tex(:argmin))_{p ∈ $(_tex(:Cal, "C"))} f(p)
```

where ``$(_tex(:Cal, "C")) ⊂ $(_math(:Manifold))`` is a convex closed subset.

# Fields

* `objective::AbstractManifoldObjective` the (unconstrained) objective, which
  contains ``f`` and for example its gradient ``$(_tex(:grad)) f``.
* `project!::PF` a projection function ``$(_tex(:proj))_{$(_tex(:Cal, "C"))}: $(_math(:Manifold)) → $(_tex(:Cal, "C"))`` that projects onto the set ``$(_tex(:Cal, "C"))``.
* `indicator::IF` the indicator function ``ι_{$(_tex(:Cal, "C"))}(p) = $(_tex(:cases, "0 &" * _tex(:text, " for ") * "p∈" * _tex(:Cal, "C"), "∞ &" * _tex(:text, " else.")))``

# Constructor

    ManifoldConstrainedSetObjective(f, grad_f, project!; kwargs...)

Generate the constrained objective for a given function `f` its gradient `grad_f` and a projection `project!` ``$(_tex(:proj))_{$(_tex(:Cal, "C"))}``.

## Keyword arguments

* `indicator=missing`: the indicator function ``ι_{$(_tex(:Cal, "C"))}(p)``. If not provided, a test whether the projection yields the same point is performed.
  For the [`InplaceEvaluation`](@ref) this requires one allocation.
"""
struct ManifoldConstrainedSetObjective{MO <: AbstractManifoldObjective, PF, IF} <: AbstractManifoldObjective
    objective::MO
    project!::PF
    indicator::IF
end

function ManifoldConstrainedSetObjective(
        f, grad_f, project!::PF; indicator = missing, evaluation = AllocatingEvaluation(), p = missing,
    ) where {PF}
    obj = ManifoldGradientObjective(f, grad_f; evaluation = evaluation, p = p)
    proj_ = maybe_wrap_function(project!, p, evaluation; result = :Point)
    if ismissing(indicator)
        ind = function (M, p)
            q = rand(M)
            proj_(M, q, p)
            return distance(M, p, q) ≈ 0 ? 0 : Inf
        end
        ind_ = maybe_wrap_function(ind, p; result = :Number)
        return ManifoldConstrainedSetObjective{typeof(obj), typeof(proj_), typeof(ind_)}(
            obj, proj_, ind_
        )
    end
    ind_ = maybe_wrap_function(indicator, p; result = :Number)
    return ManifoldConstrainedSetObjective{typeof(obj), typeof(proj_), typeof(ind_)}(
        obj, proj_, ind_
    )
end

function get_cost(M::AbstractManifold, cso::ManifoldConstrainedSetObjective, p)
    return get_cost(M, cso.objective, p)
end
function get_cost_function(cso::ManifoldConstrainedSetObjective, recursive = false)
    return get_cost_function(cso.objective, recursive)
end
function get_gradient_function(cso::ManifoldConstrainedSetObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    return get_gradient_function(cso.objective, recursive; evaluation = evaluation)
end
function get_gradient(M::AbstractManifold, cso::ManifoldConstrainedSetObjective, p)
    return get_gradient(M, cso.objective, p)
end
function get_gradient!(M::AbstractManifold, X, cso::ManifoldConstrainedSetObjective, p)
    return get_gradient!(M, X, cso.objective, p)
end

_doc_get_projected_point = """
    get_projected_point(amp::AbstractManoptProblem, p)
    get_projected_point!(amp::AbstractManoptProblem, q, p)
    get_projected_point(M::AbstractManifold, cso::ManifoldConstrainedSetObjective, p)
    get_projected_point!(M::AbstractManifold, q, cso::ManifoldConstrainedSetObjective, p)

Project `p` with the projection that is stored within the [`ManifoldConstrainedSetObjective`](@ref).
This can be done in-place of `q`.
"""

@doc "$(_doc_get_projected_point)"
function get_projected_point(amp::AbstractManoptProblem, p)
    return get_projected_point(get_manifold(amp), get_objective(amp), p)
end
@doc "$(_doc_get_projected_point)"
function get_projected_point!(amp::AbstractManoptProblem, q, p)
    return get_projected_point!(get_manifold(amp), q, get_objective(amp), p)
end

@doc "$(_doc_get_projected_point)"
get_projected_point(M::AbstractManifold, cso::ManifoldConstrainedSetObjective, p)
function get_projected_point(
        M::AbstractManifold, cso::ManifoldConstrainedSetObjective, p
    )
    q = copy(M, p)
    cso.project!(M, q, p)
    return q
end
@doc "$(_doc_get_projected_point)"
get_projected_point!(M::AbstractManifold, q, cso::ManifoldConstrainedSetObjective, p)
function get_projected_point!(M::AbstractManifold, q, cso::ManifoldConstrainedSetObjective, p)
    cso.project!(M, q, p)
    return q
end

#
#
# ---
@doc """
    EmbeddedManifoldObjective{P, T, O2, O1<:AbstractManifoldObjective} <:
       AbstractDecoratedManifoldObjective{O2}

Declare an objective to be defined in the embedding.
This also declares the gradient to be defined in the embedding,
and especially being the Riesz representer with respect to the metric in the embedding.
The types can be used to still dispatch on also the undecorated objective type `O2`,
in case the objective being stored here is decorated, e.g. with a cache.

# Fields

* `objective`: the objective that is defined in the embedding
* `p=missing`: a point in the embedding.
* `X=missing`: a tangent vector in the embedding

When a point in the embedding `p` is provided, `embed!` is used in place of this point to reduce
memory allocations. Similarly `X` is used when embedding tangent vectors.
"""
struct EmbeddedManifoldObjective{P, T, O2, O1 <: AbstractManifoldObjective} <: AbstractDecoratedManifoldObjective{O2}
    objective::O1
    p::P
    X::T
end
function EmbeddedManifoldObjective(o::O, p::P = missing, X::T = missing) where {P, T, O <: AbstractManifoldObjective}
    return EmbeddedManifoldObjective{P, T, O, O}(o, p, X)
end
function EmbeddedManifoldObjective(
        o::O1, p::P = missing, X::T = missing
    ) where {
        P, T, O2 <: AbstractManifoldObjective, O1 <: AbstractDecoratedManifoldObjective{O2},
    }
    return EmbeddedManifoldObjective{P, T, O2, O1}(o, p, X)
end
function EmbeddedManifoldObjective(
        M::AbstractManifold, o::O;
        q = rand(M), p::P = embed(M, q), X::T = embed(M, q, rand(M; vector_at = q)),
    ) where {P, T, O <: AbstractManifoldObjective}
    return EmbeddedManifoldObjective(o, p, X)
end
# dispatch whether to do this in place or not
function local_embed!(M::AbstractManifold, ::EmbeddedManifoldObjective{Missing}, p)
    return embed(M, p)
end
function local_embed!(M::AbstractManifold, emo::EmbeddedManifoldObjective{P}, p) where {P}
    embed!(M, emo.p, p)
    return emo.p
end
@doc """
    get_cost(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)

Evaluate the cost function of an objective defined in the embedding by first embedding `p`
before calling the cost function stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_cost(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return get_cost(get_embedding(M, typeof(p)), emo.objective, q)
end
function get_cost_function(emo::EmbeddedManifoldObjective, recursive = false)
    recursive && (return get_cost_function(emo.objective, recursive))
    return (M, p) -> get_cost(M, emo, p)
end
@doc """
    get_gradient(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    get_gradient!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p)

Evaluate the gradient function of an objective defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_gradient(M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p) where {P}
    q = local_embed!(M, emo, p)
    return riemannian_gradient(M, p, get_gradient(get_embedding(M, typeof(p)), emo.objective, q))
end
function get_gradient(M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, q)
    return riemannian_gradient(M, p, emo.X)
end
function get_gradient!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective{P, Missing}, p) where {P}
    q = local_embed!(M, emo, p)
    riemannian_gradient!(M, X, p, get_gradient(get_embedding(M, typeof(p)), emo.objective, q))
    return X
end
function get_gradient!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective{P, T}, p) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, q)
    riemannian_gradient!(M, X, p, emo.X)
    return X
end
function get_gradient_function(emo::EmbeddedManifoldObjective{P, T}, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation()) where {P, T}
    recursive && (return get_gradient_function(emo.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p) -> get_gradient(M, emo, p)
    else
        return (M, X, p) -> get_gradient!(M, X, emo, p)
    end
end
@doc """
    get_hessian(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, X)
    get_hessian!(M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p, X)

Evaluate the Hessian of an objective defined in the embedding, that is embed `p` and `X`
before calling the Hessian function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned Hessian is then converted to a Riemannian Hessian calling
[`riemannian_Hessian`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library/#ManifoldDiff.riemannian_Hessian-Tuple{AbstractManifold,%20Any,%20Any,%20Any,%20Any}).
"""
function get_hessian(M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, X) where {P}
    q = local_embed!(M, emo, p)
    return riemannian_Hessian(
        M, p, get_gradient(get_embedding(M, typeof(p)), emo.objective, q),
        get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)), X,
    )
end
function get_hessian(M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, X) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, embed(M, p))
    return riemannian_Hessian(
        M, p, emo.X, get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)), X
    )
end
function get_hessian!(M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, X) where {P}
    q = local_embed!(M, emo, p)
    riemannian_Hessian!(
        M, Y, p, get_gradient(get_embedding(M, typeof(p)), emo.objective, q),
        get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)), X,
    )
    return Y
end
function get_hessian!(M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, X) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, embed(M, p))
    riemannian_Hessian!(
        M, Y, p, emo.X, get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)), X
    )
    return Y
end
function get_hessian_function(emo::EmbeddedManifoldObjective, recursive::Bool = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    recursive && (return get_hessian_function(emo.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p, X) -> get_hessian(M, emo, p, X)
    else
        return (M, Y, p, X) -> get_hessian!(M, Y, emo, p, X)
    end
end

@doc """
    get_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

Return a vector `[g, h]` of the two constraint value vectors, the inequality
constraints ``g(p)`` and the equality constraints ``h(p)``,
see also [`get_inequality_constraint`](@ref) and [`get_equality_constraint`](@ref).
"""
function get_constraints(
        M::AbstractManifold,
        co::Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective},
        p,
    )
    return [get_inequality_constraint(M, co, p, :), get_equality_constraint(M, co, p, :)]
end
@doc """
    get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)

Evaluate the `j`th equality constraint ``h_j(p)`` defined in the embedding, that is embed `p`
before calling the constraint functions stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    q = local_embed!(M, emo, p)
    return get_equality_constraint(M, emo.objective, q, j)
end
@doc """
    get_inequality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, i)

Evaluate the `i`th inequality constraint ``g_i(p)`` defined in the embedding, that is embed `p`
before calling the constraint functions stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective, p, i
    )
    q = local_embed!(M, emo, p)
    return get_inequality_constraint(M, emo.objective, q, i)
end
@doc """
    X = get_grad_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    get_grad_equality_constraint!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p, j)

Evaluate the gradient of the `j`th equality constraint ``$(_tex(:grad)) h_j(p)``
defined in the embedding, that is embed `p` before calling the gradient function stored in
the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, j::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return riemannian_gradient(M, p, Z)
end
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return [riemannian_gradient(M, p, X) for X in Z]
end
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, j::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, j)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Xs = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Ys = [riemannian_gradient(M, p, X) for X in Xs]
    return Ys
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, j::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, j::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, j)
    riemannian_gradient!(M, Y, p, emo.X)
    return Y
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
@doc """
    X = get_grad_inequality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    get_grad_inequality_constraint!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p, j)

Evaluate the gradient of the `j`th inequality constraint ``$(_tex(:grad)) g_j(p)``
defined in the embedding, that is embed `p` before calling the gradient function stored in
the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, i::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, i)
    return riemannian_gradient(M, p, Z)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return [riemannian_gradient(M, p, X) for X in Z]
end
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, i::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, i)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return [riemannian_gradient(M, p, X) for X in Z]
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, i::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, i)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, i::Integer,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, i)
    riemannian_gradient!(M, Y, p, emo.X)
    return Y
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, j,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
function show(io::IO, emo::EmbeddedManifoldObjective)
    return print(io, "EmbeddedManifoldObjective($(emo.objective), $(emo.p), $(emo.X))")
end
function status_summary(emo::EmbeddedManifoldObjective{P, T}; context::Symbol = :default) where {P, T}
    (context === :short) && return repr(emo)
    (context === :inline) && return "An embedded objective of $(status_summary(emo.objective; context = context))"
    p_str = !(ismissing(emo.p)) ? "* for a point of type $P" : ""
    X_str = !(ismissing(emo.X)) ? "* for a tangent vector of type $T" : ""
    pX_str = (length(p_str) + length(X_str) > 0) ? "\n\n## Temporary memory (in the embedding)\n$(p_str)$(length(p_str) > 0 ? "\n" : "")$(X_str)" : ""
    return """
    An embedded objective

    ## Objective
    $(_in_str(status_summary(emo.objective, context = context); indent = 1, headers = 1))$(pX_str)"""
end

@doc """
    ManifoldCachedObjective{P,O<:AbstractManifoldObjective,C<:NamedTuple{}} <: AbstractDecoratedManifoldObjective{P}

Create a cache for an objective, based on a `NamedTuple` that stores some kind of cache.

# Constructor

    ManifoldCachedObjective(M, o::AbstractManifoldObjective, caches::Vector{Symbol}; kwargs...)

Create a cache for the [`AbstractManifoldObjective`](@ref) where the Symbols in `caches` indicate,
which function evaluations to cache.

# Supported symbols

| Symbol                      | Caches calls to (incl. `!` variants)            | Comment                   |
| :-------------------------- | :---------------------------------------------- | :------------------------ |
| `:Cost`                     | [`get_cost`](@ref)                              |                           |
| `:Differential`             | [`get_differential`](@ref)`(M, p, X)`           |                           |
| `:EqualityConstraint`       | [`get_equality_constraint`](@ref)`(M, p, i)`    |                           |
| `:EqualityConstraints`      | [`get_equality_constraint`](@ref)`(M, p, :)`    |                           |
| `:GradEqualityConstraint`   | [`get_grad_equality_constraint`](@ref)          | tangent vector per (p,i)  |
| `:GradInequalityConstraint` | [`get_grad_inequality_constraint`](@ref)        | tangent vector per (p,i)  |
| `:Gradient`                 | [`get_gradient`](@ref)`(M,p)`                   | tangent vectors           |
| `:Hessian`                  | [`get_hessian`](@ref)                           | tangent vectors           |
| `:InequalityConstraint`     | [`get_inequality_constraint`](@ref)`(M, p, j)`  |                           |
| `:InequalityConstraints`    | [`get_inequality_constraint`](@ref)`(M, p, :)`  |                           |
| `:Preconditioner`           | [`get_preconditioner`](@ref)                    | tangent vectors           |
| `:ProximalMap`              | [`get_proximal_map`](@ref)                      | point per `(p,λ,i)`       |
| `:StochasticGradients`      | [`get_gradients`](@ref)                         | vector of tangent vectors |
| `:StochasticGradient`       | [`get_gradient`](@ref)`(M, p, i)`               | tangent vector per (p,i)  |
| `:SubGradient`              | [`get_subgradient`](@ref)                       | tangent vectors           |
| `:SubtrahendGradient`       | [`get_subtrahend_gradient`](@ref)               | tangent vectors           |

# Keyword arguments

* `p=rand(M)`:
  the type of the keys to be used in the caches. Defaults to the default representation on `M`.
* `value=get_cost(M, objective, p)`:
  the type of values for numeric values in the cache
* `X=zero_vector(M,p)`:
  the type of values to be cached for gradient and Hessian calls.
* `cache_size=10`:
  number of (least recently used) calls to cache
* `cache_sizes=Dict{Symbol,Int}()`:
  a dictionary specifying the sizes individually for each cache.
"""
struct ManifoldCachedObjective{P, O <: AbstractManifoldObjective, C <: NamedTuple{}} <: AbstractDecoratedManifoldObjective{P}
    objective::O
    cache::C
end
function ManifoldCachedObjective(
        M::AbstractManifold, objective::O, caches::AbstractVector{<:Symbol} = [:Cost];
        p::P = rand(M),
        value::R = get_cost(M, objective, p),
        X::T = zero_vector(M, p),
        cache_size::Int = 10,
        cache_sizes::Dict{Symbol, Int} = Dict{Symbol, Int}(),
    ) where {O <: AbstractManifoldObjective, R <: Real, P, T}
    c = init_caches(
        M, caches; p = p, value = value, X = X, cache_size = cache_size, cache_sizes = cache_sizes
    )
    return ManifoldCachedObjective{O, O, typeof(c)}(objective, c)
end
function ManifoldCachedObjective(
        M::AbstractManifold, objective::O, caches::AbstractVector{<:Symbol} = [:Cost];
        p::P = rand(M),
        value::R = get_cost(M, objective, p),
        X::T = zero_vector(M, p),
        cache_size::Int = 10,
        cache_sizes::Dict{Symbol, Int} = Dict{Symbol, Int}(),
    ) where {O2, O <: AbstractDecoratedManifoldObjective{O2}, R <: Real, P, T}
    c = init_caches(
        M, caches; p = p, value = value, X = X, cache_size = cache_size, cache_sizes = cache_sizes
    )
    return ManifoldCachedObjective{O2, O, typeof(c)}(objective, c)
end

"""
    init_caches(M::AbstractManifold, caches, T; kwargs...)

Given a vector of symbols `caches`, this function sets up the
`NamedTuple` of caches for points/vectors on `M`,
where `T` is the type of cache to use.
"""
function init_caches(M::AbstractManifold, caches, T = Nothing; kwargs...)
    return throw(
        DomainError(
            T,
            """
            No function `init_caches` available for a caches $T.
            For a good default load `LRUCache.jl`, for example:
            Enter `using LRUCache`.
            """,
        )
    )
end

#
# Default implementations: if field exists -> try to get cache
function get_cost(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Cost)) && return get_cost(M, co.objective, p) #No Cost cache
    # If so, check whether we should cache
    return get!(co.cache[:Cost], copy(M, p)) do
        get_cost(M, co.objective, p)
    end
end

function get_cost_function(co::ManifoldCachedObjective, recursive = false)
    recursive && (return get_cost_function(co.objective, recursive))
    return (M, p) -> get_cost(M, co, p)
end

function get_differential(M::AbstractManifold, co::ManifoldCachedObjective, p, X; kwargs...)
    # No Differential Cache
    !(haskey(co.cache, :Differential)) &&
        return get_differential(M, co.objective, p, X; kwargs...)
    # If so, check whether we should cache
    return get!(co.cache[:Differential], (copy(M, p), copy(M, p, X))) do
        get_differential(M, co.objective, p, X; kwargs...)
    end
end

function get_differential_function(mco::ManifoldCachedObjective, recursive = false)
    recursive && (return get_differential_function(mco.objective, recursive))
    return (M, p, X; kwargs...) -> get_differential(M, mco, p, X; kwargs...)
end

function get_gradient(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Gradient)) && return get_gradient(M, co.objective, p)
    return copy(
        M,
        p,
        get!(co.cache[:Gradient], copy(M, p)) do
            get_gradient(M, co.objective, p)
        end,
    )
end
function get_gradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :Gradient)) && return get_gradient!(M, X, co.objective, p)
    copyto!(
        M,
        X,
        p,
        get!(co.cache[:Gradient], copy(M, p)) do
            # This evaluates in place of X
            get_gradient!(M, X, co.objective, p)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end

function get_gradient_function(
        sco::ManifoldCachedObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation()
    )
    # recursive: Unwrap cache
    recursive && (return get_gradient_function(sco.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p) -> get_gradient(M, get_objective(sco), p)
    else
        return (M, X, p) -> get_gradient!(M, X, get_objective(sco), p)
    end
end

function get_cost_and_gradient(M::AbstractManifold, mco::ManifoldCachedObjective, p)
    #Neither cost not grad cached -> evaluate normally
    all(.!(haskey.(Ref(mco.cache), [:Cost, :Gradient]))) &&
        return get_cost_and_gradient(M, mco.objective, p)
    # Otherwise -> check whether any of the does not have this index:
    # No cost case or no grad case
    nc = !haskey(mco.cache, :Cost) || !haskey(mco.cache[:Cost], p)
    ng = !haskey(mco.cache, :Gradient) || !haskey(mco.cache[:Gradient], p)
    if nc || ng # one of them does not exist, either full cache or entry -> eval
        c, X = get_cost_and_gradient(M, mco.objective, p)
        # Cache if cache present
        haskey(mco.cache, :Cost) && setindex!(mco.cache[:Cost], c, copy(M, p))
        haskey(mco.cache, :Gradient) &&
            setindex!(mco.cache[:Gradient], copy(M, p, X), copy(M, p))
        return c, X
    else # both exist and are cached, return them
        return mco.cache[:Cost][p], copy(M, p, mco.cache[:Gradient][p])
    end
end
function get_cost_and_gradient!(M::AbstractManifold, X, mco::ManifoldCachedObjective, p)
    #Neither cost not grad cached -> evaluate normally
    all(.!(haskey.(Ref(mco.cache), [:Cost, :Gradient]))) &&
        return get_cost_and_gradient!(M, X, mco.objective, p)
    # Otherwise -> check whether any of the does not have this index:
    # No cost case or no grad case
    nc = !haskey(mco.cache, :Cost) || !haskey(mco.cache[:Cost], p)
    ng = !haskey(mco.cache, :Gradient) || !haskey(mco.cache[:Gradient], p)
    if nc || ng # one of them does not exist, either full cache or entry -> eval
        c, _ = get_cost_and_gradient!(M, X, mco.objective, p)
        # Cache if cache present
        haskey(mco.cache, :Cost) && setindex!(mco.cache[:Cost], c, copy(M, p))
        haskey(mco.cache, :Gradient) &&
            setindex!(mco.cache[:Gradient], copy(M, p, X), copy(M, p))
        return c, X
    else # both exist and are cached, return them
        copyto!(M, X, p, mco.cache[:Gradient][p])
        return mco.cache[:Cost][p], X
    end
end

function get_equality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, j::Integer
    )
    (!haskey(co.cache, :EqualityConstraint)) &&
        return get_equality_constraint(M, co.objective, p, j)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:EqualityConstraint], (copy(M, p), j)) do
            get_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_equality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, i::Colon
    )
    (!haskey(co.cache, :EqualityConstraints)) &&
        return get_equality_constraint(M, co.objective, p, i)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:EqualityConstraints], copy(M, p)) do
            get_equality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_equality_constraint(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    key = copy(M, p)
    if haskey(co.cache, :EqualityConstraints) # full constraints are stored
        if haskey(co.cache[:EqualityConstraints], key)
            return co.cache[:EqualityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :EqualityConstraint) # storing the index constraints
        return [
            copy(
                get!(co.cache[:EqualityConstraint], (key, j)) do
                    get_equality_constraint(M, co.objective, p, j)
                end,
            ) for j in _to_iterable_indices(1:equality_constraints_length(co.objective), i)
        ]
    end # neither cache: pass down to objective
    return get_equality_constraint(M, co.objective, p, i)
end
function get_inequality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, i::Integer
    )
    (!haskey(co.cache, :InequalityConstraint)) &&
        return get_inequality_constraint(M, co.objective, p, i)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:InequalityConstraint], (copy(M, p), i)) do
            get_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_inequality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, i::Colon
    )
    (!haskey(co.cache, :InequalityConstraints)) &&
        return get_inequality_constraint(M, co.objective, p, i)
    return copy(# Return a copy of the version in the cache
        get!(co.cache[:InequalityConstraints], copy(M, p)) do
            get_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_inequality_constraint(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    key = copy(M, p)
    if haskey(co.cache, :InequalityConstraints) # full constraints are stored
        if haskey(co.cache[:InequalityConstraints], key)
            return co.cache[:InequalityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :InequalityConstraint) # storing the index constraints
        return [
            copy(
                get!(co.cache[:InequalityConstraint], (key, j)) do
                    get_inequality_constraint(M, co.objective, p, j)
                end,
            ) for
                j in _to_iterable_indices(1:inequality_constraints_length(co.objective), i)
        ]
    end # neither cache: pass down to objective
    return get_inequality_constraint(M, co.objective, p, i)
end

function get_grad_equality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, j::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = nothing,
    )
    !(haskey(co.cache, :GradEqualityConstraint)) &&
        return get_grad_equality_constraint(M, co.objective, p, j)
    return copy(# Return a copy of the version in the cache
        M,
        p,
        get!(co.cache[:GradEqualityConstraint], (copy(M, p), j)) do
            get_grad_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_grad_equality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective{<:ConstrainedManifoldObjective}, p, j::Colon,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_equality_constraint(M, co.objective, p, j)
    pM = PowerManifold(M, range, length(get_objective(co, true).equality_constraints))
    P = fill(p, pM)
    return copy(# Return a copy of the version in the cache
        pM,
        P,
        get!(co.cache[:GradEqualityConstraints], (copy(M, p))) do
            get_grad_equality_constraint(M, co.objective, p, j)
        end,
    )
end
function get_grad_equality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    key = copy(M, p)
    n = _vgf_index_to_length(i, equality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    P = fill(p, pM)
    if haskey(co.cache, :GradEqualityConstraints) # full constraints are stored
        if haskey(co.cache[:GradEqualityConstraints], key)
            return co.cache[:GradEqualityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradEqualityConstraint) # storing the index constraints
        # allocate a tangent vector
        X = zero_vector(pM, P)
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i))
            copyto!(
                M, _write(pM, rep_size, X, (k,)), p,
                get!(co.cache[:GradEqualityConstraint], (key, j)) do
                    get_grad_equality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_equality_constraint(M, co.objective, p, i)
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ManifoldCachedObjective, p, j::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = nothing,
    )
    !(haskey(co.cache, :GradEqualityConstraint)) &&
        return get_grad_equality_constraint!(M, X, co.objective, p, j)
    copyto!(
        M, X, p,
        get!(co.cache[:GradEqualityConstraint], (copy(M, p), j)) do
            # This evaluates in place of X
            get_grad_equality_constraint!(M, X, co.objective, p, j)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ManifoldCachedObjective{<:ConstrainedManifoldObjective}, p, i::Colon,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    !(haskey(co.cache, :GradEqualityConstraints)) &&
        return get_grad_equality_constraint!(M, X, co.objective, p, i)
    pM = PowerManifold(M, range, length(get_objective(co, true).equality_constraints))
    P = fill(p, pM)
    copyto!(
        pM, X, P,
        get!(co.cache[:GradEqualityConstraints], (copy(M, p))) do
            # This evaluates in place of X
            get_grad_equality_constraint!(M, X, co.objective, p, i)
            copy(pM, P, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ManifoldCachedObjective, p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    key = copy(M, p)
    n = _vgf_index_to_length(i, equality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    if haskey(co.cache, :GradEqualityConstraints) # full constraints are stored
        if haskey(co.cache[:GradEqualityConstraints], key)
            # access is subsampled with j, result linear in k
            for (k, j) in zip(
                    1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i)
                )
                copyto!(
                    M,
                    _write(pM, rep_size, X, (k,)),
                    p,
                    co.cache[:GradEqualityConstraints][key][j],
                )
            end
            return X
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradEqualityConstraint) # store the index constraints
        # allocate a tangent vector
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:equality_constraints_length(co.objective), i))
            copyto!(
                M, _write(pM, rep_size, X, (k,)), p,
                get!(co.cache[:GradEqualityConstraint], (key, j)) do
                    get_grad_equality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_equality_constraint!(M, X, co.objective, p, i)
end

#
#
# Inequality Constraint
function get_grad_inequality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = nothing,
    )
    !(haskey(co.cache, :GradInequalityConstraint)) &&
        return get_grad_inequality_constraint(M, co.objective, p, i)
    return copy(
        M,
        p,
        get!(co.cache[:GradInequalityConstraint], (copy(M, p), i)) do
            get_grad_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_grad_inequality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective{<:ConstrainedManifoldObjective}, p, i::Colon,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    !(haskey(co.cache, :GradInequalityConstraints)) &&
        return get_grad_inequality_constraint(M, co.objective, p, i)
    pM = PowerManifold(M, range, length(get_objective(co, true).inequality_constraints))
    P = fill(p, pM)
    return copy(# Return a copy of the version in the cache
        pM,
        P,
        get!(co.cache[:GradInequalityConstraints], (copy(M, p))) do
            get_grad_inequality_constraint(M, co.objective, p, i)
        end,
    )
end
function get_grad_inequality_constraint(
        M::AbstractManifold, co::ManifoldCachedObjective, p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    key = copy(M, p)
    n = _vgf_index_to_length(i, inequality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    P = fill(p, pM)
    if haskey(co.cache, :GradInequalityConstraints) # full constraints are stored
        if haskey(co.cache[:GradInequalityConstraints], key)
            return co.cache[:GradInequalityConstraints][key][i]
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradInequalityConstraint) # storing the index constraints
        # allocate a tangent vector
        X = zero_vector(pM, P)
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:inequality_constraints_length(co.objective), i))
            copyto!(
                M, _write(pM, rep_size, X, (k,)), p,
                get!(co.cache[:GradInequalityConstraint], (key, j)) do
                    get_grad_inequality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_inequality_constraint(M, co.objective, p, i)
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ManifoldCachedObjective, p,
        i::Integer, range::Union{AbstractPowerRepresentation, Nothing} = nothing,
    )
    !(haskey(co.cache, :GradInequalityConstraint)) &&
        return get_grad_inequality_constraint!(M, X, co.objective, p, i)
    copyto!(
        M, X, p,
        get!(co.cache[:GradInequalityConstraint], (copy(M, p), i)) do
            # This evaluates in place of X
            get_grad_inequality_constraint!(M, X, co.objective, p, i)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ManifoldCachedObjective{<:ConstrainedManifoldObjective}, p, j::Colon,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    !(haskey(co.cache, :GradInequalityConstraints)) &&
        return get_grad_inequality_constraint!(M, X, co.objective, p, j)
    pM = PowerManifold(M, range, length(get_objective(co, true).inequality_constraints))
    P = fill(p, pM)
    copyto!(
        pM,
        X,
        P,
        get!(co.cache[:GradInequalityConstraints], (copy(M, p))) do
            # This evaluates in place of X
            get_grad_inequality_constraint!(M, X, co.objective, p, j)
            copy(pM, P, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ManifoldCachedObjective, p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = NestedPowerRepresentation(),
    )
    key = copy(M, p)
    n = _vgf_index_to_length(i, inequality_constraints_length(co.objective))
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    if haskey(co.cache, :GradInequalityConstraints) # full constraints are stored
        if haskey(co.cache[:GradInequalityConstraints], key)
            # access is subsampled with j, result linear in k
            for (k, j) in zip(
                    1:n, _to_iterable_indices(1:inequality_constraints_length(co.objective), i)
                )
                copyto!(
                    M,
                    _write(pM, rep_size, X, (k,)),
                    p,
                    co.cache[:GradInequalityConstraints][key][j],
                )
            end
            return X
            #but caching is not possible here, since that requires evaluating all
        end
    end
    if haskey(co.cache, :GradInequalityConstraint) # storing the index constraints
        # access is subsampled with j, result linear in k
        for (k, j) in
            zip(1:n, _to_iterable_indices(1:inequality_constraints_length(co.objective), i))
            copyto!(
                M,
                _write(pM, rep_size, X, (k,)),
                p,
                get!(co.cache[:GradInequalityConstraint], (key, j)) do
                    get_grad_inequality_constraint(M, co.objective, p, j)
                end,
            )
        end
        return X
    end # neither cache: pass down to objective
    return get_grad_inequality_constraint!(M, X, co.objective, p, i)
end

function get_hessian(M::AbstractManifold, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Hessian)) && return get_hessian(M, co.objective, p, X)
    return copy(
        M, p,
        get!(co.cache[:Hessian], (copy(M, p), copy(M, p, X))) do
            get_hessian(M, co.objective, p, X)
        end,
    )
end
function get_hessian!(M::AbstractManifold, Y, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Hessian)) && return get_hessian!(M, Y, co.objective, p, X)
    copyto!(
        M, Y, p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:Hessian], (copy(M, p), copy(M, p, X))) do
            get_hessian!(M, Y, co.objective, p, X)
            copy(M, p, Y) #store a copy of Y
        end,
    )
    return Y
end

function get_hessian_function(
        mco::ManifoldCachedObjective, recursive::Bool = false; evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    )
    recursive && (return get_hessian_function(mco.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p, X) -> get_hessian(M, mco, p, X)
    else
        return (M, Y, p, X) -> get_hessian!(M, Y, mco, p, X)
    end
end
function get_preconditioner(M::AbstractManifold, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Preconditioner)) && return get_preconditioner(M, co.objective, p, X)
    return copy(
        M, p,
        get!(co.cache[:Preconditioner], (copy(M, p), copy(M, p, X))) do
            get_preconditioner(M, co.objective, p, X)
        end,
    )
end
function get_preconditioner!(M::AbstractManifold, Y, co::ManifoldCachedObjective, p, X)
    !(haskey(co.cache, :Preconditioner)) &&
        return get_preconditioner!(M, Y, co.objective, p, X)
    copyto!(
        M, Y, p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:Preconditioner], (copy(M, p), copy(M, p, X))) do
            get_preconditioner!(M, Y, co.objective, p, X)
            copy(M, p, Y)
        end,
    )
    return Y
end

function get_proximal_map(M::AbstractManifold, co::ManifoldCachedObjective, λ, p, i)
    !(haskey(co.cache, :ProximalMap)) && return get_proximal_map(M, co.objective, λ, p, i)
    return copy(
        M,
        get!(co.cache[:ProximalMap], (copy(M, p), λ, i)) do  # use the tuple (p,i) as key
            get_proximal_map(M, co.objective, λ, p, i)
        end,
    )
end
function get_proximal_map!(M::AbstractManifold, q, co::ManifoldCachedObjective, λ, p, i)
    !(haskey(co.cache, :ProximalMap)) &&
        return get_proximal_map!(M, q, co.objective, λ, p, i)
    copyto!(
        M, q,
        get!(co.cache[:ProximalMap], (copy(M, p), λ, i)) do
            get_proximal_map!(M, q, co.objective, λ, p, i) #compute in-place of q
            copy(M, q) #store copy of q
        end,
    )
    return q
end

function get_gradient(M::AbstractManifold, co::ManifoldCachedObjective, p, i)
    !(haskey(co.cache, :StochasticGradient)) && return get_gradient(M, co.objective, p, i)
    return copy(
        M, p,
        get!(co.cache[:StochasticGradient], (copy(M, p), i)) do
            get_gradient(M, co.objective, p, i)
        end,
    )
end
function get_gradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p, i)
    !(haskey(co.cache, :StochasticGradient)) && return get_gradient!(M, X, co.objective, p, i)
    copyto!(
        M, X, p,
        get!(co.cache[:StochasticGradient], (copy(M, p), i)) do
            # This evaluates in place of X
            get_gradient!(M, X, co.objective, p, i)
            copy(M, p, X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end

function get_gradients(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :StochasticGradients)) && return get_gradients(M, co.objective, p)
    return copy.(
        Ref(M), Ref(p),
        get!(co.cache[:StochasticGradients], copy(M, p)) do
            get_gradients(M, co.objective, p)
        end,
    )
end
function get_gradients!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :StochasticGradients)) && return get_gradients!(M, X, co.objective, p)
    copyto!.(
        Ref(M),
        X,
        Ref(p),
        get!(co.cache[:StochasticGradients], copy(M, p)) do
            # This evaluates in place of X
            get_gradients!(M, X, co.objective, p)
            copy.(Ref(M), Ref(p), X) #this creates a copy to be placed in the cache
        end, #and copy the values back to X
    )
    return X
end

function get_subgradient(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubGradient)) && return get_subgradient(M, co.objective, p)
    return copy(
        M, p,
        get!(co.cache[:SubGradient], copy(M, p)) do
            get_subgradient(M, co.objective, p)
        end,
    )
end
function get_subgradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubGradient)) && return get_subgradient!(M, X, co.objective, p)
    copyto!(
        M, X, p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:SubGradient], copy(M, p)) do
            get_subgradient!(M, X, co.objective, p)
            copy(M, p, X)
        end,
    )
    return X
end
# Subtrahend gradient (from DC)
function get_subtrahend_gradient(M::AbstractManifold, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubtrahendGradient)) && return get_subtrahend_gradient(M, co.objective, p)
    return copy(
        M, p,
        get!(co.cache[:SubtrahendGradient], copy(M, p)) do
            get_subtrahend_gradient(M, co.objective, p)
        end,
    )
end
function get_subtrahend_gradient!(M::AbstractManifold, X, co::ManifoldCachedObjective, p)
    !(haskey(co.cache, :SubtrahendGradient)) && return get_subtrahend_gradient!(M, X, co.objective, p)
    copyto!(
        M, X,
        p, # perform an in-place cache evaluation, see also `get_gradient!`
        get!(co.cache[:SubtrahendGradient], copy(M, p)) do
            get_subtrahend_gradient!(M, X, co.objective, p)
            copy(M, p, X)
        end,
    )
    return X
end

function show(io::IO, mco::ManifoldCachedObjective)
    return print(io, "$(status_summary(mco))")
end
function show(
        io::IO, t::Tuple{<:ManifoldCachedObjective, S}
    ) where {S <: AbstractManoptSolverState}
    return print(io, "$(t[2])\n\n$(status_summary(t[1]))")
end
function status_summary(mco::ManifoldCachedObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(mco))
    s = "## Cache\n"
    s2 = status_summary(mco.objective; context = context)
    (length(s2) > 0) && (s2 = "$(s2)\n\n")
    length(mco.cache) == 0 && return "$(s2)$(s)    No caches active"
    longest_key_length = max(length.(["$k" for k in keys(mco.cache)])...)
    cache_strings = [
        "  * :" *
            rpad("$k", longest_key_length, " ") *
            " : $(v.currentsize)/$(v.maxsize) entries of type $(valtype(v)) used" for
            (k, v) in zip(keys(mco.cache), values(mco.cache))
    ]
    return "$(s2)$(s)$(join(cache_strings, "\n"))\n"
end

#
#
# ---
@doc """
    ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}

Specify an [`AbstractManifoldObjective`](@ref) that does only have information about
the cost function ``f:  $(_math(:Manifold)) → ℝ`` implemented as a function `(M, p) -> c`
to compute the cost value `c` at `p` on the manifold `M`.

# Fields

* `cost`: a function ``f: $(_math(:Manifold)) → ℝ`` to minimize

# Constructors

    ManifoldCostObjective(f; p::P = missing) where {P}
    ManifoldCostObjective(f, ::Type{P}) where {P}

Generate a [`ManifoldCostObjective`](@ref) with cost function `f`.

## Keyword Arguments
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.

## See also
[`NelderMead`](@ref), [`particle_swarm`](@ref)
"""
struct ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}
    cost::F
    function ManifoldCostObjective(f, ::Type{P}) where {P}
        f_ = maybe_wrap_function(f, P; result = :Number)
        return new{typeof(f_)}(f_)
    end
end
ManifoldCostObjective(f; p::P = missing) where {P} = ManifoldCostObjective(f, P)
function show(io::IO, mco::ManifoldCostObjective{F}) where {F}
    return print(io, "ManifoldCostObjective($(mco.cost))")
end
function status_summary(::ManifoldCostObjective{F}; context::Symbol = :default) where {F}
    return "A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
end

"""
    ManifoldCountObjective{P,O<:AbstractManifoldObjective,I<:Union{<:Integer,AbstractVector{<:Integer}}} <: AbstractDecoratedManifoldObjective{P}

A wrapper for any [`AbstractManifoldObjective`](@ref) of type `O` to count different calls
to parts of the objective.

# Fields

* `counts` a dictionary of symbols mapping to integers keeping the counted values
* `objective` the wrapped objective

# Supported symbols

| Symbol                      | Counts calls to (incl. `!` variants)   | Comment                      |
| :-------------------------- | :------------------------------------- | :--------------------------- |
| `:Cost`                     | [`get_cost`](@ref)                     |                              |
| `:Differential`             | [`get_differential`](@ref)             |                              |
| `:EqualityConstraint`       | [`get_equality_constraint`](@ref)      | requires vector of counters  |
| `:EqualityConstraints`      | [`get_equality_constraint`](@ref)      | when evaluating all of them with `:` |
| `:GradEqualityConstraint`   | [`get_grad_equality_constraint`](@ref) | requires vector of counters  |
| `:GradEqualityConstraints`  | [`get_grad_equality_constraint`](@ref) | when evaluating all of them with `:`  |
| `:GradInequalityConstraint` | [`get_grad_inequality_constraint`](@ref) | requires vector of counters  |
| `:GradInequalityConstraints`| [`get_grad_inequality_constraint`](@ref) | when evaluating all of them with `:`  |
| `:Gradient`                 | [`get_gradient`](@ref)`(M,p)`          |                              |
| `:Hessian`                  | [`get_hessian`](@ref)                  |                              |
| `:InequalityConstraint`     | [`get_inequality_constraint`](@ref)    | requires vector of counters  |
| `:InequalityConstraints`    | [`get_inequality_constraint`](@ref)    | when evaluating all of them with `:`  |
| `:Preconditioner`           | [`get_preconditioner`](@ref)           |                              |
| `:ProximalMap`              | [`get_proximal_map`](@ref)             |                              |
| `:StochasticGradients`      | [`get_gradients`](@ref)                |                              |
| `:StochasticGradient`       | [`get_gradient`](@ref)`(M, p, i)`      |                              |
| `:SubGradient`              | [`get_subgradient`](@ref)              |                              |
| `:SubtrahendGradient`       | [`get_subtrahend_gradient`](@ref)      |                              |

# Constructors

    ManifoldCountObjective(objective::AbstractManifoldObjective, counts::Dict{Symbol, <:Integer})

Initialize the `ManifoldCountObjective` to wrap `objective`, initializing the set of counts.

    ManifoldCountObjective(M::AbstractManifold, objective::AbstractManifoldObjective, count::AbstractVector{Symbol}, init=0)

Count function calls on `objective` using the symbols in `count` initializing all entries to `init`.
"""
struct ManifoldCountObjective{
        P, O <: AbstractManifoldObjective, I <: Union{<:Integer, AbstractVector{<:Integer}},
    } <: AbstractDecoratedManifoldObjective{P}
    counts::Dict{Symbol, I}
    objective::O
end
ManifoldCountObjective(::AbstractManifold, o::AbstractManifoldObjective, c) = ManifoldCountObjective(o, c)
function ManifoldCountObjective(
        objective::O, counts::Dict{Symbol, I}
    ) where {
        I <: Union{<:Integer, AbstractVector{<:Integer}}, O <: AbstractManifoldObjective,
    }
    return ManifoldCountObjective{O, O, I}(counts, objective)
end
# Store the undecorated type of the input is decorated
function ManifoldCountObjective(
        objective::O, counts::Dict{Symbol, I}
    ) where {
        I <: Union{<:Integer, AbstractVector{<:Integer}},
        P <: AbstractManifoldObjective, O <: AbstractDecoratedManifoldObjective{P},
    }
    return ManifoldCountObjective{P, O, I}(counts, objective)
end
function ManifoldCountObjective(
        M::AbstractManifold, objective::O, count::AbstractVector{Symbol}, init::I = 0;
        p::P = rand(M),
    ) where {P, I <: Integer, O <: AbstractManifoldObjective}
    # Infer the sizes of the counters from the symbols if possible
    counts = Pair{Symbol, Union{I, Vector{I}}}[]
    for symbol in count
        l = _get_counter_size(M, objective, symbol, p)
        push!(counts, Pair(symbol, l == 1 ? init : fill(init, l)))
    end
    return ManifoldCountObjective(objective, Dict(counts))
end

function _get_counter_size(
        M::AbstractManifold, o::O, s::Symbol, p::P = rand(M)
    ) where {P, O <: AbstractManifoldObjective}
    # vectorial counting cases
    (s === :EqualityConstraint) && (return length(get_equality_constraint(M, o, p, :)))
    (s === :GradEqualityConstraint) && (return length(get_equality_constraint(M, o, p, :)))
    (s === :InequalityConstraint) && (return length(get_inequality_constraint(M, o, p, :)))
    (s === :GradInequalityConstraint) &&
        (return length(get_inequality_constraint(M, o, p, :)))
    # For now this only appears in ProximalMapObjective, access its field
    if s === :ProximalMap
        pm = get_objective(o).proximal_maps!
        return (pm isa Union{Tuple, AbstractVector}) ? length(pm) : 1
    end
    (s === :StochasticGradient) && (return length(get_gradients(M, o, p)))
    return 1 #number - default
end

function _count_if_exists(co::ManifoldCountObjective, s::Symbol)
    return haskey(co.counts, s) && (co.counts[s] += 1)
end
function _count_if_exists(co::ManifoldCountObjective, s::Symbol, i::Integer)
    return if haskey(co.counts, s)
        if (i == 1) && (ndims(co.counts[s]) == 0)
            return co.counts[s] += 1
        elseif length(i) == ndims(co.counts[s]) && all(i .<= size(co.counts[s]))
            return co.counts[s][i] += 1
        end
    end
end

"""
    get_count(co::ManifoldCountObjective, s::Symbol, mode::Symbol=:None)

Get the number of counts for a certain symbol `s`.

Depending on the `mode` different results appear if the symbol does not exist in the dictionary

* `:None`:  (default) silent mode, returns `-1` for non-existing entries
* `:warn`:  issues a warning if a field does not exist
* `:error`: issues an error if a field does not exist
"""
function get_count(co::ManifoldCountObjective, s::Symbol, mode::Symbol = :None)
    if !haskey(co.counts, s)
        msg = "There is no recorded count for $s."
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msg))
        return -1
    end
    return co.counts[s]
end
function get_count(o::AbstractManifoldObjective, s::Symbol, mode::Symbol = :None)
    return _get_count(o, dispatch_objective_decorator(o), s, mode)
end
function _get_count(o::AbstractManifoldObjective, ::Val{false}, s, m)
    return error("It seems $o does not provide access to a `ManifoldCountObjective`.")
end
function _get_count(o::AbstractManifoldObjective, ::Val{true}, s, m)
    return get_count(get_objective(o, false), s, m)
end

function get_count(co::ManifoldCountObjective, s::Symbol, i, mode::Symbol = :None)
    if !haskey(co.counts, s)
        msg = "There is no recorded count for :$s."
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msg))
        return -1
    end
    if !(ndims(i) == 0 && ndims(co.counts[s]) == 1) && ndims(i) != ndims(co.counts[s])
        msg = "The entry for :$s has $(ndims(co.counts[s])) dimensions but the index you provided has $(ndims(i))"
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msg))
        return -1
    end
    if ndims(i) == 0 && ndims(co.counts[s]) == 0
        if i > 1
            msg = "The entry for :$s is a number, but you provided the index $i > 1"
            (mode === :warn) && (@warn msg)
            (mode === :error) && (error(msg))
            return -1
        end
        return co.counts[s]
    end
    if any(i .> size(co.counts[s]))
        msg = "The index $i is out of range for the stored counts in :$s ($(size(co.counts[s])))."
        (mode === :warn) && (@warn msg)
        (mode === :error) && (error(msg))
        return -1
    end
    return co.counts[s][i...]
end
function get_count(o::AbstractManifoldObjective, s::Symbol, i, mode::Symbol = :None)
    return _get_count(o, dispatch_objective_decorator(o), s, i, mode)
end
function _get_count(o::AbstractManifoldObjective, ::Val{false}, s, i, m)
    return error("It seems $o does not provide access to a `ManifoldCountObjective`.")
end
function _get_count(o::AbstractManifoldObjective, ::Val{true}, s, i, m)
    return get_count(get_objective(o, false), s, i, m)
end

"""
    reset_counters!(co::ManifoldCountObjective, value::Integer=0)

Reset all values in the count objective to `value`.
"""
function reset_counters!(co::ManifoldCountObjective, value::Integer = 0)
    for s in keys(co.counts)
        if (ndims(co.counts[s]) == 0)
            co.counts[s] = value
        else
            co.counts[s] .= value
        end
    end
    return co
end
function reset_counters!(o::AbstractDecoratedManifoldObjective, value::Integer = 0)
    return reset_counters!(get_objective(o, false), value)
end
function reset_counters!(o::AbstractManifoldObjective, value::Integer = 0)
    return error("It seems $o does not provide access to a `ManifoldCountObjective`.")
end
# Default, just count cost
function get_cost(M::AbstractManifold, co::ManifoldCountObjective, p)
    _count_if_exists(co, :Cost)
    return get_cost(M, co.objective, p)
end
# Passthrough
function get_cost_function(co::ManifoldCountObjective, recursive = false)
    recursive && return get_cost_function(co.objective, recursive)
    return (M, p) -> get_cost(M, co, p)
end

function get_cost_and_gradient(M::AbstractManifold, co::ManifoldCountObjective, p)
    _count_if_exists(co, :Cost)
    _count_if_exists(co, :Gradient)
    return get_cost_and_gradient(M, co.objective, p)
end


function get_cost_and_gradient!(M::AbstractManifold, X, co::ManifoldCountObjective, p)
    _count_if_exists(co, :Cost)
    _count_if_exists(co, :Gradient)
    return get_cost_and_gradient!(M, X, co.objective, p)
end

function get_differential(M::AbstractManifold, co::ManifoldCountObjective, p, X; kwargs...)
    _count_if_exists(co, :Differential)
    return get_differential(M, co.objective, p, X; kwargs...)
end
# Passthrough
function get_differential_function(co::ManifoldCountObjective, recursive = false)
    recursive && return get_differential_function(co.objective, recursive)
    return (M, p, X; kwargs...) -> get_differential(M, co, p, X; kwargs...)
end

function get_gradient_function(mco::ManifoldCountObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    recursive && return get_gradient_function(mco.objective, recursive; evaluation = evaluation)
    # Otherwise, keep count
    if evaluation isa AllocatingEvaluation
        return (M, p) -> get_gradient(M, mco, p)
    else
        return (M, X, p) -> get_gradient!(M, X, mco, p)
    end
end

function get_gradient(M::AbstractManifold, co::ManifoldCountObjective, p)
    _count_if_exists(co, :Gradient)
    return get_gradient(M, co.objective, p)
end
function get_gradient!(M::AbstractManifold, X, co::ManifoldCountObjective, p)
    _count_if_exists(co, :Gradient)
    get_gradient!(M, X, co.objective, p)
    return X
end


function get_hessian(M::AbstractManifold, co::ManifoldCountObjective, p, X)
    _count_if_exists(co, :Hessian)
    return get_hessian(M, co.objective, p, X)
end
function get_hessian!(M::AbstractManifold, Y, co::ManifoldCountObjective, p, X)
    _count_if_exists(co, :Hessian)
    get_hessian!(M, Y, co.objective, p, X)
    return Y
end
function get_hessian_function(
        sco::ManifoldCountObjective, recursive::Bool = false; evaluation::AbstractEvaluationType = AllocatingEvaluation()
    )
    recursive && return get_hessian_function(sco.objective, recursive; evaluation = evaluation)
    if evaluation isa AllocatingEvaluation
        return (M, p, X) -> get_hessian(M, sco, p, X)
    else
        return (M, Y, p, X) -> get_hessian!(M, Y, sco, p, X)
    end
end

function get_preconditioner(M::AbstractManifold, co::ManifoldCountObjective, p, X)
    _count_if_exists(co, :Preconditioner)
    return get_preconditioner(M, co.objective, p, X)
end
function get_preconditioner!(M::AbstractManifold, Y, co::ManifoldCountObjective, p, X)
    _count_if_exists(co, :Preconditioner)
    get_preconditioner!(M, Y, co.objective, p, X)
    return Y
end

#
# Constraint
function get_equality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, c::Colon
    )
    _count_if_exists(co, :EqualityConstraints)
    return get_equality_constraint(M, co.objective, p, c)
end
function get_equality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, j::Integer
    )
    _count_if_exists(co, :EqualityConstraint, j)
    return get_equality_constraint(M, co.objective, p, j)
end
function get_equality_constraint(M::AbstractManifold, co::ManifoldCountObjective, p, i)
    for j in _to_iterable_indices(1:equality_constraints_length(co.objective), i)
        _count_if_exists(co, :EqualityConstraint, j)
    end
    return get_equality_constraint(M, co.objective, p, i)
end

function get_inequality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, i::Colon
    )
    _count_if_exists(co, :InequalityConstraints)
    return get_inequality_constraint(M, co.objective, p, i)
end
function get_inequality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, i::Integer
    )
    _count_if_exists(co, :InequalityConstraint, i)
    return get_inequality_constraint(M, co.objective, p, i)
end
function get_inequality_constraint(M::AbstractManifold, co::ManifoldCountObjective, p, i)
    for j in _to_iterable_indices(1:inequality_constraints_length(co.objective), i)
        _count_if_exists(co, :InequalityConstraint, j)
    end
    return get_inequality_constraint(M, co.objective, p, i)
end

function get_grad_equality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, i::Colon
    )
    _count_if_exists(co, :GradEqualityConstraints)
    return get_grad_equality_constraint(M, co.objective, p, i)
end
function get_grad_equality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, j::Integer
    )
    _count_if_exists(co, :GradEqualityConstraint, j)
    return get_grad_equality_constraint(M, co.objective, p, j)
end
function get_grad_equality_constraint(M::AbstractManifold, co::ManifoldCountObjective, p, i)
    for j in _to_iterable_indices(1:equality_constraints_length(co.objective), i)
        _count_if_exists(co, :GradEqualityConstraint, j)
    end
    return get_grad_equality_constraint(M, co.objective, p, i)
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ManifoldCountObjective, p, i::Colon
    )
    _count_if_exists(co, :GradEqualityConstraints)
    return get_grad_equality_constraint!(M, X, co.objective, p, i)
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ManifoldCountObjective, p, j::Integer
    )
    _count_if_exists(co, :GradEqualityConstraint, j)
    return get_grad_equality_constraint!(M, X, co.objective, p, j)
end
function get_grad_equality_constraint!(
        M::AbstractManifold, X, co::ManifoldCountObjective, p, i
    )
    for j in _to_iterable_indices(1:equality_constraints_length(co.objective), i)
        _count_if_exists(co, :GradEqualityConstraint, j)
    end
    return get_grad_equality_constraint!(M, X, co.objective, p, i)
end

function get_grad_inequality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, i::Colon
    )
    _count_if_exists(co, :GradInequalityConstraints)
    return get_grad_inequality_constraint(M, co.objective, p, i)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, i::Integer
    )
    _count_if_exists(co, :GradInequalityConstraint, i)
    return get_grad_inequality_constraint(M, co.objective, p, i)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, co::ManifoldCountObjective, p, i
    )
    for j in _to_iterable_indices(1:inequality_constraints_length(co.objective), i)
        _count_if_exists(co, :GradInequalityConstraint, j)
    end
    return get_grad_inequality_constraint(M, co.objective, p, i)
end

function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ManifoldCountObjective, p, i::Colon
    )
    _count_if_exists(co, :GradInequalityConstraints)
    return get_grad_inequality_constraint!(M, X, co.objective, p, i)
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ManifoldCountObjective, p, i::Integer
    )
    _count_if_exists(co, :GradInequalityConstraint, i)
    return get_grad_inequality_constraint!(M, X, co.objective, p, i)
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, X, co::ManifoldCountObjective, p, i
    )
    for j in _to_iterable_indices(1:inequality_constraints_length(co.objective), i)
        _count_if_exists(co, :GradInequalityConstraint, j)
    end
    return get_grad_inequality_constraint!(M, X, co.objective, p, i)
end

#
# proximal maps
function get_proximal_map(M::AbstractManifold, co::ManifoldCountObjective, λ, p)
    _count_if_exists(co, :ProximalMap)
    return get_proximal_map(M, co.objective, λ, p)
end
function get_proximal_map!(M::AbstractManifold, q, co::ManifoldCountObjective, λ, p)
    _count_if_exists(co, :ProximalMap)
    return get_proximal_map!(M, q, co.objective, λ, p)
end
function get_proximal_map(M::AbstractManifold, co::ManifoldCountObjective, λ, p, i)
    _count_if_exists(co, :ProximalMap, i)
    return get_proximal_map(M, co.objective, λ, p, i)
end
function get_proximal_map!(M::AbstractManifold, q, co::ManifoldCountObjective, λ, p, i)
    _count_if_exists(co, :ProximalMap, i)
    return get_proximal_map!(M, q, co.objective, λ, p, i)
end

#
# DC
function get_subtrahend_gradient(M::AbstractManifold, co::ManifoldCountObjective, p)
    _count_if_exists(co, :SubtrahendGradient)
    return get_subtrahend_gradient(M, co.objective, p)
end
function get_subtrahend_gradient!(M::AbstractManifold, X, co::ManifoldCountObjective, p)
    _count_if_exists(co, :SubtrahendGradient)
    return get_subtrahend_gradient!(M, X, co.objective, p)
end

#
# Subgradient
function get_subgradient(M::AbstractManifold, co::ManifoldCountObjective, p)
    _count_if_exists(co, :SubGradient)
    return get_subgradient(M, co.objective, p)
end
function get_subgradient!(M::AbstractManifold, X, co::ManifoldCountObjective, p)
    _count_if_exists(co, :SubGradient)
    return get_subgradient!(M, X, co.objective, p)
end

#
# Stochastic Gradient
function get_gradients(M::AbstractManifold, co::ManifoldCountObjective, p)
    _count_if_exists(co, :StochasticGradients)
    return get_gradients(M, co.objective, p)
end
function get_gradients!(M::AbstractManifold, X, co::ManifoldCountObjective, p)
    _count_if_exists(co, :StochasticGradients)
    return get_gradients!(M, X, co.objective, p)
end
function get_gradient(M::AbstractManifold, co::ManifoldCountObjective, p, i)
    _count_if_exists(co, :StochasticGradient, i)
    return get_gradient(M, co.objective, p, i)
end
function get_gradient!(M::AbstractManifold, X, co::ManifoldCountObjective, p, i)
    _count_if_exists(co, :StochasticGradient, i)
    return get_gradient!(M, X, co.objective, p, i)
end

function objective_count_factory(
        M::AbstractManifold, o::AbstractManifoldObjective, counts::Vector{<:Symbol}
    )
    return ManifoldCountObjective(M, o, counts)
end

# change default – do not unwrap but call the one below
function status_summary(io::IO, co::ManifoldCountObjective; kwargs...)
    return print(io, status_summary(co; kwargs...))
end
function status_summary(co::ManifoldCountObjective; context::Symbol = :default)
    so = status_summary(co.objective; context = context)
    if _is_inline(context)
        return "$so (statistics: $(join([ ":$(c[1])=$(c[2])" for c in co.counts ], ", ")))"
    end
    s = "## Statistics on function calls\n"
    (length(so) > 0) && (so = "$(so)")
    length(co.counts) == 0 && return "$(s)    No counters active\n$(so)"
    longest_key_length = max(length.(["$c" for c in keys(co.counts)])...)
    count_strings = [
        "  * :$(rpad("$(c[1])", longest_key_length)) : $(c[2])" for c in co.counts
    ]
    return "$(so)\n\n$(s)$(join(count_strings, "\n"))"
end
function status_summary(t::Tuple{<:ManifoldCountObjective, S}; context::Symbol = :default) where {S <: AbstractManoptSolverState}
    return "$(status_summary(t[2], context = context))\n\n$(status_summary(t[1]; context = context))"
end
function show(io::IO, co::ManifoldCountObjective)
    return print(io, "ManifoldCountObjective($(repr(co.objective)), $(repr(co.counts)))")
end


#
#
# ---
@doc """
    ManifoldFirstOrderObjective{F<:NamedTuple} <: AbstractManifoldFirstOrderObjective{F, F}

Specify an objective containing a cost and its gradient or differential.

# Fields

* `functions::F`: a function or a tuple of functions containing the cost and first order information.

Currently the following cases are covered, sorted by their popularity

1. a single function `fg` representing a combined
    function `(M, X, p) -> (c, X)` that computes the cost `c=cost(M,p)` and gradient `X=grad_f(M, X, p)`;
2. a single function `fdf` representing a combined function
    `(M, p, X) -> (c, d)` that computes the cost `c=cost(M,p)` and the differential `d=diff_f(M, p, X)` in direction `X`;
3. pairs of single functions `(f, g)`, `(f, df)` of a cost function `f` and either its
    gradient `g` or its differential `d`, respectively
4. The function `(fg, d)` and `(fdf, g)`  from 1 and 2, respectively joined by
    the other missing third information, the differential for the first or the gradient for the second
5. a tuple `(f, g, d)` of three functions, computing cost `f`, gradient `g`,
    and differential `d` separately

In all cases a gradient and/or a differential that is present is assumed to work in-place,
see the [`InplaceManifoldFunction`](@ref) wrapper for alternatives.

The cases of a common `fg` function for cost and gradient and the tuple `(f,g)` are the most common ones.
They can also be addressed by their alternate constructors
[`ManifoldCostGradientObjective`](@ref)`(fg)` and [`ManifoldGradientObjective`](@ref)`(f,g)`, respectively.

# Constructors
    ManifoldFirstOrderObjective(; kwargs...)

## Keyword arguments

* `cost = missing` the cost function `c = f(M,p)`
* `costdifferential = missing` the combined cost and differential function  `fdf(M, p, X)`
* `costgradient = missing` the combined cost and gradient function `fg(M,p)` or in-place `fg!(M, X, p)`
* `differential = missing` the differential `d = df(M, p, X)`
$(_kwargs(:evaluation))
* `gradient=missing` the gradient function `g(M, p)` or in-place `g!(M, X, p)`
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.

Where:
 * At least one of `cost`, `costgradient` or `costdifferential` must be provided.
 * Either `gradient`, `costgradient`, `differential` or `costdifferential` must be provided.
 * If more than one function provides the same thing (e.g. cost), it is assumed that all
   such functions return the same value. Optimization algorithms will attempt to make the
   most efficient use of provided functions fitting for the access required.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldFirstOrderObjective{F <: NamedTuple} <: AbstractManifoldFirstOrderObjective{F, F}
    functions::F
end
function ManifoldFirstOrderObjective(;
        cost = missing, differential = missing, gradient = missing,
        costgradient = missing, costdifferential = missing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
    )
    no_cost = ismissing(cost)
    no_diff = ismissing(differential)
    no_grad = ismissing(gradient)
    ncg = ismissing(costgradient)
    ncd = ismissing(costdifferential)

    if no_cost && ncg && ncd
        throw(
            ArgumentError(
                "Either cost, costgradient or costdifferential keyword argument needs to be provided",
            ),
        )
    end
    if no_grad && ncg && no_diff && ncd
        throw(
            ArgumentError(
                "Either gradient, costgradient, differential or costdifferential keyword argument needs to be provided",
            ),
        )
    end
    nt = (;)
    if !no_cost
        nt = merge(nt, (; cost = maybe_wrap_function(cost, p; result = :Number)))
    end
    if !no_grad
        nt = merge(nt, (; gradient = maybe_wrap_function(gradient, p, evaluation; result = :TangentVector)))
    end
    if !no_diff
        nt = merge(nt, (; differential = maybe_wrap_function(differential, p; result = :Number)))
    end
    if !ncg
        nt = merge(nt, (; costgradient = maybe_wrap_function(costgradient, p, evaluation; result = :NumberAndTangentVector)))
    end
    if !ncd
        nt = merge(nt, (; costdifferential = maybe_wrap_function(costdifferential, p; result = :Number)))
    end
    return ManifoldFirstOrderObjective{typeof(nt)}(nt)
end

const ManifoldGradientObjective{F, G} = ManifoldFirstOrderObjective{
    <:Union{
        NamedTuple{Tuple{:cost, :gradient}, Tuple{F, G}},
        NamedTuple{Tuple{:cost, :gradient, :differential}, Tuple{F, G, D where {D}}},
    },
}
@doc """
    ManifoldGradientObjective(cost, gradient; kwargs...)

Generate an objective with a function `cost` and its `gradient`.
The gradient is assumed to work in-place

* as a function `(M, X, p) -> X` that works in place of `X`, an [`InplaceEvaluation`](@ref)

Internally this is stored in a [`ManifoldFirstOrderObjective`](@ref). The `kwargs...`
are also passed to this representation, which allows to add a special function
to evaluate the `differential`.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
function ManifoldGradientObjective(cost, grad; kwargs...)
    return ManifoldFirstOrderObjective(; cost = cost, gradient = grad, kwargs...)
end

const ManifoldCostGradientObjective{FG} = ManifoldFirstOrderObjective{
    <:Union{
        NamedTuple{Tuple{:costgradient}, Tuple{FG}},
        NamedTuple{Tuple{:costgradient, :differential}, Tuple{FG, D where {D}}},
    },
}
@doc """
    ManifoldCostGradientObjective(costgrad; kwargs...)

Create an objective containing one function to perform a combined computation of cost and its gradient

Internally this is stored in a [`ManifoldFirstOrderObjective`](@ref). The `kwargs...`
are also passed to this representation, which allows to add a special function
to evaluate the `differential`.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
function ManifoldCostGradientObjective(cost_grad; kwargs...)
    return ManifoldFirstOrderObjective(; costgradient = cost_grad, kwargs...)
end

# accessors
function get_cost(M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p)
    haskey(mfo.functions, :cost) && (return mfo.functions[:cost](M, p))
    X = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, p, X)[1]
    end
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[1])
    return error("$mfo does not seem to provide a cost")
end

"""
    get_cost_and_differential(
        M, mfo::ManifoldFirstOrderObjective, p, X;
        gradient = missing, evaluated::Bool = false,
    )

Evaluate cost function and differential simultaneously.

This combined evaluation is especially beneficial when you provided a combined `costdifferential` function within
the [`ManifoldFirstOrderObjective`](@ref) `mfo`.
When there is no separate differential, the evaluation falls back to evaluating the gradient and an inner product.

## Input
$(_args(:M))
* `mfo` an [`ManifoldFirstOrderObjective`](@ref)
$(_args([:p, :X]))

## Keyword Arguments
* `gradient = missing` provide a cache/memory to evaluate the gradient in.
  if not provided, this function will allocate one tangent vector
* `evaluated = false` if the cache is provided, this boolean specifies, whether
  the cache already contains the evaluated gradient and does not need to evaluate the gradient again, if possible.
  For example for a combined `costgrad` function, the evaluation can not be avoided, since we need the cost value.
"""
function get_cost_and_differential(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p, X;
        gradient = missing, evaluated = false
    )
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, p, X)
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :differential)
        return (mfo.functions[:cost](M, p), mfo.functions[:differential](M, p, X))
    elseif haskey(mfo.functions, :costgradient)
        _Y = ismissing(gradient) ? zero_vector(M, p) : gradient
        # here we can not avoid the evaluation of the gradient even if it was already evaluated
        cost, grad = mfo.functions[:costgradient](M, _Y, p)
        return (cost, real(inner(M, p, X, grad)))
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        cost = mfo.functions[:cost](M, p)
        _Y = ismissing(gradient) ? zero_vector(M, p) : gradient
        # if we have no cache or it has not been evaluated: evaluate the gradient
        (ismissing(gradient) || !evaluated) && (mfo.functions[:gradient](M, _Y, p))
        return (cost, real(inner(M, p, X, _Y)))
    end
    return error("$mfo does not provide a cost and a differential")
end
function get_cost_and_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective, p
    )
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p))
    if haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        return mfo.functions[:cost](M, p), mfo.functions[:gradient](M, X, p)
    end
    Y = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential) && haskey(mfo.functions, :gradient)
        return (
            mfo.functions[:costdifferential](M, p, Y)[1], mfo.functions[:gradient](M, X, p),
        )
    end
    return error("$mfo seems to either have no access to a cost or a gradient")
end
# Interaction with cache
function get_cost_and_gradient(
        M::AbstractManifold,
        co::ManifoldCountObjective{<:ManifoldFirstOrderObjective{<:NamedTuple}},
        p,
    )
    _count_if_exists(co, :Cost)
    _count_if_exists(co, :Gradient)
    return get_cost_and_gradient(M, co.objective, p)
end
function get_cost_and_gradient!(
        M::AbstractManifold,
        X,
        co::ManifoldCountObjective{<:ManifoldFirstOrderObjective{<:NamedTuple}},
        p,
    )
    _count_if_exists(co, :Cost)
    _count_if_exists(co, :Gradient)
    return get_cost_and_gradient!(M, X, co.objective, p)
end

function get_cost_function(mfo::ManifoldFirstOrderObjective, recursive::Bool = false)
    if haskey(mfo.functions, :cost)
        return mfo.functions[:cost]
    else
        return (M, p) -> get_cost(M, mfo, p)
    end
end

function get_differential(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p, X;
        gradient = missing, evaluated::Bool = false, kwargs...,
    )
    # If we have a differential – evaluate that
    haskey(mfo.functions, :differential) && (return mfo.functions[:differential](M, p, X))
    haskey(mfo.functions, :costdifferential) &&
        (return mfo.functions[:costdifferential](M, p, X)[2])
    # default: inner with gradient
    # (a) we have gradient but it is not evaluated -> eval
    (!evaluated && !ismissing(gradient)) && (get_gradient!(M, gradient, mfo, p))
    # if grad is missing -> allocated gradient
    ismissing(gradient) && (gradient = get_gradient(M, mfo, p))
    # -> we have a gradient!
    return real(inner(M, p, gradient, X))
end

function get_differential_function(
        mfo::ManifoldFirstOrderObjective, recursive::Bool = false
    )
    if haskey(mfo.functions, :differential)
        return mfo.functions[:differential]
    else
        return (M, p, X; kwargs...) -> get_differential(M, mfo, p, X; kwargs...)
    end
end
function get_gradient(
        M::AbstractManifold, co::ManifoldCountObjective{<:ManifoldFirstOrderObjective{<:NamedTuple}}, p,
    )
    _count_if_exists(co, :Gradient)
    fs = get_objective(co.objective, true).functions
    haskey(fs, :costgradient) && _count_if_exists(co, :Cost)
    haskey(fs, :costgradientdifferential) && _count_if_exists(co, :Cost)
    haskey(fs, :costgradientdifferential) && _count_if_exists(co, :Differential)
    haskey(fs, :gradientdifferential) && _count_if_exists(co, :Differential)
    return get_gradient(M, co.objective, p)
end
function get_gradient!(
        M::AbstractManifold, X,
        co::ManifoldCountObjective{<:ManifoldFirstOrderObjective{<:NamedTuple}},
        p,
    )
    _count_if_exists(co, :Gradient)
    fs = get_objective(co.objective, true).functions
    haskey(fs, :costgradient) && _count_if_exists(co, :Cost)
    haskey(fs, :costgradientdifferential) && _count_if_exists(co, :Cost)
    haskey(fs, :costgradientdifferential) && _count_if_exists(co, :Differential)
    haskey(fs, :gradientdifferential) && _count_if_exists(co, :Differential)
    get_gradient!(M, X, co.objective, p)
    return X
end
function get_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{<:NamedTuple}, p,
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, X, p))
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[2])
    return error("$mfo does not seem to provide a gradient")
end

function get_gradient_function(
        mfo::ManifoldFirstOrderObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation()
    )
    if evaluation isa AllocatingEvaluation # Since internally we have an inplace one: wrap
        return (M, p) -> get_gradient(M, mfo, p)
    else
        # (a) do we have access to a “pure” gradient function? Return that
        haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
        # (b) otherwise generate an anonymous function of correct type
        return (M, X, p) -> get_gradient!(M, X, mfo, p)
    end
end

function status_summary(mfo::ManifoldFirstOrderObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(mfo))
    return "A first order objective with $(length(mfo.functions)) provided functions.\n\n" * join([ "* $k:$(_MANOPT_INDENT) $(v)" for (k, v) in zip(keys(mfo.functions), mfo.functions) ], "\n")
end
function Base.show(io::IO, mfo::ManifoldFirstOrderObjective)
    print(io, "ManifoldFirstOrderObjective(; ")
    print(io, join([ "$k = $v" for (k, v) in zip(keys(mfo.functions), mfo.functions)], ", "))
    return print(io, ")")
end

#
#
# ---
@doc """
    ManifoldHessianObjective{C,G,H,Pre} <: AbstractManifoldHessianObjective{C,G,H}

Specify a problem for Hessian based algorithms.

# Fields

* `cost`:           a function ``f:$(_math(:Manifold))→ℝ`` to minimize
* `gradient!`:      the gradient ``$(_tex(:grad))f:$(_math(:Manifold)) → $(_math(:TangentBundle))`` of the cost function ``f``
* `hessian!`:       the Hessian ``$(_tex(:Hess))f(p)[⋅]: $(_math(:TangentSpace)) → $(_math(:TangentSpace))`` of the cost function ``f``
* `preconditioner!`: the symmetric, positive definite preconditioner
  as an approximation of the inverse of the Hessian of ``f``, a map with the same
  input variables as the `hessian` to numerically stabilize iterations when the Hessian is
  ill-conditioned

# Constructor
    ManifoldHessianObjective(
        f, grad_f, Hess_f, preconditioner = missing;
        p = missing, evaluation=AllocatingEvaluation()
    )

Generate a [`ManifoldHessianObjective`](@ref) from a cost function `f`, its gradient `grad_f`, its Hessian `Hess_f`, and an optional preconditioner `preconditioner`.
The `evaluation` keyword argument can be used to specify whether the functions are in-place or allocating, a point `p` can be used to specify the domain of the functions,
which only has a “wrapping” effect if `p` is an immutable number.

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct ManifoldHessianObjective{C, G, H, Pre} <: AbstractManifoldHessianObjective{C, G, H}
    cost::C
    gradient!::G
    hessian!::H
    preconditioner!::Pre
    function ManifoldHessianObjective(
            cost::C, grad::G, hess::H, precond = missing;
            p = missing, evaluation = AllocatingEvaluation(),
        ) where {C, G, H}
        cost_ = maybe_wrap_function(cost, p; result = :Number)
        grad_ = maybe_wrap_function(grad, p, evaluation; result = :TangentVector)
        hess_ = maybe_wrap_function(hess, p, evaluation; result = :TangentVector)
        precond_ = ismissing(precond) ? missing : maybe_wrap_function(precond, p, evaluation; result = :TangentVector)
        return new{typeof(cost_), typeof(grad_), typeof(hess_), typeof(precond_)}(cost_, grad_, hess_, precond_)
    end
end
function get_gradient(M::AbstractManifold, mho::ManifoldHessianObjective, p)
    X = zero_vector(M, p)
    mho.gradient!(M, X, p)
    return X
end
function get_gradient!(M::AbstractManifold, Y, mho::ManifoldHessianObjective, p)
    return mho.gradient!(M, Y, p)
end
function get_gradient_function(mho::ManifoldHessianObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    if evaluation isa AllocatingEvaluation # We have to wrap anyways
        return mho.gradient! isa InplaceManifoldFunction ? mho.gradient!.f : (M, p) -> mho.gradient!(M, zero_vector(M, p), p)
    else
        return mho.gradient!
    end
end
function get_hessian(M::AbstractManifold, mho::ManifoldHessianObjective, p, X)
    Y = zero_vector(M, p)
    mho.hessian!(M, Y, p, X)
    return Y
end
function get_hessian!(M::AbstractManifold, Y, mho::ManifoldHessianObjective, p, X)
    mho.hessian!(M, Y, p, X)
    return Y
end

function get_preconditioner end
@doc """
    get_preconditioner(M::AbstractManifold, mho::ManifoldHessianObjective, p, X)

Evaluate the preconditioner of the [`ManifoldHessianObjective`](@ref) `mho`
at the point `p`, applied to the tangent vector `X`.

It usually is a symmetric, positive definite approximation of the inverse of the Hessian of the cost function `f`.
"""
function get_preconditioner(M::AbstractManifold, mho::ManifoldHessianObjective, p, X)
    Y = zero_vector(M, p)
    ismissing(mho.preconditioner!) && return copyto!(M, Y, p, X)
    mho.preconditioner!(M, Y, p, X)
    return Y
end
function get_preconditioner(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_preconditioner(M, get_objective(admo, false), p, X)
end
function get_preconditioner!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, p, X
    )
    return get_preconditioner!(M, Y, get_objective(admo, false), p, X)
end
function get_preconditioner!(M::AbstractManifold, Y, mho::ManifoldHessianObjective, p, X)
    return ismissing(mho.preconditioner!) ? copyto!(M, Y, p, X) : mho.preconditioner!(M, Y, p, X)
end

update_hessian!(M, f, p, p_proposal, X) = f

update_hessian_basis!(M, f, p) = f

function status_summary(mho::ManifoldHessianObjective{E}; context::Symbol = :default) where {E}
    _is_inline(context) && return "A second order objective with cost, gradient$(ismissing(mho.preconditioner!) ? ", and" : ",") Hessian$(ismissing(mho.preconditioner!) ? "" : ", and a preconditioner")"
    precon_str = ismissing(mho.preconditioner!) ? "" : "\n* preconditioner: $(mho.preconditioner!)"
    return """
    A second order objective providing a cost, a gradient$(ismissing(mho.preconditioner!) ? ", and" : ",") a Hessian$(ismissing(mho.preconditioner!) ? "" : ", and a preconditioner")

    ## Functions
    * cost:    $(_MANOPT_INDENT)$(mho.cost)
    * gradient:$(_MANOPT_INDENT)$(mho.gradient!)
    * Hessian: $(_MANOPT_INDENT)$(mho.hessian!)$(precon_str)"""
end

function Base.show(io::IO, mho::ManifoldHessianObjective)
    print(io, "ManifoldHessianObjective(")
    print(io, "$(mho.cost), ")
    print(io, "$(mho.gradient!), ")
    print(io, "$(mho.hessian!)")
    !ismissing(mho.preconditioner!) && print(io, ", $(mho.preconditioner!)")
    return print(io, ")")
end


#
#
# ---
@doc """
    ManifoldNonlinearLeastSquaresObjective{VFV,RFV,TVC} <: AbstractManifoldFirstOrderObjective{VFV, VFV}

An objective to model the robustified nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

# Fields

* `objective`: a vector of [`AbstractFirstOrderVectorFunction`](@ref)s, one for each
  block component cost function ``F_i``, which might internally also be a vector of component costs ``(F_i)_j``,
  as well as their Jacobian ``J_{F_i}`` or a vector of gradients ``$(_tex(:grad)) (F_i)_j``
  depending on the specified [`AbstractVectorialType`](@ref)s.
* `robustifier`: a vector of [`AbstractRobustifierFunction`](@ref)s, one for each
  block component cost function ``F_i``.
* `value_cache::AbstractVector` an internal cache to store the result of evaluating the cost functions

# Constructors

    ManifoldNonlinearLeastSquaresObjective(f, jacobian, range_dimension::Integer, robustifier=IdentityRobustifier(); kwargs...)

Create a nonlinear least squares objective for a single vectorial function `f` and its `jacobian`,
where `range_dimension` is the dimension of the vector space `f` maps into. These three are internally
wrapped into a [`VectorGradientFunction`](@ref) and this then calls the following constructor.

    ManifoldNonlinearLeastSquaresObjective(vf::AbstractFirstOrderVectorFunction, robustifier::AbstractRobustifierFunction=IdentityRobustifier())

Create a nonlinear least squares objective for a given vectorial function.
Note that for this constructor the `robustifier` is applied componentwise to each component of `vf`,
i.e. wrapped in a [`ComponentwiseRobustifierFunction`](@ref).
Internally this wraps both `vf` and `robustifier` in an array and calls the next constructor.
Hence to not use the componentwise robustifier but a global one, pass `[vf,]` and `[robustifier,]` instead.

    ManifoldNonlinearLeastSquaresObjective(fs::Vector{<:AbstractFirstOrderVectorFunction}, robustifiers::Vector{<:AbstractRobustifierFunction}=fill(IdentityRobustifier(), length(fs)))

Given a vector of [`AbstractFirstOrderVectorFunction`](@ref)s to represent the single blocks
and a vector of robustifiers, one for each block, create the corresponding nonlinear least squares objective.

# Keyword arguments

The first constructor, that is the variant for a single block, accepts the following keyword
arguments, which are passed on to the corresponding [`VectorGradientFunction`](@ref) constructor.

* `function_type::`[`AbstractVectorialType`](@ref)`=`[`FunctionVectorialType`](@ref)`()`: specify
  the format the residuals are given in. By default a function returning a vector.
* `jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis()`: shortcut to specify
  the basis the Jacobian matrix is build with.
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoefficientVectorialType`](@ref)`(jacobian_tangent_basis)`:
  specify the format the Jacobian is given in. By default a matrix of the differential with
  respect to a certain basis of the tangent space.

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct ManifoldNonlinearLeastSquaresObjective{
        VFV <: AbstractVector{<:AbstractFirstOrderVectorFunction},
        RFV <: AbstractVector{<:AbstractRobustifierFunction},
        TVC <: AbstractVector,
    } <: AbstractManifoldFirstOrderObjective{VFV, VFV}
    objective::VFV
    robustifier::RFV
    value_cache::TVC
    # block components case constructor
    function ManifoldNonlinearLeastSquaresObjective(
            fs::VFV,
            robustifiers::RFV = fill(IdentityRobustifier(), length(fs)),
            value_cache::TVC = zeros(sum(length(f) for f in fs)),
        ) where {
            VFV <: AbstractVector{<:AbstractFirstOrderVectorFunction},
            RFV <: AbstractVector{<:AbstractRobustifierFunction},
            TVC <: AbstractVector,
        }
        # we need to check that the lengths match
        (length(fs) != length(robustifiers)) && throw(
            ArgumentError(
                "Number of functions ($(length(fs))) does not match number of robustifiers ($(length(robustifiers)))",
            ),
        )
        return new{VFV, RFV, TVC}(fs, robustifiers, value_cache)
    end
    # single component case constructor
    function ManifoldNonlinearLeastSquaresObjective(
            f::F,
            robustifier::R = IdentityRobustifier(),
            value_cache::TVC = zeros(length(f)),
        ) where {F <: AbstractFirstOrderVectorFunction, R <: AbstractRobustifierFunction, TVC <: AbstractVector}
        rs = [ComponentwiseRobustifierFunction(robustifier)]; fs = [f]
        return new{typeof(fs), typeof(rs), TVC}(fs, rs, value_cache)
    end
end
function ManifoldNonlinearLeastSquaresObjective(
        f, jacobian, range_dimension::Integer,
        robustifier::AbstractRobustifierFunction = IdentityRobustifier();
        jacobian_tangent_basis::AbstractBasis = DefaultOrthonormalBasis(),
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(jacobian_tangent_basis),
        function_type::AbstractVectorialType = FunctionVectorialType(),
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        p = missing
    )
    vgf = VectorGradientFunction(
        f, jacobian, range_dimension;
        jacobian_type = jacobian_type, function_type = function_type, evaluation = evaluation, p = p
    )
    return ManifoldNonlinearLeastSquaresObjective(vgf, robustifier)
end

"""
    get_cost(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p)

Compute the cost of the least squares objective, i.e.

```math
$(_tex(:frac, "1", "2")) $(_tex(:sum, "i=1", "m")) ρ_i $(_tex(:bigl))( $(_tex(:norm, "F_i(p)"))^2 $(_tex(:bigr))),
```

where ``F_i: $(_math(:Manifold)) → ℝ^{n_i}`` is the ``i``th block component of length ``n_i > 0``
and each ``ρ_i: ℝ → ℝ`` is a robustifier function, cf. [`AbstractRobustifierFunction`](@ref),
one for each such block component.
"""
function get_cost(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p)
    v = 0.0
    start = 0
    get_residuals!(M, nlso.value_cache, nlso, p)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        value_cache = view(nlso.value_cache, (start + 1):(start + len))
        v += _get_cost(M, o, r, p; value_cache = value_cache)
        start += len
    end
    v /= 2
    return v
end
# For a single block – or one summand in the docs of the previous function
function _get_cost(
        M, vgf::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, vgf, p)
    )
    vi = sum(abs2, value_cache)
    (a, _, _) = get_robustifier_values(r, vi)
    return a
end
# For a single vectorial function where the robustifier is applied to every index separately.
function _get_cost(
        M, vgf::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, vgf, p)
    )
    v = abs2.(value_cache)
    # componentwise robustify
    (a, _, _) = get_robustifier_values(cr, v)
    return sum(a)
end

_doc_get_gradient_nlso = """
    get_gradient(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...)
    get_gradient!(M::AbstractManifold, X, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...)

Compute the gradient for the [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso` at the point ``p ∈ M``,
i.e.

```math
$(_tex(:grad)) f(p) = $(_tex(:sum, "i=1", "m")) ρ'_i$(_tex(:bigl))($(_tex(:norm, "F_i(p)"; index = "2"))^2$(_tex(:bigr)))
$(_tex(:sum, "j=1", "n_i")) f_{i,j}(p) $(_tex(:grad)) f_{i,j}(p)
```

where ``F_i(p) ∈ ℝ^{n_i}`` is the vector of residuals for the `i`-th block component cost function
and ``f_{i,j}(p)`` its `j`-th component function.

# Keyword arguments
* `value_cache=nothing`: if provided, this vector is used to store the residuals ``F(p)``
  internally to avoid re-computations.
* `jacobian_cache=fill(nothing, length(nlso.objective))`: if provided, this is used to store
  the Jacobians of the component functions.
"""
@doc "$(_doc_get_gradient_nlso)"
function get_gradient(
        M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...,
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, nlso, p; kwargs...)
end
function get_gradient!(
        M::AbstractManifold, X, nlso::ManifoldNonlinearLeastSquaresObjective, p;
        value_cache = nothing, jacobian_cache = fill(nothing, length(nlso.objective)),
    )
    zero_vector!(M, X, p)
    start = 0
    for (o, r, jb) in zip(nlso.objective, nlso.robustifier, jacobian_cache) # for every block
        len = length(o)
        Fi = isnothing(value_cache) ? get_value(M, o, p) : view(value_cache, (start + 1):(start + len))
        _add_gradient!(M, X, o, r, p; value_cache = Fi, jacobian_cache = jb)
        start += len
    end
    return X
end
# Gradient for a single summand from above, that is a single (robustified) block
function _add_gradient!(
        M, X, vgf::AbstractFirstOrderVectorFunction, r::AbstractRobustifierFunction, p;
        value_cache = get_value(M, vgf, p), jacobian_cache = nothing
    )
    # get gradients for every component
    len = length(vgf)

    # compute robustifier derivative
    (_, b, _) = get_robustifier_values(r, sum(abs2, value_cache))
    if isnothing(jacobian_cache)
        Y = allocate(M, X)
        for j in 1:len
            get_gradient!(M, Y, vgf, p, j) # gradient of f_{i,j}
            X .+= (b * value_cache[j]) .* Y
        end
    else
        Jc = jacobian_cache' * value_cache
        Jc .*= b
        add_vector!(M, X, p, Jc, vgf.jacobian_type.basis)
    end
    return X
end
# Gradient for a single summand from above, that is a single (robustified) block where the
# robustifier is applied to every component / coordinate
function _add_gradient!(
        M, X, vgf::AbstractFirstOrderVectorFunction, cr::ComponentwiseRobustifierFunction, p;
        value_cache = get_value(M, vgf, p), jacobian_cache = nothing,
    )
    # get gradients for every component
    len = length(vgf)
    r = cr.robustifier
    Y = copy(M, p, X)
    for j in 1:len
        get_gradient!(M, Y, vgf, p, j) # gradient of f_{i,j}
        (_, b, _) = get_robustifier_values(r, abs(value_cache[j])^2)
        # compute robustifier derivative
        X .+= (b * value_cache[j]) .* Y
    end
    return X
end
function Base.show(io::IO, nlso::ManifoldNonlinearLeastSquaresObjective)
    print(io, "ManifoldNonlinearLeastSquaresObjective(")
    print(io, nlso.objective, ", ", nlso.robustifier, ", ", nlso.value_cache)
    return print(io, ")")
end
function status_summary(nlso::ManifoldNonlinearLeastSquaresObjective; context::Symbol = :default)
    (context === :short) && (return repr(nlso))
    n = length(nlso.objective)
    return ("A nonlinear least squares objective with $(n) vectorial block$(n > 1 ? "s" : "")")
end

# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, v, nlso::ManifoldNonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``F(p) ∈ ℝ^n``, ``n = $(_tex(:sum, "i=1", "m")) n_i``.
In other words this is the concatenation of the residual vectors ``F_i(p)``, ``i=1,…,m``
of the components of the [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso`
at the current point ``p`` on `M`.

This can be computed in-place of `v`.

Note that even in the presence of [`RobustifierFunction`](@ref)s, these are not applied here,
this function computes the “pure” residuals.
"""

@doc "$(_doc_get_residuals_nlso)"
function get_residuals(
        M::AbstractManifold, o::AbstractManifoldObjective, p; kwargs...
    )
    v = zeros(residuals_count(o))
    return get_residuals!(M, v, o, p; kwargs...)
end

@doc "$(_doc_get_residuals_nlso)"
function get_residuals!(
        M::AbstractManifold, v, nlso::ManifoldNonlinearLeastSquaresObjective, p; kwargs...,
    )
    start = 0
    for o in nlso.objective # for every block
        len = length(o)
        view_v = view(v, (start + 1):(start + len))
        get_value!(M, view_v, o, p)
        start += len
    end
    return v
end
function get_residuals!(M::AbstractManifold, v, admo::AbstractDecoratedManifoldObjective, p; kwargs...)
    return get_residuals!(M, v, get_objective(admo, false), p; kwargs...)
end

"""
    residuals_count(nlso::ManifoldNonlinearLeastSquaresObjective)

Return the total number of residuals in [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso`,
which is the sum of the single block components lengths.
"""
function residuals_count(nlso::ManifoldNonlinearLeastSquaresObjective)
    return sum(length(o) for o in nlso.objective)
end
residuals_count(admo::AbstractDecoratedManifoldObjective) = residuals_count(get_objective(admo, false))

#
#
# ---
@doc """
    ManifoldProximalMapObjective{TC, TP, V} <: AbstractManifoldCostObjective{TC}

Specify a problem for solvers based on the evaluation of proximal maps,
which represents proximal maps ``$(_tex(:prox))_{λf_i}`` for summands ``f = f_1 + f_2+ … + f_N`` of the cost function ``f``.

# Fields

* `cost`: a function ``f:$(_math(:Manifold))→ℝ`` to
  minimize
* `proximal_maps!`: proximal maps ``$(_tex(:prox))_{λf_i}:$(_math(:Manifold)) → $(_math(:Manifold))``
  as functions `(M, λ, p) -> q` or in-place `(M, q, λ, p)`.
* `number_of_proxes`: number of proximal maps per function,
  to specify when one of the maps is a combined one such that the proximal maps
  functions return more than one entry per function, you have to adapt this value.
  If not specified, it is set to one prox per function.

# Constructor

    ManifoldProximalMapObjective( f, proxes_f::Union{Tuple,AbstractVector}, number_of_proxes=ones(length(proxes_f)) )

Generate a proximal problem with a tuple or vector of functions, where by default every function computes a single prox
of one component of ``f``.

    ManifoldProximalMapObjective(f, prox_f)

Generate a proximal objective for ``f`` and its proximal map ``$(_tex(:prox))_{λf}``.

## Keyword Arguments

$(_kwargs(:evaluation))
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.

# See also

[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ManifoldProximalMapObjective{TC, TP, V} <: AbstractManifoldCostObjective{TC}
    cost::TC
    proximal_maps!::TP
    number_of_proxes::V
    function ManifoldProximalMapObjective(
            f, proxes_f::Union{Tuple, AbstractVector};
            evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
        )
        np = ones(length(proxes_f))
        f_ = maybe_wrap_function(f, p; result = :Number)
        proxes_f_ = [ maybe_wrap_function(pf, p, evaluation; result = :Point, point_index = 2) for pf in proxes_f]
        return new{typeof(f_), typeof(proxes_f_), typeof(np)}(
            f_, proxes_f_, np
        )
    end
    function ManifoldProximalMapObjective(
            f::F, proxes_f::Union{Tuple, AbstractVector}, nOP::Vector{<:Integer};
            evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
        ) where {F}
        return if length(nOP) != length(proxes_f)
            throw(
                ErrorException(
                    "The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxes_f))).",
                ),
            )
        else
            f_ = maybe_wrap_function(f, p; result = :Number)
            proxes_f_ = [ maybe_wrap_function(pf, p, evaluation; result = :Point, point_index = 2) for pf in proxes_f]
            return new{typeof(f_), typeof(proxes_f_), typeof(nOP)}(f_, proxes_f_, nOP)
        end
    end
    function ManifoldProximalMapObjective(
            f::F, prox_f::PF;
            evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
        ) where {F, PF}
        # Single functions each
        i = 1
        f_ = maybe_wrap_function(f, p; result = :Number)
        prox_f_ = maybe_wrap_function(prox_f, p, evaluation; result = :Point, point_index = 2)
        return new{typeof(f_), typeof(prox_f_), typeof(i)}(f_, prox_f_, i)
    end
end
function _check_prox_number(pf::Union{Tuple, Vector}, i)
    n = length(pf)
    (i > n) && throw(ErrorException("the $(i)th entry does not exists, only $n available."))
    return true
end

_doc_get_proximal_map = """
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p)
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p, i)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p, i)

Evaluate the (`i`th) proximal map of the [`ManifoldProximalMapObjective`](@ref) `mpo` at
the point `p` of `M` with parameter ``λ>0``.
"""

@doc "$(_doc_get_proximal_map)"
get_proximal_map(::AbstractManifold, ::ManifoldProximalMapObjective, ::Any...)

function get_proximal_map(
        M::AbstractManifold,
        mpo::ManifoldProximalMapObjective{F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    _check_prox_number(mpo.proximal_maps!, i)
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps![i](M, q, λ, p)
    return q
end
@doc "$(_doc_get_proximal_map)"
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    _check_prox_number(mpo.proximal_maps!, i)
    mpo.proximal_maps![i](M, q, λ, p)
    return q
end

function get_proximal_map(
        M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p
    )
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!(M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p
    )
    return mpo.proximal_maps!(M, q, λ, p)
end
function status_summary(mpo::ManifoldProximalMapObjective; context::Symbol = :default)
    (context === :short) && (return repr(mpo))
    return "A proximal map objective for a cost with $(mpo.number_of_proxes) proximal maps"
end
function Base.show(io::IO, mpo::ManifoldProximalMapObjective)
    print(io, "ManifoldProximalMapObjective(", mpo.cost, ", ", mpo.proximal_maps!, ", ")
    print(io, mpo.number_of_proxes)
    return print(io, ")")
end


@doc """
    ManifoldProximalGradientObjective{TC, TG, TGG, TP} <: AbstractManifoldCostObjective{TC}

Model an objective of the form

```math
f(p) = g(p) + h(p), $(_tex(:qquad)) p ∈ $(_math(:Manifold)),
```

where ``g: $(_math(:Manifold)) → $(_tex(:eR))`` is a differentiable function
and ``h: $(_math(:Manifold)) → $(_tex(:eR))`` is a (possibly) lower semicontinuous and proper function.

This objective provides the total cost ``f``, its smooth component ``g``,
as well as ``$(_tex(:grad)) g`` and ``$(_tex(:prox))_{λ h}``.

# Fields

* `cost`: the overall cost ``f = g + h``
* `cost_smooth`: the smooth cost component ``g``
* `gradient_g!`: the gradient ``$(_tex(:grad)) g``
* `proximal_map_h!`: the proximal map ``$(_tex(:prox))_{λ h}``

# Constructor
    ManifoldProximalGradientObjective(f, g, grad_g, prox_h = missing)

Generate the proximal gradient objective given the total cost ``f = g + h``, smooth cost ``g``, the gradient of the smooth component ``$(_tex(:grad)) g``, and the proximal map of the nonsmooth component ``$(_tex(:prox))_{λ h}``.

## Keyword Arguments

$(_kwargs(:evaluation))
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.
"""
struct ManifoldProximalGradientObjective{TC, TG, TGG, TP} <: AbstractManifoldCostObjective{TC}
    cost::TC # f = g + h
    cost_smooth::TG # smooth part
    gradient_g!::TGG
    proximal_map_h!::TP
    function ManifoldProximalGradientObjective(
            f::TC, g::TG, grad_g::TGG, prox_h::TP = missing;
            evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
        ) where {TC, TG, TGG, TP}
        f_ = maybe_wrap_function(f, p; result = :Number)
        g_ = maybe_wrap_function(g, p; result = :Number)
        grad_g_ = maybe_wrap_function(grad_g, p, evaluation; result = :TangentVector)
        prox_h_ = ismissing(prox_h) ? missing : maybe_wrap_function(prox_h, p, evaluation; result = :Point, point_index = 2)
        return new{typeof(f_), typeof(g_), typeof(grad_g_), typeof(prox_h_)}(f_, g_, grad_g_, prox_h_)
    end
end

"""
    get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, mgo::ManifoldProximalGradientObjective, p)

Evaluate the gradient of the smooth part of a [`ManifoldProximalGradientObjective`](@ref) `mgo` at `p`.
"""
function get_gradient(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, p)
    X = zero_vector(M, p)
    return get_gradient!(M, X, mpgo, p)
end

function get_gradient!(M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective, p)
    return mpgo.gradient_g!(M, X, p)
end

function Base.show(io::IO, mpgo::ManifoldProximalGradientObjective{E}) where {E}
    print(io, "ManifoldProximalGradientObjective(", mpgo.cost, ", ", mpgo.cost_smooth, ", ")
    print(io, mpgo.gradient_g!, ", ", mpgo.proximal_map_h!)
    return print(io, ")")
end

function status_summary(mpgo::ManifoldProximalGradientObjective; context::Symbol = :default)
    (context === :short) && return repr(mpgo)
    s = "A proximal gradient objective `f = g + h`, where `g` is smooth and `h` is possibly nonsmooth."
    (context === :inline) && (return s)
    return """
    $s

    # Components
    * `f`:          $(mpgo.cost)
    * `g`:          $(mpgo.cost_smooth)
    * `gradient_g`: $(mpgo.gradient_g!)
    * `prox_h`:     $(mpgo.proximal_map_h!)"""
end
"""
    get_cost_smooth(M::AbstractManifold, objective, p)

Helper function to extract the smooth part `g` of a proximal gradient objective at the point `p`.
"""
function get_cost_smooth(
        M::AbstractManifold, objective::ManifoldProximalGradientObjective, p
    )
    return objective.cost_smooth(M, p)
end

@doc """
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalGradientObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalGradientObjective, λ, p)

Evaluate the proximal map of the nonsmooth component ``h`` of the [`ManifoldProximalGradientObjective`](@ref) `mpo`
at the point `p` on `M` with parameter ``λ>0``.
"""
function get_proximal_map(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, λ, p)
    q = copy(M, p)
    return get_proximal_map!(M, q, mpgo, λ, p)
end

function get_proximal_map!(
        M::AbstractManifold, q, mpgo::ManifoldProximalGradientObjective, λ, p
    )
    return mpgo.proximal_map_h!(M, q, λ, p)
end


#
#
# ---
@doc """
    ManifoldStochasticGradientObjective{C, G} <: AbstractManifoldFirstOrderObjective{C, G}

A stochastic gradient objective consists of

* a(n optional) cost function ``f(p) = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "n")) f_i(p)``
* an array of gradients, ``$(_tex(:grad)) f_i(p), i=1,…,n`` which can be given in two forms
  * as one single function ``($(_math(:Manifold)), p) ↦ (X_1,…,X_n) ∈ ($(_math(:TangentSpace)))^n``
  * as a vector of functions ``$(_tex(:bigl))( ($(_math(:Manifold)), p) ↦ X_1, …, ($(_math(:Manifold)), p) ↦ X_n$(_tex(:bigr)))``.

Where both variants are functions `(M, X, p) -> X`, where `X` is the vector of `X1,...,Xn` and `(M, X1, p) -> X1, ..., (M, Xn, p) -> Xn`,
respectively.

# Constructors

    ManifoldStochasticGradientObjective(grad_f::Function; cost=missing)
    ManifoldStochasticGradientObjective(grad_f::AbstractVector{<:Function}; cost=missing)

Create a Stochastic gradient problem with the gradient either as one
function (returning an array of tangent vectors) or a vector of functions (each returning one tangent vector).

The optional cost can also be given as either a single function (returning a number)
or a vector of functions, each returning a value.

## Keyword Arguments

* `cost = missing` provide the cost e.g. when it is used in a stopping criterion.
$(_kwargs(:evaluation))
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.

# Used with
[`stochastic_gradient_descent`](@ref)

Note that this can also be used with a [`gradient_descent`](@ref), since the (complete) gradient
is just the sums of the single gradients.
"""
struct ManifoldStochasticGradientObjective{C, G} <: AbstractManifoldFirstOrderObjective{C, G}
    cost::C
    gradient!::G
end
function ManifoldStochasticGradientObjective(
        grad_f::G; cost::C = missing, evaluation::E = AllocatingEvaluation(), p = missing,
    ) where {G, C, E <: AbstractEvaluationType}
    if G <: AbstractVector
        grad_f_ = [ maybe_wrap_function(gf, p, evaluation; result = :TangentVector) for gf in grad_f]
    else
        grad_f_ = maybe_wrap_function(grad_f, p, evaluation; result = :TangentVectors)
    end
    if ismissing(cost)
        cost_ = cost
    elseif C <: AbstractVector
        cost_ = [maybe_wrap_function(c, p; result = :Number) for c in cost ]
    else
        cost_ = maybe_wrap_function(cost, p; result = :Number)
    end
    return ManifoldStochasticGradientObjective{typeof(cost_), typeof(grad_f_)}(cost_, grad_f_)
end
function get_cost(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{C}, p
    ) where {C <: AbstractVector}
    return sum(f(M, p) for f in sgo.cost)
end

@doc """
    get_cost(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, i)

Evaluate the `i`th summand of the cost.

If you use a single function for the stochastic cost, then only the index `i=1` is available
to evaluate the whole cost.
"""
function get_cost(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{C}, p, i
    ) where {C <: AbstractVector}
    return sgo.cost[i](M, p)
end
function get_cost(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, i)
    (i == 1) && return sgo.cost(M, p)
    return error("The cost is implemented as a single function and can not be accessed element wise at $i since the index is larger than 1.")
end

function get_gradients end
function get_gradients(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{C, <:AbstractVector}, p,
    ) where {C}
    X = [zero_vector(M, p) for _ in sgo.gradient!]
    get_gradients!(M, X, sgo, p)
    return X
end
_doc_get_gradients = """
    get_gradients(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p)
    get_gradients!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p)

Evaluate all summands gradients ``$(_math(:Sequence, "$(_tex(:grad))f", "i", "1", "n"))`` at `p` and return them as a vector.

This can be done in-place of a vector of tangent vectors `X`.
"""

@doc "$(_doc_get_gradients)"
get_gradients(M::AbstractManifold, admo::ManifoldStochasticGradientObjective, p)


# Single allocating function we wrapped ourselves – call it from intern storage, since
# we have no chance to allocate this otherwise
function get_gradients(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{C, <:InplaceManifoldFunction}, p,
    ) where {C}
    return sgo.gradient!.f(M, p)
end
function get_gradients(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradients(M, get_objective(admo, false), p)
end
# For a single function, since it is in-place, we have no chance to allocate the right amount of tangent vectors in X? Ah we can use get_cost
@doc "$(_doc_get_gradients)"
function get_gradients!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p)
    sgo.gradient!(M, X, p)
    return X
end
function get_gradients!(
        M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective{C, <:AbstractVector}, p,
    ) where {C}
    for (Xi, grad_i) in zip(X, sgo.gradient!)
        grad_i(M, Xi, p)
    end
    return X
end
function get_gradients!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradients!(M, X, get_objective(admo, false), p)
end


@doc """
    get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, k)
    get_gradient!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p, k)

Evaluate one of the summands gradients ``$(_tex(:grad))f_k``, ``k ∈ $(_tex(:set, "1,…,n"))``, at `p` (in place of `X`).

If you use a single function for the stochastic gradient, that works in-place, then [`get_gradient`](@ref) is not available,
since the length (or number of elements of the gradient required for allocation) can not be determined.
"""
function get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, k)
    X = zero_vector(M, p)
    return get_gradient!(M, X, sgo, p, k)
end
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, k)
    return get_gradient(M, get_objective(admo, false), p, k)
end
function get_gradient!(
        M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective{C, <:AbstractVector}, p, k
    ) where {C}
    return sgo.gradient![k](M, X, p)
end
# We have an allocating function and wrapped it, then we can still access that; for a purely single inplace, we have no chance
function get_gradient!(
        M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective{C, <:InplaceManifoldFunction}, p, k
    ) where {C}
    return copyto!(M, X, p, sgo.gradient!.f(M, p)[k])
end
function get_gradient!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p, k
    )
    return get_gradient!(M, X, get_objective(admo, false), p, k)
end

# Pass down from problem

@doc """
    get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p)

Evaluate the complete gradient ``$(_tex(:grad)) f = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "n")) $(_tex(:grad)) f_i(p)`` at `p` (in place of `X`).

If you use a single function for the stochastic gradient, that works in-place, then [`get_gradient`](@ref) is not available,
since the length (or number of elements of the gradient required for allocation) can not be determined.
"""
function get_gradient(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{C, <:Function}, p
    ) where {C}
    # even if the function is in-place, allocation of the full vector of tangent vectors still required
    return sum(get_gradients(M, sgo, p))
end
function get_gradient(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{C, <:AbstractVector}, p,
    ) where {C}
    X = zero_vector(M, p)
    get_gradient!(M, X, sgo, p)
    return X
end
function get_gradient!(
        M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective{C, <:AbstractVector}, p,
    ) where {C}
    zero_vector!(M, X, p)
    Y = copy(M, p, X)
    for grad_i in sgo.gradient!
        grad_i(M, Y, p)
        X .+= Y
    end
    return X
end
# Single function - since we can not allocate all at once, we have to cheat a bit.
function get_gradient!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p)
    return copyto!(M, X, p, sum(get_gradients(M, sgo, p)))
end

function Base.show(io::IO, msgo::ManifoldStochasticGradientObjective)
    print(io, "ManifoldStochasticGradientObjective(", msgo.gradient!, "; ")
    if !ismissing(msgo.cost)
        print(io, "cost = ", msgo.cost, ", ")
    end
    return print(io, ")")
end
function status_summary(msgo::ManifoldStochasticGradientObjective; context::Symbol = :default)
    (context === :short) && return repr(msgo)
    cs = ismissing(msgo.cost) ? "" : "including the cost function"
    (context === :inline) && return "A stochastic gradient objective $cs."
    ics = ismissing(msgo.cost) ? "" : "\n* cost:          $(_MANOPT_INDENT)$(msgo.cost)"
    return """
    A stochastic gradient objective

    ## Functions$(ics)
    * gradient:$(_MANOPT_INDENT)$(msgo.gradient!)"""
end

#
#
# ---
@doc """
    ManifoldSubgradientObjective{C,S} <: AbstractManifoldCostObjective{C}

A structure to store information about an objective for a subgradient based optimization problem

# Fields

* `cost`:        the function ``f`` to be minimized
* `subgradient!`: a function returning a subgradient ``∂f`` of ``f``

# Constructor

    ManifoldSubgradientObjective(f, ∂f; evaluation = AllocatingEvaluation(), p = missing)

Generate the [`ManifoldSubgradientObjective`](@ref) for a subgradient objective, consisting
of a (cost) function `f(M, p)` and a function `∂f(M, p)` that returns a not necessarily
deterministic element from the subdifferential at `p` on a manifold `M`.

## Keyword Arguments

$(_kwargs(:evaluation))
* `p = missing` provide a point to automatically ensure the functions of the objective “act” on mutating variables.
"""
struct ManifoldSubgradientObjective{C, S} <: AbstractManifoldCostObjective{C}
    cost::C
    subgradient!::S
    function ManifoldSubgradientObjective(
            cost::C, subgrad::S;
            evaluation::AbstractEvaluationType = AllocatingEvaluation(), p = missing
        ) where {C, S}
        cost_ = maybe_wrap_function(cost, p; result = :Number)
        subgrad_ = maybe_wrap_function(subgrad, p, evaluation; result = :TangentVector)
        return new{typeof(cost_), typeof(subgrad_)}(cost_, subgrad_)
    end
end

"""
    X = get_subgradient(M::AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    get_subgradient!(M::AbstractManifold, X, sgo::ManifoldSubgradientObjective, p)

Evaluate the (sub)gradient of a [`ManifoldSubgradientObjective`](@ref) `sgo`
at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(M::AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    X = zero_vector(M, p)
    return sgo.subgradient!(M, X, p)
end
function get_subgradient!(
        M::AbstractManifold, X, sgo::ManifoldSubgradientObjective, p
    )
    return sgo.subgradient!(M, X, p)
end

@doc """
    get_subgradient_function(objective::ManifoldSubgradientObjective, recursive=false; evaluation = AllocatingEvaluation())

Return the function to evaluate (just) the subgradient ``$(_tex(:subgrad)) f(p)``.
It is of the form `(M, X, p) -> X` to work in-place of `X`,
where either the subgradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.
"""
function get_subgradient_function(
        msgo::ManifoldSubgradientObjective, recursive = false;
        evaluation::AbstractEvaluationType = AllocatingEvaluation()
    )
    if evaluation isa AllocatingEvaluation
        (msgo.subgradient! isa InplaceManifoldFunction) && return msgo.subgradient!.f
        return (M, p) -> msgo.subgradient!(M, zero_vector(M, p), p)
    else
        return msgo.subgradient!
    end
end

function set_parameter!(msgo::ManifoldSubgradientObjective, ::Val{:SubGradient}, args...)
    set_parameter!(get_subgradient_function(msgo, true; evaluation = InplaceEvaluation()), args...)
    return msgo
end

function Base.show(io::IO, objective::ManifoldSubgradientObjective)
    return print(io, "ManifoldSubgradientObjective(", objective.cost, ", ", objective.subgradient!, ")")
end

function status_summary(objective::ManifoldSubgradientObjective; context::Symbol = :default)
    (context === :short) && return repr(objective)
    s = "A subgradient objective "
    (context === :inline) && (return s)
    return """
    $s

    ## Components
    * `f`:  $(objective.cost)
    * `∂f`: $(objective.subgradient!)"""
end

@doc """
    ScaledManifoldObjective{O2, O1<:AbstractManifoldObjective,F} <: AbstractDecoratedManifoldObjective{O2}

Declare an objective to be defined as a scaled version of an existing objective.

This rescales all involved functions.

For now the functions rescaled are

* the cost
* the gradient
* the Hessian

# Fields

* `objective`: the objective that is scaled
* `scale=1`: the scaling applied

# Constructors

    ScaledManifoldObjective(objective, scale::Real=1)

Generate a scaled manifold objective based on `objective`, where `scale` is `1` by default.
The unary minus and the multiplication from the left with a scalar are also overloaded.

    - objective

The single-parameter minus, that is `scale=-1`, is overloaded to have a short notation turning
a maximization problem into a minimization one, which fits the framework provided within `Manopt.jl`.

    scale * objective

Equivalent to the first constructor, but might be nicer to write in a few places.
"""
struct ScaledManifoldObjective{O2, O1 <: AbstractManifoldObjective, F} <: AbstractDecoratedManifoldObjective{O2}
    objective::O1
    scale::F
end
function ScaledManifoldObjective(
        objective::O, scale::F = 1
    ) where {O <: AbstractManifoldObjective, F <: Real}
    return ScaledManifoldObjective{O, O, F}(objective, scale)
end
function ScaledManifoldObjective(
        objective::O1, scale::F = 1
    ) where {
        F <: Real, O2 <: AbstractManifoldObjective, O1 <: AbstractDecoratedManifoldObjective{O2},
    }
    return ScaledManifoldObjective{O2, O1, F}(objective, scale)
end
Base.:-(objective::AbstractManifoldObjective) = ScaledManifoldObjective(objective, -1)
function Base.:*(scale::Real, objective::AbstractManifoldObjective)
    return ScaledManifoldObjective(objective, scale)
end

@doc """
    get_cost(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)

Evaluate the scaled objective ``s*f(p)``.
"""
function get_cost(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)
    return scaled_objective.scale * get_cost(M, scaled_objective.objective, p)
end

function get_cost_function(scaled_objective::ScaledManifoldObjective, recursive::Bool = false)
    recursive && (return get_cost_function(scaled_objective.objective, recursive))
    return (M, p) -> get_cost(M, scaled_objective, p)
end
@doc """
    get_gradient(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)
    get_gradient!(M::AbstractManifold, X, scaled_objective::ScaledManifoldObjective, p)

Evaluate the scaled gradient ``s*$(_tex(:grad))f(p)``.
"""
function get_gradient(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p)
    return scaled_objective.scale * get_gradient(M, scaled_objective.objective, p)
end
function get_gradient!(M::AbstractManifold, X, scaled_objective::ScaledManifoldObjective, p)
    get_gradient!(M, X, scaled_objective.objective, p)
    X .= scaled_objective.scale .* X
    return X
end

function get_gradient_function(scaled_objective::ScaledManifoldObjective, recursive::Bool = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    # “unwrap scaling even”
    recursive && (return get_gradient_function(scaled_objective.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p) -> get_gradient(M, scaled_objective, p)
    else
        return (M, X, p) -> get_gradient!(M, X, scaled_objective, p)
    end
end


@doc """
    get_hessian(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p, X)
    get_hessian!(M::AbstractManifold, Y, scaled_objective::ScaledManifoldObjective, p, X)

Evaluate the scaled Hessian ``s*$(_tex(:Hess))f(p)``.
"""
function get_hessian(M::AbstractManifold, scaled_objective::ScaledManifoldObjective, p, X)
    return scaled_objective.scale * get_hessian(M, scaled_objective.objective, p, X)
end
function get_hessian!(M::AbstractManifold, Y, scaled_objective::ScaledManifoldObjective, p, X)
    get_hessian!(M, Y, scaled_objective.objective, p, X)
    Y .= scaled_objective.scale .* Y
    return Y
end
function get_hessian_function(scaled_objective::ScaledManifoldObjective, recursive::Bool = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    recursive && (return get_hessian_function(scaled_objective.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p, X) -> get_hessian(M, scaled_objective, p, X)
    else
        return (M, Y, p, X) -> get_hessian!(M, Y, scaled_objective, p, X)
    end
end
function Base.show(io::IO, scaled_objective::ScaledManifoldObjective)
    return print(
        io, "ScaledManifoldObjective($(repr(scaled_objective.objective)), $(scaled_objective.scale))",
    )
end
function status_summary(scaled_objective::ScaledManifoldObjective; context::Symbol = :default)
    # short and inline
    (context === :short) && (return "$(scaled_objective.scale) * $(status_summary(scaled_objective.objective; context = context))")
    (context === :inline) && (return "$(status_summary(scaled_objective.objective; context = context)) scaled by a factor of $(scaled_objective.scale)")
    # default
    return "A scaled version of the objective\n$(status_summary(scaled_objective.objective; context = context))\nscaled by a factor of $(scaled_objective.scale)"
end

@doc """
    SimpleManifoldCachedObjective{O<:AbstractManifoldObjective, P, T, C} <: AbstractDecoratedManifoldObjective{O}

Provide a simple cache for an [`AbstractManifoldFirstOrderObjective`](@ref) that is, this cache
stores a point `p` and a gradient ``$(_tex(:grad)) f(p)`` in `X` as well as a cost value ``f(p)`` in `c`.
It can also easily evaluate the differential based on the cached gradient.

Both `X` and `c` are accompanied by booleans to keep track of their validity.

While this does not provide a cache for the differential, it uses the cached gradient
as a help to evaluate the differential, if an up-to-date gradient is available.
It otherwise does call the original differential.

This simple cache does not take into account, that some first order objectives have a
common function for cost & grad. It only caches the function that is actually called.

# Constructors

    SimpleManifoldCachedObjective(M::AbstractManifold, obj::AbstractManifoldFirstOrderObjective; kwargs...)

## Keyword arguments

* `p=`$(Manopt._link(:rand)): a point on the manifold to initialize the cache with
* `X=get_gradient(M, obj, p)` or `zero_vector(M,p)`: a tangent vector to store the gradient in,
  see also `initialized=`
* `c=`[`get_cost`](@ref)`(M, obj, p)` or `0.0`: a value to store the cost in, see also `initialized=`
* `initialized=true`: whether to initialize the cached `X` and `c` or not.

where both for `p` and `X` copies are generated before they are stored.

    SimpleManifoldCachedObjective(obj::AbstractManifoldFirstOrderObjective, p, X, c; initialized = false)

Similar as above but initializing all fields directly and without copies and `initialized` indicated whether
the three values correspond to an evaluation from `obj`.
"""
mutable struct SimpleManifoldCachedObjective{
        O <: AbstractManifoldObjective, P, T, C,
    } <: AbstractDecoratedManifoldObjective{O}
    objective::O
    p::P # a point
    X::T # a vector
    X_valid::Bool
    c::C # a value for the cost
    c_valid::Bool
end

function SimpleManifoldCachedObjective(
        M::AbstractManifold, obj::O;
        initialized = true, p = rand(M),
        X = initialized ? get_gradient(M, obj, p) : zero_vector(M, p),
        c = initialized ? get_cost(M, obj, p) : 0.0,
    ) where {O <: AbstractManifoldObjective}
    q = copy(M, p)
    return SimpleManifoldCachedObjective(obj, q, X, c; initialized = initialized)
end

function SimpleManifoldCachedObjective(
        obj::O, p, X, c; initialized::Bool = false
    ) where {O <: AbstractManifoldObjective}
    return SimpleManifoldCachedObjective{O, typeof(p), typeof(X), typeof(c)}(
        obj, p, X, initialized, c, initialized
    )
end

# Default implementations
function get_cost(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.c_valid
        # else evaluate cost, invalidate grad if p changed
        sco.c = get_cost(M, sco.objective, p)
        # for switched points, invalidate X
        scop_neq_p && (sco.X_valid = false)
        copyto!(M, sco.p, p)
        sco.c_valid = true
    end
    return sco.c
end

function get_cost_and_gradient(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid || !sco.c_valid
        sco.c, X = get_cost_and_gradient(M, sco.objective, p)
        # Update point
        copyto!(M, sco.p, p)
        sco.c_valid = true
        copyto!(M, sco.X, X)
        sco.X_valid = true
    else
        X = copy(M, p, sco.X)
    end
    return (sco.c, X)
end
function get_cost_and_gradient!(
        M::AbstractManifold, X, sco::SimpleManifoldCachedObjective, p
    )
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid || !sco.c_valid
        sco.c, _ = get_cost_and_gradient!(M, X, sco.objective, p)
        copyto!(M, sco.p, p)
        sco.c_valid = true
        copyto!(M, sco.X, p, X)
        sco.X_valid = true
    else
        copyto!(M, X, p, sco.X)
    end
    return (sco.c, X)
end

function get_cost_function(sco::SimpleManifoldCachedObjective, recursive = false)
    recursive && return get_cost_function(sco.objective, recursive)
    return (M, p) -> get_cost(M, sco, p)
end

function get_differential(
        M::AbstractManifold, sco::SimpleManifoldCachedObjective, p, X; kwargs...
    )
    scop_neq_p = sco.p != p
    # Gradient outdated -> just call differenital of the inner objective
    if scop_neq_p || !sco.X_valid
        return get_differential(M, sco.objective, p, X; kwargs...)
    end
    # otherwise use the up to date gradient and inner
    return real(inner(M, p, sco.X, X))
end
function get_differential_function(sco::SimpleManifoldCachedObjective, recursive = false)
    recursive && (return get_differential_function(sco.objective, recursive))
    return (M, p, X; kwargs...) -> get_differential(M, sco, p, X; kwargs...)
end

function get_gradient(M::AbstractManifold, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        X = get_gradient(M, sco.objective, p)
        # for switched points, invalidate c
        copyto!(M, sco.p, p)
        scop_neq_p && (sco.c_valid = false)
        copyto!(M, sco.X, X)
        sco.X_valid = true
    else
        X = copy(M, p, sco.X)
    end
    return X
end
function get_gradient!(M::AbstractManifold, X, sco::SimpleManifoldCachedObjective, p)
    scop_neq_p = sco.p != p
    if scop_neq_p || !sco.X_valid
        get_gradient!(M, X, sco.objective, p)
        # for switched points, invalidate c
        copyto!(M, sco.p, p)
        scop_neq_p && (sco.c_valid = false)
        copyto!(M, sco.X, X)
        sco.X_valid = true
    else
        copyto!(M, X, p, sco.X)
    end
    return X
end
function get_gradient_function(sco::SimpleManifoldCachedObjective, recursive = false; evaluation::AbstractEvaluationType = AllocatingEvaluation())
    recursive && (return get_gradient_function(sco.objective, recursive; evaluation = evaluation))
    if evaluation isa AllocatingEvaluation
        return (M, p) -> get_gradient(M, sco, p)
    else
        return (M, X, p) -> get_gradient!(M, X, sco, p)
    end
end
function Base.show(io::IO, smco::SimpleManifoldCachedObjective)
    print(io, "SimpleManifoldCachedObjective(")
    print(io, smco.objective); print(io, ", ")
    print(io, smco.p); print(io, ", ")
    print(io, smco.X); print(io, ", ")
    print(io, smco.c)
    return print(io, "; initialized = $(smco.X_valid && smco.c_valid))")
end
function show(
        io::IO, t::Tuple{<:SimpleManifoldCachedObjective, S}
    ) where {S <: AbstractManoptSolverState}
    return print(io, "$(status_summary(t[2]))\n\n$(status_summary(t[1]))")
end
function status_summary(smco::SimpleManifoldCachedObjective; context::Symbol = :default)
    (context === :short) && (return repr(smco))
    (context === :inline) && (return "A simple cache objective caching one p, X, and c for $(status_summary(smco.objective; context = context))")
    s = """
    ## Cache
    A `SimpleManifoldCachedObjective` to cache one point, one tangent vector, and real number
    for the iterate, the gradient, and the cost function, respectively.

    At the current iterate
    * the tangent vector is cached:$(_MANOPT_INDENT)$(smco.X_valid ? "Yes" : "No")
    * the cost is cached:$(_MANOPT_INDENT)$(smco.c_valid ? "Yes" : "No")
    """
    s2 = status_summary(smco.objective; context = context)
    length(s2) > 0 && (s2 = "$(s2)\n\n")
    return "$(s2)$(s)"
end

# Factory for both cache objectives
@doc """
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Symbol)

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o`
on the `AbstractManifold M` based on the symbol `cache`.

The following caches are available

* `:Simple` generates a [`SimpleManifoldCachedObjective`](@ref)
* `:LRU` generates a [`ManifoldCachedObjective`](@ref) where you should use the form
  `(:LRU, [:Cost, :Gradient])` to specify what should be cached or
  `(:LRU, [:Cost, :Gradient], 100)` to specify the cache size.
  Here this variant defaults to `(:LRU, [:Cost, :Gradient], 100)`,
  caching up to 100 cost and gradient values.[^1]

[^1]:
    This cache requires [`LRUCache.jl`](https://github.com/JuliaCollections/LRUCache.jl) to be loaded as well.
"""
function objective_cache_factory(M, o, cache::Symbol)
    (cache === :Simple) && return SimpleManifoldCachedObjective(M, o)
    (cache === :LRU) &&
        return ManifoldCachedObjective(M, o, [:Cost, :Gradient]; cache_size = 100)
    return o
end

@doc """
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Tuple{Symbol, Array, Array})
    objective_cache_factory(M::AbstractManifold, o::AbstractManifoldObjective, cache::Tuple{Symbol, Array})

Generate a cached variant of the [`AbstractManifoldObjective`](@ref) `o` on the `AbstractManifold M`, selected by `cache[1]`.

The second element `cache[2]` provides further arguments to the cache and
the optional third is passed down as keyword arguments.

For all available caches see the simpler variant with symbols.
"""
function objective_cache_factory(M, o, cache::Tuple{Symbol, <:AbstractArray, I}) where {I}
    (cache[1] === :Simple) && return SimpleManifoldCachedObjective(M, o; cache[3]...)
    if (cache[1] === :LRU)
        if (cache[3] isa Integer)
            return ManifoldCachedObjective(M, o, cache[2]; cache_size = cache[3])
        else
            return ManifoldCachedObjective(M, o, cache[2]; cache[3]...)
        end
    end
    return o
end
function objective_cache_factory(M, o, cache::Tuple{Symbol, <:AbstractArray})
    (cache[1] === :Simple) && return SimpleManifoldCachedObjective(M, o)
    (cache[1] === :LRU) && return ManifoldCachedObjective(M, o, cache[2])
    return o
end
