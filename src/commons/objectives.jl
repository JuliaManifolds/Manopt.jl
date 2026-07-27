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
        g=nothing, grad_g=nothing,
        h=nothing, grad_h=nothing;
        hess_f=nothing, hess_g=nothing, hess_h=nothing,
        equality_constraints=nothing,
        inequality_constraints=nothing,
        evaluation=AllocatingEvaluation(),
        M = nothing,
        p = isnothing(M) ? nothing : rand(M),
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
    (!isnothing(f) && isa(f, AbstractVector)) && return ComponentVectorialType()
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
        M::Union{AbstractManifold, Nothing} = nothing,
        p = isnothing(M) ? nothing : rand(M),
    )
    if !isnothing(g)
        if isa(function_type, ComponentVectorialType) || isa(g, AbstractVector)
            return length(g)
        end
    end
    if !isnothing(grad_g)
        if isa(jacobian_type, ComponentVectorialType) || isa(grad_g, AbstractVector)
            return length(grad_g)
        end
    end
    # These are more expensive, since they evaluate and hence allocate
    if !isnothing(M) && !isnothing(p)
        # For functions on vector representations, the last size is equal to length
        # on array power manifolds, this also yields the number of elements
        (!isnothing(g)) && (return _val_to_ncons(g(M, p)))
        (!isnothing(grad_g)) && (return _val_to_ncons(grad_g(M, p)))
    end
    return -1
end

function ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h;
        hess_f = nothing, hess_g = nothing, hess_h = nothing,
        equality_type::AbstractVectorialType = _vector_function_type_hint(h),
        equality_gradient_type::AbstractVectorialType = _vector_function_type_hint(grad_h),
        equality_hessian_type::AbstractVectorialType = _vector_function_type_hint(hess_h),
        inequality_type::AbstractVectorialType = _vector_function_type_hint(g),
        inequality_gradient_type::AbstractVectorialType = _vector_function_type_hint(grad_g),
        inequality_hessian_type::AbstractVectorialType = _vector_function_type_hint(hess_g),
        equality_constraints::Union{Integer, Nothing} = nothing,
        inequality_constraints::Union{Integer, Nothing} = nothing,
        M::Union{AbstractManifold, Nothing} = nothing, p = isnothing(M) ? nothing : rand(M), atol = 0,
    )
    if isnothing(hess_f)
        objective = ManifoldGradientObjective(f, grad_f; evaluation = evaluation)
    else
        objective = ManifoldHessianObjective(f, grad_f, hess_f; evaluation = evaluation)
    end
    num_eq = isnothing(equality_constraints) ? -1 : equality_constraints
    if isnothing(h) || isnothing(grad_h)
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
        if isnothing(hess_h)
            eq = VectorGradientFunction(
                h, grad_h, num_eq; evaluation = evaluation,
                function_type = equality_type, jacobian_type = equality_gradient_type,
            )
        else
            eq = VectorHessianFunction(
                h, grad_h, hess_h, num_eq; evaluation = evaluation,
                function_type = equality_type, jacobian_type = equality_gradient_type, hessian_type = equality_hessian_type,
            )
        end
    end
    num_ineq = isnothing(inequality_constraints) ? -1 : inequality_constraints
    if isnothing(g) || isnothing(grad_g)
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
        if isnothing(hess_g)
            ineq = VectorGradientFunction(
                g, grad_g, num_ineq; evaluation = evaluation,
                function_type = inequality_type, jacobian_type = inequality_gradient_type,
            )
        else
            ineq = VectorHessianFunction(
                g, grad_g, hess_g, num_ineq; evaluation = evaluation,
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
        f, grad_f; g = nothing, grad_g = nothing, h = nothing, grad_h = nothing, kwargs...
    )
    return ConstrainedManifoldObjective(f, grad_f, g, grad_g, h, grad_h; kwargs...)
end
function status_summary(cmo::ConstrainedManifoldObjective; context::Symbol = :default)
    _is_inline(context) && (return "A constrained objective based on $(status_summary(cmo.objective; context = context)) with $(length(cmo.equality_constraints)) equality and $(length(cmo.inequality_constraints)) inequality constraints.")
    s = status_summary(cmo.objective; context = context)
    return """
    A constrained objective with $(length(cmo.equality_constraints)) equality and $(cmo.inequality_constraints) inequality constraints.
    For verifications, the inequalities are checked with an absolute tolerance of `atol = $(cmo.atol)`

    ## Unconstrained Objective
    $(_in_str(s))

    ## Equality constrains
    $(_in_str(status_summary(cmo.equality_constraints; context = context)))

    ## Inequality constrains
    $(_in_str(status_summary(cmo.inequality_constraints; context = context)))"""
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
function get_gradient_function(co::ConstrainedManifoldObjective, recursive = false)
    return get_gradient_function(co.objective, recursive)
end

@doc """
    get_inequality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())

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

@doc """
    get_grad_equality_constraint(amp::AbstractManoptProblem, p, j)
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())
    get_grad_equality_constraint!(amp::AbstractManoptProblem, X, p, j)
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())

Evaluate the gradient or gradients  of the equality constraint ``($(_tex(:grad)) h(p))_j`` or ``$(_tex(:grad)) h_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""
function get_grad_equality_constraint end

function get_grad_equality_constraint(
        M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_grad_equality_constraint(M, get_objective(admo, false), args...)
end
function get_grad_equality_constraint(
        M::AbstractManifold,
        co::ConstrainedManifoldObjective,
        p,
        j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    if isnothing(co.equality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_gradient(M, co.equality_constraints, p, j, range)
end

function get_grad_equality_constraint!(
        amp::AbstractManoptProblem,
        X,
        p,
        j = :,
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
        M::AbstractManifold,
        X,
        co::ConstrainedManifoldObjective,
        p,
        j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    isnothing(co.equality_constraints) && (return X)
    return get_gradient!(M, X, co.equality_constraints, p, j, range)
end

@doc """
    get_grad_inequality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())
    get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, j=:)
    get_grad_inequality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())

Evaluate the gradient or gradients of the inequality constraint ``($(_tex(:grad)) g(p))_j`` or ``$(_tex(:grad)) g_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""
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
function get_hessian_function(co::ConstrainedManifoldObjective, recursive = false)
    return get_hessian_function(co.objective, recursive)
end

@doc """
    get_hess_equality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_hess_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())
    get_hess_equality_constraint!(amp::AbstractManoptProblem, X, p, j=:)
    get_hess_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())

Evaluate the Hessian or Hessians of the equality constraint ``($(_tex(:Hess)) h(p))_j`` or ``$(_tex(:Hess)) h_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the Hessian.
"""
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

@doc """
    get_hess_inequality_constraint(amp::AbstractManoptProblem, p, X, j=:)
    get_hess_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())
    get_hess_inequality_constraint!(amp::AbstractManoptProblem, Y, p, j=:)
    get_hess_inequality_constraint!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j=:, range=NestedPowerRepresentation())

Evaluate the Hessian or Hessians of the inequality constraint ``($(_tex(:Hess)) g(p)[X])_j`` or ``$(_tex(:Hess)) g_j(p)[X]``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the Hessian.
"""
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
function get_hess_inequality_constraint!(
        M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, args...
    )
    return get_hess_inequality_constraint!(M, Y, get_objective(admo, false), args...)
end
function get_hess_inequality_constraint!(
        M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j = :,
        range::AbstractPowerRepresentation = NestedPowerRepresentation(),
    )
    isnothing(co.inequality_constraints) && (return X)
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
    is_feasible(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, kwargs...)
    is_feasible(M::AbstractManifold, o::AbstractDecoratedManifoldObjective, p, kwargs...)

Evaluate whether a point `p` on `M` is feasible with respect to the [`ConstrainedManifoldObjective`](@ref) `cmo`.
That is for the provided inequality constraints ``g: $(_math(:Manifold)) → ℝ^m`` and equality constraints ``h: $(_math(:Manifold)) → ℝ^m``
from within `cmo`, the point ``p ∈ $(_math(:Manifold))`` is feasible if

```math
g_i(p) ≤ 0, $(_tex(:text, " for all ")) i=1,…,m$(_tex(:quad))\text{ and }$(_tex(:quad)) h_j(p) = 0, \text{ for all } j=1,…,n.
```

# Keyword arguments
* `check_point::Bool=true`: whether to also verify that ``p∈$(_math(:Manifold))` holds, using [`is_point`](@extref ManifoldsBase :jl:method:`ManifoldsBase.is_point-Tuple{AbstractManifold, Any, Bool}`)
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
        g = get_inequality_constraints(M, cmo, p),
        h = get_equality_constraints(M, cmo, p),
    )

Generate a message about the feasibiliy of `p` with respect to the [`ConstrainedManifoldObjective`](@ref).
You can also provide the evaluated vectors for the values of `g` and `h` as keyword arguments,
in case you had them evaluated before.
"""
function get_feasibility_status(
        M, cmo, p;
        g = get_inequality_constraints(M, cmo, p), h = get_equality_constraints(M, cmo, p),
    )
    g_violated = sum(g .> 0)
    h_violated = sum(h .!= 0)
    return """
    The point $p on $M is not feasible for the provided constants.

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
"""
    ManifoldConstrainedSetObjective{E, MO, PF, IF} <: AbstractManifoldObjective{E}

Model a constrained objective restricted to a set

```math
$(_tex(:argmin))_{p ∈ $(_tex(:Cal, "C"))} f(p)
```

where ``$(_tex(:Cal, "C")) ⊂ $(_math(:Manifold))`` is a convex closed subset.

# Fields

* `objective::AbstractManifoldObjective` the (unconstrained) objective, which
  contains ``f`` and for example its gradient ``$(_tex(:grad)) f``.
* `project!!::PF` a projection function ``$(_tex(:proj))_{$(_tex(:Cal, "C"))}: $(_math(:Manifold)) → $(_tex(:Cal, "C"))`` that projects onto the set ``$(_tex(:Cal, "C"))``.
* `indicator::IF` the indicator function ``ι_{$(_tex(:Cal, "C"))}(p) = $(_tex(:cases, "0 &" * _tex(:text, " for ") * "p∈" * _tex(:Cal, "C"), "∞ &" * _tex(:text, " else.")))``

# Constructor

    ManifoldConstrainedSetObjective(f, grad_f, project!!; kwargs...)

Generate the constrained objective for a given function `f` its gradient `grad_f` and a projection `project!!` ``$(_tex(:proj))_{$(_tex(:Cal, "C"))}``.

## Keyword arguments

* `indicator=nothing`: the indicator function ``ι_{$(_tex(:Cal, "C"))}(p)``. If not provided a test, whether the projection yields the same point is performed.
  For the [`InplaceEvaluation`](@ref) this required one allocation.
"""
struct ManifoldConstrainedSetObjective{MO <: AbstractManifoldObjective, PF, IF} <: AbstractManifoldObjective
    objective::MO
    project!!::PF
    indicator::IF
end

function ManifoldConstrainedSetObjective(
        f, grad_f, project!!::PF; indicator = nothing
    ) where {PF}
    obj = ManifoldGradientObjective(f, grad_f)
    if isnothing(indicator)
        ind = function (M, p)
            q = rand(M)
            project!!(M, q, p)
            return distance(M, p, q) ≈ 0 ? 0 : Inf
        end
        return ManifoldConstrainedSetObjective{typeof(obj), typeof(project!!), typeof(ind)}(
            obj, project!!, ind
        )
    end
    return ManifoldConstrainedSetObjective{typeof(obj), typeof(project!!), typeof(indicator)}(
        obj, project!!, indicator
    )
end

function get_cost(M::AbstractManifold, cso::ManifoldConstrainedSetObjective, p)
    return get_cost(M, cso.objective, p)
end
function get_cost_function(cso::ManifoldConstrainedSetObjective, recursive = false)
    return get_cost_function(cso.objective, recursive)
end
function get_gradient_function(cso::ManifoldConstrainedSetObjective, recursive = false)
    return get_gradient_function(cso.objective, recursive)
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
    cso.project!!(M, q, p)
    return q
end
@doc "$(_doc_get_projected_point)"
get_projected_point!(M::AbstractManifold, q, cso::ManifoldConstrainedSetObjective, p)
function get_projected_point!(M::AbstractManifold, q, cso::ManifoldConstrainedSetObjective, p)
    cso.project!!(M, q, p)
    return q
end

#
#
# ---
@doc """
    ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}

specify an [`AbstractManifoldObjective`](@ref) that does only have information about
the cost function ``f:  $(_math(:Manifold)) → ℝ`` implemented as a function `(M, p) -> c`
to compute the cost value `c` at `p` on the manifold `M`.

* `cost`: a function ``f: $(_math(:Manifold)) → ℝ`` to minimize

# Constructors

    ManifoldCostObjective(f::F)

Generate a problem. While this Problem does not have any allocating functions,

## See also
[`NelderMead`](@ref), [`particle_swarm`](@ref)
"""
struct ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}
    cost::F
end
function show(io::IO, mco::ManifoldCostObjective{F}) where {F}
    return print(io, "ManifoldCostObjective(mco.cost)")
end
function status_summary(::ManifoldCostObjective{F}; context::Symbol = :default) where {F}
    return "A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
end

#
#
# ---
@doc """
    ManifoldFirstOrderObjective{E<:AbstractEvaluationType, F} <: AbstractManifoldFirstOrderObjective{E, F}

specify an objective containing a cost and its gradient or differential,
where the [`AbstractEvaluationType`](@ref) `E` indicates the type of evaluation for a gradient.

# Fields

* `functions::F`: a function or a tuple of functions containing the cost and first order information.

Currently the following cases are covered, sorted by their popularity

1. a single function `fg`, i.e. a function, represents a combined
    function `(M, X, p) -> (c, X)` that computes the cost `c=cost(M,p)` and gradient `X=grad_f(M, X, p)`;
2. a single function `fdf`, i.e. a function, represents a combined function
    `(M, d, p) -> (c, d)` that computes the cost `c=cost(M,p)` and differential `d=diff_f(M, d, p)`;
3. pairs of single functions `(f, g)`, `(f, df)` of a cost function `f` and either its
    gradient `g` or its differential `d`, respectively
4. The function `(fg, d)` and `(fdf, g)`  from 1 and 2, respectively joined by
    the other missing third information, the differential for the first or the gradient for the second
5. a tuple `(f, g, d)` of three functions, computing cost, `f`, gradient `g`,
    and `differential `d` separately
6. a `(f, gd)` of a cost function and a combined function `(X, d) = gd(M, (X,d), p)`
    to compute gradient and differential together

For all cases where a gradient and/or a differential is present are considered to work in-place,
see [`AllocatingManifoldFunction`](@ref) for alternatives.

The cases of a common `fg` function for cost and gradient and the tuple `(f,g)` are the most common one.
They can also be addressed by their alternate constructors
[`ManifoldCostGradientObjective`](@ref)`(fg)` and [`ManifoldGradientObjective`](@ref)`(f,g)`, respectively.

# Constructors
    ManifoldFirstOrderObjective(; kwargs...)

## Keyword arguments

* `cost = nothing` the cost function `c = f(M,p)`
* `differential = nothing` the differential `d = df(M, p, X)`
* `gradient=nothing` the gradient function `g(M, p)` or in-place `g!(M, X, p)`
* `costgradient = nothing` the combined cost and gradient function `fg(M,p)` or in-place `fg!(M, X, p))`
* `costdifferential = nothing` the combined cost and differential function  `fdf(M, p, X)`

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
# TODO: Test here how to maybe handle the old evaluation= kwarg to now automatically “wrap”
# allocating variants.
function ManifoldFirstOrderObjective(;
        cost = nothing, differential = nothing, gradient = nothing,
        costgradient = nothing, costdifferential = nothing,
    )
    no_cost = isnothing(cost)
    no_diff = isnothing(differential)
    no_grad = isnothing(gradient)
    ncg = isnothing(costgradient)
    ncd = isnothing(costdifferential)

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
        nt = merge(nt, (; cost = cost))
    end
    if !no_grad
        nt = merge(nt, (; gradient = gradient))
    end
    if !no_diff
        nt = merge(nt, (; differential = differential))
    end
    if !ncg
        nt = merge(nt, (; costgradient = costgradient))
    end
    if !ncd
        nt = merge(nt, (; costdifferential = costdifferential))
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

    * as a function `(M, X, p) -> X` that work in place of `X`, an [`InplaceEvaluation`](@ref)

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
    ManifoldCostGradientObjective(costgrad; evaluation::E=AllocatingEvaluation(), kwargs...)

create an objective containing one function to perform a combined computation of cost and its gradient

Depending on the [`AbstractEvaluationType`](@ref) `E` the gradient can have to forms

* as a function `(M, p) -> (c, X)` that allocates memory for the gradient `X`, an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> (c, X)` that work in place of `X`, an [`InplaceEvaluation`](@ref)

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
function get_cost(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p
    )
    haskey(mfo.functions, :cost) && (return mfo.functions[:cost](M, p))
    X = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, X, p, X)[1]
    end
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[1])
    return error("$mfo does not seem to provide a cost")
end

#TODO: Since Y is a keyword, maybe a better name is gradient_cache? and add the evaluated bool here as well
function get_cost_and_differential(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p, X; Y = nothing,
    )
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, p, X)
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :differential)
        return (mfo.functions[:cost](M, p), mfo.functions[:differential](M, p, X))
    elseif haskey(mfo.functions, :costgradient)
        _Y = isnothing(Y) ? zero_vector(M, p) : Y
        cost, grad = mfo.functions[:costgradient](M, _Y, p)
        return (cost, real(inner(M, p, X, grad)))
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        cost = mfo.functions[:cost](M, p)
        _Y = isnothing(Y) ? zero_vector(M, p) : Y
        grad = mfo.functions[:gradient](M, _Y, p)
        return (cost, real(inner(M, p, X, grad)))
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

function get_cost_function(
        mfo::ManifoldFirstOrderObjective, recursive::Bool = false
    )
    if haskey(mfo.functions, :cost)
        return mfo.functions[:cost]
    else
        return (M, p) -> get_cost(M, mfo, p)
    end
end

function get_differential(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p, X;
        gradient = nothing, evaluated::Bool = false, kwargs...,
    )
    # If we have a differential – evaluate that
    haskey(mfo.functions, :differential) && (return mfo.functions[:differential](M, p, X))
    haskey(mfo.functions, :costdifferential) &&
        (return mfo.functions[:costdifferential](M, p, X)[2])
    # default: inner with gradient
    # (a) we have gradient but it is not evaluated -> eval
    (!evaluated && !isnothing(gradient)) && (get_gradient!(M, gradient, mfo, p))
    # if grad is nothing -> allocated gradient
    isnothing(gradient) && (gradient = get_gradient(M, mfo, p))
    # -> we have a gradient!
    return real(inner(M, p, gradient, X))
end

function get_differential_function(
        mfo::ManifoldFirstOrderObjective, recursive::Bool = false
    )
    if haskey(mfo.functions, :differential)
        return mfo.functions[:differential]
    else
        return (M, p, X; kwargs...) -> get_differential(M, mfo, p, X, kwargs...)
    end
end
function get_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{<:NamedTuple}, p,
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, X, p))
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[2])
    return error("$mfo does not seem to provide a gradient")
end

function get_gradient_function(
        mfo::ManifoldFirstOrderObjective, recursive = false
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
    return (M, X, p) -> get_gradient!(M, X, mfo, p)
end

function status_summary(mfo::ManifoldFirstOrderObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(mfo))
    return "A first order objective with $(length(mfo.functions)) provided functions.\n\n" * join([ "* $k:$(_MANOPT_INDENT) $(v)" for (k, v) in zip(keys(mfo.functions), mfo.functions) ], "\n")
end
function Base.show(io::IO, mfo::ManifoldFirstOrderObjective)
    print(io, "ManifoldFirstOrderObjective(; ")
    print(io, join([ "$k = $v" for (k, v) in zip(keys(mfo.functions), mfo.functions)], ", "))
    print(io, ", ")
    return print(io, ")")
end

#
#
# ---
@doc """
    ManifoldNonlinearLeastSquaresObjectives <: AbstractManifoldObjective

An objective to model the robustified nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

# Fields

* `objective`: a vector of [`AbstractFirstOrderVectorFunction`](@ref)`{E}`s, one for each
  block component cost function ``F_i``, which might internally also be a vector of component costs ``(F_i)_j``,
  as well as their Jacobian ``J_{F_i}`` or a vector of gradients ``$(_tex(:grad)) (F_i)_j``
  depending on the specified [`AbstractVectorialType`](@ref)s.
* `robustifier`: a vector of [`AbstractRobustifierFunction`](@ref)`s`, one for each
  block component cost function ``F_i``.
* `value_cache::AbstractVector` and internal cache to store the result of evaluating the cost functions

# Constructors

    ManifoldNonlinearLeastSquaresObjective(f, jacobian, range_dimension::Integer, robustifier=IdentityRobustifier(); kwargs...)

Create a nonlinear least squares objective for a single vectorial function `f` and its `jacobian`,
where `range_dimension` is the dimension of the vector space `f` maps into. These three are internally
wrapped into a [`VectorGradientFunction`](@ref) and calls the following constructor.

    ManifoldNonlinearLeastSquaresObjective(vf::AbstractFirstOrderVectorFunction, robustifier::AbstractRobustifierFunction=IdentityRobustifier())

Create a nonlinear least squares objective for a given vectorial function.
Note that for this constructor the `robustifier` is applied componentwise to each component of `vf`,
i.e. wrapped in a [`ComponentwiseRobustifierFunction`](@ref).
Internally this wraps both `vf` and `robustifier` in an array and calls the next constructor.
Hence to not use the componentwise robustifier but a global one, pass `[vf,]` and `[robustifier,]` instead.

    ManifoldNonlinearLeastSquaresObjective(fs::Vector{<:AbstractFirstOrderVectorFunction}, robustifiers::Vector{<:AbstractRobustifierFunction}=fill(IdentityRobustifier(), length(fs)))

Given a vector of [`AbstractFirstOrderVectorFunction`](@ref)`s to represent the single blocks
and a vector of robustifiers, one for each block, create the corresponding nonlinear least squares objective.

# Keyword arguments

The first constructor allows to pass the following keyword arguments, that are passed on to
the corresponding
the constructor of the As well as for the first variant of having a single block

* `function_type::`[`AbstractVectorialType`](@ref)`=`[`FunctionVectorialType`](@ref)`()`: specify
  the format the residuals are given in. By default a function returning a vector.
* `jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis()`; shortcut to specify
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
    )
    vgf = VectorGradientFunction(
        f, jacobian, range_dimension; jacobian_type = jacobian_type, function_type = function_type,
    )
    return ManifoldNonlinearLeastSquaresObjective(vgf, robustifier)
end

"""
    get_cost(M::AbstractManifold, nlso::ManifoldNonLinearLeastSquaresObjective, p)
ß
Compute the cost of the least squares objective, i.e.

```math
$(_tex(:frac, "1", "2")) $(_tex(:sum, "i=1", "m")) ρ_i $(_tex(:bigl))( $(_tex(:norm, "F_i(p)"))^2 $(_tex(:bigr))),
```

where ``F_i: $(_math(:Manifold)) → ℝ^{n_i}`` is the ``i``th block component of length ``n_i > 0``
and each ``ρ_i: ℝ → ℝ`` is a [* R robustifier function, cf. [`AbstractRobustifierFunction`](@ref),
for each such a block component.
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
# For a single vectorial function where the robustifier is applied to every in dex separately.
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
    zero_vector!(M, X, p)
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
    return ("A nonlinear least squares objective $(n) vectorial block$(n > 1 ? "s" : "")")
end

# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::ManifoldNonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, v, nlso::ManifoldNonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``F(p) ∈ ℝ^n``, ``n = $(_tex(:sum, "1", "m")) n_i``.
In other words this is the concatenation of the residual vectors ``F_i(p)``, ``i=1,…,m``
of the components of the the [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso`
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
    ManifoldProximalMapObjective{TC, TP, V <: Vector{<:Integer}} <: AbstractManifoldCostObjective{TC}

specify a problem for solvers based on the evaluation of proximal maps,
which represents proximal maps ``$(_tex(:prox))_{λf_i}`` for summands ``f = f_1 + f_2+ … + f_N`` of the cost function ``f``.

# Fields

* `cost`: a function ``f:$(_math(:Manifold))→ℝ`` to
  minimize
* `proxes`: proximal maps ``$(_tex(:prox))_{λf_i}:$(_math(:Manifold)) → $(_math(:Manifold))``
  as functions `(M, λ, p) -> q` or in-place `(M, q, λ, p)`.
* `number_of_proxes`: number of proximal maps per function,
  to specify when one of the maps is a combined one such that the proximal maps
  functions return more than one entry per function, you have to adapt this value.
  if not specified, it is set to one prox per function.

# Constructor

    ManifoldProximalMapObjective( f, proxes_f::Union{Tuple,AbstractVector}, number_of_proxes=onex(length(proxes)) )

Generate a proximal problem with a tuple or vector of functions, where by default every function computes a single prox
of one component of ``f``.

    ManifoldProximalMapObjective(f, prox_f)

Generate a proximal objective for ``f`` and its proxial map ``$(_tex(:prox))_{λf}``

# See also

[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ManifoldProximalMapObjective{TC, TP, V} <: AbstractManifoldCostObjective{TC}
    cost::TC
    proximal_maps!!::TP
    number_of_proxes::V
    function ManifoldProximalMapObjective(f, proxes_f::Union{Tuple, AbstractVector})
        np = ones(length(proxes_f))
        return new{typeof(f), typeof(proxes_f), typeof(np)}(
            f, proxes_f, np
        )
    end
    function ManifoldProximalMapObjective(
            f::F, proxes_f::Union{Tuple, AbstractVector}, nOP::Vector{<:Integer}
        ) where {F}
        return if length(nOP) != length(proxes_f)
            throw(
                ErrorException(
                    "The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxes_f)).",
                ),
            )
        else
            new{F, typeof(proxes_f), typeof(nOP)}(f, proxes_f, nOP)
        end
    end
    function ManifoldProximalMapObjective(f::F, prox_f::PF) where {F, PF}
        i = 1
        return new{F, PF, typeof(i)}(f, prox_f, i)
    end
end
function _check_prox_number(pf::Union{Tuple, Vector}, i)
    n = length(pf)
    (i > n) && throw(ErrorException("the $(i)th entry does not exists, only $n available."))
    return true
end

@doc """
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p)
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p, i)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p, i)

evaluate the (`i`th) proximal map of the [`ManifoldProximalMapObjective`](@ref)` mpo` at
the point `p` of `M` with parameter ``λ>0``.
"""
get_proximal_map(::AbstractManifold, ::ManifoldProximalMapObjective, ::Any...)

function get_proximal_map(
        M::AbstractManifold,
        mpo::ManifoldProximalMapObjective{InplaceEvaluation, F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    _check_prox_number(mpo.proximal_maps!!, i)
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    _check_prox_number(mpo.proximal_maps!!, i)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end

function get_proximal_map(
        M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p
    )
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!!(M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p
    )
    return mpo.proximal_maps!!(M, q, λ, p)
end
function status_summary(mpo::ManifoldProximalMapObjective; context::Symbol = :default)
    (context === :short) && (return repr(mpo))
    return "A proximal map objective for a cost with $(mpo.number_of_proxes) proximal maps"
end
function Base.show(io::IO, mpo::ManifoldProximalMapObjective)
    print(io, "ManifoldProximalMapObjective(", mpo.cost, ", ", mpo.proximal_maps!!, ", ")
    print(io, mpo.number_of_proxes)
    return print(io, ")")
end


@doc """
    ManifoldProximalGradientObjective{TC, TG, TGG, TP} <: AbstractManifoldObjective{TC,TGG}

Model an objective of the form

```math
f(p) = g(p) + h(p), $(_tex(:qquad)) p ∈ $(_math(:Manifold)),
```

where ``g: $(_math(:Manifold)) → $(_tex(:eR))`` is a differentiable function
and ``h: → $(_tex(:eR))`` is a (possibly) lower semicontinous, and proper function.

This objective provides the total cost ``f``, its smooth component ``g``,
as well as ``$(_tex(:grad)) g`` and ``$(_tex(:prox))_{λ h}``.

# Fields

* `cost`: the overall cost ``f = g + h``
* `cost_smooth`: the smooth cost component ``g``
* `gradient_g!!`: the gradient ``$(_tex(:grad)) g``
* `proximal_map_h!!`: the proximal map ``$(_tex(:prox))_{λ h}``

# Constructor
    ManifoldProximalGradientObjective(f, g, grad_g, prox_h)

Generate the proximal gradient objective given the total cost ``f = g + h``, smooth cost ``g``, the gradient of the smooth component ``$(_tex(:grad)) g``, and the proximal map of the nonsmooth component ``$(_tex(:prox))_{λ h}``.
"""
struct ManifoldProximalGradientObjective{TC, TG, TGG, TP} <: AbstractManifoldCostObjective{TC}
    cost::TC # f = g + h
    cost_smooth::TG # smooth part
    gradient_g!!::TGG
    proximal_map_h!!::TP
    function ManifoldProximalGradientObjective(
            f::TC, g::TG, grad_g::TGG, prox_h::TP
        ) where {TC, TG, TGG, TP}
        return new{TC, TG, TGG, TP}(f, g, grad_g, prox_h)
    end
end

"""
    get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, mgo::ManifoldProximalGradientObjective, p)

Evaluate the gradient of the smooth part of a [`ManifoldProximalGradientObjective`](@ref) `mgo` at `p`.
"""
get_gradient(::AbstractManifold, ::ManifoldProximalGradientObjective, p)

function get_gradient!(M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective, p)
    return mpgo.gradient_g!!(M, X, p)
end

function Base.show(io::IO, mpgo::ManifoldProximalGradientObjective{E}) where {E}
    print(io, "ManifoldProximalGradientObjective(", mpgo.cost, ", ", mpgo.cost_smooth, ", ")
    print(io, mpgo.gradient_g!!, ", ", mpgo.proximal_map_h!!)
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
    * `gradient_g`: $(mpgo.gradient_g!!)
    * `prox_h`:     $(mpgo.proximal_map_h!!)"""
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

Evaluate proximal map of the nonsmooth component ``h`` of the [`ManifoldProximalGradientObjective`](@ref)` mpo`
at the point `p` on `M` with parameter ``λ>0``.
"""
get_proximal_map(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, λ, p)

function get_proximal_map!(
        M::AbstractManifold, q, mpgo::ManifoldProximalGradientObjective, λ, p
    )
    return mpgo.proximal_map_h!!(M, q, λ, p)
end

@doc """
    ManifoldSubgradientObjective{T<:AbstractEvaluationType,C,S} <:AbstractManifoldCostObjective{T, C}

A structure to store information about a objective for a subgradient based optimization problem

# Fields

* `cost`:        the function ``f`` to be minimized
* `subgradient`: a function returning a subgradient ``∂f`` of ``f``

# Constructor

    ManifoldSubgradientObjective(f, ∂f)

Generate the [`ManifoldSubgradientObjective`](@ref) for a subgradient objective, consisting
of a (cost) function `f(M, p)` and a function `∂f(M, p)` that returns a not necessarily
deterministic element from the subdifferential at `p` on a manifold `M`.
"""
struct ManifoldSubgradientObjective{C, S} <: AbstractManifoldCostObjective{C}
    cost::C
    subgradient!!::S
    function ManifoldSubgradientObjective(cost::C, subgrad::S) where {C, S}
        return new{C, S}(cost, subgrad)
    end
end

"""
    X = get_subgradient(M;;AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    get_subgradient!(M;;AbstractManifold, X, sgo::ManifoldSubgradientObjective, p)

Evaluate the (sub)gradient of a [`ManifoldSubgradientObjective`](@ref) `sgo`
at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(M::AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    X = zero_vector(M, p)
    return sgo.subgradient!!(M, X, p)
end
function get_subgradient!(
        M::AbstractManifold, X, sgo::ManifoldSubgradientObjective, p
    )
    return sgo.subgradient!!(M, X, p)
end

@doc """
    get_subgradient_function(objective::ManifoldSubgradientObjective, recursive=false)

return the function to evaluate (just) the gradient ``$(_tex(:grad)) f(p)``
and is of the form `(M, X, p) -> X` to work in-place of `X`,
where either the gradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.
"""
function get_subgradient_function(objective::ManifoldSubgradientObjective, recursive = false)
    return objective.subgradient!!
end

function Base.show(io::IO, objective::ManifoldSubgradientObjective)
    return print(io, "ManifoldSubgradientObjective(", objective.cost, ", ", objective.subgradient!!, ")")
end

function status_summary(objective::ManifoldSubgradientObjective; context::Symbol = :default)
    (context === :short) && return repr(objective)
    s = "A subgradient objective "
    (context === :inline) && (return s)
    return """
    $s

    ## Components
    * `f`:  $(objective.cost)
    * `∂f`: $(objective.subgradient!!)"""
end
