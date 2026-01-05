_doc_IPN_subsystem = """
```math
  $(_tex(:operatorname, "J")) F(p, μ, λ, s)[X, Y, Z, W] = -F(p, μ, λ, s),
  $(_tex(:text, " where "))
  X ∈ $(_math(:TpM)), Y,W ∈ ℝ^m, Z ∈ ℝ^n
```
"""
_doc_IPN = """
    interior_point_Newton(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    interior_point_Newton(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)
    interior_point_Newton!(M, f, grad]_f, Hess_f, p; kwargs...)
    interior_point_Newton(M, ConstrainedManifoldObjective, p; kwargs...)

perform the interior point Newton method following [LaiYoshise:2024](@cite).

In order to solve the constrained problem

$(_problem(:Constrained))

This algorithms iteratively solves the linear system based on extending the KKT system
by a slack variable `s`.

$(_doc_IPN_subsystem)

see [`CondensedKKTVectorFieldJacobian`](@ref) and [`CondensedKKTVectorField`](@ref), respectively,
for the reduced form, this is usually solved in.
From the resulting `X` and `Z` in the reduced form, the other two, ``Y``, ``W``, are then computed.

From the gradient ``(X,Y,Z,W)`` at the current iterate ``(p, μ, λ, s)``,
a line search is performed using the [`KKTVectorFieldNormSq`](@ref) norm of the KKT vector field (squared)
and its gradient [`KKTVectorFieldNormSqGradient`](@ref) together with the [`InteriorPointCentralityCondition`](@ref).

Note that since the vector field ``F`` includes the gradients of the constraint
functions ``g, h``, its gradient or Jacobian requires the Hessians of the constraints.

For that search direction a line search is performed, that additionally ensures that
the constraints are further fulfilled.

# Input

$(_var(:Argument, :M))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))
$(_var(:Argument, :Hess_f))
$(_var(:Argument, :p))

or a [`ConstrainedManifoldObjective`](@ref) `cmo` containing `f`, `grad_f`, `Hess_f`, and the constraints

# Keyword arguments

The keyword arguments related to the constraints (the first eleven) are ignored if you
pass a [`ConstrainedManifoldObjective`](@ref) `cmo`

* `centrality_condition=missing`; an additional condition when to accept a step size.
  This can be used to ensure that the resulting iterate is still an interior point if you provide a check `(N,q) -> true/false`,
  where `N` is the manifold of the `step_problem`.
* `equality_constraints=nothing`: the number ``n`` of equality constraints.
$(_var(:Keyword, :evaluation))
* `g=nothing`: the inequality constraints
* `grad_g=nothing`: the gradient of the inequality constraints
* `grad_h=nothing`: the gradient of the equality constraints
* `gradient_range=nothing`: specify how gradients are represented, where `nothing` is equivalent to [`NestedPowerRepresentation`](@extref `ManifoldsBase.NestedPowerRepresentation`)
* `gradient_equality_range=gradient_range`: specify how the gradients of the equality constraints are represented
* `gradient_inequality_range=gradient_range`: specify how the gradients of the inequality constraints are represented
* `h=nothing`: the equality constraints
* `Hess_g=nothing`: the Hessian of the inequality constraints
* `Hess_h=nothing`: the Hessian of the equality constraints
* `inequality_constraints=nothing`: the number ``m`` of inequality constraints.
* `λ=ones(length(h(M, p)))`: the Lagrange multiplier with respect to the equality constraints ``h``
* `μ=ones(length(g(M, p)))`: the Lagrange multiplier with respect to the inequality constraints ``g``
$(_var(:Keyword, :retraction_method))
* `ρ=μ's / length(μ)`:  store the orthogonality `μ's/m` to compute the barrier parameter `β` in the sub problem.
* `s=copy(μ)`: initial value for the slack variables
* `σ=`[`calculate_σ`](@ref)`(M, cmo, p, μ, λ, s)`:  scaling factor for the barrier parameter `β` in the sub problem, which is updated during the iterations
* `step_objective`: a [`ManifoldGradientObjective`](@ref) of the norm of the KKT vector field [`KKTVectorFieldNormSq`](@ref) and its gradient [`KKTVectorFieldNormSqGradient`](@ref)
* `step_problem`: the manifold ``$(_math(:M)) × ℝ^m × ℝ^n × ℝ^m`` together with the `step_objective`
  as the problem the linesearch `stepsize=` employs for determining a step size
* `step_state`: the [`StepsizeState`](@ref) with point and search direction
$(_var(:Keyword, :stepsize; default = "[`ArmijoLinesearch`](@ref)`()`", add = " with the `centrality_condition` keyword as additional criterion to accept a step, if this is provided"))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenKKTResidualLess`](@ref)`(1e-8)`"))
  a stopping criterion, by default depending on the residual of the KKT vector field or a maximal number of steps, which ever hits first.
* `sub_kwargs=(;)`: keyword arguments to decorate the sub options, for example debug, that automatically respects the main solvers debug options (like sub-sampling) as well
* `sub_objective`: The [`SymmetricLinearSystemObjective`](@ref) modelling the system of equations to use in the sub solver,
  includes the [`CondensedKKTVectorFieldJacobian`](@ref) ``$(_tex(:Cal, "A"))(X)`` and the [`CondensedKKTVectorField`](@ref) ``b`` in ``$(_tex(:Cal, "A"))(X) + b = 0`` we aim to solve.
  $(_note(:KeywordUsedIn, "sub_problem"))
* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(manifold_dimension(M))`[` | `](@ref StopWhenAny)[`StopWhenRelativeResidualLess`](@ref)`(c,1e-8)`, where ``c = $(_tex(:norm, "b"))`` from the system to solve.
  $(_note(:KeywordUsedIn, "sub_state"))
$(_var(:Keyword, :sub_problem; default = "[`DefaultManoptProblem`](@ref)`(M, sub_objective)`"))
$(_var(:Keyword, :sub_state; default = "[`ConjugateResidualState`](@ref)"))
* `vector_space=`[`Rn`](@ref Manopt.Rn) a function that, given an integer, returns the manifold to be used for the vector space components ``ℝ^m,ℝ^n``
* `X=`[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`:
  the initial gradient with respect to `p`.
* `Y=zero(μ)`: the initial gradient with respect to `μ`
* `Z=zero(λ)`: the initial gradient with respect to `λ`
* `W=zero(s)`: the initial gradient with respect to `s`
* `is_feasible_error=:error`: specify how to handle infeasible starting points, see [`is_feasible`](@ref) for options.

As well as internal keywords used to set up these given keywords like `_step_M`, `_step_p`, `_sub_M`, `_sub_p`, and `_sub_X`,
that should not be changed.

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.

!!! note

    The `centrality_condition=missing` disables to check centrality during the line search,
    but you can pass [`InteriorPointCentralityCondition`](@ref)`(cmo, γ)`, where `γ` is a constant,
    to activate this check.

# Output

The obtained approximate constrained minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
"""

@doc "$(_doc_IPN)"
interior_point_Newton(M::AbstractManifold, args...; kwargs...)
function interior_point_Newton(
        M::AbstractManifold, f, grad_f, Hess_f, p = rand(M);
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = nothing, h = nothing,
        grad_g = nothing, grad_h = nothing,
        Hess_g = nothing, Hess_h = nothing,
        inequality_constraints::Union{Integer, Nothing} = nothing,
        equality_constraints::Union{Nothing, Integer} = nothing,
        kwargs...,
    )
    cmo = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h;
        hess_f = Hess_f, hess_g = Hess_g, hess_h = Hess_h,
        evaluation = evaluation,
        inequality_constraints = inequality_constraints,
        equality_constraints = equality_constraints,
        M = M, p = p,
    )
    return interior_point_Newton(M, cmo, p; evaluation = evaluation, kwargs...)
end
function interior_point_Newton(
        M::AbstractManifold, cmo::O, p; kwargs...
    ) where {O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(interior_point_Newton; kwargs...)
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end
calls_with_kwargs(::typeof(interior_point_Newton)) = (interior_point_Newton!,)

@doc "$(_doc_IPN)"
interior_point_Newton!(M::AbstractManifold, args...; kwargs...)

function interior_point_Newton!(
        M::AbstractManifold, f, grad_f, Hess_f, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        g = nothing, h = nothing,
        grad_g = nothing, grad_h = nothing,
        Hess_g = nothing, Hess_h = nothing,
        inequality_constraints = nothing,
        equality_constraints = nothing,
        kwargs...,
    )
    cmo = ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h;
        hess_f = Hess_f, hess_g = Hess_g, hess_h = Hess_h,
        evaluation = evaluation,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        M = M, p = p,
    )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    return interior_point_Newton!(M, dcmo, p; evaluation = evaluation, kwargs...)
end
function interior_point_Newton!(
        M::AbstractManifold, cmo::O, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        X = get_gradient(M, cmo, p),
        μ::AbstractVector = ones(inequality_constraints_length(cmo)),
        Y::AbstractVector = zero(μ),
        λ::AbstractVector = zeros(equality_constraints_length(cmo)),
        Z::AbstractVector = zero(λ),
        s::AbstractVector = copy(μ),
        W::AbstractVector = zero(s),
        ρ::Real = μ's / length(μ),
        σ::Real = calculate_σ(M, cmo, p, μ, λ, s),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        sub_kwargs = (;),
        vector_space = Rn,
        #γ=0.9,
        centrality_condition = missing, #InteriorPointCentralityCondition(cmo, γ, zero(γ), zero(γ)),
        step_objective = ManifoldGradientObjective(
            KKTVectorFieldNormSq(cmo), KKTVectorFieldNormSqGradient(cmo); evaluation = evaluation
        ),
        _step_M::AbstractManifold = ProductManifold(
            M,
            vector_space(length(μ)),
            vector_space(length(λ)),
            vector_space(length(s)),
        ),
        step_problem = DefaultManoptProblem(_step_M, step_objective),
        _step_p = rand(_step_M),
        step_state = StepsizeState(_step_p, zero_vector(_step_M, _step_p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = ArmijoLinesearch(
            _step_M;
            retraction_method = default_retraction_method(_step_M),
            initial_guess = interior_point_initial_guess,
            additional_decrease_condition = if ismissing(centrality_condition)
                (M, p) -> true
            else
                centrality_condition
            end,
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(800) |
            StopWhenKKTResidualLess(1.0e-8),
        _sub_M = ProductManifold(M, vector_space(length(λ))),
        _sub_p = rand(_sub_M),
        _sub_X = rand(_sub_M; vector_at = _sub_p),
        sub_objective = decorate_objective!(
            TangentSpace(_sub_M, _sub_p),
            SymmetricLinearSystemObjective(
                CondensedKKTVectorFieldJacobian(cmo, μ, s, σ * ρ),
                CondensedKKTVectorField(cmo, μ, s, σ * ρ),
            ),
            sub_kwargs...,
        ),
        sub_stopping_criterion::StoppingCriterion = StopAfterIteration(manifold_dimension(M)) |
            StopWhenRelativeResidualLess(
            norm(_sub_M, _sub_p, get_b(TangentSpace(_sub_M, _sub_p), sub_objective)), 1.0e-8
        ),
        sub_state::St = decorate_state!(
            ConjugateResidualState(
                TangentSpace(_sub_M, _sub_p),
                sub_objective;
                X = _sub_X,
                stop = sub_stopping_criterion,
                sub_kwargs...,
            );
            sub_kwargs...,
        ),
        sub_problem::Pr = DefaultManoptProblem(TangentSpace(_sub_M, _sub_p), sub_objective),
        is_feasible_error = :error,
        kwargs...,
    ) where {
        O <: Union{ConstrainedManifoldObjective, AbstractDecoratedManifoldObjective},
        St <: AbstractManoptSolverState,
        Pr <: Union{F, AbstractManoptProblem} where {F},
    }
    !is_feasible(M, cmo, p; error = is_feasible_error)
    keywords_accepted(interior_point_Newton!; kwargs...)
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointNewtonState(
        M, cmo, sub_problem, sub_state;
        p = p, X = X, Y = Y, Z = Z, W = W, μ = μ, λ = λ, s = s,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        step_problem = step_problem, step_state = step_state,
        stepsize = _produce_type(stepsize, _step_M),
        is_feasible_error = is_feasible_error,
        kwargs...,
    )
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end
calls_with_kwargs(::typeof(interior_point_Newton!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(amp::AbstractManoptProblem, ips::InteriorPointNewtonState)
    M = get_manifold(amp)
    cmo = get_objective(amp)
    !is_feasible(M, cmo, ips.p; error = ips.is_feasible_error)
    return ips
end

function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointNewtonState, k)
    M = get_manifold(amp)
    cmo = get_objective(amp)
    N = base_manifold(get_manifold(ips.sub_problem))
    q = base_point(get_manifold(ips.sub_problem))
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), zero_vector(N, q))

    set_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)
    set_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_parameter!(ips.sub_problem, :Objective, :β, ips.ρ * ips.σ)
    # product manifold on which to perform linesearch

    X2 = get_solver_result(solve!(ips.sub_problem, ips.sub_state))
    ips.X, ips.Z = submanifold_components(N, X2) #for p and λ

    # Compute the remaining part of the solution
    m, n = length(ips.μ), length(ips.λ)
    if m > 0
        g = get_inequality_constraint(amp, ips.p, :)
        grad_g = get_grad_inequality_constraint(amp, ips.p, :)
        β = ips.ρ * ips.σ
        # for s and μ
        ips.W .= [-inner(M, ips.p, grad_g[i], ips.X) for i in 1:m] .- g .- ips.s
        ips.Y .= (β .- ips.μ .* (ips.s + ips.W)) ./ ips.s
    end

    N = get_manifold(ips.step_problem)
    # generate current full iterate in step state
    q = get_iterate(ips.step_state)
    q1, q2, q3, q4 = submanifold_components(N, q)
    copyto!(N[1], q1, get_iterate(ips))
    q2 .= ips.μ
    q3 .= ips.λ
    q4 .= ips.s
    set_iterate!(ips.step_state, M, q)
    # generate current full gradient in step state
    X = get_gradient(ips.step_state)
    copyto!(N[1], X[N, 1], ips.X)
    (m > 0) && (copyto!(N[2], X[N, 2], ips.Z))
    (n > 0) && (copyto!(N[3], X[N, 3], ips.Y))
    (m > 0) && (copyto!(N[4], X[N, 4], ips.W))
    set_gradient!(ips.step_state, M, q, X)
    # Update centrality factor – Maybe do this as an update function?
    γ = get_parameter(ips.stepsize, :DecreaseCondition, :γ)
    if !isnothing(γ)
        set_parameter!(ips.stepsize, :DecreaseCondition, :γ, (γ + 0.5) / 2)
    end
    set_parameter!(ips.stepsize, :DecreaseCondition, :τ, N, q)
    # determine stepsize
    α = ips.stepsize(ips.step_problem, ips.step_state, k; gradient = X)
    # Update Parameters and slack
    retract!(M, ips.p, ips.p, α * ips.X, ips.retraction_method)
    if m > 0
        ips.μ .+= α .* ips.Y
        ips.s .+= α .* ips.W
        ips.ρ = ips.μ'ips.s / m
        # we can use the memory from above still
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s; N = N, q = q)
    end
    (n > 0) && (ips.λ .+= α .* ips.Z)
    return ips
end

get_solver_result(ips::InteriorPointNewtonState) = ips.p
