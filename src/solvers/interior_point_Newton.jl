_doc_IPN = raw"""
    interior_point_Newton(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    interior_point_Newton(M, cmo::ConstrainedManifoldObjective, p=rand(M); kwargs...)
    interior_point_Newton!(M, f, grad_f, Hess_f, p; kwargs...)
    interior_point_Newton(M, ConstrainedManifoldObjective, p; kwargs...)

perform the interior point Newton method following [LaiYoshise:2024](@cite).

In order to solve the constrained problem

```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```

This algorithms iteratively solves the linear system based on extending the KKT system
by a slack variable `s`.

```math
\operatorname{J} F(p, μ, λ, s)[X, Y, Z, W] = -F(p, μ, λ, s),
\text{ where }
X ∈ T_p\mathcal M, Y,W ∈ ℝ^m, Z ∈ ℝ^n,
```

see [`CondensedKKTVectorFieldJacobian`](@ref) and [`CondensedKKTVectorField`](@ref), respectively,
for the reduced form, this is usually solved in.
From the resulting `X` and `Z` in the reeuced form, the other two, ``Y``, ``W``, are then computed.

From the gradient ``(X,y,Z,W)`` at the current iterate ``(p, μ, λ, s)``,
a line search is performed using the [`KKTVectorFieldNormSq`](@ref) norm of the KKT vector field (squared)
and its gradient [`KKTVectorFieldNormSqGradient`](@ref) together with the [`InteriorPointCentralityCondition`](@ref).

Note that since the vector field ``F`` includes the gradients of the constraint
functions ``g,h`, its gradient or Jacobian requires the Hessians of the constraints.

For that seach direction a line search is performed, that additionally ensures that
the constraints are further fulfilled.

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``f : \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad} f : \mathcal M → T \mathcal M`` of ``f``
* `Hess_f`: the Hessian ``\operatorname{Hess}f(p): T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}f(p)[X] = ∇_X\operatorname{grad}f(p)``
* `p=`[`rand`](@extref Base.rand-Tuple{AbstractManifold})`(M)`: an initial value ``p  ∈  \mathcal M``

or a [`ConstrainedManifoldObjective`](@ref) `cmo` containing `f`, `grad_f`, `Hess_f`, and the constraints

# Keyword arguments

The keyword arguments related to the constraints (the first eleven) are ignored if you
pass a [`ConstrainedManifoldObjective`](@ref) `cmo`

* `centrality_condition=missing`; an additional condition when to accept a step size.
  This can be used to ensure that the resulting iterate is still an interior point if you provide a check `(N,q) -> true/false`,
  where `N` is the manifold of the `step_problem`.
* `equality_constraints=nothing`: the number ``n`` of equality constraints.
* `evaluation=`[`AllocatingEvaluation`](@ref)`()`:
  specify whether the functions that return an array, for example a point or a tangent vector, work by allocating its result ([`AllocatingEvaluation`](@ref)) or whether they modify their input argument to return the result therein ([`InplaceEvaluation`](@ref)). Since usually the first argument is the manifold, the modified argument is the second."
* `g=nothing`: the inequality constraints
* `grad_g=nothing`: the gradient of the inequality constraints
* `grad_h=nothing`: the gradient of the equality constraints
* `gradient_range=nothing`: specify how gradients are represented, where `nothing` is equivalent to [`NestedPowerRepresentation`](@extref)
* `gradient_equality_range=gradient_range`: specify how the gradients of the equality constraints are represented
* `gradient_inequality_range=gradient_range`: specify how the gradients of the inequality constraints are represented
* `h=nothing`: the equality constraints
* `Hess_g=nothing`: the Hessian of the inequality constraints
* `Hess_h=nothing`: the Hessian of the equality constraints
* `inequality_constraints=nothing`: the number ``m`` of inequality constraints.
* `λ=ones(length(h(M,x),1))`: the Lagrange multiplier with respect to the equality constraints ``h``
* `μ=ones(length(g(M,x)))`: the Lagrange multiplier with respect to the inequality constraints ``g``
* `retraction_method=`[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`:
  the retraction to use, defaults to the default set `M` with respect to the representation for `p` chosen.
* `ρ=μ's / length(μ)`:  store the orthogonality `μ's/m` to compute the barrier parameter `β` in the sub problem.
* `s=copy(μ)`: initial value for the slack variables
* `σ=`[`calculate_σ`](@ref)`(M, cmo, p, μ, λ, s)`:  scaling factor for the barrier parameter `β` in the sub problem, which is updated during the iterations
* `step_objective`: a [`ManifoldGradientObjective`](@ref) of the norm of the KKT vector field [`KKTVectorFieldNormSq`](@ref) and its gradient [`KKTVectorFieldNormSqGradient`](@ref)
* `step_problem`: the manifold ``\mathcal M × ℝ^m × ℝ^n × ℝ^m`` together with the `step_objective`
  as the problem the linesearch `stepsize=` employs for determining a step size
* `step_state`: the [`StepsizeState`](@ref) with point and search direction
* `stepsize` an [`ArmijoLinesearch`](@ref) with the [`InteriorPointCentralityCondition`](@ref) as
  additional condition to accept a step. Note that this step size operates on its own `step_problem`and `step_state`
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(200)`[` | `](@ref StopWhenAny)[`StopWhenKKTResidualLess`](@ref)`(1e-8)`:
  a stopping criterion, by default depending on the residual of the KKT vector field or a maximal number of steps, which ever hits first.
* `sub_kwargs=(;)`: keyword arguments to decorate the sub options, for example debug, that automatically respects the main solvers debug options (like sub-sampling) as well
* `sub_objective`: The [`SymmetricLinearSystemObjective`](@ref) modelling the system of equations to use in the sub solver,
  includes the [`CondensedKKTVectorFieldJacobian`](@ref) ``\mathcal A(X)`` and the [`CondensedKKTVectorField`](@ref) ``b`` in ``\mathcal A(X) + b = 0`` we aim to solve.
  This is used to setup the `sub_problem`. If you set the `sub_problem` directly, this keyword has no effect.
* `sub_stopping_criterion=`[`StopAfterIteration`](@ref)`(manifold_dimension(M))`[` | `](@ref StopWhenAny)[`StopWhenRelativeResidualLess`](@ref)`(c,1e-8)`, where ``c = \lVert b \rVert`` from the system to solve.
  This keyword is used in the `sub_state`. If you set that keyword diretly, this keyword does not have an effect.
* `sub_problem`: combining the `sub_objective` and the tangent space at ``(p,λ)``` on the manifold ``\mathcal M × ℝ^n`` to a manopt problem.
   This is the manifold and objective for the sub solver.
* `sub_state=`[`ConjugateResidualState`](@ref): a state specifying the subsolver. This default is also decorated with the `sub_kwargs...`.
* `vector_space=`[`Rn`](@ref Manopt.Rn): specify which manifold to use for the vector space components ``ℝ^m,ℝ^n``
* `X=`[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`:
  th initial gradient with respect to `p`.
* `Y=zero(μ)`:  the initial gradient with respct to `μ`
* `Z=zero(λ)`:  the initial gradient with respct to `λ`
* `W=zero(s)`:  the initial gradient with respct to `s`

As well as internal keywords used to set up these given keywords like `_step_M`, `_step_p`, `_sub_M`, `_sub_p`, and `_sub_X`,
that should not be changed.

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.

!!! note

    The `centrality_condition=mising` disables to check centrality during the line search,
    but you can pass [`InteriorPointCentralityCondition`](@ref)`(cmo, γ)`, where `γ` is a constant,
    to activate this check.

# Output

The obtained approximate constrained minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
"""

@doc "$(_doc_IPN)"
interior_point_Newton(M::AbstractManifold, args...; kwargs...)
function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p=rand(M);
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    inequality_constrains::Union{Integer,Nothing}=nothing,
    equality_constrains::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    q = copy(M, p)
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        hess_f=Hess_f,
        hess_g=Hess_g,
        hess_h=Hess_h,
        evaluation=evaluation,
        inequality_constrains=inequality_constrains,
        equality_constrains=equality_constrains,
        M=M,
        p=p,
    )
    return interior_point_Newton!(M, cmo, q; evaluation=evaluation, kwargs...)
end
function interior_point_Newton(
    M::AbstractManifold, cmo::O, p; kwargs...
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end

@doc "$(_doc_IPN)"
interior_point_Newton!(M::AbstractManifold, args...; kwargs...)

function interior_point_Newton!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    inequality_constrains=nothing,
    equality_constrains=nothing,
    kwargs...,
)
    cmo = ConstrainedManifoldObjective(
        f,
        grad_f,
        g,
        grad_g,
        h,
        grad_h;
        hess_f=Hess_f,
        hess_g=Hess_g,
        hess_h=Hess_h,
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        M=M,
        p=p,
    )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    return interior_point_Newton!(
        M,
        dcmo,
        p;
        evaluation=evaluation,
        equality_constrains=equality_constrains,
        inequality_constrains=inequality_constrains,
        kwargs...,
    )
end
function interior_point_Newton!(
    M::AbstractManifold,
    cmo::O,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    X=get_gradient(M, cmo, p),
    μ::Vector=ones(length(get_inequality_constraint(M, cmo, p, :))),
    Y=zero(μ),
    λ::Vector=zeros(length(get_equality_constraint(M, cmo, p, :))),
    Z=zero(λ),
    s=copy(μ),
    W=zero(s),
    ρ=μ's / length(μ),
    σ=calculate_σ(M, cmo, p, μ, λ, s),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    sub_kwargs=(;),
    vector_space=Rn,
    #γ=0.9,
    centrality_condition=missing, #InteriorPointCentralityCondition(cmo, γ, zero(γ), zero(γ)),
    step_objective=ManifoldGradientObjective(
        KKTVectorFieldNormSq(cmo), KKTVectorFieldNormSqGradient(cmo); evaluation=evaluation
    ),
    _step_M=M × vector_space(length(μ)) × vector_space(length(λ)) × vector_space(length(s)),
    step_problem=DefaultManoptProblem(_step_M, step_objective),
    _step_p=rand(_step_M),
    step_state=StepsizeState(_step_p, zero_vector(_step_M, _step_p)),
    stepsize::Stepsize=ArmijoLinesearch(
        _step_M;
        retraction_method=default_retraction_method(_step_M),
        initial_guess=interior_point_initial_guess,
        additional_decrease_condition=if ismissing(centrality_condition)
            (M, p) -> true
        else
            centrality_condition
        end,
    ),
    stopping_criterion::StoppingCriterion=StopAfterIteration(800) |
                                          StopWhenKKTResidualLess(1e-8),
    _sub_M=M × vector_space(length(λ)),
    _sub_p=rand(_sub_M),
    _sub_X=rand(_sub_M; vector_at=_sub_p),
    sub_objective=decorate_objective!(
        TangentSpace(_sub_M, _sub_p),
        SymmetricLinearSystemObjective(
            CondensedKKTVectorFieldJacobian(cmo, μ, s, σ * ρ),
            CondensedKKTVectorField(cmo, μ, s, σ * ρ),
        ),
        sub_kwargs...,
    ),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(M)) |
                                              StopWhenRelativeResidualLess(
        norm(_sub_M, _sub_p, get_b(TangentSpace(_sub_M, _sub_p), sub_objective, _sub_X)),
        1e-8,
    ),
    sub_state::St=decorate_state!(
        ConjugateResidualState(
            TangentSpace(_sub_M, _sub_p),
            sub_objective;
            X=_sub_X,
            stop=sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs...,
    ),
    sub_problem::Pr=DefaultManoptProblem(TangentSpace(_sub_M, _sub_p), sub_objective),
    kwargs...,
) where {
    O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective},
    St<:AbstractManoptSolverState,
    Pr<:Union{F,AbstractManoptProblem} where {F},
}
    !is_feasible(M, cmo, p; error=:error)
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointNewtonState(
        M,
        cmo,
        p,
        sub_problem,
        sub_state;
        X=X,
        Y=Y,
        Z=Z,
        W=W,
        μ=μ,
        λ=λ,
        s=s,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        step_problem=step_problem,
        step_state=step_state,
        stepsize=stepsize,
        kwargs...,
    )
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end
function initialize_solver!(::AbstractManoptProblem, ips::InteriorPointNewtonState)
    return ips
end

function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointNewtonState, i)
    M = get_manifold(amp)
    cmo = get_objective(amp)
    N = base_manifold(get_manifold(ips.sub_problem))
    q = base_point(get_manifold(ips.sub_problem))
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), zero_vector(N, q))

    set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)
    set_manopt_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_manopt_parameter!(ips.sub_problem, :Objective, :β, ips.ρ * ips.σ)
    # product manifold on which to perform linesearch

    X2 = get_solver_result(solve!(ips.sub_problem, ips.sub_state))
    ips.X, ips.Z = X2[N, 1], X2[N, 2] #for p and λ

    # Compute the remaining part of the solution
    m, n = length(ips.μ), length(ips.λ)
    if m > 0
        g = get_inequality_constraint(amp, ips.p, :)
        grad_g = get_grad_inequality_constraint(amp, ips.p, :)
        β = ips.ρ * ips.σ
        # for s and μ
        ips.W = -[inner(M, ips.p, grad_g[i], ips.X) for i in 1:m] - g - ips.s
        ips.Y = (β .- ips.μ .* (ips.s + ips.W)) ./ ips.s
    end

    N = get_manifold(ips.step_problem)
    # generate current full iterate in step state
    q = get_iterate(ips.step_state)
    copyto!(N[1], q[N, 1], get_iterate(ips))
    copyto!(N[2], q[N, 2], ips.μ)
    copyto!(N[3], q[N, 3], ips.λ)
    copyto!(N[4], q[N, 4], ips.s)
    set_iterate!(ips.step_state, M, q)
    # generate current full gradient in step state
    X = get_gradient(ips.step_state)
    copyto!(N[1], X[N, 1], ips.X)
    (m > 0) && (copyto!(N[2], X[N, 2], ips.Z))
    (n > 0) && (copyto!(N[3], X[N, 3], ips.Y))
    (m > 0) && (copyto!(N[4], X[N, 4], ips.W))
    set_gradient!(ips.step_state, M, q, X)
    # Update centrality factor – Maybe do this as an update function?
    γ = get_manopt_parameter(ips.stepsize, :DecreaseCondition, :γ)
    if !isnothing(γ)
        set_manopt_parameter!(ips.stepsize, :DecreaseCondition, :γ, (γ + 0.5) / 2)
    end
    set_manopt_parameter!(ips.stepsize, :DecreaseCondition, :τ, N, q)
    # determine stepsize
    α = ips.stepsize(ips.step_problem, ips.step_state, i)
    # Update Parameters and slack
    retract!(M, ips.p, ips.p, α * ips.X, ips.retraction_method)
    if m > 0
        ips.μ += α * ips.Y
        ips.s += α * ips.W
        ips.ρ = ips.μ'ips.s / m
        # we can use the memory from above still
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s; N=N, q=q)
    end
    (n > 0) && (ips.λ += α * ips.Z)
    return ips
end

get_solver_result(ips::InteriorPointNewtonState) = ips.p
