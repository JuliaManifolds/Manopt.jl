@doc raw"""
    interior_point_Newton(M, f,. grad_f, Hess_f; kwargs...)

Perform the interior point Newton method following [LaiYoshise:2024](@cite).
"""
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
    num_eq = if isnothing(equality_constrains)
        _number_of_constraints(h, grad_h; M=M, p=p)
    else
        inequality_constrains
    end
    num_ineq = if isnothing(inequality_constrains)
        _number_of_constraints(g, grad_g; M=M, p=p)
    else
        inequality_constrains
    end
    cmo = ConstrainedManifoldObjective(
        M,
        f,
        grad_f;
        hess_f=Hess_f,
        g=g,
        grad_g=grad_g,
        hess_g=Hess_g,
        h=h,
        grad_h=grad_h,
        hess_h=Hess_h,
        evaluation=evaluation,
        inequality_constrains=num_ineq,
        equality_constrains=num_eq,
        M=M,
        p=p,
    )
    return interior_point_Newton!(
        M,
        cmo,
        Hess_g,
        Hess_h,
        q;
        evaluation=evaluation,
        inequality_constrains=num_ineq,
        equality_constrains=num_eq,
        kwargs...,
    )
end

# not-in-place
# case where dim(M) = 1 and in particular p is a number
function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    h=nothing,
    grad_g=nothing,
    grad_h=nothing,
    Hess_g=nothing,
    Hess_h=nothing,
    kwargs...,
)
    p_ = [p]
    f_(M, p) = f(M, p[])

    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    Hess_f_ = _to_mutating_gradient(Hess_f, evaluation)

    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    Hess_g_ = isnothing(Hess_g) ? nothing : _to_mutating_gradient(Hess_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)
    Hess_h_ = isnothing(Hess_h) ? nothing : _to_mutating_gradient(Hess_h, evaluation)

    rs = interior_point_Newton(
        M,
        f_,
        grad_f_,
        Hess_f_,
        p_;
        evaluation=evaluation,
        g=g_,
        h=h_,
        grad_g=grad_g_,
        grad_h=grad_h_,
        Hess_g=Hess_g_,
        Hess_h=Hess_h_,
        kwargs...,
    )
    return (typeof(p_) == typeof(rs)) ? rs[] : rs
end

# not-in-place
# takes only manifold, constrained objetive and initial point
function interior_point_Newton(
    M::AbstractManifold, cmo::O, p; kwargs...
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end

# in-place
# takes M, f, grad_f, Hess_f and possibly constreint functions and their graidents
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
    if isnothing(inequality_constrains)
        inequality_constrains = _number_of_constraints(g, grad_g; M=M, p=p)
    end
    if isnothing(equality_constrains)
        equality_constrains = _number_of_constraints(h, grad_h; M=M, p=p)
    end
    cmo = ConstrainedManifoldObjective(
        M,
        f,
        grad_f;
        hess_f=Hess_f,
        g=g,
        grad_g=grad_g,
        hess_g=Hess_g,
        h=h,
        grad_h=grad_h,
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
    λ::Vector=zeros(length(get_equality_constraint(M, cmo, p, :))),
    s=μ,
    ρ=μ's / length(μ),
    σ=calculate_σ(M, cmo, p, μ, λ, s),
    stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    _N=M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s),
    stepsize::Stepsize=InteriorPointLinesearch(
        _N;
        retraction_method=default_retraction_method(_N),
        additional_decrease_condition=ConstraintLineSearchCheckFunction(
            cmo,
            length(μ) * minimum(μ .* s) / sum(μ .* s),
            sum(μ .* s) / sqrt(MeritFunction(_N, cmo, p, μ, λ, s)),
            0.1,
        ),
        initial_stepsize=1.0,
    ),
    sub_kwargs=(;),
    sub_objective=decorate_objective!(
        TangentSpace(M × ℝ^length(λ), rand(M × ℝ^length(λ))),
        SymmetricLinearSystemObjective(
            ReducedLagrangianHess(cmo, Hess_g, Hess_h, μ, λ, s),
            NegativeReducedLagrangianGrad(cmo, μ, λ, s, ρ * σ),
        ),
        sub_kwargs...,
    ),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(20) |
                                              StopWhenGradientNormLess(1e-5),
    sub_state::Union{AbstractEvaluationType,AbstractManoptSolverState}=decorate_state!(
        ConjugateResidualState(
            TangentSpace(M × ℝ^length(λ), rand(M × ℝ^length(λ))),
            sub_objective;
            stop=sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs...,
    ),
    sub_problem::Union{F,AbstractManoptProblem}=DefaultManoptProblem(
        TangentSpace(M × ℝ^length(λ), rand(M × ℝ^length(λ))), sub_objective
    ),
    kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective},F}
    !is_feasible(M, cmo, p) && throw(ErrorException("Starting point p must be feasible."))
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointState(
        M,
        cmo,
        p,
        sub_problem,
        sub_state;
        X=X,
        μ=μ,
        λ=λ,
        s=s,
        stop=stop,
        retraction_method=retraction_method,
        stepsize=stepsize,
        kwargs...,
    )
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end

# inititializer, might add more here
function initialize_solver!(::AbstractManoptProblem, ips::InteriorPointState)
    return ips
end

# step solver
function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointState, i)
    M = get_manifold(amp)
    cmo = get_objective(amp)

    m, n = length(ips.μ), length(ips.λ)
    g = get_inequality_constraints(amp, ips.p)
    grad_g = get_grad_inequality_constraints(amp, ips.p)

    N = M × ℝ^n
    q = rand(N)
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.λ)
    TpM = TangentSpace(M, ips.p)
    TqN = TangentSpace(N, q)

    # make deterministic as opposed to random?

    set_iterate!(ips.sub_state, get_manifold(ips.sub_problem), rand(TqN))

    set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, q)

    set_manopt_parameter!(ips.sub_problem, :Objective, :μ, ips.μ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :λ, ips.λ)
    set_manopt_parameter!(ips.sub_problem, :Objective, :s, ips.s)
    set_manopt_parameter!(ips.sub_problem, :Objective, :barrier_param, ips.ρ * ips.σ)

    # product manifold on which to perform linesearch
    K = M × ℝ^m × ℝ^n × ℝ^m
    X = allocate_result(K, rand)

    Y = get_solver_result(solve!(ips.sub_problem, ips.sub_state))
    Xp, Xλ = Y[N, 1], Y[N, 2]

    if m > 0
        b = ips.ρ * ips.σ
        Xs = -[inner(M, ips.p, grad_g[i], Xp) for i in 1:m] - g - ips.s
        Xμ = (b .- ips.μ .* (ips.s + Xs)) ./ ips.s
    end

    copyto!(K[1], X[N, 1], Xp)
    (m > 0) && (copyto!(K[2], X[K, 2], Xμ))
    (n > 0) && (copyto!(K[3], X[K, 3], Xλ))
    (m > 0) && (copyto!(K[4], X[K, 4], Xs))

    α = ips.stepsize(amp, ips, i, X)

    # update params
    retract!(M, ips.p, ips.p, α * Xp, ips.retraction_method)
    if m > 0
        ips.μ += α * Xμ
        ips.s += α * Xs
        ips.ρ = ips.μ'ips.s / m
        ips.σ = calculate_σ(M, cmo, ips.p, ips.μ, ips.λ, ips.s)
    end
    (n > 0) && (ips.λ += α * Xλ)

    get_gradient!(M, ips.X, cmo, ips.p)

    return ips
end

get_solver_result(ips::InteriorPointState) = ips.p
