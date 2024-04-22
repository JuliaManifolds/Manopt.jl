mutable struct InteriorPointState{
    P,
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TStepsize<:Stepsize,
} <: AbstractManoptSolverState
    p::P
    X::T
    μ::T
    λ::T
    s::T
    ρ::R
    σ::R
    stop::TStop
    retraction_method::TRTM
    stepsize::TStepsize
    function InteriorPointState(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        p::P;
        X::T = get_gradient(M, cmo, p),
        μ::T = rand(length(get_inequality_constraints(M, cmo, p))),
        λ::T = zeros(length(get_equality_constraints(M, cmo, p))),
        s::T = rand(length(get_inequality_constraints(M, cmo, p))),
        ρ::R = μ's/length(get_inequality_constraints(M, cmo, p)),
        σ::R = rand(), 
        stop::StoppingCriterion = StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        stepsize::Stepsize = ArmijoLinesearch(
            M; retraction_method=retraction_method, initial_stepsize=1.0
            ),
    ) where{P,T,R}
        ips = new{
            P,T,R,
            typeof(stop),
            typeof(retraction_method),
            typeof(stepsize)}()
        ips.p = p
        ips.X = X
        ips.μ = μ
        ips.λ = λ
        ips.s = s
        ips.ρ = ρ
        ips.σ = σ
        ips.stop = stop
        ips.retraction_method = retraction_method
        ips.stepsize = stepsize
        return ips
    end
end

get_iterate(ips::InteriorPointState) = ips.p
function set_iterate!(ips::InteriorPointState, ::AbstractManifold, p)
    ips.p = p
    return ips
end
get_gradient(ips::InteriorPointState) = ips.X
function set_gradient!(ips::InteriorPointState, ::AbstractManifold, X)
    ips.X = X
    return ips
end

function get_message(ips::InteriorPointState)
    return get_message(ips.stepsize)
end

function show(io::IO, ips::InteriorPointState)

    i = get_count(ips, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ips.stop) ? "Yes" : "No"

    s = """
    # Solver state for `Manopt.jl`s Interior Point Newton Method
    $Iter
    ## Parameters
    * p: $(ips.p)
    * μ: $(ips.μ)
    * λ: $(ips.λ)
    * s: $(ips.s)
    * ρ: $(ips.ρ)
    * σ: $(ips.σ)

    ## Stopping criterion
    $(status_summary(ips.stop))

    * retraction method: $(ips.retraction_method)
    
    ## Stepsize
    $(ips.stepsize)
    
    This indicates convergence: $Conv
    """

    return print(io, s)
end

function interior_point_Newton(
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
    kwargs...,
)
    q = copy(M, p)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(
        mho, g, grad_g, h, grad_h; evaluation=evaluation
    )
    return interior_point_Newton!(
        M, cmo, q; evaluation=evaluation, kwargs...
    )
end

function interior_point_Newton(
    M::AbstractManifold,
    f,
    grad_f,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    g=nothing,
    grad_g=nothing,
    grad_h=nothing,
    h=nothing,
    kwargs...,
) 
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    Hess_f_ = _to_mutating_gradient(Hess_f, evaluation)
    g_ = isnothing(g) ? nothing : (M, p) -> g(M, p[])
    grad_g_ = isnothing(grad_g) ? nothing : _to_mutating_gradient(grad_g, evaluation)
    h_ = isnothing(h) ? nothing : (M, p) -> h(M, p[])
    grad_h_ = isnothing(grad_h) ? nothing : _to_mutating_gradient(grad_h, evaluation)
    mho = ManifoldHessianObjective(f_, grad_f_, Hess_f_)
    cmo = ConstrainedManifoldObjective(
        mho, g_, grad_g_, h_, grad_h_; evaluation=evaluation
    )
    rs = augmented_Lagrangian_method(M, cmo, q; evaluation=evaluation, kwargs...)
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
    
function interior_point_Newton(
    M::AbstractManifold, cmo::O, p; kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return interior_point_Newton!(M, cmo, q; kwargs...)
end

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
    kwargs...,
)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(
        mho, g, grad_g, h, grad_h; evaluation=evaluation
    )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    return interior_point_Newton!(
        M, dcmo, p; evaluation=evaluation, kwargs...
    )
end

function interior_point_Newton!(
    M::AbstractManifold, 
    cmo::O, 
    p; 
    stop::StoppingCriterion = StopAfterIteration(200) | StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod = default_retraction_method(M),
    stepsize::Stepsize = ArmijoLinesearch(M; retraction_method=retraction_method, initial_stepsize=1.0),
    kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    !is_feasible(M, cmo, p) && throw(
        ErrorException(
            "Starting point p must be feasible."
        )
    )
    ips = InteriorPointState(
        M, cmo, p;
        stop = stop,
        retraction_method = retraction_method,
        stepsize = stepsize,
        )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = decorate_state!(ips; kwargs...)
    solve!(dmp, ips)
    return get_solver_return(get_objective(dmp), ips)
end

function initialize_solver!(amp::AbstractManoptProblem, ips::InteriorPointState)
    ips.σ = calculate_σ(amp, ips)
    return ips
end

function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointState, i)

    g_p = get_inequality_constraints(amp, ips.p)
    dg_p = get_grad_inequality_constraints(amp, ips.p)

    m = length(ips.μ)
    n = length(ips.λ)

    rhs = RHS(amp, ips)
    lhs = X -> LHS(amp, ips, X)

    if n > 0
        TpM = TangentSpace(get_manifold(amp) × ℝ^m, ArrayPartition(ips.p, ips.λ))
    else
        TpM = TangentSpace(get_manifold(amp), ips.p)
    end

    mho = ManifoldHessianObjective(
        (TpM, x)    -> metric(X, lhs(x)) - metric(rhs, x),
        (TpM, x)    -> lhs(x) - rhs,
        (TpM, x, y) -> lhs(y)
    )

    if n > 0
        X_p, X_λ = conjugate_residual!(TpM, mho, rand(TpM))
    else
        X_p = conjugate_residual!(TpM, mho, rand(TpM))
    end

    X_μ = ((dg_p*X_p + g_p).*(ips.μ) .+ (ips.ρ)*(ips.σ) - ips.μ)./(ips.s)
    X_s = ((ips.ρ)*(ips.σ) .- (ips.μ).*(ips.s) - (ips.s).*X_μ) ./ (ips.μ)

    # update iterate
    
    α = get_stepsize(amp, ips, i)
    retract!(get_manifold(amp), ips.p, ips.p, X_p, ips.retraction_method)
    (m > 0) && (ips.μ += X_μ)
    (m > 0) && (ips.s += X_s)
    (n > 0) && (ips.λ += X_λ)
    ips.ρ = ips.μ'ips.s / m
    ips.σ = calculate_σ(amp, ips)

    return ips
end

get_solver_result(ips::InteriorPointState) = ips.p


#-----------------------------------------------------------------------#


function calculate_σ(amp::AbstractManoptProblem, ips::InteriorPointState)

    g_p = get_inequality_constraints(amp, ips.p)
    h_p = get_equality_constraints(amp, ips.p)
    dg_p = get_grad_inequality_constraints(amp, ips.p)
    dh_p = get_grad_equality_constraints(amp, ips.p)

    
    F_p = get_gradient(amp, ips.p)
    d = inner(get_manifold(amp), ips.p, F_p, F_p)

    m = length(g_p)
    n = length(h_p)

    (m > 0) && (d += inner(get_manifold(amp), ips.p, dg_p'ips.μ, dg_p'ips.μ) + norm(g_p+ips.s)^2 + norm(ips.μ.*ips.s)^2)
    (n > 0) && (d += inner(get_manifold(amp), ips.p, dh_p'ips.μ, dg_p'ips.λ) + norm(h_p)^2) 

    return min(0.5, d^(1/4))
end

function RHS(amp::AbstractManoptProblem, ips::InteriorPointState)

    g_p = get_inequality_constraints(amp, ips.p)
    dg_p = get_grad_inequality_constraints(amp, ips.p)
    dh_p = get_grad_equality_constraints(amp, ips.p)

    X_1 = get_gradient(amp, ips.p)

    m = length(ips.μ)
    n = length(ips.λ)

    (m > 0) && (X_1 += dg_p' * ((g_p + ips.s).*(ips.μ) .+ (ips.ρ)*(ips.σ) .- ips.μ) ./ ips.s)
    (n > 0) && (X_2 = dh_p'*(ips.λ))

    if n == 0
        return -X_1
    else
        return -ArrayPartition(X_1, X_2)
    end
end

function LHS(amp::AbstractManoptProblem, ips::InteriorPointState, X)

    dg_p = get_grad_inequality_constraints(amp, ips.p)
    dh_p = get_grad_equality_constraints(amp, ips.p)

    Y_1 = get_hessian(amp, ips.p, X)

    m = length(ips.μ)
    n = length(ips.λ)

    (m > 0) && (Y_1 += (dg_p'*dg_p*X).*(ips.μ)./(ips.s)) # plus Hess g(p)
    (n > 0) && (Y_2 = dh_p*X)

    if n == 0
        return Y_1
    else
        return ArrayPartition(Y_1, Y_2)
    end
end

function is_feasible(M, cmo, p)

    # evaluate constraint functions at p
    g_p = get_inequality_constraints(M, cmo, p)
    h_p = get_equality_constraints(M, cmo, p)

    # check feasibility
    return is_point(M, p) && all(g_p .<= 0) && all(h_p .== 0)
end