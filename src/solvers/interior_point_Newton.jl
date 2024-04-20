mutable struct InteriorPointState{
    P,
    T,
    R,
    # Pr<:AbstractManoptProblem,
    # St<:AbstractManoptSolverState,
    TStop<:StoppingCriterion,
    TStepsize<:Stepsize,
    TRTM<:AbstractRetractionMethod
} <: AbstractManoptSolverState
    p::P
    X::T
    μ::T
    λ::T
    s::T
    ρ::R
    σ::R
    # sub_problem::Pr
    # sub_state::St
    stop::TStop
    retraction_method::TRTM
    stepsize::TStepsize    
    function InteriorPointState(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        p::P;
        X::T = get_gradient(M, cmo, p),
        # sub_problem::Pr,
        # sub_state::St, 
        μ::T = rand(length(get_inequality_constraints(M, cmo, p))),
        λ::T = zeros(length(get_equality_constraints(M, cmo, p))),
        s::T = rand(length(get_inequality_constraints(M, cmo, p))),
        ρ::R = μ's/length(get_inequality_constraints(M, cmo, p)),
        σ::R = calculate_σ(M, cmo, p, μ, λ, s),
        stop::StoppingCriterion = StopAfterIteration(200) | StopWhenChangeLess(1e-5),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        stepsize::Stepsize = ArmijoLinesearch(M; retraction_method=retraction_method, initial_stepsize=1.0),
        kwargs...,
    ) where{P,T,R}
        ips = new{P,T,R,typeof(stop),typeof(stepsize),typeof(retraction_method)}()
        ips.p = p
        ips.X = X
        ips.μ = μ
        ips.λ = λ
        ips.s = s
        ips.ρ = ρ
        ips.σ = σ
        ips.stop = stop
        ips.stepsize = stepsize
        ips.retraction_method = retraction_method
        return ips
    end
end

get_iterate(ips::InteriorPointState) = ips.p
function set_iterate!(ips::InteriorPointState, ::AbstractManifold, p)
    ips.p = p
    return ips
end

get_gradient(ips::InteriorPointState) = ips.X
function set_gradient!(ips::InteriorPointState, M, p, X)
    copyto!(M, ips.X, p, X)
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
    * retraction method: $(ips.retraction_method)

    ## Stepsize
    $(ips.stepsize)

    ## Stopping criterion

    $(status_summary(ips.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

# Do this one on the DefaultProblem & IPState ??? better with current inputs?
function calculate_σ(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)

    g_p = get_inequality_constraints(M, cmo, p)
    h_p = get_equality_constraints(M, cmo, p)
    dg_p = get_grad_inequality_constraints(M, cmo, p)
    dh_p = get_grad_equality_constraints(M, cmo, p)

    
    F_p = get_gradient(M, cmo, p)
    d = inner(M, p, F_p, F_p)

    m = length(g_p)
    n = length(h_p)

    (m > 0) && (d += inner(M, p, dg_p'μ, dg_p'μ) + norm(g_p+s)^2 + norm(μ.*s)^2)
    (n > 0) && (d += inner(M, p, dh_p'μ, dg_p'λ) + norm(h_p)^2) 

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
        return X_1
    else
        return ArrayPartition(X_1, X_2)
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

# obnjugate residudal method, inpace
#  subsolver!(M, x, p, A, x0 b)

# obnjugate residudal method,
function subsolver(M::AbstractManifold, p, A, b)

    # iteration count
    k = 0

    # random initial vector
    x_0 = rand(M, vector_at = p)

    #initial residual
    r_0 = b - A(x_0)

    q_0 = r_0
    Ar_0 = A(r_0)
    Aq_0 = A(q_0)

    r = r_0
    q = q_0
    Ar = Ar_0
    Aq = Aq_0

    x = x_0

    metric = (X, Y) -> inner(M, p, X, Y)

    print("k = 0: ", metric(r, r), '\n')

    while metric(r, r) > 1e-5

        α = metric(r, Ar) / metric(Aq, Aq)
        x += α*q

        k += 1
        
        r_next = r - α*Aq
        print("k = ", k, ": ", metric(r_next, r_next), '\n')
        Ar_next = A(r_next)
        β = metric(r_next, Ar_next) / metric(r, Ar)
        q_next = r_next + β*q
        Aq_next = Ar_next + β*Aq

        r  = r_next
        q  = q_next
        Ar = Ar_next
        Aq = Aq_next
    end
    return x
end

function interior_point_Newton!(
    M::AbstractManifold, cmo::O, p; kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}
    !is_feasible(M, cmo, p) && throw(
        ErrorException(
            "Starting point p must be feasible."
        )
    )
    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = InteriorPointState(M, cmo, p; kwargs...)
    dips = decorate_state!(ips)
    solve!(dmp, dips)
    return get_solver_return(get_objective(dmp), ips)
end

function initialize_solver!(::AbstractManoptProblem, ips::InteriorPointState)
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
        X_p, X_λ = subsolver(get_manifold(amp) × ℝ^m, ArrayPartition(ips.p, ips.λ), lhs, -rhs).x
    else
        X_p = subsolver(get_manifold(amp), ips.p, lhs, -rhs)
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
    ips.σ = calculate_σ(get_manifold(amp), get_objective(amp), ips.p, ips.μ, ips.λ, ips.s)

    return ips
end