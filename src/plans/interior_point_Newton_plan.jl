include("../solvers/interior_point_Newton.jl")

function interior_point_initial_guess(
    mp::AbstractManoptProblem, ips::InteriorPointState, ::Int, l::Real
)
    N = get_manifold(mp) × ℝ^length(ips.μ) × ℝ^length(ips.λ) × ℝ^length(ips.s)
    q = rand(N)
    copyto!(N[1], q[N,1], ips.p)
    copyto!(N[2], q[N,2], ips.μ)
    copyto!(N[3], q[N,3], ips.λ)
    copyto!(N[4], q[N,4], ips.s)
    Y = GradMeritFunction(N, get_objective(mp), q)
    grad_norm = norm(N, q, Y)
    max_step = max_stepsize(N, q)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

mutable struct InteriorPointLinesearch{TRM<:AbstractRetractionMethod,Q,F} <: Linesearch
    candidate_point::Q
    contraction_factor::Float64
    initial_guess::F
    initial_stepsize::Float64
    last_stepsize::Float64
    message::String
    retraction_method::TRM
    sufficient_decrease::Float64
    stop_when_stepsize_less::Float64
    stop_when_stepsize_exceeds::Float64
    stop_increasing_at_step::Int
    stop_decreasing_at_step::Int
    function InteriorPointLinesearch(
        N::AbstractManifold=DefaultManifold();
        candidate_point::Q=allocate_result(N, rand),
        contraction_factor::Real=0.95,
        initial_stepsize::Real=1.0,
        initial_guess=interior_point_initial_guess,
        retraction_method::TRM=default_retraction_method(N),
        stop_when_stepsize_less::Real=0.0,
        stop_when_stepsize_exceeds::Real=max_stepsize(N),
        stop_increasing_at_step::Int=100,
        stop_decreasing_at_step::Int=1000,
        sufficient_decrease=0.1,
    ) where {TRM,Q}
        return new{TRM,Q,typeof(initial_guess)}(
            candidate_point,
            contraction_factor,
            initial_guess,
            initial_stepsize,
            initial_stepsize,
            "", # initialize an empty message
            retraction_method,
            sufficient_decrease,
            stop_when_stepsize_less,
            stop_when_stepsize_exceeds,
            stop_increasing_at_step,
            stop_decreasing_at_step,
        )
    end
end
function (ipls::InteriorPointLinesearch)(
    mp::AbstractManoptProblem,
    ips::InteriorPointState,
    i::Int,
    η;
    kwargs...,
)
    N = get_manifold(mp) × ℝ^length(ips.μ) × ℝ^length(ips.λ) × ℝ^length(ips.s)
    q = allocate_result(N, rand)
    copyto!(N[1], q[N,1], get_iterate(ips))
    copyto!(N[2], q[N,2], ips.μ)
    copyto!(N[3], q[N,3], ips.λ)
    copyto!(N[4], q[N,4], ips.s)
    X = GradMeritFunction(N, get_objective(mp), q)
    (ipls.last_stepsize, ipls.message) = linesearch_backtrack!(
        N,
        ipls.candidate_point,
        (N, q) -> MeritFunction(N, get_objective(mp), q),
        q,
        X,
        ipls.initial_guess(mp, ips, i, ipls.last_stepsize),
        ipls.sufficient_decrease,
        ipls.contraction_factor,
        η;
        retraction_method=ipls.retraction_method,
        stop_when_stepsize_less=ipls.stop_when_stepsize_less / norm(N, q, η),
        stop_when_stepsize_exceeds=ipls.stop_when_stepsize_exceeds /
                                   norm(N, q, η),
        stop_increasing_at_step=ipls.stop_increasing_at_step,
        stop_decreasing_at_step=ipls.stop_decreasing_at_step,
    )
    return ipls.last_stepsize
end
get_initial_stepsize(ipls::InteriorPointLinesearch) = ipls.initial_stepsize
function show(io::IO, ipls::InteriorPointLinesearch)
    return print(
        io,
        """
        InteriorPointLinesearch() with keyword parameters
          * initial_stepsize    = $(ipls.initial_stepsize)
          * retraction_method   = $(ipls.retraction_method)
          * contraction_factor  = $(ipls.contraction_factor)
          * sufficient_decrease = $(ipls.sufficient_decrease)""",
    )
end
function status_summary(ipls::InteriorPointLinesearch)
    return "$(ipls)\nand a computed last stepsize of $(ipls.last_stepsize)"
end
get_message(ipls::InteriorPointLinesearch) = ipls.message

function get_last_stepsize(
    ::AbstractManoptProblem,
    ::AbstractManoptSolverState,
    step::InteriorPointLinesearch,
    ::Any...,
)
    return step.last_stepsize
end

mutable struct NegativeReducedLagrangianGrad{T,R}
    cmo::ConstrainedManifoldObjective
    μ::T
    λ::T
    s::T
    barrier_param::R
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:μ}, μ)
    nrlg.μ = μ
    return nrlg
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:λ}, λ)
    nrlg.λ = λ
    return nrlg
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:s}, s)
    nrlg.s = s
    return nrlg
end

function set_manopt_parameter!(
    nrlg::NegativeReducedLagrangianGrad, ::Val{:barrier_param}, barrier_param
)
    nrlg.barrier_param = barrier_param
    return nrlg
end

function (nrlg::NegativeReducedLagrangianGrad)(N::AbstractManifold, q)
    m, n = length(nrlg.μ), length(nrlg.λ)
    g = get_inequality_constraints(N[1], nrlg.cmo, q[N,1])
    Jg = get_grad_inequality_constraints(N[1], nrlg.cmo, q[N,1])
    Jh = get_grad_equality_constraints(N[1], nrlg.cmo, q[N,1])
    X = zero_vector(N, q)
    grad = get_gradient(N[1], nrlg.cmo, q[N,1])
    (m > 0) && (grad += Jg' * (nrlg.μ + (nrlg.μ .* g .+ nrlg.barrier_param) ./ nrlg.s))
    (n > 0) && (grad +=  Jh'*λ)
    copyto!(N[1], X[N,1], grad)
    (n > 0) && (copyto!(N[2], X[N,2], Jh'*λ))
    return -X
end

mutable struct ReducedLagrangianHess{T}
    cmo::ConstrainedManifoldObjective
    μ::T
    λ::T
    s::T
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:μ}, μ)
    rlh.μ = μ
    return rlh
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:λ}, λ)
    rlh.λ = λ
    return rlh
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:s}, s)
    rlh.s = s
    return rlh
end

function (rlh::ReducedLagrangianHess)(N::AbstractManifold, q, Y)
    TqN = TangentSpace(N, q)
    m, n = length(rlh.μ), length(rlh.λ)
    Jg = get_grad_inequality_constraints(N[1], rlh.cmo, q[N,1])
    Jh = get_grad_equality_constraints(N[1], rlh.cmo, q[N,1])
    X = zero_vector(N, q)
    hess = get_hessian(N[1], rlh.cmo, q[N,1], Y[N,1])
    (m > 0) && (hess += Jg' * Diagonal(rlh.μ ./ rlh.s) * Jg * Y[N,1]) # plus Hess g and Hess h
    copyto!(N[1], X[N,1], hess)
    (n > 0) && (copyto!(N[2], X[N,2], Jh*Y[N,1]))
    return X
end

function MeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    p, μ, λ, s = q.x
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(N[1], cmo, p)
    h = get_equality_constraints(N[1], cmo, p)
    dg = get_grad_inequality_constraints(N[1], cmo, p)
    dh = get_grad_equality_constraints(N[1], cmo, p)
    F = get_gradient(N[1], cmo, p)
    (m > 0) && (F += dg'μ)
    (n > 0) && (F += dh'λ)
    d = inner(N[1], p, F, F)
    (m > 0) && (d += norm(g + s)^2 + norm(μ .* s)^2)
    (n > 0) && (d += norm(h)^2)
    return d
end

function calculate_σ(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    N = M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s)
    q = allocate_result(N, rand)
    copyto!(N[1], q[N,1], p)
    copyto!(N[2], q[N,2], μ)
    copyto!(N[3], q[N,3], λ)
    copyto!(N[4], q[N,4], s)
    return min(0.5, MeritFunction(N, cmo, q)^(1/4))
    
    G = NegativeReducedLagrangianGrad(
        get_objective(amp), ips.μ, ips.λ, ips.s, ips.ρ*ips.σ
    )
    
    return minG(N, q)
end

function GradMeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    p, μ, λ, s = q.x
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(N[1], cmo, p)
    h = get_equality_constraints(N[1], cmo, p)
    dg = get_grad_inequality_constraints(N[1], cmo, p)
    dh = get_grad_equality_constraints(N[1], cmo, p)
    grad = get_gradient(N[1], cmo, p)
    X = zero_vector(N, q)
    (m > 0) && (grad += dg'μ)
    (n > 0) && (grad += dh'λ)
    copyto!(N[1], X[N,1], get_hessian(N[1], cmo, p, grad))
    (m > 0) && copyto!(N[2], X[N,2], dg*grad + μ .* s)
    (n > 0) && copyto!(N[3], X[N,3], dh*grad)
    (m > 0) && copyto!(N[4], X[N,4], s .* (g+s) + μ .* μ .* s)
    return 2*X
end

function is_feasible(M, cmo, p)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    return is_point(M, p) && all(g .<= 0) && all(h .== 0)
end