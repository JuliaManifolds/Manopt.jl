# struct for state of interior point algorithm
mutable struct InteriorPointState{
    P,
    Pr<:AbstractManoptProblem,
    St<:AbstractManoptSolverState,
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TStepsize<:Stepsize,
} <: AbstractGradientSolverState
    p::P
    sub_problem::Pr
    sub_state::St
    X::T # not sure if needed?
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
        p::P,
        sub_problem::Pr,
        sub_state::St;
        X::T=get_gradient(M, cmo, p), # not sure if needed?
        μ::T=ones(length(get_inequality_constraints(M, cmo, p))),
        λ::T=zeros(length(get_equality_constraints(M, cmo, p))),
        s::T=ones(length(get_inequality_constraints(M, cmo, p))),
        ρ::R=μ's / length(get_inequality_constraints(M, cmo, p)),
        σ::R = calculate_σ(M, cmo, p, μ, λ, s),
        stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        stepsize::Stepsize=InteriorPointLinesearch(
            M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s); retraction_method=default_retraction_method(
                M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s)
            ), initial_stepsize=1.0),
        kwargs...,
    ) where {P,Pr,St,T,R}
        ips = new{
            P,
            typeof(sub_problem),
            typeof(sub_state),
            T,
            R,
            typeof(stop),
            typeof(retraction_method),
            typeof(stepsize),
        }()
        ips.p = p
        ips.sub_problem = sub_problem
        ips.sub_state = sub_state
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

# get & set iterate
get_iterate(ips::InteriorPointState) = ips.p
function set_iterate!(ips::InteriorPointState, ::AbstractManifold, p)
    ips.p = p
    return ips
end
# get & set gradient (not sure if needed?)
get_gradient(ips::InteriorPointState) = ips.X
function set_gradient!(ips::InteriorPointState, ::AbstractManifold, X)
    ips.X = X
    return ips
end
# only message on stepsize for now
function get_message(ips::InteriorPointState)
    return get_message(ips.stepsize)
end
# pretty print state info
function show(io::IO, ips::InteriorPointState)
    i = get_count(ips, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ips.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Interior Point Newton Method
    $Iter
    ## Parameters
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