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
        σ::R=calculate_σ(M, cmo, p, μ, λ, s),
        stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        stepsize::Stepsize=InteriorPointLinesearch(
            M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s);
            retraction_method=default_retraction_method(
                M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s)
            ),
            initial_stepsize=1.0,
        ),
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

#
#
# A special linesearch for IP Newton
function interior_point_initial_guess(
    mp::AbstractManoptProblem, ips::InteriorPointState, ::Int, l::Real
)
    N = get_manifold(mp) × ℝ^length(ips.μ) × ℝ^length(ips.λ) × ℝ^length(ips.s)
    q = rand(N)
    copyto!(N[1], q[N, 1], ips.p)
    copyto!(N[2], q[N, 2], ips.μ)
    copyto!(N[3], q[N, 3], ips.λ)
    copyto!(N[4], q[N, 4], ips.s)
    Y = GradMeritFunction(N, get_objective(mp), q)
    grad_norm = norm(N, q, Y)
    max_step = max_stepsize(N, q)
    return ifelse(isfinite(max_step), min(l, max_step / grad_norm), l)
end

mutable struct InteriorPointLinesearch{TRM<:AbstractRetractionMethod,Q,F,DF,IF} <:
               Linesearch
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
    additional_decrease_condition::DF
    additional_increase_condition::IF
    function InteriorPointLinesearch(
        N::AbstractManifold=DefaultManifold();
        additional_decrease_condition::DF=(N, q) -> true,
        additional_increase_condition::IF=(N, q) -> true,
        candidate_point::Q=allocate_result(N, rand),
        contraction_factor::Real=0.95,
        initial_stepsize::Real=1.0,
        initial_guess=interior_point_initial_guess,
        retraction_method::TRM=default_retraction_method(N),
        stop_when_stepsize_less::Real=0.0,
        stop_when_stepsize_exceeds::Real=max_stepsize(N),
        stop_increasing_at_step::Int=0,
        stop_decreasing_at_step::Int=1000,
        sufficient_decrease=0.1,
    ) where {TRM,Q,DF,IF}
        return new{TRM,Q,typeof(initial_guess),DF,IF}(
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
            additional_decrease_condition,
            additional_increase_condition,
        )
    end
end
function (ipls::InteriorPointLinesearch)(
    mp::AbstractManoptProblem, ips::InteriorPointState, i::Int, η; kwargs...
)
    N = get_manifold(mp) × ℝ^length(ips.μ) × ℝ^length(ips.λ) × ℝ^length(ips.s)
    q = allocate_result(N, rand)
    copyto!(N[1], q[N, 1], get_iterate(ips))
    copyto!(N[2], q[N, 2], ips.μ)
    copyto!(N[3], q[N, 3], ips.λ)
    copyto!(N[4], q[N, 4], ips.s)
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
        stop_when_stepsize_exceeds=ipls.stop_when_stepsize_exceeds / norm(N, q, η),
        stop_increasing_at_step=ipls.stop_increasing_at_step,
        stop_decreasing_at_step=ipls.stop_decreasing_at_step,
        additional_decrease_condition=ipls.additional_decrease_condition,
        additional_increase_condition=ipls.additional_increase_condition,
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

function get_last_stepsize(step::InteriorPointLinesearch, ::Any...)
    return step.last_stepsize
end

#
#
# Subproblem gradient and hessian
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
    g = get_inequality_constraints(N[1], nrlg.cmo, q[N, 1])
    Jg = get_grad_inequality_constraints(N[1], nrlg.cmo, q[N, 1])
    Jh = get_grad_equality_constraints(N[1], nrlg.cmo, q[N, 1])
    X = zero_vector(N, q)
    grad = get_gradient(N[1], nrlg.cmo, q[N, 1])
    v = (nrlg.μ + (nrlg.μ .* g .+ nrlg.barrier_param) ./ nrlg.s)
    (m > 0) && (grad += sum(Jg[i] * v[i] for i in 1:m))
    (n > 0) && (grad += sum(Jh[j] * λ[j] for j in 1:n))
    copyto!(N[1], X[N, 1], grad)
    (n > 0) && (copyto!(N[2], X[N, 2], sum(Jh[j] * λ[j] for j in 1:n)))
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
    Jg = get_grad_inequality_constraints(N[1], rlh.cmo, q[N, 1])
    Jh = get_grad_equality_constraints(N[1], rlh.cmo, q[N, 1])
    X = zero_vector(N, q)
    hess = get_hessian(N[1], rlh.cmo, q[N, 1], Y[N, 1])
    (m > 0) && (hess += Jg' * Diagonal(rlh.μ ./ rlh.s) * Jg * Y[N, 1]) # plus Hess g and Hess h
    copyto!(N[1], X[N, 1], hess)
    (n > 0) && (copyto!(N[2], X[N, 2], Jh * Y[N, 1]))
    return X
end

function MeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    return MeritFunction(N, cmo, q[N, 1], q[N, 2], q[N, 3], q[N, 4])
end
function MeritFunction(N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, q)
    return MeritFunction(N, get_objective(cmo, true), q[N, 1], q[N, 2], q[N, 3], q[N, 4])
end
function MeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    m, n = length(μ), length(λ)
    gp = get_inequality_constraints(N[1], cmo, p)
    hp = get_equality_constraints(N[1], cmo, p)
    grad_gp = get_grad_inequality_constraints(N[1], cmo, p)
    grad_hp = get_grad_equality_constraints(N[1], cmo, p)
    grad_pL = get_gradient(N[1], cmo, p)
    (m > 0) && (grad_pL += sum([μ[i] * grad_gp[i]  for i in 1:m]))
    (n > 0) && (grad_pL += sum([λ[j] * grad_hp[j]  for j in 1:n]))
    ϕq = inner(N[1], p, grad_pL, grad_pL)
    (m > 0) && (ϕq += norm(gp + s)^2 + norm(μ .* s)^2)
    (n > 0) && (ϕq += norm(hp)^2)
    return ϕq
end

function calculate_σ(N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, p, μ, λ, s)
    return calculate_σ(N, get_objective(cmo, true), p, μ, λ, s)
end
function calculate_σ(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    N = M × ℝ^length(μ) × ℝ^length(λ) × ℝ^length(s)
    q = allocate_result(N, rand)
    copyto!(N[1], q[N, 1], p)
    copyto!(N[2], q[N, 2], μ)
    copyto!(N[3], q[N, 3], λ)
    copyto!(N[4], q[N, 4], s)
    return min(0.5, MeritFunction(N, cmo, q)^(1 / 4))
end

function GradMeritFunction(N::AbstractManifold, cmo::AbstractDecoratedManifoldObjective, q)
    GradMeritFunction(N, get_objective(cmo, true), q)
end
function GradMeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    p, μ, λ, s = q[N, 1], q[N, 2], q[N, 3], q[N, 4]
    m, n = length(μ), length(λ)
    gp = get_inequality_constraints(N[1], cmo, p)
    hp = get_equality_constraints(N[1], cmo, p)
    grad_gp = get_grad_inequality_constraints(N[1], cmo, p)
    grad_hp = get_grad_equality_constraints(N[1], cmo, p)

    # gradₚℒ(𝑞)
    grad_pLq = get_gradient(N[1], cmo, p)
    (m > 0) && (grad_pLq += sum([μ[i] * grad_gp[i] for i in 1:m]))
    (n > 0) && (grad_pLq += sum([λ[j] * grad_hp[j] for j in 1:n]))

    X = allocate_result(TangentSpace(N, q), rand)

    copyto!(N[1], X[N, 1], get_hessian(N[1], cmo, p, grad_pLq))
    (m > 0) && (X[N, 1] += sum([(gp + s)[i] * grad_gp[i] for i in 1:m]))
    (n > 0) && (X[N, 1] += sum([hp[j] * grad_hp[j] for j in 1:n]))
    
    (m > 0) && copyto!(N[2], X[N, 2], [inner(N[1], p, grad_gp[i], grad_pLq) for i in 1:m] + μ .* s .* s)
    (n > 0) && copyto!(N[3], X[N, 3], [inner(N[1], p, grad_hp[j], grad_pLq) for j in 1:n])
    (m > 0) && copyto!(N[4], X[N, 4], gp + s + μ .* μ .* s)
    return 2 * X
end

function is_feasible(M, cmo, p)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    return is_point(M, p) && all(g .<= 0) && all(h .== 0)
end

mutable struct ConstraintLineSearchCheckFunction{CO}
    cmo::CO
    τ1::Float64
    τ2::Float64
    γ::Float64
end
# ConstraintLineSearchCheckFunction(N::ProductManifold, q0) = ....
function (clcf::ConstraintLineSearchCheckFunction)(N, q)
    #p = q[N,1]
    μ = q[N, 2]
    #λ = q[N,3]
    s = q[N, 4]
    (minimum(μ .* s) - clcf.γ * clcf.τ1 / length(μ) < 0) && return false
    (sum(μ .* s) - clcf.γ * clcf.τ2 * sqrt(MeritFunction(N, clcf.cmo, q)) < 0) &&
        return false
    return true
end
