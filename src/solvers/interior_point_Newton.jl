# struct for state of interior point algorithm
mutable struct InteriorPointState{
    P,
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    TStepsize<:Stepsize
} <: AbstractGradientSolverState
    p::P
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
        p::P;
        X::T=get_gradient(M, cmo, p), # not sure if needed?
        μ::T=rand(length(get_inequality_constraints(M, cmo, p))),
        λ::T=zeros(length(get_equality_constraints(M, cmo, p))),
        s::T=rand(length(get_inequality_constraints(M, cmo, p))),
        ρ::R=μ's / length(get_inequality_constraints(M, cmo, p)),
        σ::R=rand(),
        stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-8),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        stepsize::Stepsize=ArmijoLinesearch(
            M; retraction_method=retraction_method, initial_stepsize=1.0
        ),
    ) where {P,T,R}

        ips = new{P,T,R,typeof(stop),typeof(retraction_method),typeof(stepsize)}()
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

# not-in-place,
# takes M, f, grad_f, Hess_f and possibly constreint functions and their graidents
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
    cmo = ConstrainedManifoldObjective(mho, g, grad_g, h, grad_h; evaluation=evaluation)
    return interior_point_Newton!(M, cmo, q; evaluation=evaluation, kwargs...)
end

# not-in-place
# case where dim(M) = 1 and thus p is a number
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
    cmo = ConstrainedManifoldObjective(mho, g_, grad_g_, h_, grad_h_; evaluation=evaluation)

    rs = interior_point_Newton(M, cmo, q; evaluation=evaluation, kwargs...)

    return (typeof(q) == typeof(rs)) ? rs[] : rs
end

# not-in-place
# takes only manifold, constraine objetive and initial point
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
    kwargs...,
)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    cmo = ConstrainedManifoldObjective(mho, g, grad_g, h, grad_h; evaluation=evaluation)
    dcmo = decorate_objective!(M, cmo; kwargs...)

    return interior_point_Newton!(M, dcmo, p; evaluation=evaluation, kwargs...)
end

# MAIN SOLVER
function interior_point_Newton!(
    M::AbstractManifold,
    cmo::O,
    p;
    stop::StoppingCriterion=StopAfterIteration(200) | StopWhenChangeLess(1e-5),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stepsize::Stepsize=ArmijoLinesearch(
        M; retraction_method=retraction_method, initial_stepsize=1.0
    ),
    kwargs...,
) where {O<:Union{ConstrainedManifoldObjective,AbstractDecoratedManifoldObjective}}

    !is_feasible(M, cmo, p) && throw(ErrorException("Starting point p must be feasible."))

    ips = InteriorPointState(
        M, cmo, p; stop=stop, retraction_method=retraction_method, stepsize=stepsize
    )

    dcmo = decorate_objective!(M, cmo; kwargs...)
    dmp = DefaultManoptProblem(M, dcmo)
    ips = decorate_state!(ips; kwargs...)

    solve!(dmp, ips)

    return get_solver_return(get_objective(dmp), ips)

end

# inititializer, might add more here
function initialize_solver!(amp::AbstractManoptProblem, ips::InteriorPointState)
    ips.σ = calculate_σ(amp, ips)
    return ips
end

# step solver
function step_solver!(amp::AbstractManoptProblem, ips::InteriorPointState, i)

    # state parameters
    p, μ, λ, s = ips.p, ips.μ, ips.λ, ips.s
    b = ips.ρ * ips.σ

    # get inequality constraints and their gradients
    g = get_inequality_constraints(amp, p)
    Jg = get_grad_inequality_constraints(amp, p)

    # constraint dimensions
    m, n = length(μ), length(λ)

    # if equality constrains are present, work on product manifold
    if n == 0
        M = get_manifold(amp)
    else
        M = get_manifold(amp) × ℝ^n
        p = ArrayPartition(p, λ)
    end

    # tangent space
    TpM = TangentSpace(M, p)
    # Instead
    # set_manopt_parameter!(ips.sub_problem, :Manifold, :Basepoint, p)
    # set_iterate!(ips.sub_state, TpM, rand(TpM))
    # new_p = get_solver_result(solve!(alms.sub_problem, alms.sub_state))

    # get left- and right-hand side of Newton eq
    A, B = get_Newtons(amp, ips)

    # objective for subsolver
    mho = ManifoldHessianObjective(
        (TpM, X) -> inner(M, p, X, A(X)) - inner(M, p, B, X),
        (TpM, X) -> A(X) - B,
        (TpM, X, Y) -> A(Y),
    )

    # get subsolver result
    X = conjugate_residual!(TpM, mho, rand(TpM))

    # get either one or two tangent vectors depending on if equality constrains are present
    if n > 0
        Xp, Xλ = X
    else
        Xp = X
    end

    # stepsize
    α = get_stepsize(amp, ips, i)

    # update p
    retract!(get_manifold(amp), ips.p, ips.p, α * Xp, ips.retraction_method)

    # update μ, s and aux parameters
    if m > 0
        Xμ = (μ .* (Jg * Xp .+ g)) ./ s
        Xs = b ./ μ - s - s .* Xμ ./ μ

        ips.μ += α * Xμ
        ips.s += α * Xs

        ips.ρ = ips.μ'ips.s / m
        ips.σ = calculate_σ(amp, ips)
    end

    # update λ
    (n > 0) && (ips.λ += α * Xλ)

    return ips
end

get_solver_result(ips::InteriorPointState) = ips.p

#-----------------------------------------------------------------------#

# calculates σ for a given state, ref Lai & Yoshise first paragraph Section 8.1
# might add more calculation methods for σ
function calculate_σ(amp::AbstractManoptProblem, ips::InteriorPointState)

    M = get_manifold(amp)

    p, μ, λ, s = ips.p, ips.μ, ips.λ, ips.s

    m, n = length(μ), length(λ)

    g = get_inequality_constraints(amp, p)
    h = get_equality_constraints(amp, p)
    dg = get_grad_inequality_constraints(amp, p)
    dh = get_grad_equality_constraints(amp, p)

    F = get_gradient(amp, p)
    d = inner(M, p, F, F)

    (m > 0) && (d += inner(M, p, dg'μ, dg'μ) + norm(g + s)^2 + norm(μ .* s)^2)
    (n > 0) && (d += inner(M, p, dh'μ, dh'λ) + norm(h)^2)

    return min(0.5, d^(1/4))

end

# returns left- and right-hand sides of Newton equation
# to be used as input for subsolver
function get_Newtons(amp::AbstractManoptProblem, ips::InteriorPointState)

    p, μ, λ, s = ips.p, ips.μ, ips.λ, ips.s
    b = ips.ρ .* ips.σ

    m, n = length(μ), length(λ)

    g = get_inequality_constraints(amp, p)
    h = get_inequality_constraints(amp, p)
    Jg = get_grad_inequality_constraints(amp, p)
    Jh = get_grad_equality_constraints(amp, p)

    # THIS IS ONLY FOR THE CASE OF NO EQUALITY CONSTRAINTS
    A = X -> get_hessian(amp, p, X) + Jg' * Diagonal(μ ./ s) * Jg * X
    b = -get_gradient(amp, p) - Jg' * (μ + (μ .* g .+ b) ./ s)

    # idea below: build the function A like a variable
    # don't seem to get convergence this way, why?

    # if m > 0
    #     A_ = A
    #     A = X -> A_(X) + Jg'*Diagonal(μ./s)*Jg*X
    #     b -= Jg'*(μ + (μ.*g .+ b)./s)
    # end

    # if n > 0
    #     A_ = A
    #     A = X -> ArrayPartition(A_(X), Jh*X)
    #     b = ArrayPartition(b - Jh'λ, Jh'λ)
    # end

    return A, b

end

# similarly
# struct ReducedLagrangianCost{H,V,R}
# struct ReducedLagrangianGrad{H,V,R}
struct ReducedLagrangianHessian{H,V,R}
    mho::H
    μ::V
    λ::V
    s::V
    barrier::R
end
# This replaces A
# function (L::ReducedLagrangianHessian)(TpM::AbstractManifold, p, X)
#    Y = zero_vector(base_manifold(TpM), p)
#    (m > 0) && (Y .+= lk    hwefjkqhwbef)
# end


function is_feasible(M, cmo, p)

    # evaluate constraint functions at p
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)

    # check feasibility
    return is_point(M, p) && all(g .<= 0) && all(h .== 0)

end
