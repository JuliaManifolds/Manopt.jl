@doc raw"""
    AdaptiveRegularizationState{P,T} <: AbstractHessianSolverState

Describes ... algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `η1`, `η2`           – (`0.1`, `0.9`) bounds for evaluating the regularization parameter
* `γ1`, `γ2`           – (`0.1`, `2.0`) shrinking and exansion factors for regularization parameter `σ`
* `H`                  – (`zero_vector(M,p)`) the current hessian, $\operatorname{Hess}F(p)[⋅]$
* `p`                  – (`rand(M)` the current iterate
* `X`                  – (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``
* `s`                  - (`zero_vector(M,p)`) the tangent vector step resulting from minimizing the model
  problem in the tangent space ``\mathcal T_{p} \mathcal M``
* `σσ`                 – the current cubic regularization parameter
* `σmin`               – (`1e-7`) lower bound for the cubic regularization parameter
* `ρ`                  – the current regularized ratio of actual improvement and model improvement.
* `ρ_regularization`   – (1e3) regularization paramter for computing ρ. As we approach convergence the ρ may be difficult to compute with numerator and denominator approachign zero. Regularizing the the ratio lets ρ go to 1 near convergence.
* `retraction_method`  – (`default_retraction_method(M)`) the retraction to use
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `sub_problem`      -
* `sub_state`          - sub state for solving the sub problem
* `θ`                  – reduction factor for the norm ``\lVert Y \rVert_p`` compared to the gradient of the model.

Furthermore the following interal fields are defined

* `q`                  - (`copy(M,p)`) a point for the candidates to evaluate model and ρ

# Constructor

    AdaptiveRegularizationState(M, p=rand(M); X=zero_vector(M, p); kwargs...)


"""
mutable struct AdaptiveRegularizationState{
    P,
    T,
    Pr<:AbstractManoptProblem,
    St<:AbstractManoptSolverState,
    TStop<:StoppingCriterion,
    R,
    TRTM<:AbstractRetractionMethod,
} <: AbstractManoptSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    q::P
    H::T
    S::T
    σ::R
    ρ::R
    ρ_regularization::R
    stop::TStop
    retraction_method::TRTM
    σmin::R
    θ::R
    η1::R
    η2::R
    γ1::R
    γ2::R
end

function AdaptiveRegularizationState(
    M::AbstractManifold,
    p::P=rand(M),
    X::T=zero_vector(M, p);
    sub_state::St=LanczosState(M, p),
    sub_objective=nothing,
    sub_problem::Pr=if isnothing(sub_objective)
        nothing
    else
        DefaultManoptProblem(M, sub_objective)
    end,
    σ::R=100.0 / sqrt(manifold_dimension(M)),# Had this to inital value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ::R=1.0,
    ρ_regularization::R=1e3,
    stop::SC=StopAfterIteration(100),
    retraction_method::RTM=default_retraction_method(M),
    σmin::R=1e-10, #Set the below to appropriate default vals.
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
) where {
    P,
    T,
    R,
    Pr<:Union{<:AbstractManoptProblem,<:Function,Nothing},
    St<:Union{<:AbstractManoptSolverState,<:AbstractEvaluationType},
    SC<:StoppingCriterion,
    RTM<:AbstractRetractionMethod,
}
    isnothing(sub_problem) && error("No sub_problem provided,")

    return AdaptiveRegularizationState{P,T,Pr,St,SC,R,RTM}(
        p,
        X,
        sub_problem,
        sub_state,
        copy(M, p),
        copy(M, p, X),
        copy(M, p, X),
        σ,
        ρ,
        ρ_regularization,
        stop,
        retraction_method,
        σmin,
        θ,
        η1,
        η2,
        γ1,
        γ2,
    )
end
@doc raw"""
    adaptive_regularization_with_cubics(M, f, grad_f, Hess_f, p=rand(M); kwargs...)


"""
function adaptive_regularization_with_cubics(
    M::AbstractManifold, f::TF, grad_f::TDF, Hess_f::THF, p=rand(M); kwargs...
) where {TF,TDF,THF}
    q = copy(M, p)
    return adaptive_regularization_with_cubics!(M, f, grad_f, Hess_f, q; kwargs...)
end

function adaptive_regularization_with_cubics!(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    Hess_f::THF,
    p=rand(M);
    initial_tangent_vector::T=zero_vector(M, p),
    σ::R=100.0 / sqrt(manifold_dimension(M)),
    maxIterLanczos=200,
    ρ::R=0.0, #1.0
    ρ_regularization::R=1e3,
    stop::StoppingCriterion=StopAfterIteration(40) |
                            StopWhenGradientNormLess(1e-9) |
                            StopWhenAllLanczosVectorsUsed(maxIterLanczos - 1),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    σmin::R=1e-10,
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    sub_state::Union{<:AbstractManoptSolverState,<:AbstractEvaluationType}=LanczosState(
        M, copy(M, p); maxIterLanczos=maxIterLanczos, σ=σ
    ),
    sub_cost=ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation),
    sub_problem=DefaultManoptProblem(M, sub_cost),
    kwargs...,
) where {T,R,TF,TDF,THF}
    X = copy(M, p, initial_tangent_vector)
    mho = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        p,
        X;
        sub_state=sub_state,
        sub_problem=sub_problem,
        σ=σ,
        ρ=ρ,
        ρ_regularization=ρ_regularization,
        stop=stop,
        retraction_method=retraction_method,
        σmin=σmin,
        θ=θ,
        η1=η1,
        η2=η2,
        γ1=γ1,
        γ2=γ2,
    )
    arcs = decorate_state!(arcs; kwargs...)
    solve!(dmp, arcs)
    return get_solver_return(get_objective(dmp), arcs)
end
get_iterate(s::AdaptiveRegularizationState) = s.p
function set_iterate!(s::AdaptiveRegularizationState, p)
    s.p = p
    return s
end
get_gradient(s::AdaptiveRegularizationState) = s.X
function set_gradient!(s::AdaptiveRegularizationState, X)
    s.X = X
    return s
end

function initialize_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState)
    get_gradient!(dmp, arcs.X, arcs.p)
    return arcs
end
function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    # Update sub state
    set_iterate!(arcs.sub_state, M, copy(M, arcs.p))
    set_manopt_parameter!(arcs.sub_state, :σ, arcs.σ)
    #Solve the sub_problem – via dispatch depending on type
    solve_arc_subproblem!(M, arcs.S, arcs.sub_problem, arcs.sub_state, arcs.p)
    # Compute ρ
    retract!(M, arcs.q, arcs.p, arcs.S, arcs.retraction_method)
    cost = get_cost(M, mho, arcs.p)
    ρ_num = cost - get_cost(M, mho, arcs.q)
    ρ_vec = get_gradient(M, mho, arcs.p) + 0.5 * get_hessian(M, mho, arcs.p, arcs.S)
    ρ_den = -inner(M, arcs.p, arcs.S, ρ_vec)
    ρ_reg = arcs.ρ_regularization * eps(Float64) * max(abs(cost), 1)
    arcs.ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)

    # if  (ρ_den + ρ_reg <= 0) -> add a warning Debug that is there by default

    #Update iterate
    if arcs.ρ >= arcs.η1
        copyto!(M, arcs.p, arcs.q)
        get_gradient!(dmp, arcs.X, arcs.p) #only compute gradient when we update the point
    end
    #Update regularization parameter - in the mid interval between η1 and η2 we leave it as is
    if arcs.ρ >= arcs.η2 #very successful, reduce
        arcs.σ = max(arcs.σmin, arcs.γ1 * arcs.σ)
    elseif arcs.ρ < arcs.η1 # unsuccessful
        arcs.σ = arcs.γ2 * arcs.σ
    end
    return arcs
end

# Dispatch on different forms of sub_solvers
function solve_arc_subproblem!(
    M, s, problem::P, state::S, p
) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
    solve!(problem, state)
    copyto!(M, s, p, get_solver_result(state))
    return s
end
function solve_arc_subproblem!(
    M, s, problem::P, ::AllocatingEvaluation, p
) where {P<:Function}
    copyto!(M, p, s, problem(M, p))
    return s
end
function solve_arc_subproblem!(
    M, s, problem!::P, ::InplaceEvaluation, p
) where {P<:AbstractManoptProblem}
    problem!(M, s, p)
    return s
end

#
# Lanczos sub solver
#

@doc raw"""
    LanczosState{P,T,SC,B,I,R,TM,V,Y} <: AbstractManoptSolverState

Solve the adaptive regularized subproblem with a Lanczos iteration

# Fields
* `p` the current iterate
* `stop` – the stopping criterion
* `σ` – the current regularization parameter
* `X` the current gradient
* `Lanczos_vectors` – the obtained Lanczos vectors
* `tridig_matrix` the tridigonal coefficient matrix T
* `coefficients` the coefficients `y_1,...y_k`` that deteermine the solution
* `Hp` – a temporary vector containing the evaluation of the Hessian
* `Hp_residual` – a temporary vector containing the residual to the Hessian
* `S` – the current obtained / approximated solution
"""
mutable struct LanczosState{P,T,R,SC,SCN,B,TM,C} <: AbstractManoptSolverState
    p::P
    X::T
    σ::R
    stop::SC           # Notation in ABBC
    stop_newton::SCN
    Lanczos_vectors::B # qi
    tridig_matrix::TM  # T
    coefficients::C     # y
    Hp::T              # Hess_f A temporary vector for evaluations of the hessian
    Hp_residual::T     # A residual vector
    # Maybe not necessary?
    S::T               # store the tangent vector that solves the minimization problem
end
function LanczosState(
    M::AbstractManifold,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    maxIterLanczos=200,
    stopping_criterion::SC=StopAfterIteration(maxIterLanczos) |
                           StopWhenLanczosModelGradLess(0.5),
    stopping_criterion_newtown::SCN=StopAfterIteration(200),
    σ::R=10.0,
) where {P,T,SC<:StoppingCriterion,SCN<:StoppingCriterion,R}
    tridig = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    coeffs = zeros(maxIterLanczos)
    Lanczos_vectors = typeof(X)[]
    return LanczosState{P,T,R,SC,SCN,typeof(Lanczos_vectors),typeof(tridig),typeof(coeffs)}(
        p,
        X,
        σ,
        stopping_criterion,
        stopping_criterion_newtown,
        Lanczos_vectors,
        tridig,
        coeffs,
        copy(M, p, X),
        copy(M, p, X),
        copy(M, p, X),
    )
end
function get_solver_result(ls::LanczosState)
    return ls.S
end
function set_iterate!(ls::LanczosState, M, p)
    ls.p = p
    return ls
end
function set_manopt_parameter!(ls::LanczosState, ::Val{:p}, p)
    ls.p = p
    return ls
end
function set_manopt_parameter!(ls::LanczosState, ::Val{:σ}, σ)
    ls.σ = σ
    return ls
end

function initialize_solver!(dmp::AbstractManoptProblem, ls::LanczosState)
    M = get_manifold(dmp)
    # Maybe better to allocate once and just reset the number of vectors k?
    maxIterLanczos = size(ls.tridig_matrix, 1)
    ls.tridig_matrix = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    ls.coefficients = zeros(maxIterLanczos)
    for X in ls.Lanczos_vectors
        zero_vector!(M, X, ls.p)
    end
    zero_vector!(M, ls.Hp, ls.p)
    get_gradient!(dmp, ls.X, ls.p)
    zero_vector!(M, ls.Hp_residual, ls.p)
    return ls
end

#step solver for the LanczosState (will change to LanczosState when its done and its correct)
function step_solver!(dmp::AbstractManoptProblem, ls::LanczosState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    if i == 1 #we can easily compute the first Lanczos vector
        nX = norm(M, ls.p, ls.X)
        if length(ls.Lanczos_vectors) == 0
            push!(ls.Lanczos_vectors, ls.X ./ nX)
        else
            copyto!(M, ls.Lanczos_vectors[1], ls.p, ls.X ./ nX)
        end
        get_hessian!(M, ls.Hp, mho, ls.p, ls.Lanczos_vectors[1])
        α = inner(M, ls.p, ls.Lanczos_vectors[1], ls.Hp)
        # This is also the first coefficient in the tridigianoal matrix
        ls.tridig_matrix[1, 1] = α
        ls.Hp_residual .= ls.Hp - α * ls.Lanczos_vectors[1]
        #argmin of one dimensional model
        ls.coefficients[1] = (α - sqrt(α^2 + 4 * ls.σ * nX)) / (2 * ls.σ)
    else # i > 1
        β = norm(M, ls.p, ls.Hp_residual)
        if β > 1e-12 # Obtained new orth Lanczos long enough cf. to num stability
            if length(ls.Lanczos_vectors) < i
                push!(ls.Lanczos_vectors, ls.Hp_residual ./ β)
            else
                copyto!(M, ls.Lanczos_vectors[i], ls.p, ls.Hp_residual ./ β)
            end
        else # Generate new random vec and MGS of new vec wrt. Q
            rand!(M, ls.Hp_residual; vector_at=ls.p)
            for k in 1:(i - 1)
                ls.Hp_residual .=
                    ls.Hp_residual -
                    inner(M, ls.p, ls.Lanczos_vectors[k], ls.Hq_resudial) *
                    ls.Lanczos_vectors[k]
            end
            if length(ls.Lanczos_vectors) < i
                push!(ls.Lanczos_vectors, ls.Hp_residual ./ norm(M, ls.p, ls.Hp_residual))
            else
                copyto!(
                    M,
                    ls.Lanczos_vectors[i],
                    ls.p,
                    ls.Hp_residual ./ norm(M, ls.p, ls.Hp_residual),
                )
            end
        end
        # Update Hessian and residual
        get_hessian!(M, ls.Hp, mho, ls.p, ls.Lanczos_vectors[i])
        ls.Hp_residual .= ls.Hp - β * ls.Lanczos_vectors[i - 1]
        α = inner(M, ls.p, ls.Hp_residual, ls.Lanczos_vectors[i])
        ls.Hp_residual .= ls.Hp_residual - α * ls.Lanczos_vectors[i]
        # Update tridiagonal matric
        ls.tridig_matrix[i, i] = α
        ls.tridig_matrix[i - 1, i] = β
        ls.tridig_matrix[i, i - 1] = β
        min_cubic_Newton!(dmp, ls, i)
    end
    copyto!(M, ls.S, ls.p, sum(ls.Lanczos_vectors[k] * ls.coefficients[k] for k in 1:i))
    return ls
end
#
# Solve Lanczos sub problem
#
function min_cubic_Newton!(mp::AbstractManoptProblem, ls::LanczosState, i)
    M = get_manifold(mp)
    tol = 1e-6 # TODO: Put into a stopping criterion

    gvec = zeros(i)
    gvec[1] = norm(M, ls.p, ls.X)
    λ = opnorm(Array(@view ls.tridig_matrix[1:i, 1:i])) + 2
    T_λ = @view(ls.tridig_matrix[1:i, 1:i]) + λ * I

    λ_min = eigmin(Array(@view ls.tridig_matrix[1:i, 1:i]))
    lower_barrier = max(0, -λ_min)
    k = 0
    y = zeros(i)
    while !ls.stop_newton(mp, ls, k)
        k += 1
        y = -(T_λ \ gvec)
        ynorm = norm(y, 2)
        ϕ = 1 / ynorm - ls.σ / λ #when ϕ is "zero", y is the solution.
        if abs(ϕ) < tol * ynorm
            break
        end
        #compute the newton step
        ψ = ynorm^2
        Δy = -(T_λ) \ y
        ψ_prime = 2 * dot(y, Δy)
        # Quadratic polynomial coefficients
        p0 = 2 * ls.σ * ψ^(1.5)
        p1 = -2 * ψ - λ * ψ_prime
        p2 = ψ_prime
        #Polynomial roots
        r1 = (-p1 + sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)
        r2 = (-p1 - sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)

        Δλ = max(r1, r2) - λ

        if λ + Δλ <= lower_barrier #if we jumped past the lower barrier for λ, jump to midpoint between current and lower λ.
            Δλ = -0.5 * (λ - lower_barrier)
        end
        if abs(Δλ) <= eps(λ) #if the steps we make are to small, terminate -> Stopping criterion?
            break
        end
        T_λ = T_λ + Δλ * I
        λ = λ + Δλ
    end
    ls.coefficients[1:i] .= y
    return ls.coefficients
end

#
# Stopping Criteria
#
mutable struct StopWhenLanczosModelGradLess <: StoppingCriterion
    relative_threshold::Float64
    reason::String
    StopWhenLanczosModelGradLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenLanczosModelGradLess)(
    dmp::AbstractManoptProblem, ls::LanczosState, i::Int
)
    if (i == 0)
        c.reason = ""
        return false
    end
    #Update Gradient
    M = get_manifold(dmp)
    get_gradient!(dmp, ls.X, ls.p)
    y = @view(ls.coefficients[1:i])
    model_grad_norm = norm(
        norm(M, ls.p, ls.X) .* ones(i + 1) +
        @view(ls.tridig_matrix[1:(i + 1), 1:i]) * y +
        ls.σ * norm(y) * [y..., 0],
    )
    if (i > 0) && model_grad_norm <= c.relative_threshold * norm(y, 2)^2
        c.reason = "The algorithm has reduced the model grad norm by $(c.relative_threshold).\n"
        return true
    end
    return false
end

#A new stopping criterion that deals with the scenario when a step needs more Lanczos vectors than preallocated.
#Previously this would just cause an error due to out of bounds error. So this stopping criterion deals both with the scenario
#of too few allocated vectors and stagnation in the solver.
mutable struct StopWhenAllLanczosVectorsUsed <: StoppingCriterion
    maxInnerIter::Int64
    reason::String
    StopWhenAllLanczosVectorsUsed(maxIts::Int64) = new(maxIts, "")
end
function (c::StopWhenAllLanczosVectorsUsed)(
    ::AbstractManoptProblem,
    arcs::AdaptiveRegularizationState{P,T,Pr,<:LanczosState},
    i::Int,
) where {P,T,Pr}
    (i == 0) && (c.reason = "") # reset on init
    if (i > 0) && size(arcs.sub_state.tridig_matrix, 1) == c.maxInnerIter
        c.reason = "The algorithm used all preallocated Lanczos vectors and may have stagnated. Allocate more by variable maxIterLanczos.\n"
        return true
    end
    return false
end

#
# Old code, not yet reviewed nor reworked.
#

#
# Nowehere used?
# Maybe use in a ARCNewtonState?

mutable struct CubicSubCost{Y,T,I,R}
    k::I #number of Lanczos vectors
    gradnorm::R
    Tmatrix::T #submatrix
    y::Y # Solution of of argmin m(s), s= sum y[i]q[i]
end
function (C::CubicSubCost)(::AbstractManifold, y)# Ronny: M is Euclidean (R^k) but p should be y. I can change it to y a just input c.y when computing the subcost
    #C.y[1]*C.gradnorm + 0.5*dot(C.y[1:C.k],@view(C.Tmatrix[1:C.k,1:C.k])*C.y[1:C.k]) + C.σ/3*norm(C.y[1:C.k],2)^3
    return y[1] * C.gradnorm +
           0.5 * dot(y, @view(C.Tmatrix[1:(C.k), 1:(C.k)]) * y) +
           C.σ / 3 * norm(y, 2)^3
end

#Sub cost set_manopt_parameter!'s
function set_manopt_parameter!(s::CubicSubCost, ::Val{:k}, k)
    s.k = k
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:gradnorm}, gradnorm)
    s.gradnorm = gradnorm
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:σ}, σ)
    s.σ = σ
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:Tmatrix}, Tmatrix)
    s.Tmatrix = Tmatrix
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:y}, y)
    s.y = y
    return s
end
