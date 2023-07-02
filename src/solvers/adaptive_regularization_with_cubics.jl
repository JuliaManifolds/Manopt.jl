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
    TStop<:StoppingCriterion,
    SSt<:AbstractManoptSolverState,
    SPr<:AbstractManoptProblem,
    R,
    TRTM<:AbstractRetractionMethod,          #,
} <: AbstractManoptSolverState#AbstractHessianSolverState
    p::P
    q::P
    sub_state::SSt
    sub_problem::SPr
    X::T
    H::T
    S::T
    σ::R
    ρ::R
    ρ_regularization::R
    stop::TStop
    retraction_method::TRTM
    #   optimized_updating_rule::Bool
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
    sub_problem::Ob=ManifoldCostObjective(subcost),
    H::T=zero_vector(M, p),
    S::T=zero_vector(M, p),
    σ::R=100.0 / sqrt(manifold_dimension(M)),# Had this to inital value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ::R=1.0,
    ρ_regularization::R=1e3,
    stop::SC=StopAfterIteration(100),
    retraction_method::RTM=default_retraction_method(M),
    optimized_updating_rule::Bool=false,
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
    St<:Union{<:AbstractManoptSolverState,<:Function},
    Ob<:Union{<:AbstractManifoldObjective,<:AbstractEvaluationType},
    SC<:StoppingCriterion,
    RTM<:AbstractRetractionMethod,
}
    return AdaptiveRegularizationState{P,St,SCO,SPR,T,R,RTM}(
        M,
        p,
        copy(M, p),
        sub_state,
        subcost,
        sub_problem,
        X,
        H,
        S,
        σ,
        ρ,
        ρ_regularization,
        stop,
        retraction_method,
        optimized_updating_rule,
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
    X::T=zero_vector(M, p),
    H::T=zero_vector(M, p),
    S::T=zero_vector(M, p),
    σ::R=100.0 / sqrt(manifold_dimension(M)),
    maxIterLanczos=200,
    ρ::R=0.0, #1.0
    ρ_regularization::R=1e3,
    stop::StoppingCriterion=StopAfterIteration(40) |
                            StopWhenGradientNormLess(1e-9) |
                            StopWhenAllLanczosVectorsUsed(maxIterLanczos - 1),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    optimized_updating_rule::Bool=false,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    σmin::R=1e-10,
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    sub_state::Union{<:AbstractManoptSolverState,<:AbstractEvaluationType}=LanczosState(
        M,
        copy(M, p);       #tried adding copy
        maxIterLanczos=maxIterLanczos,
        θ=θ,
        σ=σ,
    ),
    subcost=ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation),
    sub_problem=DefaultManoptProblem(M, ManifoldCostObjective(subcost)),
    kwargs...,
) where {T,R,TF,TDF,THF}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        p;
        sub_state=sub_state,
        subcost=subcost,
        sub_problem=sub_problem,
        X=X,
        H=H,
        S=S,
        σ=σ,
        ρ=ρ,
        ρ_regularization=ρ_regularization,
        stop=stop,
        retraction_method=retraction_method,
        optimized_updating_rule=optimized_updating_rule,
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

function initialize_solver!(dmp::AbstractManoptProblem, s::AdaptiveRegularizationState)
    get_gradient!(dmp, s.X, s.p)
    return s
end
function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, i) #old dummy
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
        copyto!(M, arcs.p, arsc, q)
        get_gradient!(dmp, arcs.X, arcs.p) #only compute gradient when we update the point
        #retract(M, arcs.p, arcs.S) #changed to .=
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
function solve_arc_subproblem(
    M, s, problem::P, state::S, p
) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
    solve!(problem, state)
    copyto!(M, p, s, get_solver_result(arcs.sub_problem))
    return s
end
function solve_arc_subproblem(
    M, s, problem::P, ::AllocatingEvaluation, p
) where {P<:Function}
    copyto!(M, p, s, problem(M, p))
    return s
end
function solve_arc_subproblem(
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
* `coeffcients` the coefficients `y_1,...y_k`` that deteermine the solution
* `Y` – a temporary vector containing the evaluation of the Hessian
* `S` – the current obtained / approximated solution
"""
mutable struct LanczosState{P,T,R,SC,B,TM,C} <: AbstractManoptSolverState
    p::P
    X::T
    σ::R
    stop::SC
    Lanczos_vectors::B # qi
    tridig_matrix::TM # T
    coeffcients::C
    Y::T # A temporary vector for evaluations of the hessian
    S::T # store the tangent vector that solves the minimization problem
end
function LanczosState(
    M::AbstractManifold,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    maxIterLanczos=200,
    stopping_criterion::SC=StopAfterIteration(maxIterLanczos) |
                           StopWhenLanczosModelGradLess(0.5),
    σ::R=10.0,
) where {P,T,SC<:StoppingCriterion,R}
    tridig = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    coeffs = zeros(maxIterLanczos)
    Lanczos_vectors = [zero_vector(M, p) for _ in 1:maxIterLanczos]
    return LanczosState{P,T,R,SC,typeof(Lanczos_vectors),typeof(tridig),typeof(coeffs)}(
        p,
        X,
        σ,
        stopping_criterion,
        Lanczos_vectors,
        T,
        coeffs,
        copy(M, p, X),
        copy(M, p, X),
    )
end
get_solver_result(ls::LanczosState) = ls.S

function initialize_solver!(dmp::AbstractManoptProblem, ls::LanczosState)
    # Ronny: No let's to the first iteration in the first step.
    # adapt to follow closer to the Paper by Cartis, Boumal et al.
    #in the intialization we set the first orthonormal vector, the first element of the Tmatrix and the r vector.
    M = get_manifold(dmp)
    mho = ls.objective

    g = get_gradient(M, mho, ls.p)   #added ! and s.X
    ls.gradnorm = norm(M, ls.p, g)

    #q = g / s.gradnorm
    #s.Q[1] .= q #store it directly above
    ls.Q[1] .= g / ls.gradnorm

    r = get_hessian(M, mho, ls.p, ls.Q[1])#changed from q to       #save memory here use s.r, and below use @. s.r = s.r -...
    α = inner(M, ls.p, ls.Q[1], r) #q change
    ls.Tmatrix[1, 1] = α
    ls.r = r - α * ls.Q[1] #q change

    #idea in the initalize_solver we set dim of subspace sol to d=1.

    #argmin of one dimensional model
    ls.y = [(α - sqrt(α^2 + 4 * ls.σ * ls.gradnorm)) / (2 * ls.σ)] # store y in the state.
    return ls
end

#step solver for the LanczosState (will change to LanczosState when its done and its correct)
function step_solver!(dmp::AbstractManoptProblem, ls::LanczosState, j)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    β = norm(M, ls.p, ls.r)
    #Had to move it here to avoid logic error: earlier we computed only new y if stopping criterion failed, however this would lead to updating the y in s.y, and when the stopping_criterion is called after the step, it would be checked with a new y.
    if j > 1
        ls.y = min_cubic_Newton(ls, j)
    end

    #Note: not doing MGS causes fast loss of orthogonality. Do full orthogonalization for robustness?
    if β > 1e-12  # β large enough-> Do regular procedure: MGS of r wrt. Q
        ls.Q[j + 1] .= project(M, ls.p, ls.r / β) #s.r/β
    #for i in 1:j
    #    s.r=s.r-inner(M,s.p,s.Q[i],s.r)*s.Q[i]
    #end
    # s.Q[j + 1] .= project(M,s.p,s.r/norm(M,s.p,s.r))    #q=r/norm(M,s.p,r) #/β                                      #s.r / β # project(M::Grassmann, p, X)
    else # Generate new random vec and MGS of new vec wrt. Q
        println("maxed out! gen rand vec")
        r = rand(M; vector_at=ls.p)
        for i in 1:j
            r .= r - inner(M, ls.p, ls.Q[i], r) * ls.Q[i]  #use @.
        end
        ls.Q[j + 1] .= project(M, ls.p, r / norm(M, ls.p, r))  #r / norm(M, s.p, r)                            # r / norm(M, s.p, r)
    end

    rh = get_hessian(M, mho, ls.p, ls.Q[j + 1])
    r = rh - β * ls.Q[j] #also store this in s.r to save memory
    α = inner(M, ls.p, r, ls.Q[j + 1])
    ls.r = r - α * ls.Q[j + 1]

    ls.Tmatrix[j + 1, j + 1] = α
    ls.Tmatrix[j, j + 1] = β
    ls.Tmatrix[j + 1, j] = β

    #Compute the norm of the gradient of the model.

    #Do this vcat(gradnorm,zeros(3)) instead (3 was just arbitarly chosen number?
    e1 = zeros(j + 1) #vcat(s.gradnorm,zeros(j))
    e1[1] = 1

    #temporary way of doing it.
    #This was only necessary since
    if j == 1
        modelGradnorm = norm(
            ls.gradnorm * e1 +
            @view(ls.Tmatrix[1:(j + 1), 1:j]) * ls.y' +
            ls.σ * norm(ls.y, 2) * vcat(ls.y, 0),
            2,
        )
    else
        modelGradnorm = norm(
            ls.gradnorm * e1 +
            @view(ls.Tmatrix[1:(j + 1), 1:j]) * ls.y +
            ls.σ * norm(ls.y, 2) * vcat(ls.y, 0),
            2,
        )
    end
    if modelGradnorm <= ls.θ * norm(ls.y, 2)^2
        #The condition is satisifed. Assemble the optimal tangent vector
        project!(M, ls.S, ls.p, sum(ls.Q[i] * ls.y[i] for i in 1:j))
    end
    return ls
end
get_iterate(s::LanczosState) = s.S
function set_manopt_parameter!(s::LanczosState, ::Val{:p}, p)
    s.p = p
    return s
end
function set_manopt_parameter!(s::LanczosState, ::Val{:σ}, σ)
    s.σ = σ
    return s
end

mutable struct StopWhenLanczosModelGradLess <: StoppingCriterion
    relative_threshold::Float64
    reason::String
    StopWhenLanczosModelGradLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenLanczosModelGradLess)(::AbstractManoptProblem, s::LanczosState, i::Int)
    (i == 0) && (c.reason = "") # reset on init
    # Ronny: maybe s.y[1:s.k] ?
    if (i > 0) && s.modelGradnorm <= c.relative_threshold * norm(s.y, 2)^2
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
    ::AbstractManoptProblem, s::AdaptiveRegularizationState, i::Int
)
    (i == 0) && (c.reason = "") # reset on init
    if (i > 0) && s.subcost.k == c.maxInnerIter
        c.reason = "The algorithm used all preallocated Lanczos vectors and may have stagnated. Allocate more by variable maxIterLanczos.\n"
        return true
    end
    return false
end

#
# Old code, not yet reviewed nor reworked.
#
function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, i, j) #old dummy
    M = get_manifold(dmp)
    mho = get_objective(dmp)

    #Set iterate and update the regularization parameter
    set_iterate!(arcs.sub_state, M, copy(M, arcs.p))
    set_manopt_parameter!(arcs.sub_state, :σ, arcs.σ)

    #Solve the sub_problem
    solve!(arcs.sub_problem, decorate_state!(arcs.sub_state))
    arcs.S = get_solver_result(arcs.sub_problem)

    if !arcs.optimized_updating_rule #check if we want to use the optimized procedure
        #Regular updating procedure
        #Computing the regularized ratio between actual improvement and model improvement.
        retrx = retract(M, arcs.p, arcs.S, arcs.retraction_method)
        cost = get_cost(M, mho, arcs.p)
        ρ_num = cost - get_cost(M, mho, retrx)
        ρ_vec = get_gradient(M, mho, arcs.p) + 0.5 * get_hessian(M, mho, arcs.p, arcs.S)
        ρ_den = -inner(M, arcs.p, arcs.S, ρ_vec)
        ρ_reg = arcs.ρ_regularization * eps(Float64) * max(abs(cost), 1)
        ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)

        arcs.ρ = ρ
        sub_fail = (ρ_den + ρ_reg <= 0)
        if sub_fail
            println("sub_problem failure!")    #if this is the case we should reject the step!
        end

        #Update iterate
        if arcs.ρ >= arcs.η1
            arcs.p = retrx                    #retract(M, arcs.p, arcs.S) #changed to .=
            get_gradient!(dmp, arcs.X, arcs.p) #only compute gradient when we update the point
        end

        #Update regularization parameter
        if arcs.ρ >= arcs.η2 #very successful
            arcs.σ = max(arcs.σmin, arcs.γ1 * arcs.σ)
        elseif arcs.η1 <= arcs.ρ < arcs.η2
            #leave regParam unchanged
        else #unsuccessful
            arcs.σ = arcs.γ2 * arcs.σ
        end

    else #optimized updating procedure - where is this from or where described?

        #temporarly set the parameters here. Will move them to the arc state so they can be adjusted.
        ϵχ = 1e-10
        β = 0.01
        δ1 = 0.1
        δ2 = 0.1
        δ3 = 2.0
        δmax = 100.0
        αmax = 2
        η = arcs.η1

        #compute retraction and cost
        retrx = retract(M, arcs.p, arcs.S, arcs.retraction_method)
        cost = get_cost(M, mho, arcs.p)
        newcost = get_cost(M, mho, retrx)

        #compute ρ (not regularized)
        ρ_num = cost - newcost
        ρ_vec = get_gradient(M, mho, arcs.p) + 0.5 * get_hessian(M, mho, arcs.p, arcs.S)
        ρ_den = -inner(M, arcs.p, arcs.S, ρ_vec)
        ρ = ρ_num / ρ_den
        arcs.ρ = ρ

        #Update iterate
        if arcs.ρ >= arcs.η1
            arcs.p = retrx                    #retract(M, arcs.p, arcs.S) #changed to .=
            get_gradient!(dmp, arcs.X, arcs.p) #only compute gradient when we update the point
        end

        #useful variables
        ck = (arcs.subcost)(M, arcs.subcost.y) #compute subcost  (arcs.sub_state.y)
        qk = ck - arcs.σ / 3 * norm(arcs.subcost.y, 2)^3 #compute quadratic model
        χk = ck - max(newcost, qk) #compute gap
        pk = newcost - qk
        gs = arcs.subcost.y[1] * arcs.subcost.gradnorm
        sHs =
            0.5 * dot(
                arcs.subcost.y,
                @view(arcs.subcost.Tmatrix[1:(arcs.subcost.k), 1:(arcs.subcost.k)]) *
                arcs.subcost.y,
            )

        if arcs.ρ >= 1 && χk >= ϵχ
            if newcost >= qk
                #solve cubic equation (3.29) given by 3*pk*α^3 + sHs*α^2 + gs*α + 3*β*χk=0
                roots = solvecubic(3 * pk, sHs, gs, 3 * β * χk)
                realrootsvec = realroots(roots)
                A = realrootsvec[realrootsvec .>= cbrt(β)]
                if length(A) == 0
                    arcs.σ = max(δ1 * arcs.σ, eps())
                end
                if length(A) > 0
                    Aβ = A .- cbrt(β)
                    min_index = argmin(Aβ) #computes αβ=argmin{(α-cbrt(β))| α ∈ A }
                    αβ = A[min_index]

                    if αβ <= αmax
                        σβ =
                            arcs.σ +
                            3.0 * χk / (norm(arcs.subcost.y, 2)^3) * ((β - αβ^3) / (αβ^3))                      #σβ =arcs.σ*β/(αβ)^3
                        arcs.σ = max(σβ, eps())
                    end
                    if αβ > αmax
                        arcs.σ = max(δ1 * arcs.σ, eps())
                    end
                end
            elseif newcost < qk
                # solve quadratic equation (3.34) sHs*α^2 + gs*α + 3*β*χk=0
                disc = (gs)^2 - 4 * sHs * 3 * β * χk #compute discriminant
                real_roots = []
                if isapprox(disc, 0.0; atol=1e-15, rtol=0)
                    r1 = -gs / (2 * sHs)
                    append!(real_roots, r1)
                elseif disc > 0
                    r1 = (-gs + sqrt(disc)) / (2 * sHs)
                    r2 = (-gs - sqrt(disc)) / (2 * sHs)
                    append!(real_roots, r1)
                    append!(real_roots, r2)
                end
                A = real_roots[real_roots .>= cbrt(β)]

                if length(A) == 0
                    arcs.σ = max(δ1 * arcs.σ, eps())
                end
                if length(A) > 0
                    Aβ = A .- cbrt(β)
                    min_index = argmin(Aβ) #computes αβ=argmin{(α-cbrt(β))| α ∈ A }
                    αβ = A[min_index]
                    if αβ <= αmax
                        σβ = arcs.σ * β / (αβ)^3
                        arcs.σ = max(σβ, eps())
                    end
                    if αβ > αmax
                        arcs.σ = max(δ1 * arcs.σ, eps())
                    end
                end
            end
        elseif arcs.ρ >= 1.0 && χk < ϵχ
            arcs.σ = max(δ1 * arcs.σ, eps())
        elseif arcs.η2 <= arcs.ρ < 1.0
            arcs.σ = max(δ2 * arcs.σ, eps())
        elseif arcs.η1 <= arcs.ρ < arcs.η2
            println("we enter unchange elseif")
            #leave unchanged
        elseif 0 <= arcs.ρ < arcs.η1
            arcs.σ = δ3 * arcs.σ
        else
            arcs.ρ < 0
            #solve the quadratic equation (3.38) 6*pk*α^2 + (3-η)*sHs*α + 2*(3-2*η)*gs = 0
            disc = ((3 - η) * sHs)^2 - 48 * pk * (3 - 2 * η) * gs
            r1 = (-(3 - η) * sHs + sqrt(disc)) / (12 * pk)
            r2 = (-(3 - η) * sHs - sqrt(disc)) / (12 * pk)
            αη = max(r1, r2)
            ση = -(gs + sHs * αη) / (αη^2 * norm(arcs.subcost.y, 2)^3)
            arcs.σ = min(max(ση, δ3 * arcs.σ), δmax * arcs.σ)
        end
    end

    return arcs
end

#solver for cubic equations taken from github since we dont have have Polynomial roots package.
#needed for the optimized updating rule

function solvecubic(a, b, c, d)
    if a == 0 && b == 0                    # Case for handling Liner Equation
        return [(-d * 1.0) / c]# Returning
    elseif a == 0                             # Case for handling Quadratic
        D = c * c - 4.0 * b * d                       # Helper Temporary Variable
        if D >= 0
            D = sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else
            D = sqrt(-D)
            x1 = (-c + D * im) / (2.0 * b)
            x2 = (-c - D * im) / (2.0 * b)
        end
        return [x1, x2]
    end
    # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    if f == 0 && g == 0 && h == 0            # All 3 Roots are Real and Equal
        if (d / a) >= 0
            x = (d / (1.0 * a))^(1 / 3.0) * -1
        else
            x = (-d / (1.0 * a))^(1 / 3.0)
        end
        return [x, x, x]           # Returning Equal Roots as numpy array.

    elseif h <= 0                               # All 3 roots are Real
        i = sqrt(((g^2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i^(1 / 3.0)                      # Helper Temporary Variable
        k = acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = cos(k / 3.0)                   # Helper Temporary Variable
        N = sqrt(3) * sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return [x1, x2, x3]          # Returning Real Roots as numpy array.

    elseif h > 0                               # One Real Root and two Complex Roots
        R = -(g / 2.0) + sqrt(h)           # Helper Temporary Variable
        if R >= 0
            S = R^(1 / 3.0)                  # Helper Temporary Variable
        else
            S = (-R)^(1 / 3.0) * -1
        end                 # Helper Temporary Variable
        T = -(g / 2.0) - sqrt(h)
        if T >= 0
            U = (T^(1 / 3.0))                # Helper Temporary Variable
        else
            U = ((-T)^(1 / 3.0)) * -1
        end# Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * sqrt(3) * 0.5 * im
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * sqrt(3) * 0.5 * im
        return [x1, x2, x3]
    end
end
# Returning One Real Root and two Complex Roots
# Helper function to return float value of f.
function findF(a, b, c)
    return ((3.0 * c / a) - ((b^2.0) / (a^2.0))) / 3.0
end
# Helper function to return float value of g.
function findG(a, b, c, d)
    return (((2.0 * (b^3.0)) / (a^3.0)) - ((9.0 * b * c) / (a^2.0)) + (27.0 * d / a)) / 27.0
end
# Helper function to return float value of h.
function findH(g, f)
    return ((g^2.0) / 4.0 + (f^3.0) / 27.0)
end

#find the real roots from the cubic solver
function realroots(rootvec)
    #find the real roots of the cubic equation ax^3+bx^2+cx+d=0
    #input rootvec=[x1,x2,x3]
    realrootvec = []
    im_part = imag.(rootvec)
    for i in 1:length(rootvec)
        if isapprox(im_part[i], 0.0; atol=1e-15, rtol=0) # check if im part is zero
            append!(realrootvec, real(rootvec[i]))
        end
    end
    return realrootvec
end

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
