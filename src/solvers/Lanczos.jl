#
# Lanczos sub solver
#
@doc """
    LanczosState{P,T,SC,B,I,R,TM,V,Y} <: AbstractManoptSolverState

Solve the adaptive regularized subproblem with a Lanczos iteration

# Fields

$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:stopping_criterion, name = "stop_newton"))
  used for the inner Newton iteration
* `σ`:               the current regularization parameter
* `X`:               the Iterate
* `Lanczos_vectors`: the obtained Lanczos vectors
* `tridig_matrix`:   the tridiagonal coefficient matrix T
* `coefficients`:    the coefficients ``y_1,...y_k`` that determine the solution
* `Hp`:              a temporary tangent vector containing the evaluation of the Hessian
* `Hp_residual`:     a temporary tangent vector containing the residual to the Hessian
* `S`:               the current obtained / approximated solution

# Constructor

    LanczosState(TpM::TangentSpace; kwargs...)

## Keyword arguments

$(_kwargs(:X; add_properties = [:as_Iterate]))
* `maxIterLanzcos=200`: shortcut to set the maximal number of iterations in the ` stopping_crtierion=`
* `θ=0.5`: set the parameter in the [`StopWhenFirstOrderProgress`](@ref) within the default `stopping_criterion=`.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(maxIterLanczos)`$(_sc(:Any))[`StopWhenFirstOrderProgress`](@ref)`(θ)"))
$(_kwargs(:stopping_criterion; name = "stopping_criterion_newton", default = "`[`StopAfterIteration`](@ref)`(200)"))
  used for the inner Newton iteration
* `σ=10.0`: specify the regularization parameter
"""
mutable struct LanczosState{T, R, SC, SCN, B, TM, C} <: AbstractManoptSolverState
    X::T
    σ::R
    stop::SC
    stop_newton::SCN
    Lanczos_vectors::B # ``q_i``
    tridig_matrix::TM  # `T``
    coefficients::C     # `y``
    Hp::T              # `Hess_f`` A temporary vector for evaluations of the Hessian
    Hp_residual::T     # A residual vector
    S::T               # store the tangent vector that solves the minimization problem
end
function LanczosState(
        TpM::TangentSpace;
        X::T = zero_vector(TpM.manifold, TpM.point),
        maxIterLanczos = 200,
        θ = 0.5,
        stopping_criterion::SC = StopAfterIteration(maxIterLanczos) |
            StopWhenFirstOrderProgress(θ),
        stopping_criterion_newton::SCN = StopAfterIteration(200),
        σ::R = 10.0,
    ) where {T, SC <: StoppingCriterion, SCN <: StoppingCriterion, R}
    tridig = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    coeffs = zeros(maxIterLanczos)
    Lanczos_vectors = typeof(X)[]
    return LanczosState{T, R, SC, SCN, typeof(Lanczos_vectors), typeof(tridig), typeof(coeffs)}(
        X,
        σ,
        stopping_criterion,
        stopping_criterion_newton,
        Lanczos_vectors,
        tridig,
        coeffs,
        copy(TpM, X),
        copy(TpM, X),
        copy(TpM, X),
    )
end
function get_solver_result(ls::LanczosState)
    return ls.S
end
function set_iterate!(ls::LanczosState, M, X)
    ls.X .= X
    return ls
end
function set_parameter!(ls::LanczosState, ::Val{:σ}, σ)
    ls.σ = σ
    return ls
end
function status_summary(ls::LanczosState; context = default)
    _is_inline(context) && return repr(ls)
    i = get_count(ls, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ls.stop) ? "Yes" : "No"
    vectors = length(ls.Lanczos_vectors)
    return """
    # Solver state for `Manopt.jl`s Lanczos Iteration
    $Iter
    ## Parameters
    * σ                         : $(ls.σ)
    * # of Lanczos vectors used : $(vectors)

    ## Stopping criteria
    (a) For the Lanczos Iteration
    $(status_summary(ls.stop))
    (b) For the Newton sub solver
    $(status_summary(ls.stop_newton))
    This indicates convergence: $Conv"""
end

#
# The Lanczos Subsolver implementation
#
function initialize_solver!(dmp::AbstractManoptProblem{<:TangentSpace}, ls::LanczosState)
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = TpM.point
    maxIterLanczos = size(ls.tridig_matrix, 1)
    ls.tridig_matrix = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    ls.coefficients = zeros(maxIterLanczos)
    for X in ls.Lanczos_vectors
        zero_vector!(M, X, p)
    end
    zero_vector!(M, ls.Hp, p)
    zero_vector!(M, ls.Hp_residual, p)
    return ls
end

function step_solver!(dmp::AbstractManoptProblem{<:TangentSpace}, ls::LanczosState, k)
    TpM = get_manifold(dmp)
    p = TpM.point
    M = base_manifold(TpM)
    arcmo = get_objective(dmp)
    if k == 1 #the first is known directly
        nX = norm(M, p, ls.X)
        if length(ls.Lanczos_vectors) == 0
            push!(ls.Lanczos_vectors, ls.X ./ nX)
        else
            copyto!(M, ls.Lanczos_vectors[1], p, ls.X ./ nX)
        end
        get_objective_hessian!(M, ls.Hp, arcmo, p, ls.Lanczos_vectors[1])
        α = inner(M, p, ls.Lanczos_vectors[1], ls.Hp)
        # This is also the first coefficient in the tridiagonal matrix
        ls.tridig_matrix[1, 1] = α
        ls.Hp_residual .= ls.Hp - α * ls.Lanczos_vectors[1]
        #this is the minimiser of one dimensional model
        ls.coefficients[1] = (α - sqrt(α^2 + 4 * ls.σ * nX)) / (2 * ls.σ)
    else # `i > 1`
        β = norm(M, p, ls.Hp_residual)
        if β > 1.0e-12 # Obtained new orthogonal Lanczos long enough with respect to numerical stability
            if length(ls.Lanczos_vectors) < k
                push!(ls.Lanczos_vectors, ls.Hp_residual ./ β)
            else
                copyto!(M, ls.Lanczos_vectors[k], p, ls.Hp_residual ./ β)
            end
        else # Generate new random vector and
            # modified Gram Schmidt of new vector with respect to Q
            rand!(M, ls.Hp_residual; vector_at = p)
            for k in 1:(k - 1)
                ls.Hp_residual .=
                    ls.Hp_residual -
                    inner(M, p, ls.Lanczos_vectors[k], ls.Hp_residual) *
                    ls.Lanczos_vectors[k]
            end
            if length(ls.Lanczos_vectors) < k
                push!(ls.Lanczos_vectors, ls.Hp_residual ./ norm(M, p, ls.Hp_residual))
            else
                copyto!(
                    M,
                    ls.Lanczos_vectors[k],
                    p,
                    ls.Hp_residual ./ norm(M, p, ls.Hp_residual),
                )
            end
        end
        # Update Hessian and residual
        get_objective_hessian!(M, ls.Hp, arcmo, p, ls.Lanczos_vectors[k])
        ls.Hp_residual .= ls.Hp - β * ls.Lanczos_vectors[k - 1]
        α = inner(M, p, ls.Hp_residual, ls.Lanczos_vectors[k])
        ls.Hp_residual .= ls.Hp_residual - α * ls.Lanczos_vectors[k]
        # Update tridiagonal matrix
        ls.tridig_matrix[k, k] = α
        ls.tridig_matrix[k - 1, k] = β
        ls.tridig_matrix[k, k - 1] = β
        min_cubic_Newton!(dmp, ls, k)
    end
    copyto!(M, ls.S, p, sum(ls.Lanczos_vectors[k] * ls.coefficients[k] for k in 1:k))
    return ls
end
#
# Solve Lanczos sub problem
#
function min_cubic_Newton!(mp::AbstractManoptProblem{<:TangentSpace}, ls::LanczosState, k)
    TpM = get_manifold(mp)
    p = TpM.point
    M = base_manifold(TpM)
    tol = 1.0e-16

    gvec = zeros(k)
    gvec[1] = norm(M, p, ls.X)
    λ = opnorm(Array(@view ls.tridig_matrix[1:k, 1:k])) + 2
    T_λ = @view(ls.tridig_matrix[1:k, 1:k]) + λ * I

    λ_min = eigmin(Array(@view ls.tridig_matrix[1:k, 1:k]))
    lower_barrier = max(0, -λ_min)
    j = 0
    y = zeros(k)
    while !ls.stop_newton(mp, ls, j)
        j += 1
        y = -(T_λ \ gvec)
        ynorm = norm(y, 2)
        ϕ = 1 / ynorm - ls.σ / λ # when ϕ is "zero" then y is the solution.
        (abs(ϕ) < tol * ynorm) && break
        # compute the Newton step
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

        #instead of jumping past the lower barrier for λ,
        # jump to midpoint between current and lower λ.
        (λ + Δλ <= lower_barrier) && (Δλ = -0.5 * (λ - lower_barrier))
        #if the steps are too small, exit
        (abs(Δλ) <= eps(λ)) && break
        T_λ = T_λ + Δλ * I
        λ = λ + Δλ
    end
    ls.coefficients[1:k] .= y
    return ls.coefficients
end

#
# Stopping Criteria
#
_math_sc_firstorder = raw"""
```math
m(X_k) ≤ m(0)
\quad\text{ and }\quad
\lVert \operatorname{grad} m(X_k) \rVert ≤ θ \lVert X_k \rVert^2
```
"""

@doc """
    StopWhenFirstOrderProgress <: StoppingCriterion

A stopping criterion related to the Riemannian adaptive regularization with cubics (ARC)
solver indicating that the model function at the current (outer) iterate,

$_doc_ARC_model

defined on the tangent space ``$(_math(:TangentSpace))`` fulfils at the current iterate ``X_k`` that

$_math_sc_firstorder

# Fields

* `θ`:      the factor ``θ`` in the second condition
$(_fields(:at_iteration))

# Constructor

    StopWhenAllLanczosVectorsUsed(θ)

"""
mutable struct StopWhenFirstOrderProgress{F} <: StoppingCriterion
    θ::F
    at_iteration::Int
    StopWhenFirstOrderProgress(θ::F) where {F} = new{F}(θ, -1)
end
function (c::StopWhenFirstOrderProgress)(
        dmp::AbstractManoptProblem{<:TangentSpace}, ls::LanczosState, k::Int
    )
    if (k == 0)
        if norm(ls.X) == zero(eltype(ls.X))
            c.at_iteration = 0
            return true
        end
        c.at_iteration = -1
        return false
    end
    #Update Gradient
    TpM = get_manifold(dmp)
    p = TpM.point
    M = base_manifold(TpM)
    nX = norm(M, p, get_gradient(dmp, p))
    y = @view(ls.coefficients[1:(k - 1)])
    Ty = @view(ls.tridig_matrix[1:k, 1:(k - 1)]) * y
    ny = norm(y)
    model_grad_norm = norm(nX .* [1, zeros(k - 1)...] + Ty + ls.σ * ny * [y..., 0])
    prog = (model_grad_norm <= c.θ * ny^2)
    (prog) && (c.at_iteration = k)
    return prog
end
function get_reason(c::StopWhenFirstOrderProgress)
    if c.at_iteration > 0
        return "The algorithm has reduced the model grad norm by a factor $(c.θ)."
    end
    if c.at_iteration == 0 # gradient 0
        return "The gradient of the gradient is zero."
    end
    return ""
end
function (c::StopWhenFirstOrderProgress)(
        dmp::AbstractManoptProblem{<:TangentSpace}, ams::AbstractManoptSolverState, k::Int
    )
    if (k == 0)
        c.at_iteration = -1
        return false
    end
    TpM = get_manifold(dmp)
    p = TpM.point
    q = get_iterate(ams)
    # Update Gradient and compute norm
    nG = norm(base_manifold(TpM), p, get_gradient(dmp, q))
    # norm of current iterate
    nX = norm(base_manifold(TpM), p, q)
    prog = (nG <= c.θ * nX^2)
    prog && (c.at_iteration = k)
    return prog
end
function status_summary(c::StopWhenFirstOrderProgress; context = :default)
    (context == :short) && return repr(sc)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    _is_inline(context) && return "First order progress with θ=$(c.θ):$(_MANOPT_INDENT)$s"
    return "A stopping criterion to stop when the Lanczos model has fpund a certain first order progress with θ=$(c.θ):$(_MANOPT_INDENT)$s"
end
indicates_convergence(c::StopWhenFirstOrderProgress) = true
function show(io::IO, c::StopWhenFirstOrderProgress)
    return print(io, "StopWhenFirstOrderProgress($(repr(c.θ)))\n    $(status_summary(c))")
end

@doc """
    StopWhenAllLanczosVectorsUsed <: StoppingCriterion

When an inner iteration has used up all Lanczos vectors, then this stopping criterion is
a fallback / security stopping criterion to not access a non-existing field
in the array allocated for vectors.

Note that this stopping criterion (for now) is only implemented for the case that an
[`AdaptiveRegularizationState`](@ref) when using a [`LanczosState`](@ref) subsolver

# Fields

* `maxLanczosVectors`: maximal number of Lanczos vectors
* `at_iteration` indicates at which iteration (including `i=0`) the stopping criterion
  was fulfilled and is `-1` while it is not fulfilled.

# Constructor

    StopWhenAllLanczosVectorsUsed(maxLancosVectors::Int)

"""
mutable struct StopWhenAllLanczosVectorsUsed <: StoppingCriterion
    maxLanczosVectors::Int
    at_iteration::Int
    StopWhenAllLanczosVectorsUsed(maxLanczosVectors::Int) = new(maxLanczosVectors, -1)
end
function (c::StopWhenAllLanczosVectorsUsed)(
        ::AbstractManoptProblem,
        arcs::AdaptiveRegularizationState{P, T, Pr, <:LanczosState},
        i::Int,
    ) where {P, T, Pr}
    (i == 0) && (c.at_iteration = -1) # reset on init
    if (i > 0) && length(arcs.sub_state.Lanczos_vectors) == c.maxLanczosVectors
        c.at_iteration = i
        return true
    end
    return false
end
function get_reason(c::StopWhenAllLanczosVectorsUsed)
    if (c.at_iteration >= 0)
        return "The algorithm used all ($(c.maxLanczosVectors)) preallocated Lanczos vectors and may have stagnated.\n Consider increasing this value.\n"
    end
    return ""
end
function status_summary(c::StopWhenAllLanczosVectorsUsed)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "" : "Stop when all Lanczos vectors are used\n$(_MANOPT_INDENT)":"All Lanczos vectors ($(c.maxLanczosVectors)) used:$(_MANOPT_INDENT)") * s
end
indicates_convergence(c::StopWhenAllLanczosVectorsUsed) = false
function show(io::IO, c::StopWhenAllLanczosVectorsUsed)
    return print(io, "StopWhenAllLanczosVectorsUsed($(repr(c.maxLanczosVectors)))")
end
