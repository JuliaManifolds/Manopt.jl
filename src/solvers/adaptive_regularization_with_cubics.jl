@doc raw"""
    AdaptiveRegularizationState{P,T} <: AbstractHessianSolverState

Describes ... algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `p` – (`rand(M)` the current iterate
* `X` – (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``, initialised to zero vector.
* `H` – (`zero_vector(M,p)`) the current hessian, $\operatorname{Hess}F(p)[⋅]$, initialised to zero vector.
* `s` (`zero_vector(M,p)`)- the tangent vector step resulting from minimizing the model problem in the tangent space \mathcal T_{p} \mathcal M
* `ς`– the current regularization parameter
* `ρ`– the current regularized ratio of actual improvement and model improvement.
* `ρ_regularization`– (1e3) regularization paramter for computing ρ. As we approach convergence the ρ may be difficult to compute with numerator and denominator approachign zero. Regularizing the the ratio lets ρ go to 1 near convergence.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `retraction_method` – (`default_retraction_method(M)`) the retraction to use, defaults to
  the default set for your manifold.
* `ςmin`
* `θ`
* `η1`
* `η2`
* `γ1`
* `γ2`
* `γ3`

# Constructor

    AdaptiveRegularizationState(M, p=rand(M); X=zero_vector(M, p), H=zero_vector(M, p), kwargs...)


"""

mutable struct AdaptiveRegularizationState{
    P,
    St<:AbstractManoptSolverState,
    SCO,
    SPR,
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,          #,
} <: AbstractManoptSolverState#AbstractHessianSolverState
    p::P
    substate::St
    subcost::SCO # Ronny: not necessary because its included in the cubproblem
    subprob::SPR
    X::T
    H::T
    S::T
    ς::R
    ρ::R
    ρ_regularization::R
    stop::TStop
    retraction_method::TRTM
    optimized_updating_rule::Bool
    ςmin::R
    θ::R
    η1::R
    η2::R
    γ1::R
    γ2::R
    γ3::R
end

function AdaptiveRegularizationState(
    M::AbstractManifold,
    p::P=rand(M);
    substate::St=LanczosState(M, p),
    subcost::SCO=substate.subcost, # ManifoldHessianObjective((M,p)->0,(M,p)->0,(M,p,X)->0)
    subprob::SPR=DefaultManoptProblem(M, ManifoldCostObjective(subcost)),
    X::T=zero_vector(M, p),
    H::T=zero_vector(M, p),
    S::T=zero_vector(M, p),
    ς::R=100.0 / sqrt(manifold_dimension(M)),# Had this to inital value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ::R=1.0,
    ρ_regularization::R=1e3,
    stop::StoppingCriterion=StopAfterIteration(100),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    optimized_updating_rule::Bool=false,
    ςmin::R=1e-10, #Set the below to appropriate default vals.
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    γ3::R=2.0,
) where {P,St,SCO,SPR,T,R}
    return AdaptiveRegularizationState{P,St,SCO,SPR,T,R}(
        M,
        p,
        substate,
        subcost,
        subprob,
        X,
        H,
        S,
        ς,
        ρ,
        ρ_regularization,
        stop,
        retraction_method,
        optimized_updating_rule,
        ςmin,
        θ,
        η1,
        η2,
        γ1,
        γ2,
        γ3,
    )
end

@doc raw"""
    adaptive_regularization_with_cubicsM, f, grad_f, Hess_f, p=rand(M); kwargs...)


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
    ς::R=100.0 / sqrt(manifold_dimension(M)),
    maxIterLanczos=200,
    ρ::R=0.0, #1.0
    ρ_regularization::R=1e3,
    stop::StoppingCriterion=StopAfterIteration(40) |
                            StopWhenGradientNormLess(1e-9) |
                            StopWhenAllLanczosVectorsUsed(maxIterLanczos - 1),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    optimized_updating_rule::Bool=false,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ςmin::R=1e-10,
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    γ3::R=2.0,
    substate::AbstractManoptSolverState=LanczosState(
        M,
        copy(M, p);       #tried adding copy
        maxIterLanczos=maxIterLanczos,
        θ=θ,
        ς=ς,
        objective=ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation),
    ),
    subcost=substate.subcost,
    subprob=DefaultManoptProblem(M, ManifoldCostObjective(subcost)),
    kwargs...,
) where {T,R,TF,TDF,THF}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        p;
        substate=substate,
        subcost=subcost,
        subprob=subprob,
        X=X,
        H=H,
        S=S,
        ς=ς,
        ρ=ρ,
        ρ_regularization=ρ_regularization,
        stop=stop,
        retraction_method=retraction_method,
        optimized_updating_rule=optimized_updating_rule,
        ςmin=ςmin,
        θ=θ,
        η1=η1,
        η2=η2,
        γ1=γ1,
        γ2=γ2,
        γ3=γ3,
    )
    arcs = decorate_state!(arcs; kwargs...)
    solve!(dmp, arcs)
    return get_solver_return(get_objective(dmp), arcs)
end
get_iterate(s::AdaptiveRegularizationState) = s.p
function set_iterate!(s::AdaptiveRegularizationState, p)
    copyto!(M, s.p, p)
    return s
end
get_gradient(s::AdaptiveRegularizationState) = s.X
function set_gradient!(s::AdaptiveRegularizationState, X)
    copyto!(M, s.X, s.p, X)
    return s
end

function initialize_solver!(dmp::AbstractManoptProblem, s::AdaptiveRegularizationState)
    get_gradient!(dmp, s.X, s.p)
    return s
end
function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)

    #Set iterate and update the regularization parameter
    set_iterate!(arcs.substate, M, copy(M, arcs.p))
    set_manopt_parameter!(arcs.substate, :ς, arcs.ς)

    #Solve the subproblem
    arcs.S = get_solver_result(solve!(arcs.subprob, decorate_state!(arcs.substate)))

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
            println("subproblem failure!")    #if this is the case we should reject the step!
        end

        #Update iterate
        if arcs.ρ >= arcs.η1
            arcs.p = retrx                    #retract(M, arcs.p, arcs.S) #changed to .=
            get_gradient!(dmp, arcs.X, arcs.p) #only compute gradient when we update the point
        end

        #Update regularization parameter
        if arcs.ρ >= arcs.η2 #very successful
            arcs.ς = max(arcs.ςmin, arcs.γ1 * arcs.ς)
        elseif arcs.η1 <= arcs.ρ < arcs.η2
            #leave regParam unchanged
        else #unsuccessful
            arcs.ς = arcs.γ2 * arcs.ς
        end

    else
        #optimized updating procedure

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
        ck = (arcs.subcost)(M, arcs.subcost.y) #compute subcost  (arcs.substate.y)
        qk = ck - arcs.ς / 3 * norm(arcs.subcost.y, 2)^3 #compute quadratic model
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
                    arcs.ς = max(δ1 * arcs.ς, eps())
                end
                if length(A) > 0
                    Aβ = A .- cbrt(β)
                    min_index = argmin(Aβ) #computes αβ=argmin{(α-cbrt(β))| α ∈ A }
                    αβ = A[min_index]

                    if αβ <= αmax
                        ςβ =
                            arcs.ς +
                            3.0 * χk / (norm(arcs.subcost.y, 2)^3) * ((β - αβ^3) / (αβ^3))                      #ςβ =arcs.ς*β/(αβ)^3
                        arcs.ς = max(ςβ, eps())
                    end
                    if αβ > αmax
                        arcs.ς = max(δ1 * arcs.ς, eps())
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
                    arcs.ς = max(δ1 * arcs.ς, eps())
                end
                if length(A) > 0
                    Aβ = A .- cbrt(β)
                    min_index = argmin(Aβ) #computes αβ=argmin{(α-cbrt(β))| α ∈ A }
                    αβ = A[min_index]
                    if αβ <= αmax
                        ςβ = arcs.ς * β / (αβ)^3
                        arcs.ς = max(ςβ, eps())
                    end
                    if αβ > αmax
                        arcs.ς = max(δ1 * arcs.ς, eps())
                    end
                end
            end
        elseif arcs.ρ >= 1.0 && χk < ϵχ
            arcs.ς = max(δ1 * arcs.ς, eps())
        elseif arcs.η2 <= arcs.ρ < 1.0
            arcs.ς = max(δ2 * arcs.ς, eps())
        elseif arcs.η1 <= arcs.ρ < arcs.η2
            println("we enter unchange elseif")
            #leave unchanged
        elseif 0 <= arcs.ρ < arcs.η1
            arcs.ς = δ3 * arcs.ς
        else
            arcs.ρ < 0
            #solve the quadratic equation (3.38) 6*pk*α^2 + (3-η)*sHs*α + 2*(3-2*η)*gs = 0
            disc = ((3 - η) * sHs)^2 - 48 * pk * (3 - 2 * η) * gs
            r1 = (-(3 - η) * sHs + sqrt(disc)) / (12 * pk)
            r2 = (-(3 - η) * sHs - sqrt(disc)) / (12 * pk)
            αη = max(r1, r2)
            ςη = -(gs + sHs * αη) / (αη^2 * norm(arcs.subcost.y, 2)^3)
            arcs.ς = min(max(ςη, δ3 * arcs.ς), δmax * arcs.ς)
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

mutable struct LanczosState{P,SCO,SC,I,R,T,TM,V,Y} <: AbstractManoptSolverState
    p::P
    objective::ManifoldHessianObjective
    subcost::SCO # Ronny: Why is that here?
    stop::SC #maximum number of iterations
    maxIterNewton::I
    ς::R # current reg parameter, must update with set manopt parameter.
    θ::R
    gradnorm::R # norm of gradient at current point p
    modelGradnorm::R
    tolNewton::R
    Q::T #store orthonormal basis
    Tmatrix::TM #store triddiagonal matrix
    r::V # store the r vector
    S::V # store the tangent vector that solves the minimization problem
    y::Y # store the y vector
    d::I #number of dimensions of current subspace solution
end

function LanczosState(
    M::AbstractManifold,
    p::P=rand(M);
    # Ronny: Maybe not necessary?
    objective::ManifoldHessianObjective=ManifoldHessianObjective(
        (M, p) -> 0, (M, p) -> 0, (M, p, X) -> 0
    ),
    θ=0.5,
    maxIterLanczos=200,
    #maxIterLanczos-1 since the first Lanczos vector is constructed in the intialization
    stopping_criterion::SC=StopAfterIteration(maxIterLanczos - 1) |
                           StopWhenLanczosModelGradLess(θ),
    maxIterNewton::I=100,
    ς::R=10.0,
    gradnorm::R=1.0,
    modelGradnorm::R=Inf,
    tolNewton::R=1e-16,
    Q::T=[zero_vector(M, p) for _ in 1:maxIterLanczos],
    Tmatrix::TM=spdiagm(
        -1 => zeros(maxIterLanczos - 1),
        0 => zeros(maxIterLanczos),
        1 => zeros(maxIterLanczos - 1),
    ),
    r::V=zero_vector(M, p),
    S::V=zero_vector(M, p),
    y::Y=[0.0],
    d=1,                     #Y=zeros(maxIterLanczos),
    subcost::SCO=CubicSubCost(1, gradnorm, ς, Tmatrix, y),
) where {P,SCO,SC<:StoppingCriterion,I,R,T,TM,V,Y}
    return LanczosState{P,SCO,SC,I,R,T,TM,V,Y}(
        M,
        p,
        objective,
        subcost,
        stopping_criterion,
        maxIterNewton,
        ς,
        θ,
        gradnorm,
        modelGradnorm,
        tolNewton,
        Q,
        Tmatrix,
        r,
        S,
        y,
        d,
    )
end

get_solver_result(ls::LanczosState) = ls.S

function initialize_solver!(dmp::AbstractManoptProblem, s::LanczosState)
    #in the intialization we set the first orthonormal vector, the first element of the Tmatrix and the r vector.
    M = get_manifold(dmp)
    mho = s.objective

    g = get_gradient(M, mho, s.p)   #added ! and s.X
    s.gradnorm = norm(M, s.p, g)

    #q = g / s.gradnorm
    #s.Q[1] .= q #store it directly above
    s.Q[1] .= g / s.gradnorm

    r = get_hessian(M, mho, s.p, s.Q[1])#changed from q to       #save memory here use s.r, and below use @. s.r = s.r -...
    α = inner(M, s.p, s.Q[1], r) #q change
    s.Tmatrix[1, 1] = α
    s.r = r - α * s.Q[1] #q change

    #idea in the initalize_solver we set dim of subspace sol to d=1.

    #argmin of one dimensional model
    s.y = [(α - sqrt(α^2 + 4 * s.ς * s.gradnorm)) / (2 * s.ς)] # store y in the state.

    #update parameters for the subcost
    # Ronny: instead do set_manopt_parameter!(dmp, :Cost, :k,1)
    set_manopt_parameter!(s.subcost, :k, 1)
    set_manopt_parameter!(s.subcost, :y, s.y)
    set_manopt_parameter!(s.subcost, :Tmatrix, s.Tmatrix)
    set_manopt_parameter!(s.subcost, :ς, s.ς)
    set_manopt_parameter!(s.subcost, :gradnorm, s.gradnorm)

    return s
end

#step solver for the LanczosState (will change to LanczosState when its done and its correct)
function step_solver!(dmp::AbstractManoptProblem, s::LanczosState, j)
    M = get_manifold(dmp)
    mho = s.objective #mho = get_objective(dmp)
    β = norm(M, s.p, s.r)

    if j + 1 > length(s.Q) #j+1? Since we created the first lanczos vector in the intialization
        println(
            "We have used all the ",
            length(s.Q),
            " allocated Lanczos vectors. Allocate more by variable maxIterLanczos.",
        )
    end

    #Had to move it here to avoid logic error: earlier we computed only new y if stopping criterion failed, however this would lead to updating the y in s.y, and when the stopping_criterion is called after the step, it would be checked with a new y.
    if j > 1
        s.y = min_cubic_Newton(s, j)
    end

    #Note: not doing MGS causes fast loss of orthogonality. Do full orthogonalization for robustness?
    if β > 1e-12  # β large enough-> Do regular procedure: MGS of r wrt. Q
        s.Q[j + 1] .= project(M, s.p, s.r / β) #s.r/β
    #for i in 1:j
    #    s.r=s.r-inner(M,s.p,s.Q[i],s.r)*s.Q[i]
    #end
    # s.Q[j + 1] .= project(M,s.p,s.r/norm(M,s.p,s.r))    #q=r/norm(M,s.p,r) #/β                                      #s.r / β # project(M::Grassmann, p, X)
    else # Generate new random vec and MGS of new vec wrt. Q
        println("maxed out! gen rand vec")
        r = rand(M; vector_at=s.p)
        for i in 1:j
            r .= r - inner(M, s.p, s.Q[i], r) * s.Q[i]  #use @.
        end
        s.Q[j + 1] .= project(M, s.p, r / norm(M, s.p, r))  #r / norm(M, s.p, r)                            # r / norm(M, s.p, r)
    end

    rh = get_hessian(M, mho, s.p, s.Q[j + 1])
    r = rh - β * s.Q[j] #also store this in s.r to save memory
    α = inner(M, s.p, r, s.Q[j + 1])
    s.r = r - α * s.Q[j + 1]

    s.Tmatrix[j + 1, j + 1] = α
    s.Tmatrix[j, j + 1] = β
    s.Tmatrix[j + 1, j] = β

    #Compute the norm of the gradient of the model.

    #Do this vcat(gradnorm,zeros(3)) instead (3 was just arbitarly chosen number?
    e1 = zeros(j + 1) #vcat(s.gradnorm,zeros(j))
    e1[1] = 1

    #temporary way of doing it.
    #This was only necessary since
    if j == 1
        s.modelGradnorm = norm(
            s.gradnorm * e1 +
            @view(s.Tmatrix[1:(j + 1), 1:j]) * s.y' +
            s.ς * norm(s.y, 2) * vcat(s.y, 0),
            2,
        )
    else
        s.modelGradnorm = norm(
            s.gradnorm * e1 +
            @view(s.Tmatrix[1:(j + 1), 1:j]) * s.y +
            s.ς * norm(s.y, 2) * vcat(s.y, 0),
            2,
        )
    end

    if s.modelGradnorm <= s.θ * norm(s.y, 2)^2
        #The condition is satisifed. Assemble the optimal tangent vector
        S_opt = zero_vector(M, s.p)
        #println("number of dim in opt sol: ", j)

        for i in 1:j #length(s.y)
            S_opt = S_opt + s.Q[i] * s.y[i]    #better to do lc=s.y .* s.Q[1:length(s.y)] to do the linear comb, then sum(lc)
        end
        s.S = project(M, s.p, S_opt) #S_opt
    end

    #Update the params here.
    set_manopt_parameter!(s.subcost, :k, j) #tried changing for j+1 to j
    set_manopt_parameter!(s.subcost, :y, s.y) #     s.subcost(TangentSpaceAt(M,p), s.y)
    set_manopt_parameter!(s.subcost, :Tmatrix, s.Tmatrix)
    set_manopt_parameter!(s.subcost, :ς, s.ς)

    return s
end

mutable struct CubicSubCost{Y,T,I,R}
    k::I #number of Lanczos vectors
    gradnorm::R
    ς::R
    Tmatrix::T #submatrix
    y::Y # Solution of of argmin m(s), s= sum y[i]q[i]
end
function (C::CubicSubCost)(::AbstractManifold, y)# Ronny: M is Euclidean (R^k) but p should be y. I can change it to y a just input c.y when computing the subcost
    #C.y[1]*C.gradnorm + 0.5*dot(C.y[1:C.k],@view(C.Tmatrix[1:C.k,1:C.k])*C.y[1:C.k]) + C.ς/3*norm(C.y[1:C.k],2)^3
    return y[1] * C.gradnorm +
           0.5 * dot(y, @view(C.Tmatrix[1:(C.k), 1:(C.k)]) * y) +
           C.ς / 3 * norm(y, 2)^3
end

#How to make the below less confusing?
#The the iterates in in the Lanczos stepsolver is strictly a sequence of tangent vectors that minimize the problem in TₚM.
#However we must set the correct inital point in the LanczosState
get_iterate(s::LanczosState) = s.S

function set_iterate!(s::LanczosState, M::AbstractManifold, p)
    s.p = p                         #copyto!(M, s.p, p)
    return s
end

function set_manopt_parameter!(s::LanczosState, ::Val{:ς}, ς)
    s.ς = ς
    return s
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
function set_manopt_parameter!(s::CubicSubCost, ::Val{:ς}, ς)
    s.ς = ς
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

mutable struct StopWhenLanczosModelGradLess <: StoppingCriterion
    relative_threshold::Float64
    reason::String
    StopWhenLanczosModelGradLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenLanczosModelGradLess)(
    ::AbstractManoptProblem, s::LanczosState, i::Int
)
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
# Switch to a sub state? What‘s j?
#
function min_cubic_Newton(s::LanczosState, j)
    maxiter = s.maxIterNewton
    tol = s.tolNewton

    gvec = zeros(j)
    gvec[1] = s.gradnorm
    λ = opnorm(Array(@view s.Tmatrix[1:j, 1:j])) + 2
    T_λ = @view(s.Tmatrix[1:j, 1:j]) + λ * I

    λ_min = eigmin(Array(@view s.Tmatrix[1:j, 1:j]))
    lower_barrier = max(0, -λ_min)
    iter = 0
    y = 0

    while iter <= maxiter
        y = -(T_λ \ gvec)
        ynorm = norm(y, 2)
        ϕ = 1 / ynorm - s.ς / λ #when ϕ is "zero", y is the solution.
        if abs(ϕ) < tol * ynorm
            break
        end

        #compute the newton step
        ψ = ynorm^2
        Δy = -(T_λ) \ y
        ψ_prime = 2 * dot(y, Δy)
        # Quadratic polynomial coefficients
        p0 = 2 * s.ς * ψ^(1.5)
        p1 = -2 * ψ - λ * ψ_prime
        p2 = ψ_prime
        #Polynomial roots
        r1 = (-p1 + sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)
        r2 = (-p1 - sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)

        Δλ = max(r1, r2) - λ
        iter = iter + 1

        if λ + Δλ <= lower_barrier #if we jumped past the lower barrier for λ, jump to midpoint between current and lower λ.
            Δλ = -0.5 * (λ - lower_barrier)
        end

        if abs(Δλ) <= eps(λ) #if the steps we make are to small, terminate
            break
        end

        T_λ = T_λ + Δλ * I
        λ = λ + Δλ
    end
    return y
end
