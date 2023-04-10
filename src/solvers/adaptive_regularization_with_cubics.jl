
#Starting with writing the doc for the state for keeping an overview. Surley containing errors to be corrected.

#Idea: Add the set of orthonormal vectors to the state. With a vector of tangent space dimension containing the orthonormal vectors
# preallocated,we then we compute the k<=n orthonormal vectors needed in each iteration in-place.

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
    T,
    R,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,          #,
} <: AbstractManoptSolverState#AbstractHessianSolverState
    p::P
    substate::St
    X::T
    H::T
    S::T
    ς::R
    ρ::R
    ρ_regularization::R
    stop::TStop
    retraction_method::TRTM
    ςmin::R
    θ::R
    η1::R
    η2::R
    γ1::R
    γ2::R
    γ3::R
    function AdaptiveRegularizationState{P,St,T,R}(
        M::AbstractManifold,
        p::P=rand(M),
        substate::St=LanczosState(M),
        X::T=zero_vector(M, p),
        H::T=zero_vector(M, p),
        S::T=zero_vector(M, p),
        ς::R=1.0,
        ρ::R=1.0,
        ρ_regularization::R=1.0,
        stop::StoppingCriterion=StopAfterIteration(100),       #TRTM?
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        ςmin::R=1.0, #Set the below to appropriate default vals.
        θ::R=1.0,
        η1::R=1.0,
        η2::R=1.0,
        γ1::R=1.0,
        γ2::R=1.0,
        γ3::R=1.0,
    ) where {P,St,T,R}
        o = new{P,St,T,R,typeof(stop),typeof(retraction_method)}()
        o.p = p
        o.substate = substate
        o.X = X
        o.H = H
        o.S = S
        o.ς = ς
        o.ρ = ρ
        o.ρ_regularization = ρ_regularization
        o.stop = stop
        o.retraction_method = retraction_method
        o.ςmin = ςmin
        o.θ = θ
        o.η1 = η1
        o.η2 = η2
        o.γ1 = γ1
        o.γ2 = γ2
        o.γ3 = γ3
        return o
    end
end

function AdaptiveRegularizationState(
    M::AbstractManifold,
    p::P=rand(M);
    substate::St=LanczosState(M),
    X::T=zero_vector(M, p),
    H::T=zero_vector(M, p),
    S::T=zero_vector(M, p),
    ς::R=100.0 / sqrt(manifold_dimension(M)),# Had this to inital value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ::R=1.0,
    ρ_regularization::R=1e3,
    stop::StoppingCriterion=StopAfterIteration(100),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    ςmin::R=1e-10, #Set the below to appropriate default vals.
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    γ3::R=2.0,
) where {P,St,T,R}
    return AdaptiveRegularizationState{P,St,T,R}(
        M,
        p,
        substate,
        X,
        H,
        S,
        ς,
        ρ,
        ρ_regularization,
        stop,
        retraction_method,
        ςmin,
        θ,
        η1,
        η2,
        γ1,
        γ2,
        γ3,
    )
end

function adaptive_regularization!(
    M::AbstractManifold,
    f::TF,
    gradf::TDF,
    hessf::THF,
    p;
    substate::AbstractManoptSolverState=LanczosState(M),
    X::T=zero_vector(M, p),
    H::T=zero_vector(M, p),
    S::T=zero_vector(M, p),
    ς::R=100.0 / sqrt(manifold_dimension(M)),
    ρ::R=1.0,
    ρ_regularization::R=1e3,
    stop::StoppingCriterion=StopAfterIteration(40) | StopWhenGradientNormLess(1e-9),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ςmin::R=1e-10,
    θ::R=2.0,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    γ3::R=2.0,
    kwargs...,
) where {T,R,TF,TDF,THF}
    mho = ManifoldHessianObjective(f, gradf, hessf; evaluation=evaluation)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        p;
        substate=substate,
        X=X,
        H=H,
        S=S,
        ς=ς,
        ρ=ρ,
        ρ_regularization=ρ_regularization,
        stop=stop,
        retraction_method=retraction_method,
        ςmin=ςmin,
        θ=θ,
        η1=η1,
        η2=η2,
        γ1=γ1,
        γ2=γ2,
        γ3=γ3,
    )
    arcs = decorate_state!(arcs; kwargs...)
    return get_solver_return(solve!(dmp, arcs))
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

#Note to self. For invariant subspace problem on Gr(512,12) we allocate a Q vector of length 6000 (manifold dim) where each element is a 512x12 matrix. In Lanczsos we would in practice only maybe use the first 100 or so of the vectors in Q.
#store triddiagonal matrix and gradnorm in LanczosState?

mutable struct LanczosState{R,T} <: AbstractManoptSolverState
    maxIterNewton::R
    tolNewton::R
    Q::T
    function LanczosState{R}(
        M::AbstractManifold, maxIterNewton::R, tolNewton::R, Q::T
    ) where {R, T}
        o = new{R,T}()
        o.maxIterNewton = maxIterNewton
        o.tolNewton = tolNewton
        o.Q = Q
        return o
    end
end

#Note on LanczosState
#Pre allocating the Q obviously does a crazy amount of allocation which we dontt need in practice. Add a maxLanczosIter variable
#Also maybe store the tridiagoal sparse matrix T.
function LanczosState(
    M::AbstractManifold,
    maxIterNewton::R=100.0,
    tolNewton::R=1e-16,
    Q=[zero_vector(M, rand(M)) for _ in 1:manifold_dimension(M)],#How should I change this?
) where {R}
    return LanczosState{R}(M, maxIterNewton, tolNewton, Q)
end

#Maybe instead of implementing the subproblem solver(and future subproblem solver I may add?) as functor just make a function #subproblem_solver!(ls::LanczosState,dmp::AbstractManoptProblem,s::AdaptiveRegularizationState). For a new solver I implement
#subproblem_solver!(ns::NewState,...)

#doing it the way of the augmented Lagrangian Solver this would be the step_solver!(LanczosState)
#which would be called to solve the subproblem in the step_solver!(AdaptiveRegularizationState)
function subproblem_solver!(
    dmp::AbstractManoptProblem, s::AdaptiveRegularizationState, ls::LanczosState
)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    dim = manifold_dimension(M)
    T = spdiagm(-1 => zeros(dim - 1), 0 => zeros(dim), 1 => zeros(dim - 1))
    g = get_gradient(M, mho, s.p)
    gradnorm = norm(M, s.p, g)
    q = g / gradnorm
    ls.Q[1] .= q

    r = get_hessian(M, mho, s.p, q)
    α = inner(M, s.p, q, r)
    T[1, 1] = α
    r = r - α * q
    # make the above part of initalize_solver?


    #argmin of one dimensional model
    y = (α - sqrt(α^2 + 4 * s.ς * gradnorm)) / (2 * s.ς)

    for j in 1:(dim - 1) # can at the most generate dim(TₓM) lanczos vectors, but created the first above so subtract
        β = norm(M, s.p, r)
        #Note: not doing MGS causes fast loss of orthogonality. We do full orthogonalization for robustness.
        if β > 1e-10  # β large enough-> Do regular procedure: MGS of r wrt. Q
            q = r / β
            #for i in 1:j
            #    r=r-inner(M,s.p,ls.Q[i],r)*ls.Q[i]
            #end
            #q=r/norm(M,s.p,r) #/β

        else # Generate new random vec and MGS of new vec wrt. Q
            r = rand(M; vector_at=s.p)
            for i in 1:j
                r .= r - inner(M, s.p, ls.Q[i], r) * ls.Q[i]   #maybe .= to avoid memory alloc
            end
            q = r / norm(M, s.p, r)
        end

        r = get_hessian(M, mho, s.p, q) - β * ls.Q[j]
        α = inner(M, s.p, r, q)
        r = r - α * q
        ls.Q[j + 1] .= q
        T[j + 1, j + 1] = α
        T[j, j + 1] = β
        T[j + 1, j] = β

        #We have created the j+1'th orthonormal vector and the (j+1)x(j+1) T matrix, creating one additional lanczos vector
        #to the dimension we minimized in so that checking the stopping criterion is cheaper.
        #Now compute the gradient corresponding to the j dimensional model in the basis of Q[1],...,Q[j+1].

        e1 = zeros(j + 1)#
        e1[1] = 1
        model_gradnorm = norm(
            gradnorm * e1 + T[1:(j + 1), 1:j] * y + s.ς * norm(y, 2) * vcat(y, 0), 2
        )

        if model_gradnorm <= s.θ * norm(y, 2)^2    #the loop stopped here. Temp comment out
            #println("number of dims in tangent space solution ", j)
            break
        end

        #Stopping condition not satisifed. Proceed to minimize the (j+1) dimensional model by in the subspace of TₚM spanned by
        #the (j+1) orthogonal vectors

        y = min_cubic_Newton(ls, gradnorm, T[1:(j + 1), 1:(j + 1)], s.ς)
    end

    #Assemble the tangent vector
    S_opt = zero_vector(M, s.p)
    for i in 1:length(y)
        S_opt = S_opt + ls.Q[i] * y[i]   #views here?
    end

    s.S = S_opt
    return S_opt
end

function min_cubic_Newton(ls::LanczosState, gradnorm, T, σ)
    #Implements newton again, but this time I compute the Newton step the same way as in MATLAB code.
    #minimize the cubic in R^n.
    n = size(T)[1]
    maxiter = ls.maxIterNewton
    tol = ls.tolNewton

    gvec = zeros(n)
    gvec[1] = gradnorm
    λ = opnorm(T, 1) + 2
    T_λ = T + λ * I
    λ_min = eigmin(Array(T))
    lower_barrier = max(0, -λ_min)
    iter = 0
    y = 0
    while iter <= maxiter
        y = -(T_λ \ gvec)
        ynorm = norm(y, 2)
        ϕ = 1 / ynorm - σ / λ #when ϕ is "zero", y is the solution.
        if abs(ϕ) < tol * ynorm
            break
        end

        #compute the newton step
        ψ = ynorm^2
        Δy = -(T_λ) \ y
        ψ_prime = 2 * dot(y, Δy)
        # Quadratic polynomial coefficients
        p0 = 2 * σ * ψ^(1.5)
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

function initialize_solver!(dmp::AbstractManoptProblem, s::AdaptiveRegularizationState)
    get_gradient!(dmp, s.X, s.p)
    return s
end

function step_solver!(dmp::AbstractManoptProblem, s::AdaptiveRegularizationState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)

    #Subproblem solver
    #solves the problem of computing minimizer of m(s) over TpM. Lanzos is the default subproblem solver.
    subproblem_solver!(dmp, s, s.substate)

    #Computing the regularized ratio between actual improvement and model improvement.
    cost = get_cost(M, mho, s.p)
    ρ_num = cost - get_cost(M, mho, retract(M, s.p, s.S))
    ρ_vec = get_gradient(M, mho, s.p) + 0.5 * get_hessian(M, mho, s.p, s.S)
    ρ_den = -inner(M, s.p, s.S, ρ_vec)
    ρ_reg = s.ρ_regularization * eps(Float64) * max(abs(cost), 1)
    ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)
    s.ρ = ρ
    sub_fail = (ρ_den + ρ_reg <= 0)
    if sub_fail
        println("subproblem failure!")
    end

    #Update iterate
    if s.ρ >= s.η1
        #println("Updated iterate")
        s.p = retract(M, s.p, s.S)
    end

    #Update regularization parameter
    if s.ρ >= s.η2 #very successful
        #println("very successful")
        s.ς = max(s.ςmin, s.γ1 * s.ς)
    elseif s.η1 <= s.ρ < s.η2
        #println("successful")
        #leave regParam unchanged
    else #unsuccessful
        #println("unsuccessful")
        s.ς = s.γ2 * s.ς
    end
    #ComputeRegRatio!(M,mho,s)
    #UpdateRegParameterAndIterate!(M,s)
    get_gradient!(dmp, s.X, s.p)
    return s
end














#################################################################################################################
#Below here I write some new functions that will be used when I formulate the subproblem solver in a similar manner as
# in the AugmentedLagrangianMethod


mutable struct NewLanczosState{P,I,R,T,TM,V,Y} <: AbstractManoptSolverState
    p::P
    maxIterLanczos::I #maximum number of iterations
    maxIterNewton::I
    tolNewton::R
    Q::T #store orthonormal basis
    Tmatrix::TM #store triddiagonal matrix
    r::V # store the r vector
    y::Y # store the y vector
    function NewLanczosState{P,I,R,T,TM,V,Y}(
        M::AbstractManifold,p::P,maxIterLanczos::R, maxIterNewton::R, tolNewton::R, Q::T,Tmatrix::TM,r::V,y::Y
    ) where {P,I,R,T,TM,V,Y}
        o = new{P,I,R,T,TM,V,Y}()
        o.p=p
        o.maxIterLanczos=maxIterLanczos
        o.maxIterNewton = maxIterNewton
        o.tolNewton = tolNewton
        o.Q = Q
        o.Tmatrix=Tmatrix
        o.r=r
        o.y=y
        return o
    end
end

function NewLanczosState(
    M::AbstractManifold,
    p::P=rand(M);
    maxIterLanczos::I=200,
    maxIterNewton::I=100,
    tolNewton::R=1e-16,
    Q::T=[zero_vector(M, rand(M)) for _ in 1:maxIterLanczos],
    Tmatrix::TM=spdiagm(-1 => zeros(maxIterLanczos - 1), 0 => zeros(maxIterLanczos), 1 => zeros(maxIterLanczos - 1)),
    r::V=zero_vector(M,p),
    y::Y=0.0
) where{P,I,R,T,TM,V,Y}
    return NewLanczosState{P,I,R,T,TM,V,Y}(M,p, maxIterNewton, tolNewton, Q, Tmatrix, r, y)
end


#NewLanczosState will be the substate of the ArcState

#also add stopping crit to state

#stopping_criterion::StoppingCriterion=StopAfterIteration(maxLanczosIter)| #make custom Stop for 





function initialize_solver!(dmp::AbstractManoptProblem, s::NewLanczosState)
    #in the intialization we set the first orthonormal vector, the first element of the Tmatrix and the r vector.
    M = get_manifold(dmp)
    mho = get_objective(dmp)

    g = get_gradient(M, mho, s.p)
    gradnorm = norm(M, s.p, g)
    q = g / gradnorm

    s.Q[1] .= q
    r = get_hessian(M, mho, s.p, q)
    α = inner(M, s.p, q, r)
    s.Tmatrix[1,1]=α
    s.r = r - α * q
    #argmin of one dimensional model
    s.y = (α - sqrt(α^2 + 4 * s.ς * gradnorm)) / (2 * s.ς) # store y in the state. 
    return s
end


#step solver for the NewLanczosState (will change to LanczosState when its done and its correct)
function step_solver!(dmp::AbstractManoptProblem, s::NewLanczosState, j)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    β = norm(M, s.p, s.r)

    #Note: not doing MGS causes fast loss of orthogonality. Do full orthogonalization for robustness?
    if β > 1e-10  # β large enough-> Do regular procedure: MGS of r wrt. Q
        q = r / β
    else # Generate new random vec and MGS of new vec wrt. Q
        r = rand(M; vector_at=s.p)
        for i in 1:j
            r .= r - inner(M, s.p, s.Q[i], r) * s.Q[i]
        end
        q = r / norm(M, s.p, r)
    end

    r = get_hessian(M, mho, s.p, q) - β * ls.Q[j] #also store this in s.r to save memory
    α = inner(M, s.p, r, q)
    s.r = r - α * q
    s.Q[j + 1] .= q

    s.Tmatrix[j + 1, j + 1] = α
    s.Tmatrix[j, j + 1] = β
    s.Tmatrix[j + 1, j] = β

    #here we dont have to write any stopping criterion stuff, that is handled by the solve! function when we 
    #have defined or own stopping criterion.

    # How to define stopping crit? Use stop if less or equal?
    #Should I call stop_solver! here so that I can compute y then if necessary (i.e if we should not stop, proceed to compute y)

    #IMPORTANT HERE WE SOLVE for the next y if we did not stop above


    #This is the stopping crit
    #e1 = zeros(j + 1)#
    #e1[1] = 1
    #model_gradnorm = norm(
    #    gradnorm * e1 + T[1:(j + 1), 1:j] * y + s.ς * norm(y, 2) * vcat(y, 0), 2
    #)

    #if model_gradnorm <= s.θ * norm(y, 2)^2    #the loop stopped here. Temp comment out
    #    #println("number of dims in tangent space solution ", j)
    #    brea
end





#TODO
#1. Add any remaning stuff we need to lanczosState. 
    #a) gradient? copied over from ArcState.
    #b) ς
#2 Fix the stopping criterion
#3 Fix the nice way of allocating memory for Q and T.