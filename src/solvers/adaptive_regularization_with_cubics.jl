
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
    ςmin::R
    θ::R
    η1::R
    η2::R
    γ1::R
    γ2::R
    γ3::R
    function AdaptiveRegularizationState{P,St,SCO,SPR,T,R}(
        M::AbstractManifold,
        p::P=rand(M),
        substate::St=NewLanczosState(M,p), # Why i subcost in the state?
        subcost::SCO=substate.subcost,
        subprob::SPR=DefaultManoptProblem(M,ManifoldCostObjective(subcost)),
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
    ) where {P,St,SCO,SPR,T,R}
        o = new{P,St,SCO,SPR,T,R,typeof(stop),typeof(retraction_method)}()
        o.p = p
        o.substate = substate
        o.subcost=subcost
        o.subprob=subprob
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
    substate::St=NewLanczosState(M,p),
    subcost::SCO=substate.subcost, # ManifoldHessianObjective((M,p)->0,(M,p)->0,(M,p,X)->0)
    subprob::SPR=DefaultManoptProblem(M,ManifoldCostObjective(subcost)),
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
        ςmin,
        θ,
        η1,
        η2,
        γ1,
        γ2,
        γ3,
    )
end


function adaptive_regularization_with_cubics(
    M::AbstractManifold, f::TF, gradf::TDF,hessf::THF, p0=rand(M); kwargs...
) where {TF,TDF,THF}
    q0 = copy(M, p0)
    return adaptive_regularization_with_cubics!(M, f, gradf,hessf, q0; kwargs...)
end



function adaptive_regularization_with_cubics!(
    M::AbstractManifold,
    f::TF,
    gradf::TDF,
    hessf::THF,
    p0=rand(M);
    X::T=zero_vector(M, p0),
    H::T=zero_vector(M, p0),
    S::T=zero_vector(M, p0),
    ς::R=100.0 / sqrt(manifold_dimension(M)),
    ρ::R=0.0, #1.0
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
    substate::AbstractManoptSolverState=NewLanczosState(M, p0; θ=θ,ς=ς,objective=ManifoldHessianObjective(f,gradf,hessf)),
    subcost=substate.subcost,
    subprob=DefaultManoptProblem(M,ManifoldCostObjective(subcost)),
    kwargs...,
) where {T,R,TF,TDF,THF}
    mho = ManifoldHessianObjective(f, gradf, hessf; evaluation=evaluation)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    arcs = AdaptiveRegularizationState(
        M,
        p0;
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

#removed here
function initialize_solver!(dmp::AbstractManoptProblem, s::AdaptiveRegularizationState)
    get_gradient!(dmp, s.X, s.p)
    return s
end

function step_solver!(dmp::AbstractManoptProblem, arcs::AdaptiveRegularizationState, i)
    M = get_manifold(dmp)
    mho = get_objective(dmp)

    #Set iterate and update the regularization parameter
    set_iterate!(arcs.substate, M, arcs.p)
    set_manopt_parameter!(arcs.substate,:ς,arcs.ς)


    #Solve the subproblem
    arcs.S = get_solver_result(solve!(arcs.subprob, decorate_state!(arcs.substate)))
    println("step S" ,arcs.S)

    #Computing the regularized ratio between actual improvement and model improvement.
    cost = get_cost(M, mho, arcs.p)
    ρ_num = cost - get_cost(M, mho, retract(M, arcs.p, arcs.S))
    ρ_vec = get_gradient(M, mho, arcs.p) + 0.5 * get_hessian(M, mho, arcs.p, arcs.S)
    ρ_den = -inner(M, arcs.p, arcs.S, ρ_vec)
    ρ_reg = arcs.ρ_regularization * eps(Float64) * max(abs(cost), 1)
    ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)
    println("ρ ", ρ)
    arcs.ρ = ρ
    sub_fail = (ρ_den + ρ_reg <= 0)
    if sub_fail
        println("subproblem failure!")    #if this is the case we should reject the step!
    end

    #Update iterate
    if arcs.ρ >= arcs.η1
        #println("Updated iterate")
        arcs.p = retract(M, arcs.p, arcs.S)
        get_gradient!(dmp, arcs.X, arcs.p)
        #            Move the get_gradient! below up here so we only compute it when the point is new.
    end

    #Update regularization parameter
    if arcs.ρ >= arcs.η2 #very successful
        #println("very successful")
        arcs.ς = max(arcs.ςmin, arcs.γ1 * arcs.ς)
    elseif arcs.η1 <= arcs.ρ < arcs.η2
        #println("successful")
        #leave regParam unchanged
    else #unsuccessful
        #println("unsuccessful")
        arcs.ς = arcs.γ2 * arcs.ς
    end

    #get_gradient!(dmp, arcs.X, arcs.p)
    return arcs
end
















#################################################################################################################
#Below here I write some new functions that will be used when I formulate the subproblem solver in a similar manner as
# in the AugmentedLagrangianMethod


mutable struct NewLanczosState{P,SCO,SC,I,R,T,TM,V,Y} <: AbstractManoptSolverState
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
    function NewLanczosState{P,SCO,SC,I,R,T,TM,V,Y}(                               #sc_lanzcos
        M::AbstractManifold,p::P, objective::ManifoldHessianObjective,subcost::SCO,stop::SC, maxIterNewton::I,ς::R,θ::R,gradnorm::R,modelGradnorm::R, tolNewton::R, Q::T,Tmatrix::TM,r::V,S::V,y::Y
    ) where {P,SCO,SC,I,R,T,TM,V,Y}
        s = new{P,SCO,SC,I,R,T,TM,V,Y}()
        s.p=p
        s.objective=objective
        s.subcost=subcost
        s.stop=stop      #sc_lanzcos
        s.maxIterNewton = maxIterNewton # ? better: NewtonSate
        s.ς=ς
        s.θ=θ
        s.gradnorm=gradnorm
        s.modelGradnorm=modelGradnorm
        s.tolNewton = tolNewton
        s.Q = Q
        s.Tmatrix=Tmatrix
        s.r=r
        s.S=S
        s.y=y
        return s
    end
end

function NewLanczosState(
    M::AbstractManifold,
    p::P=rand(M);
    # Ronny: Maybe not necessary?
    objective::ManifoldHessianObjective = ManifoldHessianObjective((M,p)->0,(M,p)->0,(M,p,X)->0),
    θ=0.5,
    maxIterLanczos=200,                     #maxIterLanczos
    stopping_criterion::SC=StopAfterIteration(maxIterLanczos) | StopWhenLanczosModelGradLess(θ),
    maxIterNewton::I=100,
    ς::R=10.0,
    gradnorm::R=1.0,
    modelGradnorm::R=Inf,
    tolNewton::R=1e-16,
    Q::T=[zero_vector(M, p) for _ in 1:maxIterLanczos],
    Tmatrix::TM=spdiagm(-1 => zeros(maxIterLanczos - 1), 0 => zeros(maxIterLanczos), 1 => zeros(maxIterLanczos - 1)),
    r::V=zero_vector(M,p),
    S::V=zero_vector(M,p),
    y::Y=[0.0],                     #Y=zeros(maxIterLanczos),
    subcost::SCO=CubicSubCost(1,gradnorm,ς,Tmatrix,y)
) where{P,SCO,SC<:StoppingCriterion,I,R,T,TM,V,Y}
    return NewLanczosState{P,SCO,SC,I,R,T,TM,V,Y}(M, p, objective, subcost, stopping_criterion, maxIterNewton,ς,θ,gradnorm,modelGradnorm, tolNewton, Q, Tmatrix, r,S,y)
end


get_solver_result(ls::NewLanczosState) = ls.S



function initialize_solver!(dmp::AbstractManoptProblem, s::NewLanczosState)
    #in the intialization we set the first orthonormal vector, the first element of the Tmatrix and the r vector.
    M = get_manifold(dmp)
    mho =s.objective

    g = get_gradient(M, mho, s.p)
    s.gradnorm = norm(M, s.p, g)
    #q = g / s.gradnorm
    #s.Q[1] .= q #store it directly above
    s.Q[1] .= g / s.gradnorm



    r = get_hessian(M, mho, s.p, s.Q[1])#changed from q to       #save memory here use s.r, and below use @. s.r = s.r -...
    α = inner(M, s.p, s.Q[1], r) #q change
    s.Tmatrix[1,1]=α
    s.r = r - α * s.Q[1] #q change
    #argmin of one dimensional model
    s.y = [(α - sqrt(α^2 + 4 * s.ς * s.gradnorm)) / (2 * s.ς)] # store y in the state.



    #update parameters for the subcost
    # Ronny: instead do set_manopt_parameter!(dmp, :Cost, :k,1)
    set_manopt_parameter!(s.subcost,:k,1)
    set_manopt_parameter!(s.subcost,:y,s.y)
    set_manopt_parameter!(s.subcost,:Tmatrix,s.Tmatrix)
    set_manopt_parameter!(s.subcost,:ς,s.ς)
    set_manopt_parameter!(s.subcost,:gradnorm,s.gradnorm)

    return s
end


#step solver for the NewLanczosState (will change to LanczosState when its done and its correct)
function step_solver!(dmp::AbstractManoptProblem, s::NewLanczosState, j)
    M = get_manifold(dmp)
    mho =s.objective #mho = get_objective(dmp)
    β = norm(M, s.p, s.r)

    if j+1 > length(s.Q) #j+1? Since we created the first lanczos vector in the intialization
        println("We have used all the allocated Lanczos vectors")
    end

    #Note: not doing MGS causes fast loss of orthogonality. Do full orthogonalization for robustness?
    if β > 1e-12  # β large enough-> Do regular procedure: MGS of r wrt. Q
        s.Q[j + 1] .= s.r / β
    else # Generate new random vec and MGS of new vec wrt. Q
        r = rand(M; vector_at=s.p)
        for i in 1:j
            r .= r - inner(M, s.p, s.Q[i], r) * s.Q[i]  #use @.
        end
        s.Q[j + 1] .= r / norm(M, s.p, r)
    end

    r = get_hessian(M, mho, s.p,s.Q[j + 1]) - β * s.Q[j] #also store this in s.r to save memory
    α = inner(M, s.p, r, s.Q[j + 1])
    s.r = r - α * s.Q[j + 1]
    #s.Q[j + 1] .= q #it would be better to just store it here in the if/else above at once, and acess it at the index where we need it above.

    s.Tmatrix[j + 1, j + 1] = α
    s.Tmatrix[j, j + 1] = β
    s.Tmatrix[j + 1, j] = β


    #Compute the norm of the gradient of the model.

    #Do this vcat(gradnorm,zeros(3)) instead (3 was just arbitarly chosen number?
    e1 = zeros(j + 1) #vcat(s.gradnorm,zeros(j))
    e1[1] = 1

    #temporary way of doing it.
    #This was only necessary since
    if j==1
        s.modelGradnorm = norm(
            s.gradnorm * e1 + @view(s.Tmatrix[1:(j+1),1:j]) * s.y' + s.ς * norm(s.y, 2) * vcat(s.y, 0), 2)
    else
        s.modelGradnorm = norm(
            s.gradnorm * e1 + @view(s.Tmatrix[1:(j+1),1:j]) * s.y + s.ς * norm(s.y, 2) * vcat(s.y, 0), 2)
    end

    #Old modelgrad
    #s.modelGradnorm = norm(
    #    s.gradnorm * e1 + @view(s.Tmatrix[1:(j+1),1:j]) * s.y[1:j] + s.ς * norm(s.y[1:j], 2) * vcat(s.y[1:j], 0), 2)


    if s.modelGradnorm > s.θ * norm(s.y, 2)^2
        s.y=min_cubic_Newton(s,j+1)
        #S is initalized as zero_vector(M,p) so dont have to set it.
    else
        println("stopping crit satisifed at iteration:",j)
        #The condition is satisifed. Assemble the optimal tangent vector
        S_opt=zero_vector(M,s.p)
        for i in 1:j #length(s.y)
            S_opt = S_opt + s.Q[i] * s.y[i] # save memory here
        end
        s.S=S_opt
    end
    #Update the params here.
    set_manopt_parameter!(s.subcost,:k,j+1)
    set_manopt_parameter!(s.subcost,:y,s.y) #     s.subcost(TangentSpaceAt(M,p), s.y)
    set_manopt_parameter!(s.subcost,:Tmatrix,s.Tmatrix)
    set_manopt_parameter!(s.subcost,:ς,s.ς)
    return s
end










mutable struct CubicSubCost{Y,T,I,R}
    k::I #number of Lanczos vectors
    gradnorm::R
    ς::R
    Tmatrix::T #submatrix
    y::Y # Solution of of argmin m(s), s= sum y[i]q[i]
end
function (C::CubicSubCost)(M::AbstractManifold,p)  # Ronny: M is Euclidean (R^k) but p should be y
    #C.y[1]*C.gradnorm + 0.5*dot(C.y[1:C.k],@view(C.Tmatrix[1:C.k,1:C.k])*C.y[1:C.k]) + C.ς/3*norm(C.y[1:C.k],2)^3
    return C.y[1]*C.gradnorm + 0.5*dot(C.y,@view(C.Tmatrix[1:C.k,1:C.k])*C.y) + C.ς/3*norm(C.y,2)^3
end




#How to make the below less confusing?
#The the iterates in in the Lanczos stepsolver is strictly a sequence of tangent vectors that minimize the problem in TₚM.
#However we must set the correct inital point in the LanczosState
get_iterate(s::NewLanczosState) = s.S

function set_iterate!(s::NewLanczosState, M::AbstractManifold, p)
    copyto!(M, s.p, p)
    return s
end


function set_manopt_parameter!(s::NewLanczosState, ::Val{:ς}, ς)
    s.ς=ς
    return s
end




#Sub cost set_manopt_parameter!'s

function set_manopt_parameter!(s::CubicSubCost, ::Val{:k}, k)
    s.k=k
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:gradnorm}, gradnorm)
    s.gradnorm=gradnorm
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:ς}, ς)
    s.ς=ς
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:Tmatrix}, Tmatrix)
    s.Tmatrix=Tmatrix
    return s
end
function set_manopt_parameter!(s::CubicSubCost, ::Val{:y}, y)
    s.y=y
    return s
end




mutable struct StopWhenLanczosModelGradLess <: StoppingCriterion
    relative_threshold::Float64
    reason::String
    StopWhenLanczosModelGradLess(ε::Float64) = new(ε, "")
end
function (c::StopWhenLanczosModelGradLess)(
    ::AbstractManoptProblem, s::NewLanczosState, i::Int
)
    (i == 0) && (c.reason = "") # reset on init
    # Ronny: maybe s.y[1:s.k] ?
    if (i > 0) && s.modelGradnorm <= c.relative_threshold*norm(s.y, 2)^2
        c.reason = "The algorithm has reduced the model grad norm by $(c.relative_threshold).\n"
        return true
    end
    return false
end





function min_cubic_Newton(s::NewLanczosState,j)
    maxiter = s.maxIterNewton
    tol = s.tolNewton

    gvec=zeros(j)
    gvec[1]=s.gradnorm
    λ=opnorm(Array(@view s.Tmatrix[1:j,1:j]))+2
    T_λ = @view(s.Tmatrix[1:j,1:j]) + λ * I


    λ_min = eigmin(Array(@view s.Tmatrix[1:j,1:j]))
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
