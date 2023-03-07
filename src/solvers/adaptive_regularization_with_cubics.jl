
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
* `ρ`– the current regularized ratio of actual improvement
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
    P,T,TStop<:StoppingCriterion,TRTM<:AbstractRetractionMethod
} <: AbstractHessianSolverState
    p::P
    X::T
    H::T
    S::T
    ς::Real
    ρ::Real
    stop::TStop
    retraction_method::TRTM
    ςmin::Real
    θ::Real
    η1::Real
    η2::Real
    γ1::Real
    γ2::Real
    γ3::Real
    function AdaptiveRegularizationState{P,T}(
        M::AbstractManifold,
        p::P =rand(M),
        X::T=zero_vector(M, p),
        H::T=zero_vector(M, p),
        S::T=zero_vector(M, p),
        ς::Real=1,
        ρ::Real=1,
        stop::StoppingCriterion=StopAfterIteration(100),       #TRTM?
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        ςmin::Real=1, #Set the below to appropriate default vals.
        θ::Real=1,
        η1::Real=1,
        η2::Real=1,
        γ1::Real=1,
        γ2::Real=1,
        γ3::Real=1
    ) where{P,T}
        o = new{P,T,typeof(stop),typeof(retraction_method)}()
        o.p=p
        o.X=X
        o.H=H
        o.S=S
        o.ς=ς
        o.ρ=ρ
        o.stop=stop
        o.retraction_method=retraction_method
        o.ςmin=ςmin
        o.θ=θ
        o.η1=η1
        o.η2=η2
        o.γ1=γ1
        o.γ2=γ2
        o.γ3=γ3
        return o
    end
end

function AdaptiveRegularizationState(
    M::AbstractManifold,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    H::T=zero_vector(M, p),
    S::T=zero_vector(M, p),
    ς::Real=0.01,# find sensible value for inital ς
    ρ::Real=1,
    stop::StoppingCriterion=StopAfterIteration(100),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    ςmin::Real=1e-10, #Set the below to appropriate default vals.
    θ::Real=1,
    η1::Real=0.1,
    η2::Real=0.9,
    γ1::Real=0.1,
    γ2::Real=2,
    γ3::Real=2
    ) where{P,T}
    return AdaptiveRegularizationState{P,T}(M, p, X, H, S, ς, ρ, stop, retraction_method, ςmin , θ, η1, η2, γ1, γ2, γ3)
end




function Lanczos!(M::AbstractManifold,mho::ManifoldHessianObjective,s::AdaptiveRegularizationState)
	dim=manifold_dimension(M)
	T=spdiagm(-1=>zeros(dim-1),0=>zeros(dim),1=>zeros(dim-1))
	Q = [zero_vector(M,s.p) for _ in 1:dim]
	g=get_gradient(M,mho,s.p)
	gradnorm=norm(M,s.p,g)
	q=g/gradnorm
	Q[1] .= q
	r=get_hessian(M,mho,s.p,q)
	α=inner(M,s.p,q,r)
	T[1,1]=α
	r=r-α*q

	#argmin of one dimensional model
	y=(α-sqrt(α^2+4*s.ς*gradnorm))/(2*s.ς) #verifed this

	for j in 1:dim-1
		β=norm(M,s.p,r)
		#Note: not doing MGS causes fast loss of orthogonality.
		if β>1e-10  # β large enough-> Do regular procedure: MGS of r wrt. Q
			for i in 1:j
				r=r-inner(M,s.p,Q[i],r)*Q[i]
			end
			q=r/β

		else # Generate new random vec and MGS of new vec wrt. Q
			r=rand(M,vector_at=s.p)
			for i in 1:j
				r=r-inner(M,s.p,Q[i],r)*Q[i]
			end
			q=r/norm(M,s.p,r)
		end
		r=get_hessian(M,mho,s.p,q)-β*Q[j]
		α=inner(M,s.p,r,q)
		r=r-α*q
		Q[j+1].=q
		T[j+1,j+1]=α
		T[j,j+1]=β
		T[j+1,j]=β

		#We have created the j+1'th orthonormal vector and the (j+1)x(j+1) T matrix.
		#Now compute the gradient corresponding to the j dimensional model.

		e1=zeros(j+1)#what is the standard Julia way?
		e1[1]=1
		model_gradnorm=norm(gradnorm*e1+T[1:j+1,1:j]*y+s.ς*norm(y,2)*vcat(y,0),2)

		#check stopping condition
		#print(model_gradnorm," <= ",Θ*norm(y,2)^2)

		#if model_gradnorm <= s.θ*norm(y,2)^2    #the loop stopped here. Temp comment out
		#	break
		#end

		#Minimize the (j+1) dimensional model by in the subspace of TₚM spanned by the (j+1) orthogonal vectors
		y=minimize_cubic_grad_descent(gradnorm,T[1:j+1,1:j+1],s.ς) # verified this in 2-dim.

	end
	#Assemble the tangent vector
	S_opt=zero_vector(M,s.p)
	for i in 1:length(y)
		S_opt=S_opt+Q[i]*y[i]
	end
    s.S=S_opt
	return S_opt
end


function minimize_cubic_grad_descent(gradnorm,T,ς)
	#minimize the cubic in the k-dimensional subspace of TₚM spanned by {q₁,...,qₖ} Lanczos vectors.
    #Equivalently minimize cubic in R^k with respect to the coordinates.
	#input: g, the current gradient norm.
	#T, the sparse kxk matrix that tridiagonalizes the hessian
	k=size(T)[1] # Dimension of subspace of TₚM spanned by k lanczos vectors.
	Mₑ=Euclidean(k)
	function cost(M,p)
		return gradnorm*p[1] + 0.5*p'*T*p +ς/3*norm(M,p,p)^3
	end
	function grad(M,p)
        X = T*p +ς*p*norm(M,p,p)
        X[1] += gradnorm
		return X
	end
	return gradient_descent(Mₑ,cost,grad,rand(Mₑ))
end

#Update the Regularization parameter in the same way as its done in Numerical Section of Boumal equation (39)
function UpdateRegParameterAndIterate!(M::AbstractManifold,s::AdaptiveRegularizationState)

    #Update iterate
    if s.ρ>=s.η1
        println("Updated iterate")
        s.p=retract(M,s.p,s.S)
    end

    #Update regularization parameter
    if s.ρ >= s.η2 #very successful
        println("very successful")
        s.ς=max(s.ςmin,s.γ1*s.ς)
    elseif s.η1<=s.ρ<s.η2
        println("successful")
        #leave regParam unchanged
    else #unsuccessful
        println("unsuccessful")
        s.ς=s.γ2*s.ς                #IMPORTANT:Adding safeguards?
    end
end

function ComputeRegRatio!(M::AbstractManifold,
    mho::ManifoldHessianObjective,
    s::AdaptiveRegularizationState
    )
    tmp1=get_cost(M,mho,s.p)-get_cost(M,mho,retract(M,s.p,s.S)) #change so that we denote optimal tanvec by s and not S.
    tmp2=-inner(M,s.p,get_gradient(M,mho,s.p)+get_hessian(M,mho,s.p,s.S),s.S)
    s.ρ=tmp1/tmp2
end

function step_solver!(dmp::AbstractManoptProblem,s::AdaptiveRegularizationState)
    M = get_manifold(dmp)
    mho = get_objective(dmp)
    Lanczos!(M,mho,s)
    ComputeRegRatio!(M,mho,s)
    UpdateRegParameterAndIterate!(M,s)
    return s
end















function OLDLanczos(M,mho::ManifoldHessianObjective,s::AdaptiveRegularizationState) #will alter function input
	dim=manifold_dimension(M)
    T=spdiagm(-1=>zeros(dim-1),0=>zeros(dim),1=>zeros(dim-1)) #Tridiag sp.matrix to store α's and β's
    qvecs=Vector(undef,dim) #Store orthonormal vecs. find better way. store in state?

    v=get_gradient(M,mho,s.p)
    q=v/norm(M,s.p,v)
	qvecs[1]=q

	Hq=get_hessian(M,mho,s.p,q)
	α=inner(M,s.p,q,Hq)
    T[1,1]=α
    #Here solve the k=1 case analytically.

	Hq_perp=Hq-α*q #must i embed?
	for j in 2:dim
		β=norm(Hq_perp,2)

        #In the MATLAB Code they check beta
		#if β>1e-12
		#	compute q
		#else
		#	reorthogonalize
		#end

		q=1/β*Hq_perp
        u=get_hessian(M,mho,s.p,q)-β*qvecs[j-1]
        α=inner(M,s.p,u,q)
		Hq_perp=u-α*q

        qvecs[j]=q
        T[j,j]=α
        T[j-1,j]=β
        T[j,j-1]=β

        #solve the problem here

        model_gradnorm=abc#write the modelnorm

	end
	return qvecs

end
