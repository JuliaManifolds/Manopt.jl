
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
    s::T
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
        p::P,
        X::T,
        H::T,
        s::T,
        ς::Real,
        ρ::Real,
        stop::StoppingCriterion=StopAfterIteration(100),       #TRTM?
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
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
        o.s=s
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



#write outer constructur, then try it out in pluto.


function AdaptiveRegularizationState{P,T}(
    p::P,
    X::T,
    H::T,
    s::T,
    ς::Real,
    ρ::Real,
    stop::StoppingCriterion=StopAfterIteration(100),    
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    ςmin::Real=1, #Set the below to appropriate default vals.
    θ::Real=1,
    η1::Real=1,
    η2::Real=1,
    γ1::Real=1,
    γ2::Real=1,
    γ3::Real=1
) where{P,T}





#Whats best to input?
function Lanczos(M,mho::ManifoldHessianObjective,s::AdaptiveRegularizationState) #will alter function input 
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

#just to see if I could call function in Plutonotebook. 
function testfunction(number)
    return 2*number
end






#function ComputeRegRatio(
#    mho::ManifoldHessianObjective,
#    s::AdaptiveRegularizationState
#    )
#    s.ρ=get_cost(M,mho,s.p)-get_cost(M,mho,retract)  
#end 

#function UpdateRegParameter(
#    mho::ManifoldHessianObjective,
#    s::AdaptiveRegularizationState
#    )
