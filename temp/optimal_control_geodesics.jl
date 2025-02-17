### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 6362be32-ed1a-11ef-0352-c7ccb14267de
using Pkg; Pkg.activate();

# ╔═╡ bc7188d7-38a4-49a3-9271-c2ad4b4ebf91
begin
	using LinearAlgebra
	using SparseArrays
	using Manopt
	using ManoptExamples
	using Manifolds
	using OffsetArrays
	using Random
	using RecursiveArrayTools
    using WGLMakie, Makie, GeometryTypes, Colors
	#using CairoMakie
	#using FileIO
end;

# ╔═╡ aeceb735-da1b-4db5-9964-538b316441c2
begin
	N=10
	
	st1 = 0.0
	halt1 = 1.0
	st = 0.5
	halt = pi/2

	h = (halt-st)/(N+1)
	#halt = pi - st
	Omega1 = range(; start=st, stop = halt, length=N+2)[2:end-1]
	Omega2 = range(; start=st, stop = halt, length=N+2)[2:end-1]
	Omega3 = range(; start=st, stop = halt, length=N+2)[2:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y01 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT1 = [sin(halt),0,cos(halt)] # endpoint of geodesic

	y02 = [1,0,0] # start u
	yT2 = [1,0,0] # end u

	y03 = [0,0,0] # start lambda
	yT3 = [0,0,0] # end lambda

end;

# ╔═╡ b211e406-61df-427c-885c-8217adc42540
function y1(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ bb07fd2d-df5f-4e4a-bd5f-c58bf2f1990f
function u(t)
	#return [sin(t*pi/2+pi/4), cos(t*pi/2+pi/4), 0]
	return [1.0, 0, 0]
end;

# ╔═╡ 930ead70-8331-4e78-b15a-f67cc6042acb
function λ(t)
	return [0.1, 0.1, 0.1]
end;

# ╔═╡ ee844858-db04-4a3f-947b-18130e3bf160
discretized_y = [y1(Ωi) for Ωi in Omega1];

# ╔═╡ b0e6d152-91cc-4f29-940a-ec23a342ab59
discretized_u = [u(Ωi) for Ωi in Omega3];

# ╔═╡ 347f61fc-fb14-47d0-8602-8550cf881ceb
discretized_λ = [λ(Ωi) for Ωi in Omega2];

# ╔═╡ 33c572fe-0103-4e8e-82fa-4035955347b8
disc_y = ArrayPartition(discretized_y, discretized_u, discretized_λ)

# ╔═╡ 9b589eb8-c42c-45cf-a2d4-89fd1d522759
"""
Such a structure has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""
mutable struct DifferentiableMapping{M<:AbstractManifold, N<:AbstractManifold,F1<:Function,F2<:Function}
	domain::M
	precodomain::N
	value::F1
	derivative::F2
	#scaling::T
end


# ╔═╡ b22850cc-26fa-4fbe-a919-0ae95c8571d6
"""
 The following two routines define the vector transport, its derivative and its second derivative. The first derivative is needed to obtain covariant derivative from the ordinary derivative.
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ 3ee0ae2b-200e-4ce4-8dc9-b873bb6873f4
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ 777c9eab-0e5d-41d5-a73a-685ab24b82ff
function transport_by_proj_doubleprime(S, p, X, dq1, dq2)
	return (-dq1*dq2' - dq2 * dq1')*X
end

# ╔═╡ 8cc250f5-f66e-4746-9855-664be0bd3408
function zerotrans_prime(S, p, X, dq)
	return 0.0*X
end

# ╔═╡ ecd3cb2a-887f-4cb2-bcb6-2bb58366ff07
function identitytrans(S, p, X, q)
	return X
end

# ╔═╡ a3a135f0-c5e8-4825-8653-cbd68bbe795e
md"""
State equation \
$$(A(y) - f(y)u)δλ = ∫_\Omega⟨ẏ, \dot{\delta \lambda}⟩ + ⟨u, δλ⟩\; dt = 0 \; ∀ δλ$$\
and linearized state equation $$\rightsquigarrow L_{\lambda y}$$ and adjoint equation (rhs)\
$$\int_\Omega\langle \dot{\delta y_2}, \dot{\delta \lambda_1}\rangle \;dt$$
"""

# ╔═╡ 423eadae-bc19-48a7-a468-3506e20a784c
function state_equation_at(Integrand, y, ydot, T, Tdot)
	return ydot.x[1]'*Tdot + y.x[2]'*T
end

# ╔═╡ ded804d7-792b-42c1-8283-237b3ec3faac
function state_prime_lambda_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return Bdot'*Tdot
end

# ╔═╡ acea974a-d978-469a-86e2-7c50a91696e7
md"""
Stationary equation \
$$∫_\Omegaα⟨u, δu⟩ + ⟨δu, λ⟩ dt = 0 \; ∀δu$$\
and derivative w.r.t. u $$\rightsquigarrow L_{uu}$$ using zerotransport\
$$\alpha\int_\Omega \langle \delta u_2, \delta u_1\rangle \; dt$$
"""

# ╔═╡ d1ae1ab8-9815-44f5-8478-14ac21d9b15f
α = 0.5;

# ╔═╡ db2cc3ff-e823-469f-9c8f-cd42b5cc837f
function stationary_equation_at(Integrand, y, ydot, T, Tdot)
	return α*y.x[2]'*T + T'*y.x[3]
end

# ╔═╡ 50fc9f8e-6ab6-43c7-9762-7393ddca8827
function stationary_prime_u_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return α*Bdot'*Tdot
end

# ╔═╡ 312cb447-a613-4180-a1bf-0a844c07fa82
md"""
for $$L_{u\lambda}$$ (using zerotransport):
"""

# ╔═╡ 2037b581-7e46-4bce-ac57-5081951f2661
function stationary_prime_lambda_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return - T'*B
end

# ╔═╡ a5145e5e-78ad-4493-954f-9fd6b5c6fc5d
md"""
Derivative of the Adjoint equation\
$$∫_\Omega ⟨y-y_d, δy⟩ + ⟨δẏ, λ̇⟩ dt + (A(y) - f(y)u)(P'(y)δyλ) = 0 \;∀δy$$\
w.r.t. u $$\rightsquigarrow L_{yu}$$ using zerotransport\
$$-\int_\Omega \langle\delta u_2, P'(y)\delta y_1\lambda\rangle \; dt$$
"""

# ╔═╡ 1973b655-4368-4310-a218-be7b5be9f64e
function adjoint_prime_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return -B'* transport_by_proj_prime(Integrand.domain, y.x[1], T, y.x[3])
end

# ╔═╡ 79a1738b-450b-4cfa-822f-1811eb5a5905
md"""
First and second (euclidean) derivatives of the Lagrangian function w.r.t. y:\
$$L'(y)\delta y = \int_\Omega \langle y-y_d,\delta y \rangle + \langle \dot{\delta y},\dot{\lambda} \rangle\; dt$$\
$$L''(y)\delta y_2\delta y_1 = \int_\Omega \langle\delta y_2, \delta y_1\rangle\; dt$$\
$$\rightsquigarrow$$ ein Summand in $$L_{yy}$$
"""

# ╔═╡ 953e02ea-4c40-4b5a-a4ec-102ae84a8627
yd = [y1(Ωi) for Ωi in Omega1];

# ╔═╡ ff8d7450-5c29-4030-9d70-8b149cc27837
function L_prime_at(Integrand, y, ydot, T, Tdot)
	  return (y.x[1] - yd[1])'*T + Tdot'*ydot.x[3]
end

# ╔═╡ dbb9fbbd-38d5-45da-bc8c-770b9a91b675
function L_doubleprime_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return B'*T
end

# ╔═╡ 208bcc35-4258-4aa4-9302-df0b44999f5f
md"""
Mannigfaltigkeiten für Variablen
$$(y_{disc}, u_{disc}, \lambda_{disc}) \in (\mathbb{S}^2)^N \times (\mathbb{R}^3)^N \times (T_{y(t)}\mathbb{S}^2)^N $$\
$$(y(t),u(t),\lambda(t)) \in \mathbb{S}^2 \times \mathbb{R}^3 \times T_{y(t)}\mathbb{S}^2$$
"""

# ╔═╡ 06d1f25c-80d8-4a07-8b53-3cabc90e70a9
begin
S = Manifolds.Sphere(2)
R3 = Manifolds.Euclidean(3)	
TS = TangentBundle(S)
	
powerS = PowerManifold(S, NestedPowerRepresentation(), N) #y
powerR3 = PowerManifold(R3, NestedPowerRepresentation(), N) #u
powerR3lambda = PowerManifold(R3, NestedPowerRepresentation(), N) #λ
product = ProductManifold(powerS, powerR3, powerR3lambda)
end;

# ╔═╡ c1ae6bcf-fab5-4456-8a11-7f36f125fffa
zerotransport=DifferentiableMapping(R3,R3,identitytrans,zerotrans_prime)

# ╔═╡ 8552ff47-9c4c-4028-b6cd-635f144ae522
md"""
Integranden
"""

# ╔═╡ 4d79c53a-06ee-4c42-8fdf-678ca6a8c7e8
begin
integrand_adjoint_equation=DifferentiableMapping(S,S,state_equation_at,L_prime_at)
integrand_L_prime = DifferentiableMapping(S,S, L_prime_at, L_doubleprime_at)
integrand_Lyu = DifferentiableMapping(S, R3, state_equation_at, adjoint_prime_at) # state_equation_at wird nicht gebraucht, bei Assemblierung zerotransport verwenden!
integrand_Luλ = DifferentiableMapping(R3, S, stationary_equation_at, stationary_prime_lambda_at) # stationary_equation_at wird nicht gebraucht, bei Assemblierung zerotransport verwenden!
integrand_Lλy = DifferentiableMapping(S,S, state_equation_at, state_prime_lambda_at)
integrand_Luu = DifferentiableMapping(R3,R3, stationary_equation_at, stationary_prime_u_at)
end;

# ╔═╡ db5d5c75-d4f6-4737-b5dc-a839a81395ac
md"""
Vektortransport"""

# ╔═╡ 7d6721b5-df0e-4d58-bd6f-aa7dc4634c41
transport=DifferentiableMapping(S,S,transport_by_proj,transport_by_proj_prime)

# ╔═╡ 75cc0c62-293c-4e21-bdf4-7156244a9d68
md"""
Assemblierungsroutinen:
"""

# ╔═╡ 697c0fbc-3717-4158-8d95-06893c143276
"""
 A:      Matrix to be written into\\
row_idx: row index of block inside system\\
col_idx: column index of block inside system\\

h:       length of interval\\
i:       index of interval\\

yl:      left value of iterate\\
yr:      right value of iterate\\

B:       basis vector for basis function\\
bfl:     0/1 scaling factor at left boundary\\
bfr:     0/1 scaling factor at right boundary \\

T:       basis vector for test function\\
tfl:     0/1 scaling factor at left boundary\\
tfr:     0/1 scaling factor at right boundary \\
...
"""
function assemble_local_Jac_with_connection!(A,row_idx, col_idx, h, i, yl, yr, B, bfl, bfr, T, tfl, tfr, integrand, transport)
 dim = manifold_dimension(integrand.domain)
 dimc = manifold_dimension(integrand.precodomain)
if tfr == 1
	idxc=dimc*(i-1)
else 
	idxc=dimc*(i-2)
end
if bfr == 1
	idx=dim*(i-1)
else 
	idx=dim*(i-2)
end

 ydot=(yr-yl)/h
 quadwght=0.5*h
 nA1=size(A,1)
 nA2=size(A,2)
 #	Schleife über Komponenten der Testfunktion
 for k in 1:dimc
    # Schleife über Komponenten der Basisfunktion
	for j in 1:dim
		# Sicherstellen, dass wir in Indexgrenzen der Matrix bleiben
        if idx+j >= 1 && idxc+k >= 1 && idx+j <= nA2 && idxc+k <= nA1

		# Zeit-Ableitungen der Basis- und Testfunktionen (=0 am jeweils anderen Rand)
     	Tdot=(tfr-tfl)*T[k]/h
		Bdot=(bfr-bfl)*B[j]/h


    	# Ableitung in der Einbettung am rechten und linken Quadraturpunkt
        tmp=integrand.derivative(integrand,yl,ydot,bfl*B[j],Bdot,tfl*T[k],Tdot)	
			
		tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,tfr*T[k],Tdot)	
			
		# Modifikation für Kovariante Ableitung:	
		# y-Ableitungen der Projektionen am linken Punkt
		# P'(yl)bfl*B[j] (tfl*T(k))
		Pprime=transport.derivative(integrand.domain,yl.x[row_idx],bfl*B[j],tfl*T[k])
		# Zeit- und y-Ableitungen der Projektionen
		Pprimedot=(bfr-bfl)*Pprime/h
		# Einsetzen in die rechte Seite am rechten und linken Quadraturpunkt
		tmp+=integrand.value(integrand,yl,ydot,bfl*Pprime,Pprimedot)
		tmp+=integrand.value(integrand,yr,ydot,bfr*Pprime,Pprimedot)
			
		# y-Ableitungen der Projektionen am rechten Punkt
		Pprime=transport.derivative(integrand.domain,yr.x[row_idx],bfr*B[j],tfr*T[k])
		Pprimedot=(bfr-bfl)*Pprime/h			
		tmp+=integrand.value(integrand,yl,ydot,bfl*Pprime,Pprimedot)
		tmp+=integrand.value(integrand,yr,ydot,bfr*Pprime,Pprimedot)
			
        # Update des Matrixeintrags
			
		A[idxc+k,idx+j]+=quadwght*tmp
		end
	end
 end
end

# ╔═╡ 85400860-8a15-4f6e-bec4-108bdb8da275
"""
If no vector transport is needed, leave it away, then a zero dummy transport is used
"""
function get_Jac!(A,row_idx,degT,col_idx,degB,h,nCells,y,integrand)
	get_Jac!(A,row_idx,degT,col_idx,degB,h,nCells,y,integrand,zerotransport)
end

# ╔═╡ 6354afbc-28a1-4b33-a2ba-5878eb9d7d03
"""
  T          : Test function
  tlf        : 1/0 : scaling of test function at left interval point
  trf        : 1/0 : scaling of test function at right interval point
"""
function assemble_local_rhs!(b,row_idx, h, i, yl, yr, T, tlf, trf, integrand)
    dimc = manifold_dimension(integrand.precodomain)
	if trf == 1
     idx=dimc*(i-1)
	else 
	 idx=dimc*(i-2)
	end
	ydotl=(yr-yl)/h
	ydotr=(yr-yl)/h
	# Trapezregel
	quadwght = 0.5*h   
	for k in 1:dimc
		# finite differences, taking into account values of test function at both endpoints
        if idx+k > 0 && idx+k <= length(b)
			
			Tdot = (trf-tlf)*T[k]/h
			#linker Quadraturpunkt
			tmp =  integrand.value(integrand,yl,ydotl,tlf*T[k],Tdot)	
			#rechter Quadraturpunkt
			tmp += integrand.value(integrand,yr,ydotr,trf*T[k],Tdot)	
			# Update der rechten Seite
		  	b[idx+k]+= quadwght*tmp	
		end
	end
end


# ╔═╡ 707d23d1-f7b4-49cb-b0ff-11ec536939fa
function evaluate(y, i, tloc)
	return ArrayPartition(
		(1.0-tloc)*y.x[1][i-1]+tloc*y.x[1][i],
		(1.0-tloc)*y.x[2][i-1]+tloc*y.x[2][i],
		y.x[3][i]
	)
end

# ╔═╡ cd866f05-b106-4cb7-a520-03f0bf8c4402
"""
 A:      Matrix to be written into\\
row_idx: row index of block inside system\\
detT:    degree of test function: 1: linear, 0: constant\\
col_idx: column index of block inside system\\
detB:    degree of basis function: 1: linear, 0: constant\\
h:       length of interval\\
nCell:    total number of intervals\\
y:       iterate\\
...
"""
function get_Jac!(A,row_idx,degT,col_idx,degB,h, nCells,y,integrand,transport)
	M = integrand.domain
	N = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells

		# Evaluation of the current iterate. This routine has to be provided from outside, because Knowledge about the basis functions is needed
		yl=evaluate(y,i,0.0)
		yr=evaluate(y,i,1.0)

		#yl=ArrayPartition(getindex.(y.x, (i-1...,)))
		#yr=ArrayPartition(getindex.(y.x, (i...,)))

		Bcl=get_basis(M,yl.x[col_idx],DefaultOrthonormalBasis())
	    Bl = get_vectors(M,yl.x[col_idx], Bcl)
		Bcr=get_basis(M,yr.x[col_idx],DefaultOrthonormalBasis())
	    Br = get_vectors(M,yr.x[col_idx], Bcr)

		Tcl=get_basis(N,yl.x[row_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(N,yl.x[row_idx], Tcl)
		Tcr=get_basis(N,yr.x[row_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(N,yr.x[row_idx], Tcr)

        # In the following, all combinations of test and basis functions have to be considered.
		
		# The case, where both test and basis functions are linear. We have 2x2=4 combinations, since there are two test/basis functions on each interval
		if degT==1 && degB == 1
    	    assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  			Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  Tr,0,1, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tr,0,1, integrand, transport)		
		end
		# The case, where both test functions are linear and basis functions are piecewies constant. We have 1x2=2 combinations, since there are are two test functions and 1 basis function on each interval
		if degT==1 && degB == 0
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,1,1,  Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,1,1,  Tr,0,1, integrand, transport)		
		end
		# Other cases could be added here. I did not need them, thus I havent implented them
	end
end

# ╔═╡ b8880b9b-047d-4a84-8494-c99f6b25d151
function get_rhs_row!(b,row_idx,degT,h,nCells,y,integrand)
	CoDom = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells
		yl=evaluate(y,i,0.0)
		yr=evaluate(y,i,1.0)
		#yl=ArrayPartition(getindex.(y.x, (i-1...,)))
		#yr=ArrayPartition(getindex.(y.x, (i...,)))

		Tcl=get_basis(CoDom,yl.x[row_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(CoDom, yl.x[row_idx], Tcl)
		Tcr=get_basis(CoDom,yr.x[row_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(CoDom, yr.x[row_idx], Tcr)
		if degT == 1
			assemble_local_rhs!(b,row_idx, h, i, yl, yr, Tl, 1, 0, integrand)		
        	assemble_local_rhs!(b,row_idx, h, i, yl, yr, Tr, 0, 1, integrand)		
		end
		if degT == 0
        	assemble_local_rhs!(b,row_idx, h, i, yl, yr, Tr, 1, 1, integrand)		
		end
	end
end

# ╔═╡ 9fbe80b4-6371-41d3-b381-7fcd9dd9ece9
function get_rhs_simplified!(b,row_idx,degT,h,nCells,y,y_trial,integrand,transport)
	S = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells
			yl=evaluate(y,i,0.0)
		    yr=evaluate(y,i,1.0)

			#yl=ArrayPartition(getindex.(y.x, (i-1...,)))
			#yr=ArrayPartition(getindex.(y.x, (i...,)))
			
			yl_trial=evaluate(y,i,0.0)
			yr_trial=evaluate(y,i,1.0)
		
			#yl_trial=ArrayPartition(getindex.(y_trial.x, (i-1...,)))
			#yr_trial=ArrayPartition(getindex.(y_trial.x, (i...,)))
		
			Tcl=get_basis(S,yl.x[row_idx],DefaultOrthonormalBasis())
			Tl=get_vectors(S, yl.x[row_idx],Tcl)
		
			Tcr=get_basis(S,yr.x[row_idx],DefaultOrthonormalBasis())
	    	Tr = get_vectors(S, yr.x[row_idx], Tcr)
		
			dim = manifold_dimension(S)
		# Transport der Testfunktionen auf $T_{x_k}S$
            for k=1:dim
				Tl[k]=transport.value(S,yl.x[row_idx],Tl[k],yl_trial.x[row_idx])
				Tr[k]=transport.value(S,yr.x[row_idx],Tr[k],yr_trial.x[row_idx])
			end
			if degT == 1
			assemble_local_rhs!(b,row_idx, h, i, yl_trial, yr_trial, Tl, 1, 0, integrand)		
        	assemble_local_rhs!(b,row_idx, h, i, yl_trial, yr_trial, Tr, 0, 1, integrand)		
			end
			if degT == 0
        	assemble_local_rhs!(b,row_idx, h, i, yl_trial, yr_trial, Tr, 1, 1, integrand)		
			end
	end
end

# ╔═╡ bd557dde-2dcc-42fc-9394-803b8d5af9b3
md"""
Für Aufruf Newton
"""

# ╔═╡ e0b255cc-7ccd-4171-8550-e04191e77bf3
"""
Dummy
"""
function bundlemap(M, y)
		# Include boundary points
end

# ╔═╡ a53b96cc-7093-4ce7-b3f2-7c94ddbc0a5a
"""
Dummy
"""
function connection_map(E, q)
    return q
end

# ╔═╡ 45141a59-510a-4177-bab1-31be7450fcd9
function solve_linear_system(M, p, state, prob)
	obj = get_objective(prob)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	Ac11::SparseMatrixCSC{Float64,Int32} =spzeros(n1,n1)
	Ac12::SparseMatrixCSC{Float64,Int32} =spzeros(n1,n2) #Ac21 = Ac12'
	Ac13::SparseMatrixCSC{Float64,Int32} =spzeros(n1,n3) #Ac31 = Ac13'
	Ac22::SparseMatrixCSC{Float64,Int32} =spzeros(n2,n2)
	Ac23::SparseMatrixCSC{Float64,Int32} =spzeros(n2,n3) #Ac32 = Ac23'
	Ac33::SparseMatrixCSC{Float64,Int32} =spzeros(n3,n3)
	
	bc1 = zeros(n1)
	bc2 = zeros(n2)
	bc3 = zeros(n3)
	
	bctrial1=zeros(n1)
	bctrial2=zeros(n2)
	bctrial3=zeros(n3)
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(Omega1)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(Omega2)+1))
	Oy3 = OffsetArray([y03, p[M, 3]..., yT3], 0:(length(Omega3)+1))
	Oy = ArrayPartition(Oy1,Oy2,Oy3);

    println("Iter: ",norm(Oy))
	
	Oytrial1 = OffsetArray([y01, state.p_trial[M,1]..., yT1], 0:(length(Omega1)+1))
	Oytrial2 = OffsetArray([y02, state.p_trial[M,2]..., yT2], 0:(length(Omega2)+1))
	Oytrial3 = OffsetArray([y03, state.p_trial[M,3]..., yT3], 0:(length(Omega3)+1))

	Oytrial = ArrayPartition(Oytrial1,Oytrial2,Oytrial3);
	
	println("Assemble:")
	nCells = length(Omega3)

	get_Jac!(Ac11,1,1,1,1,h,nCells,Oy,integrand_L_prime,transport)
	
	get_Jac!(Ac12,1,1,2,1,h,nCells,Oy, integrand_Lyu)
	get_Jac!(Ac23,2,1,3,1,h,nCells,Oy,integrand_Luλ)
	get_Jac!(Ac13,1,1,3,1,h,nCells,Oy,integrand_Lλy,transport)
	get_Jac!(Ac22,2,1,2,1,h,nCells,Oy,integrand_Luu)
	
	#get_rhs_row!(bc1,1,1,h,nCells,Oy,integrand1)
	get_rhs_row!(bc2,2,1,h,nCells,Oy,integrand_Lyu)
	get_rhs_row!(bc3,3,1,h,nCells,Oy,integrand_Luλ)
	
	Ac = vcat(hcat(Ac11 , Ac12 , Ac13), 
			  hcat(Ac12', Ac22 , Ac23), 
			  hcat(Ac13', Ac23', Ac33))

	if state.is_same == true
		bcsys = vcat(bc1, bc2, bc3)
		#println("rhs", bcsys)
	else
		#get_rhs_simplified!(bctrial1,1,1,h,nCells,Oy,Oytrial,integrand1)
		get_rhs_simplified!(bctrial2,2,1,h,nCells,Oy,Oytrial,integrand_Lyu,transport)
		get_rhs_simplified!(bctrial3,3,1,h,nCells,Oy,Oytrial,integrand_Luλ)
		bctrial = vcat(bctrial1,bctrial2, bctrail3)
    	bcsys=bctrial-(1.0 - state.stepsize.alpha)*vcat(bc1, bc2,bc3)
		println("alpha: ", state.stepsize.alpha)
	end
    #println(bcsys)
	println("Solve:")
	Xc = (Ac) \ (-bcsys)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	res_c = get_vector(M, p, Xc, B)
	println("norm:", norm(res_c))
	#println("res:", res_c)
	return res_c
end

# ╔═╡ a231448c-8819-42c8-9826-59183c47526e
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

# ╔═╡ b3d4e21b-8183-4522-a463-63bef1810357
""" Initial geodesic """
y_0 = copy(product, disc_y)

# ╔═╡ 5220a0dd-9484-4556-a7d0-ecf94955bf6c
st_res = vectorbundle_newton(product, TangentBundle(product), bundlemap, bundlemap, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(product,1e-13; outer_norm=Inf)),
	#retraction_method=ProductRetraction(),
stepsize=ConstantLength(1.0),
#stepsize=Manopt.AffineCovariantStepsize(product),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)

# ╔═╡ Cell order:
# ╠═6362be32-ed1a-11ef-0352-c7ccb14267de
# ╠═bc7188d7-38a4-49a3-9271-c2ad4b4ebf91
# ╠═aeceb735-da1b-4db5-9964-538b316441c2
# ╠═b211e406-61df-427c-885c-8217adc42540
# ╠═bb07fd2d-df5f-4e4a-bd5f-c58bf2f1990f
# ╠═930ead70-8331-4e78-b15a-f67cc6042acb
# ╠═ee844858-db04-4a3f-947b-18130e3bf160
# ╠═b0e6d152-91cc-4f29-940a-ec23a342ab59
# ╠═347f61fc-fb14-47d0-8602-8550cf881ceb
# ╠═33c572fe-0103-4e8e-82fa-4035955347b8
# ╠═9b589eb8-c42c-45cf-a2d4-89fd1d522759
# ╟─b22850cc-26fa-4fbe-a919-0ae95c8571d6
# ╠═3ee0ae2b-200e-4ce4-8dc9-b873bb6873f4
# ╠═777c9eab-0e5d-41d5-a73a-685ab24b82ff
# ╠═8cc250f5-f66e-4746-9855-664be0bd3408
# ╠═ecd3cb2a-887f-4cb2-bcb6-2bb58366ff07
# ╠═c1ae6bcf-fab5-4456-8a11-7f36f125fffa
# ╟─a3a135f0-c5e8-4825-8653-cbd68bbe795e
# ╠═423eadae-bc19-48a7-a468-3506e20a784c
# ╠═ded804d7-792b-42c1-8283-237b3ec3faac
# ╟─acea974a-d978-469a-86e2-7c50a91696e7
# ╠═d1ae1ab8-9815-44f5-8478-14ac21d9b15f
# ╠═db2cc3ff-e823-469f-9c8f-cd42b5cc837f
# ╠═50fc9f8e-6ab6-43c7-9762-7393ddca8827
# ╟─312cb447-a613-4180-a1bf-0a844c07fa82
# ╠═2037b581-7e46-4bce-ac57-5081951f2661
# ╟─a5145e5e-78ad-4493-954f-9fd6b5c6fc5d
# ╠═1973b655-4368-4310-a218-be7b5be9f64e
# ╟─79a1738b-450b-4cfa-822f-1811eb5a5905
# ╠═953e02ea-4c40-4b5a-a4ec-102ae84a8627
# ╠═ff8d7450-5c29-4030-9d70-8b149cc27837
# ╠═dbb9fbbd-38d5-45da-bc8c-770b9a91b675
# ╟─208bcc35-4258-4aa4-9302-df0b44999f5f
# ╠═06d1f25c-80d8-4a07-8b53-3cabc90e70a9
# ╟─8552ff47-9c4c-4028-b6cd-635f144ae522
# ╠═4d79c53a-06ee-4c42-8fdf-678ca6a8c7e8
# ╟─db5d5c75-d4f6-4737-b5dc-a839a81395ac
# ╠═7d6721b5-df0e-4d58-bd6f-aa7dc4634c41
# ╟─75cc0c62-293c-4e21-bdf4-7156244a9d68
# ╠═697c0fbc-3717-4158-8d95-06893c143276
# ╠═cd866f05-b106-4cb7-a520-03f0bf8c4402
# ╠═85400860-8a15-4f6e-bec4-108bdb8da275
# ╠═6354afbc-28a1-4b33-a2ba-5878eb9d7d03
# ╠═b8880b9b-047d-4a84-8494-c99f6b25d151
# ╠═9fbe80b4-6371-41d3-b381-7fcd9dd9ece9
# ╠═707d23d1-f7b4-49cb-b0ff-11ec536939fa
# ╟─bd557dde-2dcc-42fc-9394-803b8d5af9b3
# ╠═e0b255cc-7ccd-4171-8550-e04191e77bf3
# ╠═a53b96cc-7093-4ce7-b3f2-7c94ddbc0a5a
# ╠═a231448c-8819-42c8-9826-59183c47526e
# ╠═45141a59-510a-4177-bab1-31be7450fcd9
# ╠═b3d4e21b-8183-4522-a463-63bef1810357
# ╠═5220a0dd-9484-4556-a7d0-ecf94955bf6c
