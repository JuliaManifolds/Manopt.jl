### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 6362be32-ed1a-11ef-0352-c7ccb14267de
using Pkg; Pkg.activate();

# ╔═╡ bc7188d7-38a4-49a3-9271-c2ad4b4ebf91
begin
	using LinearAlgebra
	using SparseArrays
	using Manopt
	#using ManoptExamples
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
	N=200
	
	st1 = 0.0
	halt1 = 1.0
	st = 0.5
	halt = pi/2

	h = (halt-st)/(N+1)
	#halt = pi - st
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1]
	
	y01 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT1 = [sin(halt),0,cos(halt)] # endpoint of geodesic

	y02 = [0,0,0] # start u
	yT2 = [0,0,0] # end u

	y03 = [0,0,0] # start lambda
	yT3 = [0,0,0] # end lambda

	α = 0.00031;
	yd = [1/sqrt(3)*[1.0,1.0,1.0] for Ωi in Omega];
	#yd = [[1.0,1.0,1.0] for Ωi in Omega];
end;

# ╔═╡ b211e406-61df-427c-885c-8217adc42540
function y1(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ bb07fd2d-df5f-4e4a-bd5f-c58bf2f1990f
function u(t)
	#return [sin(t*pi/2+pi/4), cos(t*pi/2+pi/4), 0]
	return [0.0, 0, 0]
end;

# ╔═╡ 930ead70-8331-4e78-b15a-f67cc6042acb
function λ(t)
	return [0.0, 0.0, 0.0]
end;

# ╔═╡ ee844858-db04-4a3f-947b-18130e3bf160
discretized_y = [y1(Ωi) for Ωi in Omega];

# ╔═╡ b0e6d152-91cc-4f29-940a-ec23a342ab59
discretized_u = [u(Ωi) for Ωi in Omega];

# ╔═╡ 347f61fc-fb14-47d0-8602-8550cf881ceb
discretized_λ = [λ(Ωi) for Ωi in Omega];

# ╔═╡ 33c572fe-0103-4e8e-82fa-4035955347b8
disc_y = ArrayPartition(discretized_y, discretized_u, discretized_λ);

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
function transport_by_proj_prime(S, p, dq, X)
	return (- dq*p' - p*dq')*X
end;

# ╔═╡ 777c9eab-0e5d-41d5-a73a-685ab24b82ff
function transport_by_proj_doubleprime(S, p, dq1, dq2, X)
	return (-dq1*dq2' - dq2 * dq1')*X
end;

# ╔═╡ 8cc250f5-f66e-4746-9855-664be0bd3408
function zerotrans_prime(S, p, X, dq)
	return 0.0*dq
end;

# ╔═╡ ecd3cb2a-887f-4cb2-bcb6-2bb58366ff07
function identitytrans(S, p, X, q)
	return X
end;

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
end;

# ╔═╡ ded804d7-792b-42c1-8283-237b3ec3faac
function state_equation_prime_y_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return Bdot'*Tdot
end;

# ╔═╡ 73b9e9eb-5048-4494-ade1-15ed552f1d75
function state_equation_at_with_dummyB(Integrand, y, ydot, B, Bdot, T, Tdot)
	return ydot.x[1]'*Tdot + y.x[2]'*T
end;

# ╔═╡ acea974a-d978-469a-86e2-7c50a91696e7
md"""
Stationary equation \
$$∫_\Omegaα⟨u, δu⟩ + ⟨δu, λ⟩ dt = 0 \; ∀δu$$\
and derivative w.r.t. u $$\rightsquigarrow L_{uu}$$ using zerotransport\
$$\alpha\int_\Omega \langle \delta u_2, \delta u_1\rangle \; dt$$
"""

# ╔═╡ db2cc3ff-e823-469f-9c8f-cd42b5cc837f
function stationary_equation_at(Integrand, y, ydot, T, Tdot)
	return α*y.x[2]'*T + T'*y.x[3]
end;

# ╔═╡ 50fc9f8e-6ab6-43c7-9762-7393ddca8827
function stationary_prime_u_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return α*B'*T
end;

# ╔═╡ 312cb447-a613-4180-a1bf-0a844c07fa82
md"""
and derivative w.r.t. $$\lambda$$ $$\rightsquigarrow L_{u\lambda}$$ using zerotransport
"""

# ╔═╡ 2037b581-7e46-4bce-ac57-5081951f2661
function stationary_prime_lambda_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return T'*B
end;

# ╔═╡ a5145e5e-78ad-4493-954f-9fd6b5c6fc5d
md"""
Derivative of the Adjoint equation\
$$∫_\Omega ⟨y-y_d, δy⟩ + ⟨δẏ, \dot{\lambda}⟩ dt + (A(y) - f(y)u)(P'(y)δyλ) = 0 \;∀δy$$\
w.r.t. u $$\rightsquigarrow L_{yu}$$ using zerotransport\
$$\int_\Omega \langle\delta u_2, P'(y)\delta y_1\lambda\rangle \; dt$$
"""

# ╔═╡ 1973b655-4368-4310-a218-be7b5be9f64e
function adjoint_prime_u_at(Integrand, y, ydot, B, Bdot, T, Tdot)
	return B'* transport_by_proj_prime(Integrand.domain, y.x[1], T, y.x[3])
end;

# ╔═╡ 79a1738b-450b-4cfa-822f-1811eb5a5905
md"""
First and second (euclidean) derivatives of the Lagrangian function w.r.t. y:\
$$L'(y)\delta y = \int_\Omega \langle y-y_d,\delta y \rangle + \langle \dot{\delta y},\dot{\lambda} \rangle\; dt$$\
$$L''(y)\delta y_2\delta y_1 = \int_\Omega \langle\delta y_2, \delta y_1\rangle\; dt$$\
$$\rightsquigarrow$$ ein Summand in $$L_{yy}$$
"""

# ╔═╡ ff8d7450-5c29-4030-9d70-8b149cc27837
function L_prime_at(Integrand, y, ydot, T, Tdot)
	  return (y.x[1] - yd[1])'*T + Tdot'*ydot.x[3]
end;

# ╔═╡ dbb9fbbd-38d5-45da-bc8c-770b9a91b675
function L_doubleprime_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return B'*T
end;

# ╔═╡ 53b242b6-745c-43fd-95cd-a05ca3e10a2f
function J1_prime_at(Integrand, y, ydot, T, Tdot)
	return (y.x[1] - yd[1])'*T
end;

# ╔═╡ a8d6c6d0-2b12-493a-9827-48a9706a20a2
function J1_doubleprime_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return B'*T
end;

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
# wir brauchen für λ am Ende dann einen Vektortransport (Projektion) in der Retraktion
product = ProductManifold(powerS, powerR3, powerR3lambda)
end;

# ╔═╡ c1ae6bcf-fab5-4456-8a11-7f36f125fffa
zerotransport=DifferentiableMapping(R3,R3,identitytrans,zerotrans_prime)

# ╔═╡ 259c5f1b-5447-4690-a706-5dcc4c9fa5bb
function P_prime_test_lambda(Integrand,y,B,T)
	return transport_by_proj_prime(S, y.x[1], T, y.x[3])
end;

# ╔═╡ a80c462e-b999-4f69-b6d5-536e94e7f536
function P_doubleprime_et_al(Integrand,y,B,T)
	return transport_by_proj_doubleprime(S, y.x[1], T, B, y.x[3]) + transport_by_proj_prime(S, y.x[1], transport_by_proj_prime(S, y.x[1], B, T), y.x[3])
	#return [0.0,0.0,0.0]
end;

# ╔═╡ 8552ff47-9c4c-4028-b6cd-635f144ae522
md"""
Integranden
"""

# ╔═╡ 4d79c53a-06ee-4c42-8fdf-678ca6a8c7e8
begin
	
# Vektortransport durch Orthogonalprojektion:
	transport=DifferentiableMapping(S,S,transport_by_proj,transport_by_proj_prime)

# für den L_yy-Block:
	integrand_L_prime = DifferentiableMapping(S,S, L_prime_at, L_doubleprime_at)
	# ergibt die ersten drei Summanden durch kovariante Ableitung, Assemblierung mit Vektortransport durch Projektion
	
	integrand_Lyy_1 = DifferentiableMapping(S,S,state_equation_at,state_equation_prime_y_at)
	transport_Lyy_1 = DifferentiableMapping(S,S,transport_by_proj,P_prime_test_lambda)
	# ergibt den vierten (durch Transponieren) und fünften Summanden

	integrand_Lyy_2 = DifferentiableMapping(S,S,state_equation_at,state_equation_at_with_dummyB)
	transport_Lyy_2 = DifferentiableMapping(S,S,transport_by_proj,P_doubleprime_et_al)
	# ergibt den sechsten und siebten Summanden

# für den L_yu-Block:
	integrand_Lyu = DifferentiableMapping(R3, S, state_equation_at, adjoint_prime_u_at) # state_equation_at wird nicht gebraucht, bei Assemblierung zerotransport verwenden!

# für den L_λy-Block:
	integrand_Lλy = DifferentiableMapping(S,S, state_equation_at, state_equation_prime_y_at)

# für den L_uλ-Block:
	integrand_Luλ = DifferentiableMapping(S, R3, stationary_equation_at, stationary_prime_lambda_at) # stationary_equation_at wird nicht gebraucht, bei Assemblierung zerotransport verwenden!

# für den L_uu-Block:
	integrand_Luu = DifferentiableMapping(R3,R3, stationary_equation_at, stationary_prime_u_at)


# für die rechte Seite:
integrandJ1 = DifferentiableMapping(S,S,J1_prime_at, J1_doubleprime_at)
	
integrand_state_eq = DifferentiableMapping(S,S,state_equation_at, state_equation_prime_y_at)

	
end;

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
		Pprimel=transport.derivative(integrand.domain,yl.x[row_idx],bfl*B[j],tfl*T[k])
		Pprimer=transport.derivative(integrand.domain,yr.x[row_idx],bfr*B[j],tfr*T[k])
			
		# Zeit- und y-Ableitungen der Projektionen
		Pprimedotl=(bfr-bfl)*Pprimel/h
		Pprimedotr=(bfr-bfl)*Pprimer/h	

		Pprimedot_neu = (Pprimer - Pprimel)/h
			
		# Einsetzen in die rechte Seite am rechten und linken Quadraturpunkt
		tmp+=integrand.value(integrand,yl,ydot,bfl*Pprimel,Pprimedot_neu)
		#tmp+=integrand.value(integrand,yr,ydot,bfr*Pprimel,Pprimedot_neu)
			
		# y-Ableitungen der Projektionen am rechten Punkt
		
		#tmp+=integrand.value(integrand,yl,ydot,bfl*Pprimer,Pprimedot_neu)
		tmp+=integrand.value(integrand,yr,ydot,bfr*Pprimer,Pprimedot_neu)
			
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
function get_Jac!(A,row_idx,col_idx,h,nCells,y,integrand)
	get_Jac!(A,row_idx,col_idx,h,nCells,y,integrand,zerotransport)
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


# ╔═╡ afc8df19-7226-4a37-b59a-236400d37cff
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
function assemble_local_Jac_Lyy!(A,row_idx, col_idx, h, i, yl, yr, B, bfl, bfr, T, tfl, tfr, integrand, transport)
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
			
		# Modifikation für Kovariante Ableitung:	
		# y-Ableitungen der Projektionen am linken Punkt
		# P'(yl)bfl*B[j] (tfl*T(k))

		Pprimel=transport.derivative(integrand.domain,yl,bfl*B[j],tfl*T[k])
		Pprimer=transport.derivative(integrand.domain,yr,bfr*B[j],tfr*T[k])
			
		# Zeit- und y-Ableitungen der Projektionen
		Pprimedotl=(bfr-bfl)*Pprimel/h
		Pprimedotr=(bfr-bfl)*Pprimer/h	

		Pprimedot_neu = (Pprimer - Pprimel)/h
			
		# Einsetzen in die rechte Seite am rechten und linken Quadraturpunkt

		tmp = integrand.derivative(integrand,yl,ydot,bfl*B[j],Bdot,bfl*Pprimel,Pprimedot_neu)
		#tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,bfr*Pprimel,Pprimedot_neu)
			
		
        # Update des Matrixeintrags

		#tmp+=integrand.derivative(integrand,yl,ydot,bfl*B[j], Bdot,bfl*Pprimer,Pprimedot_neu)
		tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,bfr*Pprimer,Pprimedot_neu)
			
		A[idxc+k,idx+j]+=quadwght*tmp
		end
	end
 end
end

# ╔═╡ 707d23d1-f7b4-49cb-b0ff-11ec536939fa
function evaluate(y, i, tloc)
	return ArrayPartition(
		(1.0-tloc)*y.x[1][i-1]+tloc*y.x[1][i],
		(1.0-tloc)*y.x[2][i-1]+tloc*y.x[2][i],
		(1.0-tloc)*y.x[3][i-1]+tloc*y.x[3][i]
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
function get_Jac!(A,row_idx,col_idx,h, nCells,y,integrand,transport)
	M = integrand.domain
	N = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells

		# Evaluation of the current iterate. This routine has to be provided from outside, because Knowledge about the basis functions is needed
		yl=evaluate(y,i,0.0)
		yr=evaluate(y,i,1.0)

		col_base_idx = col_idx

		if col_idx == 3
			col_base_idx = 1
		end

		Bcl=get_basis(M,yl.x[col_base_idx],DefaultOrthonormalBasis())
	    Bl = get_vectors(M,yl.x[col_base_idx], Bcl)
		Bcr=get_basis(M,yr.x[col_base_idx],DefaultOrthonormalBasis())
	    Br = get_vectors(M,yr.x[col_base_idx], Bcr)

		Tcl=get_basis(N,yl.x[row_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(N,yl.x[row_idx], Tcl)
		Tcr=get_basis(N,yr.x[row_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(N,yr.x[row_idx], Tcr)

        # In the following, all combinations of test and basis functions have to be considered.
		
    	assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  			Tl,1,0, integrand, transport)		
		assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tl,1,0, integrand, transport)		
		assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  Tr,0,1, integrand, transport)		
		assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tr,0,1, integrand, transport)		
	end
end

# ╔═╡ b8880b9b-047d-4a84-8494-c99f6b25d151
function get_rhs_row!(b,row_idx,h,nCells,y,integrand)
	CoDom = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells
		yl=evaluate(y,i,0.0)
		yr=evaluate(y,i,1.0)

		row_base_idx = row_idx

		if row_idx == 3
			row_base_idx = 1
		end

		Tcl=get_basis(CoDom,yl.x[row_base_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(CoDom, yl.x[row_base_idx], Tcl)
		Tcr=get_basis(CoDom,yr.x[row_base_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(CoDom, yr.x[row_base_idx], Tcr)
		
		assemble_local_rhs!(b,row_idx, h, i, yl, yr, Tl, 1, 0, integrand)		
        assemble_local_rhs!(b,row_idx, h, i, yl, yr, Tr, 0, 1, integrand)		
	end
end

# ╔═╡ 9fbe80b4-6371-41d3-b381-7fcd9dd9ece9
function get_rhs_simplified!(b,row_idx,h,nCells,y,y_trial,integrand,transport)
	S = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells
			yl=evaluate(y,i,0.0)
		    yr=evaluate(y,i,1.0)
		
			yl_trial=evaluate(y,i,0.0)
			yr_trial=evaluate(y,i,1.0)
		
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

			assemble_local_rhs!(b,row_idx, h, i, yl_trial, yr_trial, Tl, 1, 0, integrand)		
        	assemble_local_rhs!(b,row_idx, h, i, yl_trial, yr_trial, Tr, 0, 1, integrand)		
	end
end

# ╔═╡ d098fbc7-2407-45f7-99bd-21a77705bc16
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
function get_Jac_Lyy!(A,row_idx,col_idx,h, nCells,y,integrand,transport)
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
		
        assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0, Tl,1,0, integrand, transport)		
		assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1, Tl,1,0, integrand, transport)		
		assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0, Tr,0,1, integrand, transport)		
		assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1, Tr,0,1, integrand, transport)		

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
	#n3 = Int(manifold_dimension(submanifold(M, 3)))
	n3 = n1 # Dim muss 2 sein (aktuell liegt λ im R3, deswegen hier per Hand)

	Ac11::SparseMatrixCSC{Float64,Int32} = spzeros(n1,n1)
	Ac11_helper::SparseMatrixCSC{Float64,Int32} = spzeros(n1,n1)
	Ac12::SparseMatrixCSC{Float64,Int32} = spzeros(n1,n2) #Ac21 = Ac12'
	Ac13::SparseMatrixCSC{Float64,Int32} = spzeros(n1,n3) #Ac31 = Ac13'
	Ac22::SparseMatrixCSC{Float64,Int32} = spzeros(n2,n2)
	Ac23::SparseMatrixCSC{Float64,Int32} = spzeros(n2,n3) #Ac32 = Ac23'
	Ac33::SparseMatrixCSC{Float64,Int32} = spzeros(n3,n3)
	
	bc1 = zeros(n1)
	bc2 = zeros(n2)
	bc3 = zeros(n3)
	
	bctrial1=zeros(n1)
	bctrial2=zeros(n2)
	bctrial3=zeros(n3)

	projected_λ = project(powerS, p[M,1], p[M, 3]) 
	p[M, 3] = projected_λ
	# Retraktion für lambda (VT) per hand (λ+ = P(y+)(λ+δλ)), TODO: Im Newton die Retraktion richtig setzen
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(Omega)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(Omega)+1))
	Oy3 = OffsetArray([y03, p[M, 3]..., yT3], 0:(length(Omega)+1))
	Oy = ArrayPartition(Oy1,Oy2,Oy3);

	
	Oytrial1 = OffsetArray([y01, state.p_trial[M,1]..., yT1], 0:(length(Omega)+1))
	Oytrial2 = OffsetArray([y02, state.p_trial[M,2]..., yT2], 0:(length(Omega)+1))
	Oytrial3 = OffsetArray([y03, state.p_trial[M,3]..., yT3], 0:(length(Omega)+1))

	Oytrial = ArrayPartition(Oytrial1,Oytrial2,Oytrial3);
	
	nCells = length(Omega)+1

	get_Jac!(Ac11,1,1,h,nCells,Oy,integrand_L_prime,transport)
	get_Jac_Lyy!(Ac11_helper,1,1,h,nCells,Oy,integrand_Lyy_1,transport_Lyy_1)
	Ac11 += Ac11_helper + Ac11_helper'

	Ac11_helper *= 0.0
	get_Jac_Lyy!(Ac11_helper,1,1,h,nCells,Oy,integrand_Lyy_2,transport_Lyy_2)
	
	Ac11 += Ac11_helper
	#println(Matrix(Ac11))
	
	get_Jac!(Ac12,1,2,h,nCells,Oy, integrand_Lyu)
	get_Jac!(Ac23,2,3,h,nCells,Oy, integrand_Luλ)
	get_Jac!(Ac13,1,3,h,nCells,Oy,integrand_Lλy,transport)
	get_Jac!(Ac22,2,2,h,nCells,Oy,integrand_Luu)


	lambda_helper = get_coordinates(powerS, p[M,1], p[M,3], DefaultOrthogonalBasis())
	bc1 = Ac13 * lambda_helper
	
	get_rhs_row!(bc1,1,h,nCells,Oy,integrandJ1)
	get_rhs_row!(bc2,2,h,nCells,Oy,integrand_Luλ)
	get_rhs_row!(bc3,3,h,nCells,Oy,integrand_state_eq)

	println("norm bc1:", norm(bc1))
	println("norm bc2:", norm(bc2))
	println("norm bc3:", norm(bc3))

	
	Ac = vcat(hcat(Ac11 , Ac12 , Ac13), 
			  hcat(Ac12', Ac22 , Ac23), 
			  hcat(Ac13', Ac23', Ac33))

	if state.is_same == true
		bcsys = vcat(bc1, bc2, bc3)
	else
		#get_rhs_simplified!(bctrial1,1,h,nCells,Oy,Oytrial,integrand1)
		get_rhs_simplified!(bctrial2,2,h,nCells,Oy,Oytrial,integrand_Lyu,transport)
		get_rhs_simplified!(bctrial3,3,h,nCells,Oy,Oytrial,integrand_Luλ)
		bctrial = vcat(bctrial1,bctrial2, bctrail3)
    	bcsys=bctrial-(1.0 - state.stepsize.alpha)*vcat(bc1, bc2,bc3)
		println("alpha: ", state.stepsize.alpha)
	end
	Xc = (Ac) \ (-bcsys)
	M_helper = ProductManifold(powerS, powerR3, powerS)
	p_helper = ArrayPartition(p[M, 1], p[M, 2], p[M, 1])
	B = get_basis(M_helper, p_helper, DefaultOrthonormalBasis())
	res_c = get_vector(M_helper, p_helper, Xc, B)
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
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(product,1e-12; outer_norm=Inf)),
	#retraction_method=ProductRetraction(),
stepsize=ConstantLength(1.0),
#stepsize=Manopt.AffineCovariantStepsize(product),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)

# ╔═╡ 6920c45e-4ccb-4b60-a90c-c8df0792269c
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ 921845f5-2114-4e8f-a3c4-e1677e9134dd
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Norms of the Newton direction (semilog)"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ 817c3466-eeb7-4979-a738-1036b719361f
p_res = get_solver_result(st_res);

# ╔═╡ ac49e504-f3ba-4a5e-9a66-21160f38874e
begin
n = 45
u2 = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

#ws = [-w(p, integrand.scaling) for p in p_res]
for i in 1:n
    for j in 1:n
        sx[i,j] = cos.(u2[i]) * sin(v[j]);
        sy[i,j] = sin.(u2[i]) * sin(v[j]);
        sz[i,j] = cos(v[j]);
    end
end
fig, ax, plt = meshscatter(
  sx,sy,sz,
  color = fill(RGBA(1.,1.,1.,0.), n, n),
  shading = Makie.automatic,
  transparency=true
)
ax.show_axis = false

state_start = [y01, discretized_y ...,yT1]
state_final = [y01, p_res.x[1] ..., yT1]
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.1); transparency=true)
    π1(x) = 1.0*x[1]
    π2(x) = 1.0*x[2]
    π3(x) = 1.0*x[3]
	#arrows!(ax, π1.(p_res.x[1]), π2.(p_res.x[1]), π3.(p_res.x[1]), π1.(p_res.x[3]), π2.(p_res.x[3]), π3.(p_res.x[3]); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=2.5)
	scatterlines!(ax, π1.(state_final), π2.(state_final), π3.(state_final); markersize=8, color=:orange, linewidth=2)
	scatterlines!(ax, π1.(state_start), π2.(state_start), π3.(state_start); markersize=8, color=:blue, linewidth=2)
	scatter!(ax, π1.([y01, yT1]), π2.([y01, yT1]), π3.([y01, yT1]); markersize =10, color=:red)
	
	#scatter!(ax, 1.0,1.0,1.0; markersize =9, color=:red)
	scatter!(ax, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3); markersize =12, color=:red)
	
	#arrows!(ax, π1.(p_res.x[1]), π2.(p_res.x[1]), π3.(p_res.x[1]), π1.(-p_res.x[2]), π2.(-p_res.x[2]), π3.(-p_res.x[2]); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.05)
	fig
end

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
# ╠═73b9e9eb-5048-4494-ade1-15ed552f1d75
# ╟─acea974a-d978-469a-86e2-7c50a91696e7
# ╠═db2cc3ff-e823-469f-9c8f-cd42b5cc837f
# ╠═50fc9f8e-6ab6-43c7-9762-7393ddca8827
# ╟─312cb447-a613-4180-a1bf-0a844c07fa82
# ╠═2037b581-7e46-4bce-ac57-5081951f2661
# ╟─a5145e5e-78ad-4493-954f-9fd6b5c6fc5d
# ╠═1973b655-4368-4310-a218-be7b5be9f64e
# ╟─79a1738b-450b-4cfa-822f-1811eb5a5905
# ╠═ff8d7450-5c29-4030-9d70-8b149cc27837
# ╠═dbb9fbbd-38d5-45da-bc8c-770b9a91b675
# ╠═53b242b6-745c-43fd-95cd-a05ca3e10a2f
# ╠═a8d6c6d0-2b12-493a-9827-48a9706a20a2
# ╠═259c5f1b-5447-4690-a706-5dcc4c9fa5bb
# ╠═a80c462e-b999-4f69-b6d5-536e94e7f536
# ╟─208bcc35-4258-4aa4-9302-df0b44999f5f
# ╠═06d1f25c-80d8-4a07-8b53-3cabc90e70a9
# ╟─8552ff47-9c4c-4028-b6cd-635f144ae522
# ╠═4d79c53a-06ee-4c42-8fdf-678ca6a8c7e8
# ╟─75cc0c62-293c-4e21-bdf4-7156244a9d68
# ╠═697c0fbc-3717-4158-8d95-06893c143276
# ╠═cd866f05-b106-4cb7-a520-03f0bf8c4402
# ╠═85400860-8a15-4f6e-bec4-108bdb8da275
# ╠═6354afbc-28a1-4b33-a2ba-5878eb9d7d03
# ╠═b8880b9b-047d-4a84-8494-c99f6b25d151
# ╠═9fbe80b4-6371-41d3-b381-7fcd9dd9ece9
# ╠═d098fbc7-2407-45f7-99bd-21a77705bc16
# ╠═afc8df19-7226-4a37-b59a-236400d37cff
# ╠═707d23d1-f7b4-49cb-b0ff-11ec536939fa
# ╟─bd557dde-2dcc-42fc-9394-803b8d5af9b3
# ╠═e0b255cc-7ccd-4171-8550-e04191e77bf3
# ╠═a53b96cc-7093-4ce7-b3f2-7c94ddbc0a5a
# ╠═a231448c-8819-42c8-9826-59183c47526e
# ╠═45141a59-510a-4177-bab1-31be7450fcd9
# ╠═b3d4e21b-8183-4522-a463-63bef1810357
# ╠═5220a0dd-9484-4556-a7d0-ecf94955bf6c
# ╠═6920c45e-4ccb-4b60-a90c-c8df0792269c
# ╠═921845f5-2114-4e8f-a3c4-e1677e9134dd
# ╠═817c3466-eeb7-4979-a738-1036b719361f
# ╠═ac49e504-f3ba-4a5e-9a66-21160f38874e
