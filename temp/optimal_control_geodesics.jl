### A Pluto.jl notebook ###
# v0.20.13

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
	using RecursiveArrayTools
    using WGLMakie, Makie, GeometryTypes, Colors
	using CairoMakie
	using DataFrames, CSV
end;

# ╔═╡ 66fbe00e-1d30-468d-baa5-05e099cf329e
#Pkg.develop(path="/Users/bt308990/Documents/julia/Manopt-old.jl")

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

# ╔═╡ c6b1fd66-ef1f-427e-b8cc-d2da9c127ee3
begin
	function y(t)
		return [sin(t), 0, cos(t)]
	end

	function u(t)
		#return [sin(t*pi/2+pi/4), cos(t*pi/2+pi/4), 0]
		return [0.0, 0, 0]
	end

	function λ(t)
		return [0.0, 0.0, 0.0]
	end

	discretized_y = [y(Ωi) for Ωi in Omega];
	discretized_u = [u(Ωi) for Ωi in Omega];
	discretized_λ = [λ(Ωi) for Ωi in Omega];

	disc_y = ArrayPartition(discretized_y, discretized_u, discretized_λ);
	
end;

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
md"""
The following two routines define the vector transport, its derivative and its second derivative. The first derivative is needed to obtain covariant derivative from the ordinary derivative.
"""

# ╔═╡ 6dc5ca81-70b6-4aab-8f52-64b696e06217
begin
	function transport_by_proj(S, p, X, q)
		return X - q*(q'*X)
	end

	function transport_by_proj_prime(S, p, dq, X)
		return (- dq*p' - p*dq')*X
	end

	function transport_by_proj_doubleprime(S, p, dq1, dq2, X)
		return (-dq1*dq2' - dq2 * dq1')*X
	end

end;

# ╔═╡ 8596f05e-bc6f-4891-80ee-e45e9d730089
md"""
Helper routines
"""

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

# ╔═╡ 452addcd-d0f4-4059-8de6-448045e5b670
begin
	function zerotrans_prime(S, p, X, dq)
		return 0.0*dq
	end

	function identitytrans(S, p, X, q)
		return X
	end

	zerotransport=DifferentiableMapping(R3,R3,identitytrans,zerotrans_prime)
end;

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

# ╔═╡ 9cad614b-f5b1-43ac-a612-6f32458f7cdd
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

# ╔═╡ ea918bcc-2d49-47cf-ad9c-79ae3bc9484c
"""
If no vector transport is needed, leave it away, then a zero dummy transport is used
"""
function get_Jac!(A,row_idx,col_idx,h,nCells,y,integrand)
	get_Jac!(A,row_idx,col_idx,h,nCells,y,integrand,zerotransport)
end

# ╔═╡ f7d66dfb-d246-486a-9033-ab82fde3d586
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

# ╔═╡ 6cf5ad9a-b3dc-4a11-bf01-3aa188c9667b
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

# ╔═╡ c9b6a526-7b30-44bf-b652-ff9c17a4599c
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

		Bcl= get_basis(M,yl.x[col_base_idx],DefaultOrthonormalBasis())
	    Bl = get_vectors(M,yl.x[col_base_idx], Bcl)
		Bcr= get_basis(M,yr.x[col_base_idx],DefaultOrthonormalBasis())
	    Br = get_vectors(M,yr.x[col_base_idx], Bcr)

		Tcl= get_basis(N,yl.x[row_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(N,yl.x[row_idx], Tcl)
		Tcr= get_basis(N,yr.x[row_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(N,yr.x[row_idx], Tcr)

        # In the following, all combinations of test and basis functions have to be considered.
		
    	assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  			Tl,1,0, integrand, transport)	
		#println("A1", A)
		assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tl,1,0, integrand, transport)	
		#println("A2", A)
		assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  Tr,0,1, integrand, transport)
		#println("A3", A)
		assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tr,0,1, integrand, transport)	
		#println("A4", A)
	end
end

# ╔═╡ 9480c8ef-0f01-4983-8915-4f2513f89c5f
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

# ╔═╡ 42b3c2e4-c06e-4a9a-acd3-5a5a06a55e9a
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

# ╔═╡ b26f9e81-bd4e-48f5-b6de-31d46daac22c
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

# ╔═╡ 1c0028e5-6beb-4512-b1f3-c18780b68ad8
begin
mutable struct NewtonEquation{F1, F2, F3, F13, F23, F21, F22, J1, SE, T1, T2, T3, Om, NM, Nrhs}
	integrand_Lprime::F1
	integrand_Lyy1::F2
	integrand_Lyy2::F3
	integrand_L_yu::F13
	integrand_L_λy::F23
	integrand_L_uλ::F21
	integrand_L_uu::F22
	integrandJ_1::J1
	integrand_stateeq::SE
	VT::T1
	transport_Lyy1::T2
	transport_Lyy2::T3
	interval::Om
	A11::NM
	A12::NM
	A13::NM
	A22::NM
	A23::NM
	A33::NM
	A::NM
	b1::Nrhs
	b2::Nrhs
	b3::Nrhs
	b::Nrhs
end

function NewtonEquation(M, int1, int2, int3, int13, int23, int21, int22, intJ1, intSE, VTP, VT1, VT2, time)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = n1 # Dim muss 2 sein (aktuell liegt λ im R3, deswegen hier per Hand)
	
	A11::SparseMatrixCSC{Float64, Int32} = spzeros(n1,n1)
	A12::SparseMatrixCSC{Float64, Int32} = spzeros(n1,n2)
	A22::SparseMatrixCSC{Float64, Int32} = spzeros(n2,n2)
	A13::SparseMatrixCSC{Float64, Int32} = spzeros(n1,n3)
	A23::SparseMatrixCSC{Float64, Int32} = spzeros(n2,n3)
	A33::SparseMatrixCSC{Float64, Int32} = spzeros(n3,n3)
	A::SparseMatrixCSC{Float64, Int32} = spzeros(n1+n2+n3, n1+n2+n3)
	
	b1 = zeros(n1)
	b2 = zeros(n2)
	b3 = zeros(n3)
	b = zeros(n1+n2+n3)
	
	return NewtonEquation{typeof(int1), typeof(int2), typeof(int3), typeof(int13), typeof(int23), typeof(int21), typeof(int22), typeof(intJ1), typeof(intSE), typeof(VTP), typeof(VT1), typeof(VT2), typeof(time), typeof(A11), typeof(b1)}(int1, int2, int3, int13, int23, int21, int22, intJ1, intSE, VTP, VT1, VT2, time, A11, A12, A13, A22, A23, A33, A, b1, b2, b3, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = n1
	
	ne.A11 .= spzeros(n1,n1)
	ne.A12 .= spzeros(n1,n2)
	#ne.A11 .= Matrix{Float64}(I, n1, n1)
	A11_helper = spzeros(n1,n1)
	ne.A13 .= spzeros(n1,n3)
	ne.A22 .= spzeros(n2,n2)
	ne.A23 .= spzeros(n2,n3)
	ne.A33 .= spzeros(n3,n3)
	
	ne.b1 .= zeros(n1)
	ne.b2 .= zeros(n2)
	ne.b3 .= zeros(n3)

	projected_λ = project(powerS, p[M,1], p[M, 3]) 
	p[M, 3] = projected_λ
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(ne.interval)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(ne.interval)+1))
	Oy3 = OffsetArray([y03, p[M, 3]..., yT3], 0:(length(ne.interval)+1))
	Oy = ArrayPartition(Oy1,Oy2,Oy3);

	# Retraktion für lambda (VT) per hand (λ+ = P(y+)(λ+δλ)), TODO: Im Newton die Retraktion richtig setzen

	nCells = length(ne.interval)+1
	
	println("Assemble:")

	#ManoptExamples.get_Jac!(evaluate,ne.A11,1,1,1,1,h,nCells,Oy,ne.integrand_Lprime,ne.VT)
	get_Jac!(ne.A11,1,1,h,nCells,Oy,ne.integrand_Lprime,ne.VT)
	get_Jac_Lyy!(A11_helper,1,1,h,nCells,Oy,ne.integrand_Lyy1,ne.transport_Lyy1)
	ne.A11 += A11_helper + A11_helper'	
	A11_helper *= 0.0
	get_Jac_Lyy!(A11_helper,1,1,h,nCells,Oy,ne.integrand_Lyy2,ne.transport_Lyy2)
	ne.A11 += A11_helper
	
	#ManoptExamples.get_Jac!(evaluate,ne.A12,1,1,2,1,h,nCells,Oy, ne.integrand_L_yu, zerotransport)
	#ManoptExamples.get_Jac!(evaluate,ne.A23,2,1,3,1,h,nCells,Oy, ne.integrand_L_uλ, zerotransport)
	#ManoptExamples.get_Jac!(evaluate,ne.A13,1,1,3,1,h,nCells,Oy,ne.integrand_L_λy,ne.VT)
	#ManoptExamples.get_Jac!(evaluate,ne.A22,2,1,2,1,h,nCells,Oy,ne.integrand_L_uu, zerotransport)
	get_Jac!(ne.A12,1,2,h,nCells,Oy, ne.integrand_L_yu)
	get_Jac!(ne.A23,2,3,h,nCells,Oy, ne.integrand_L_uλ)
	get_Jac!(ne.A13,1,3,h,nCells,Oy,ne.integrand_L_λy,ne.VT)
	get_Jac!(ne.A22,2,2,h,nCells,Oy,ne.integrand_L_uu)
	
	lambda_helper = get_coordinates(powerS, p[M,1], p[M,3], DefaultOrthogonalBasis())
	ne.b1 = ne.A13 * lambda_helper
	
	#ManoptExamples.get_rhs_row!(evaluate,ne.b1,1,1,h,nCells,Oy,ne.integrandJ_1)
	#ManoptExamples.get_rhs_row!(evaluate,ne.b2,2,1,h,nCells,Oy,ne.integrand_L_uλ)
	#ManoptExamples.get_rhs_row!(evaluate,ne.b3,3,1,h,nCells,Oy,ne.integrand_stateeq)
	
	get_rhs_row!(ne.b1,1,h,nCells,Oy,ne.integrandJ_1)
	get_rhs_row!(ne.b2,2,h,nCells,Oy,ne.integrand_L_uλ)
	get_rhs_row!(ne.b3,3,h,nCells,Oy,ne.integrand_stateeq)
	
	#A33 = 0 

	ne.A .= vcat(hcat(ne.A11 , ne.A12 , ne.A13), 
			  hcat(ne.A12', ne.A22 , ne.A23), 
			  hcat(ne.A13', ne.A23', ne.A33))
		
	ne.b .= vcat(ne.b1, ne.b2, ne.b3)
	return
end


function (ne::NewtonEquation)(M, VB, p, p_trial)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	bctrial1=zeros(n1)
	bctrial2=zeros(n2)
	bctrial3=zeros(n3)

	projected_λ = project(powerS, p[M,1], p[M, 3]) 
	p[M, 3] = projected_λ
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(ne.interval)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(ne.interval)+1))
	Oy3 = OffsetArray([y03, p[M, 3]..., yT3], 0:(length(ne.interval)+1))
	Oy = ArrayPartition(Oy1,Oy2,Oy3);

	
	Oytrial1 = OffsetArray([y01, p_trial[M,1]..., yT1], 0:(length(ne.interval)+1))
	Oytrial2 = OffsetArray([y02, p_trial[M,2]..., yT2], 0:(length(ne.interval)+1))
	Oytrial3 = OffsetArray([y03, p_trial[M,3]..., yT3], 0:(length(ne.interval)+1))
	Oytrial = ArrayPartition(Oytrial1,Oytrial2,Oytrial3);

	nCells = length(ne.Omega) + 1

	get_rhs_simplified!(bctrial2,2,h,nCells,Oy,Oytrial,ne.integrand_L_yu,ne.VT)
	get_rhs_simplified!(bctrial3,3,h,nCells,Oy,Oytrial,ne.integrand_L_uλ, zerotransport)

	return vcat(bctrial1,bctrial2, bctrial3)
end
end;

# ╔═╡ 45141a59-510a-4177-bab1-31be7450fcd9
function solve_in_basis_repr(problem, newtonstate) 
	X = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	M_helper = ProductManifold(powerS, powerR3, powerS)
	p_helper = ArrayPartition(newtonstate.p[problem.manifold, 1], newtonstate.p[problem.manifold, 2], newtonstate.p[problem.manifold, 1])
	return get_vector(M_helper, p_helper, X, DefaultOrthogonalBasis())
end

# ╔═╡ b3d4e21b-8183-4522-a463-63bef1810357
""" Initial geodesic """
y_0 = copy(product, disc_y)

# ╔═╡ b7f97707-786c-47c7-a3b6-afaedf2083d3
pr = ProductRetraction(ProjectionRetraction(), ExponentialRetraction(), ExponentialRetraction())

# ╔═╡ 5220a0dd-9484-4556-a7d0-ecf94955bf6c
begin
	NE = NewtonEquation(product, integrand_L_prime, integrand_Lyy_1, integrand_Lyy_2, integrand_Lyu, integrand_Lλy, integrand_Luλ, integrand_Luu, integrandJ1, integrand_state_eq, transport, transport_Lyy_1, transport_Lyy_2, Omega)

	st_res = vectorbundle_newton(product, TangentBundle(product), NE, y_0; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(product,1e-11; outer_norm=Inf)),
	retraction_method=pr,
	#stepsize=Manopt.AffineCovariantStepsize(product, theta_des=0.5, outer_norm=Inf),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)
end

# ╔═╡ 74d553d3-b1d1-47af-ba1f-6d4297148b51
iterates = get_record(st_res, :Iteration, :Iterate)

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
  color = RGBA(1.,1.,1.,0.),
  shading = Makie.automatic,
  transparency=true
)
ax.show_axis = false

state_start = [y01, discretized_y ...,yT1]
state_final = [y01, p_res[product,1] ..., yT1]
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.1); transparency=true)
    π1(x) = 1.0*x[1]
    π2(x) = 1.0*x[2]
    π3(x) = 1.0*x[3]
	arrows!(ax, π1.(p_res.x[1]), π2.(p_res.x[1]), π3.(p_res.x[1]), π1.(p_res.x[3]), π2.(p_res.x[3]), π3.(p_res.x[3]); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=2.5)
	scatterlines!(ax, π1.(state_final), π2.(state_final), π3.(state_final); markersize=5, color=:orange, linewidth=0.1)
	#scatterlines!(ax, π1.(state_start), π2.(state_start), π3.(state_start); markersize=8, color=:blue, linewidth=1)
	scatter!(ax, π1.([y01, yT1]), π2.([y01, yT1]), π3.([y01, yT1]); markersize =7, color=:red)
	
	#scatter!(ax, 1.0,1.0,1.0; markersize =9, color=:red)
	scatter!(ax, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3); markersize =7, color=:red)
	
	#arrows!(ax, π1.(p_res.x[1]), π2.(p_res.x[1]), π3.(p_res.x[1]), π1.(-p_res.x[2]), π2.(-p_res.x[2]), π3.(-p_res.x[2]); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.05)
	fig
end

# ╔═╡ Cell order:
# ╠═6362be32-ed1a-11ef-0352-c7ccb14267de
# ╠═66fbe00e-1d30-468d-baa5-05e099cf329e
# ╠═bc7188d7-38a4-49a3-9271-c2ad4b4ebf91
# ╠═aeceb735-da1b-4db5-9964-538b316441c2
# ╠═c6b1fd66-ef1f-427e-b8cc-d2da9c127ee3
# ╠═9b589eb8-c42c-45cf-a2d4-89fd1d522759
# ╟─b22850cc-26fa-4fbe-a919-0ae95c8571d6
# ╠═6dc5ca81-70b6-4aab-8f52-64b696e06217
# ╟─8596f05e-bc6f-4891-80ee-e45e9d730089
# ╠═452addcd-d0f4-4059-8de6-448045e5b670
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
# ╠═9cad614b-f5b1-43ac-a612-6f32458f7cdd
# ╠═ea918bcc-2d49-47cf-ad9c-79ae3bc9484c
# ╠═f7d66dfb-d246-486a-9033-ab82fde3d586
# ╠═6cf5ad9a-b3dc-4a11-bf01-3aa188c9667b
# ╠═c9b6a526-7b30-44bf-b652-ff9c17a4599c
# ╠═9480c8ef-0f01-4983-8915-4f2513f89c5f
# ╠═42b3c2e4-c06e-4a9a-acd3-5a5a06a55e9a
# ╠═b26f9e81-bd4e-48f5-b6de-31d46daac22c
# ╠═707d23d1-f7b4-49cb-b0ff-11ec536939fa
# ╠═1c0028e5-6beb-4512-b1f3-c18780b68ad8
# ╠═45141a59-510a-4177-bab1-31be7450fcd9
# ╠═b3d4e21b-8183-4522-a463-63bef1810357
# ╠═b7f97707-786c-47c7-a3b6-afaedf2083d3
# ╠═5220a0dd-9484-4556-a7d0-ecf94955bf6c
# ╠═74d553d3-b1d1-47af-ba1f-6d4297148b51
# ╠═6920c45e-4ccb-4b60-a90c-c8df0792269c
# ╠═921845f5-2114-4e8f-a3c4-e1677e9134dd
# ╠═817c3466-eeb7-4979-a738-1036b719361f
# ╠═ac49e504-f3ba-4a5e-9a66-21160f38874e
