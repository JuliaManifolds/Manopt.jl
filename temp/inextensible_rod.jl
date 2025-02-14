### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ c9994bc4-b7bb-11ef-3430-8976c5eabdeb
using Pkg; Pkg.activate();

# ╔═╡ 9fb54416-3909-49c4-b1bf-cc868e580652
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

# ╔═╡ 12e32b83-ae65-406c-be51-3f21935eaae5
begin
	N=200
	
	st1 = 0.0
	halt1 = 1.0

	scale_couple=-1
	windscale=0

	h = (halt1-st1)/(N+1)
	#halt = pi - st
	Omega1 = range(; start=st1, stop = halt1, length=N+2)[2:end-1]
	Omega2 = range(; start=st1, stop = halt1, length=N+2)[2:end-1]
	Omega3 = range(; start=st1, stop = halt1, length=N+2)[1:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y01 = [0,0,0] # startpoint of rod
	yT1 = [0.8,0,0] # endpoint of rod

	y02 = [1,0,0] # start direction of rod
	yT2 = [1,0,0] # end direction of rod

	#y03 = [0,0,0] # startpoint of geodesic
	#yT3 = [0,0,0] # endpoint of geodesic

end;

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
function y1(t)
	return [t*0.8, 0.1*t*(1-t), 0]
end;

# ╔═╡ 7f6c588b-e64d-471f-9259-f3e3aeeb193a
function y2(t)
	return [sin(t*pi/2+pi/4), cos(t*pi/2+pi/4), 0]
	#return [1.0, 0, 0]
end;

# ╔═╡ 5e2e2280-fe0d-443b-8824-101a138a86a0
function y3(t)
	return [0.1, 0.1, 0.1]
end;

# ╔═╡ 5c0980c5-284e-4406-bab8-9b9aff9391ba
discretized_y1 = [y1(Ωi) for Ωi in Omega1];

# ╔═╡ 8cc3124c-dbb8-4286-aee3-3984d87868c1
discretized_y2 = [y2(Ωi) for Ωi in Omega2];

# ╔═╡ 47d26c17-d422-4e6f-a5fc-7f11a2bcba0c
discretized_y3 = [y3(Ωi) for Ωi in Omega3];

# ╔═╡ ba3051a0-078a-49ff-85b9-441eef4cb9fc
disc_y = ArrayPartition(discretized_y1, discretized_y2,discretized_y3)

# ╔═╡ bc449c2d-1f23-4c72-86ab-a46acbf64129
"""
Such a structure has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""
mutable struct DifferentiableMapping{M<:AbstractManifold, N<:AbstractManifold,F1<:Function,F2<:Function,T}
	domain::M
	precodomain::N
	value::F1
	derivative::F2
	scaling::T
end


# ╔═╡ 50a51e47-b6b1-4e43-b4b9-aad23f6ec390
"""
 The following two routines define the vector transport and its derivative. The second is needed to obtain covariant derivative from the ordinary derivative.
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ 9cdd4289-c49d-4733-8487-f471e38fc402
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ 48834792-fff9-4e96-803a-b4d07e714797
function zerotrans_prime(S, p, X, dq)
	return 0.0*X
end

# ╔═╡ 758d34df-96ad-4295-a3a1-46acd65b26e7
function identitytrans(S, p, X, q)
	return X
end

# ╔═╡ 229fa902-e125-429a-852d-0668f64c7640
function F2_at(Integrand, y, ydot, T, Tdot)
	  return ydot.x[2]'*Tdot-T'*y.x[3]
end

# ╔═╡ 9bcaa5d0-d8de-4746-8c85-0fe24a4825e2
function F3_at(Integrand, y, ydot, T, Tdot)
	  return (ydot.x[1]-y.x[2])'*T
end

# ╔═╡ 1c284f9d-f34e-435b-976d-61aaa0975fe5
function F_prime22_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return Bdot'*Tdot
end

# ╔═╡ 86fc6357-1106-48f9-8efe-fda152caf990
function F_prime13_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return Tdot'*B
end

# ╔═╡ 03c147e6-843f-47ae-924e-86ed0260cd8e
function F_prime23_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return -T'*B
end

# ╔═╡ ef0edc46-0b33-42f5-821a-373edd4cdd84
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
	idx=dimc*(i-2)
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


		# linker Quadraturpunkt

		# Ableitung in der Einbettung:	
        tmp=integrand.derivative(integrand,yl,ydot,bfl*B[j],Bdot,tfl*T[k],Tdot)
		# Modifikation für Kovariante Ableitung:	
		# y-Ableitungen der Projektionen
		Pprime=transport.derivative(integrand.domain,yl.x[row_idx],bfl*B[j],tfl*T[k])
		# Zeit- und y-Ableitungen der Projektionen
		Pprimedot=(bfr-bfl)*Pprime/h
		# Einsetzen in die rechte Seite
		tmp+=integrand.value(integrand,yl,ydot,bfl*Pprime,Pprimedot)
			
		# rechter Quadraturpunkt (analog zum linken)
			
		# Ableitung in der Einbettung:	
 	    tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,tfr*T[k],Tdot)	
		# Modifikation für Kovariante Ableitung:	
		Pprime=transport.derivative(integrand.domain,yr.x[row_idx],bfr*B[j],tfr*T[k])
		Pprimedot=(bfr-bfl)*Pprime/h			
		tmp+=integrand.value(integrand,yr,ydot,bfr*Pprime,Pprimedot)
			
        # Update des Matrixeintrags
			
		A[idxc+k,idx+j]+=quadwght*tmp
		end
	end
 end
end

# ╔═╡ 2260abfa-86c1-48ea-acdf-b641cad0433a
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


# ╔═╡ c83a1cbb-a4fa-4b0a-970d-f65f74e01615
function evaluate(y, i, tloc)
	return ArrayPartition(
		(1.0-tloc)*y.x[1][i-1]+tloc*y.x[1][i],
		(1.0-tloc)*y.x[2][i-1]+tloc*y.x[2][i],
		y.x[3][i]
	)
end

# ╔═╡ 22d7dbf4-b548-4246-9942-356571b398d0
function get_Jac!(A,row_idx,degT,col_idx,degB,h, nCells,y,integrand,transport)
	M = integrand.domain
	N = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells
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

		if degT==1 && degB == 1
    	    assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  			Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0,  Tr,0,1, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1,  Tr,0,1, integrand, transport)		
		end
		if degT==1 && degB == 0
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,1,1,  Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl,yr, Br,1,1,  Tr,0,1, integrand, transport)		
		end
	end
end

# ╔═╡ 7ee0ea56-6d68-41ae-bc10-9d1f650b3257
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

# ╔═╡ ae6c02fd-00b1-4e15-8f26-ffeca9a91ec0
function get_rhs_simplified!(b,row_idx,h,nCells,y,y_trial,integrand,transport)
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
        	assemble_local_rhs!(b,row_idx,h,i,yl_trial,yr_trial,Tl,1,0,integrand)	
		    assemble_local_rhs!(b,row_idx,h,i,yl_trial,yr_trial,Tr,0,1,integrand)		
	end
end

# ╔═╡ 808db8aa-64f7-4b36-8c6c-929ba4fa22db
"""
Force field w and its derivative. A scaling parameter is also employed.
"""
function w(p, c)
	#return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	return [0.0,0.0,0.0] #c*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
end

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
"""
The following two routines define the integrand and its ordinary derivative. They use a vector field w, wich is defined, below. A scaling parameter is also employed.
"""
function F1_at(Integrand, y, ydot, T, Tdot)
	  return w(y.x[1],Integrand.scaling)'*T+Tdot'*y.x[3]
end

# ╔═╡ 288b9637-0500-40b8-a1f9-90cb9591402b
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
	#return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
	return zeros(3,3) #c*[2*p[1]*p[2]/nenner^2 (-1.0/(nenner)+2.0*p[2]^2/nenner^2) 0.0; (1.0/nenner-2.0*p[1]^2/(nenner^2)) (-2.0*p[1]*p[2]/(nenner^2)) 0.0; 0.0 0.0 0.0]
end

# ╔═╡ ac04e6ec-61c2-475f-bb2f-83755c04bd72
function F_prime11_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return (w_prime(y.x[1],Integrand.scaling)*B)'*T
end

# ╔═╡ 684508bd-4525-418b-b89a-85d56c01b188
begin
S = Manifolds.Sphere(2)
R3 = Manifolds.Euclidean(3)	
powerS = PowerManifold(S, NestedPowerRepresentation(), N)
powerR3 = PowerManifold(R3, NestedPowerRepresentation(), N)
powerR3lambda = PowerManifold(R3, NestedPowerRepresentation(), N+1)
product = ProductManifold(powerR3, powerS, powerR3lambda)

integrand1=DifferentiableMapping(R3,R3,F1_at,F_prime11_at,windscale)
integrand2=DifferentiableMapping(S,S,F2_at,F_prime22_at,windscale)
integrand3=DifferentiableMapping(R3,R3,F3_at,F_prime11_at,windscale)
integrand13=DifferentiableMapping(R3,R3,F1_at,F_prime13_at,windscale)
integrand23=DifferentiableMapping(R3,S,F2_at,F_prime23_at,windscale)
	
transport=DifferentiableMapping(S,S,transport_by_proj,transport_by_proj_prime,nothing)
end;

# ╔═╡ 14d42ecb-6563-4d62-94ce-a36b73ed9a78
zerotransport=DifferentiableMapping(R3,R3,identitytrans,zerotrans_prime,nothing)


# ╔═╡ e2f48dcc-5c23-453d-8ff3-eb425b7b67af
"""
If no vector transport is needed, leave it away, then a zero dummy transport is used
"""
function get_Jac!(A,row_idx,degT,col_idx,degB,h,nCells,y,integrand)
	get_Jac!(A,row_idx,degT,col_idx,degB,h,nCells,y,integrand,zerotransport)
end

# ╔═╡ 98a334b1-5aa9-4e3a-a03d-f6859e77f1dc
"""
Dummy
"""
function bundlemap(M, y)
		# Include boundary points
end

# ╔═╡ 6dbfa961-639d-45af-b1d1-2622331e8455
"""
Dummy
"""
function connection_map(E, q)
    return q
end

# ╔═╡ 1abddae9-d862-4c73-a46f-7bb0a1a8f917
function solve_linear_system(M, p, state, prob)
	obj = get_objective(prob)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	Ac11::SparseMatrixCSC{Float64,Int32} =0.0*sparse(I,n1,n1)
	Ac12::SparseMatrixCSC{Float64,Int32} =spzeros(n1,n2)
	Ac22::SparseMatrixCSC{Float64,Int32} =0.0*sparse(I,n2,n2)
	Ac13::SparseMatrixCSC{Float64,Int32} =spzeros(n1,n3)
	Ac23::SparseMatrixCSC{Float64,Int32} =spzeros(n2,n3)
	Ac33::SparseMatrixCSC{Float64,Int32} =-0.00*sparse(I,n3,n3)
	
	bc1 = zeros(n1)
	bc2 = zeros(n2)
	bc3 = zeros(n3)
	
	bctrial1=zeros(n1)
	bctrial2=zeros(n2)
	bctrial3=zeros(n3)
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(Omega1)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(Omega2)+1))
	Oy3 = OffsetArray(p[M, 3], 1:length(Omega3))
	#Oy3 = OffsetArray(p[M, 3], 0:(length(Omega3)-1))
    Oy = ArrayPartition(Oy1,Oy2,Oy3);

    println("Iter: ",norm(Oy))
	
	Oytrial1 = OffsetArray([y01, state.p_trial[M,1]..., yT1], 0:(length(Omega1)+1))
	Oytrial2 = OffsetArray([y02, state.p_trial[M,2]..., yT2], 0:(length(Omega2)+1))
	Oytrial3 = OffsetArray(state.p_trial[M,3], 1:length(Omega3))

	Oytrial = ArrayPartition(Oytrial1,Oytrial2,Oytrial3);
	
	println("Assemble:")
	nCells = length(Omega3)
    get_Jac!(Ac11,1,1 ,1,1 ,h,nCells,Oy,integrand1)
	get_Jac!(Ac22,2,1, 2,1 ,h,nCells,Oy,integrand2,transport)
    get_Jac!(Ac13,1,1, 3,0 ,h,nCells,Oy,integrand13)
	get_Jac!(Ac23,2,1, 3,0 ,h,nCells,Oy,integrand23)
	# Ac12 = 0, Ac33 = 0 
	
    get_rhs_row!(bc1,1,1,h,nCells,Oy,integrand1)
	get_rhs_row!(bc2,2,1,h,nCells,Oy,integrand2)
	get_rhs_row!(bc3,3,0,h,nCells,Oy,integrand3)


	Ac = vcat(hcat(Ac11 , Ac12 , Ac13), 
			  hcat(Ac12', Ac22 , Ac23), 
			  hcat(Ac13', Ac23', Ac33))

	if state.is_same == true
		bcsys = vcat(bc1, bc2, bc3)
		#println("rhs", bcsys)
	else
		get_rhs_simplified!(bctrial1,1,h,nCells,Oy,Oytrial,integrand1)
		get_rhs_simplified!(bctrial2,2,h,nCells,Oy,Oytrial,integrand2,transport)
		get_rhs_simplified!(bctrial3,3,h,nCells,Oy,Oytrial,integrand3)
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

# ╔═╡ 7766ff98-f22f-4357-a2ea-040ac3b79651
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

# ╔═╡ c3c46cc3-7366-4724-98c3-ba94768d472b
""" Initial geodesic """
	y_0 = copy(product, disc_y)

# ╔═╡ e7a0b30c-0a7f-4094-a450-a5411c2f06af
submanifold_component(product, y_0, 1)

# ╔═╡ b49edc34-c7ed-476a-a711-37a6e0b1a21d
y_0[product,1]

# ╔═╡ cd3a0d49-2714-4c39-89d4-7fdae0f3128c
submanifold_components(disc_y)

# ╔═╡ 4bd08087-470d-456a-85ce-e92e08253cb4
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

# ╔═╡ abe5c5f3-4a28-425c-afde-64b645f3a9d9
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ 6451f8c5-7b4f-4792-87fd-9ed2635efa88
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ b0b8e87f-da09-4500-8aa9-e35934f7ef54
p_res = get_solver_result(st_res);

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
begin
n = 0
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

ws1 = [-w(p, integrand1.scaling) for p in p_res[product, 1]]
ws2 = [-w(p, integrand1.scaling) for p in p_res[product, 2]]
	
for i in 1:n
    for j in 1:n
        sx[i,j] = cos.(u[i]) * sin(v[j]);
        sy[i,j] = sin.(u[i]) * sin(v[j]);
        sz[i,j] = cos(v[j]);
    end
end
fig, ax, plt = meshscatter(
  sx,sy,sz,
  color = fill(RGBA(1.,1.,1.,0.75), n, n),
  shading = Makie.automatic,
  transparency=true
)
ax.show_axis = false
#wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.45); transparency=true)
    π1(x) = 1.02*x[1]
    π2(x) = 1.02*x[2]
    π3(x) = 1.02*x[3]
	scatter!(ax, π1.(p_res[product, 1]), π2.(p_res[product, 1]), π3.(p_res[product, 1]); markersize =8, color=:orange)
	#scatter!(ax, π1.(p_res[product, 2]), π2.(p_res[product, 2]), π3.(p_res[product, 2]); markersize =8, color=:blue)
	#scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =8, color=:blue)
	scatter!(ax, π1.([y01, yT1]), π2.([y01, yT1]), π3.([y01, yT1]); markersize =8, color=:red)
	#scatter!(ax, π1.([y02, yT2]), π2.([y02, yT2]), π3.([y02, yT2]); markersize =8, color=:red)
	#arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	fig
end

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╠═29043ca3-afe0-4280-a76a-7c160a117fdf
# ╠═7f6c588b-e64d-471f-9259-f3e3aeeb193a
# ╠═5e2e2280-fe0d-443b-8824-101a138a86a0
# ╠═5c0980c5-284e-4406-bab8-9b9aff9391ba
# ╠═8cc3124c-dbb8-4286-aee3-3984d87868c1
# ╠═47d26c17-d422-4e6f-a5fc-7f11a2bcba0c
# ╠═ba3051a0-078a-49ff-85b9-441eef4cb9fc
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╠═50a51e47-b6b1-4e43-b4b9-aad23f6ec390
# ╠═9cdd4289-c49d-4733-8487-f471e38fc402
# ╠═48834792-fff9-4e96-803a-b4d07e714797
# ╠═758d34df-96ad-4295-a3a1-46acd65b26e7
# ╠═14d42ecb-6563-4d62-94ce-a36b73ed9a78
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╠═229fa902-e125-429a-852d-0668f64c7640
# ╠═9bcaa5d0-d8de-4746-8c85-0fe24a4825e2
# ╠═ac04e6ec-61c2-475f-bb2f-83755c04bd72
# ╠═1c284f9d-f34e-435b-976d-61aaa0975fe5
# ╠═86fc6357-1106-48f9-8efe-fda152caf990
# ╠═03c147e6-843f-47ae-924e-86ed0260cd8e
# ╠═ef0edc46-0b33-42f5-821a-373edd4cdd84
# ╠═22d7dbf4-b548-4246-9942-356571b398d0
# ╠═e2f48dcc-5c23-453d-8ff3-eb425b7b67af
# ╠═2260abfa-86c1-48ea-acdf-b641cad0433a
# ╠═7ee0ea56-6d68-41ae-bc10-9d1f650b3257
# ╠═ae6c02fd-00b1-4e15-8f26-ffeca9a91ec0
# ╠═684508bd-4525-418b-b89a-85d56c01b188
# ╠═c83a1cbb-a4fa-4b0a-970d-f65f74e01615
# ╠═808db8aa-64f7-4b36-8c6c-929ba4fa22db
# ╠═288b9637-0500-40b8-a1f9-90cb9591402b
# ╠═98a334b1-5aa9-4e3a-a03d-f6859e77f1dc
# ╠═6dbfa961-639d-45af-b1d1-2622331e8455
# ╠═7766ff98-f22f-4357-a2ea-040ac3b79651
# ╠═1abddae9-d862-4c73-a46f-7bb0a1a8f917
# ╠═c3c46cc3-7366-4724-98c3-ba94768d472b
# ╠═e7a0b30c-0a7f-4094-a450-a5411c2f06af
# ╠═b49edc34-c7ed-476a-a711-37a6e0b1a21d
# ╠═cd3a0d49-2714-4c39-89d4-7fdae0f3128c
# ╠═4bd08087-470d-456a-85ce-e92e08253cb4
# ╠═abe5c5f3-4a28-425c-afde-64b645f3a9d9
# ╠═6451f8c5-7b4f-4792-87fd-9ed2635efa88
# ╠═b0b8e87f-da09-4500-8aa9-e35934f7ef54
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
