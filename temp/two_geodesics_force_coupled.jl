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
	N=50
	
	st1 = 0.5
	halt1 = pi/2

	st2 = pi/2
	halt2 = pi-0.5

	scale_couple=5
	
	h = (halt1-st1)/(N+1)
	#halt = pi - st
	Omega1 = range(; start=st1, stop = halt1, length=N+2)[2:end-1]
	Omega2 = range(; start=st1, stop = halt1, length=N+2)[2:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y01 = [sin(st1)*cos(0),0*sin(0),cos(st1)] # startpoint of geodesic
	yT1 = [sin(halt1)*cos(0),0*sin(0),cos(halt1)] # endpoint of geodesic

	y02 = [sin(st1)*cos(pi/4),sin(st1)*sin(pi/4),cos(st1)] # startpoint of geodesic
	yT2 = [sin(halt1)*cos(pi/4),sin(halt1)*sin(pi/4),cos(halt1)] # endpoint of geodesic

	#yT = [sin(st),0,cos(st)] # startpoint of geodesic: suedpol
	#y0 = [sin(halt),0,cos(halt)] # endpoint of geodesic: nordpol

	#y0 = [cos(st),sin(st),0] # startpoint of geodesic: aequator
	#yT = [cos(halt),sin(halt),0] # endpoint of geodesic: aequator
end;

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
function y1(t)
	return [sin(t)*cos(0), sin(t)*sin(0), cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
end;

# ╔═╡ 7f6c588b-e64d-471f-9259-f3e3aeeb193a
function y2(t)
	return [sin(t)*cos(pi/4), sin(t)*sin(pi/4), cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
end;

# ╔═╡ 5c0980c5-284e-4406-bab8-9b9aff9391ba
discretized_y1 = [y1(Ωi) for Ωi in Omega1];

# ╔═╡ 8cc3124c-dbb8-4286-aee3-3984d87868c1
discretized_y2 = [y2(Ωi) for Ωi in Omega2];

# ╔═╡ ba3051a0-078a-49ff-85b9-441eef4cb9fc
disc_y = ArrayPartition(discretized_y1, discretized_y2)

# ╔═╡ 0402cecb-9138-4d5b-8e09-b1a7d06e68c2
size(disc_y)

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
function zerodummy(S, p, X, dq)
	return 0.0*X
end

# ╔═╡ 86fc6357-1106-48f9-8efe-fda152caf990
function F_prime12_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return scale_couple*B1'*B2
end

# ╔═╡ b28a1cdc-50ad-4289-8ee9-ee4ccf4d653e
function F_prime21_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return scale_couple*B2'*B1
end

# ╔═╡ 2260abfa-86c1-48ea-acdf-b641cad0433a
function assemble_local_rhs!(b,component_idx, h, i, yl, yr, Bl, Br,integrand)
	dim = manifold_dimension(integrand.domain)
    idxl=dim*(i-2)
    idxr=dim*(i-1)
	ydotl=(yr-yl)/h
	ydotr=(yr-yl)/h
	# Trapezregel
	quadwght = 0.5*h   
	for k in 1:dim
		Bldot=-Bl[k]/h
		Brdot= Br[k]/h
		# linke Testfunktion
        if idxl>=0 
			#linker Quadraturpunkt
			tmp =  integrand.value(integrand,yl,ydotl,Bl[k],Bldot)	
			#rechter Quadraturpunkt
			tmp += integrand.value(integrand,yr,ydotr,0.0*Bl[k],Bldot)	
			# Update der rechten Seite
		  	b[idxl+k]+= quadwght*tmp	
		end
		# rechte Testfunktion
		if idxr < length(b)
			tmp  = integrand.value(integrand,yl,ydotl,0.0*Br[k],Brdot)	
			tmp += integrand.value(integrand,yr,ydotr,Br[k],Brdot)	
            b[idxr+k]+= quadwght*tmp
		end
	end
end


# ╔═╡ 7ee0ea56-6d68-41ae-bc10-9d1f650b3257
function get_rhs_component!(b,component_idx,h,y,integrand)
	N = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:length(y.x[1])-1
		yl=ArrayPartition(getindex.(y.x, (i-1...,)))
		yr=ArrayPartition(getindex.(y.x, (i...,)))
		Bcl=get_basis(N,yl.x[component_idx],DefaultOrthonormalBasis())
	    Bl = get_vectors(N, yl.x[component_idx], Bcl)
		Bcr=get_basis(N,yr.x[component_idx],DefaultOrthonormalBasis())
	    Br = get_vectors(N, yr.x[component_idx], Bcr)
        assemble_local_rhs!(b,component_idx, h, i, yl, yr, Bl, Br, integrand)		
	end
end

# ╔═╡ ae6c02fd-00b1-4e15-8f26-ffeca9a91ec0
function get_rhs_simplified!(b,component_idx,h,y,y_trial,integrand,transport)
	S = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:length(y.x[1])-1
			yl=ArrayPartition(getindex.(y.x, (i-1...,)))
			yr=ArrayPartition(getindex.(y.x, (i...,)))
			
			yl_trial=ArrayPartition(getindex.(y_trial.x, (i-1...,)))
			yr_trial=ArrayPartition(getindex.(y_trial.x, (i...,)))
		
			Bcl=get_basis(S,yl.x[component_idx],DefaultOrthonormalBasis())
			Bl=get_vectors(S, yl.x[component_idx],Bcl)
		
			Bcr=get_basis(S,yr.x[component_idx],DefaultOrthonormalBasis())
	    	Br = get_vectors(S, yr.x[component_idx], Bcr)
		
			dim = manifold_dimension(S)
		# Transport der Testfunktionen auf $T_{x_k}S$
            for k=1:dim
				Bl[k]=transport.value(S,yl.x[component_idx],Bl[k],yl_trial.x[component_idx])
				Br[k]=transport.value(S,yr.x[component_idx],Br[k],yr_trial.x[component_idx])
			end
        	assemble_local_rhs!(b, component_idx,h, i, yl_trial, yr_trial, Bl, Br,integrand)		
	end
end

# ╔═╡ 808db8aa-64f7-4b36-8c6c-929ba4fa22db
"""
Force field w and its derivative. A scaling parameter is also employed.
"""
function w(p, c)
	#return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	return c*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
end

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
"""
The following two routines define the integrand and its ordinary derivative. They use a vector field w, wich is defined, below. A scaling parameter is also employed.
"""
function F1_at(Integrand, y, ydot, B, Bdot)
	  return ydot.x[1]'*Bdot+w(y.x[1],Integrand.scaling)'*B+scale_couple*(y.x[2]-y.x[1])'*B
end

# ╔═╡ 229fa902-e125-429a-852d-0668f64c7640
function F2_at(Integrand, y, ydot, B, Bdot)
	  return ydot.x[2]'*Bdot+w(y.x[2],Integrand.scaling)'*B+scale_couple*(y.x[1]-y.x[2])'*B
end

# ╔═╡ 288b9637-0500-40b8-a1f9-90cb9591402b
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
	#return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
	return c*[2*p[1]*p[2]/nenner^2 (-1.0/(nenner)+2.0*p[2]^2/nenner^2) 0.0; (1.0/nenner-2.0*p[1]^2/(nenner^2)) (-2.0*p[1]*p[2]/(nenner^2)) 0.0; 0.0 0.0 0.0]
end

# ╔═╡ ac04e6ec-61c2-475f-bb2f-83755c04bd72
function F_prime11_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return B1dot'*B2dot+(w_prime(y.x[1],Integrand.scaling)*B1)'*B2-scale_couple*B1'*B2
end

# ╔═╡ 1c284f9d-f34e-435b-976d-61aaa0975fe5
function F_prime22_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return B1dot'*B2dot+(w_prime(y.x[2],Integrand.scaling)*B1)'*B2-scale_couple*B1'*B2
end

# ╔═╡ 684508bd-4525-418b-b89a-85d56c01b188
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N)
product = ProductManifold(power, power)
integrand1=DifferentiableMapping(S,S,F1_at,F_prime11_at,0.0)
integrand2=DifferentiableMapping(S,S,F2_at,F_prime22_at,0.0)
integrand12=DifferentiableMapping(S,S,F1_at,F_prime12_at,0.0)
integrand21=DifferentiableMapping(S,S,F2_at,F_prime21_at,0.0)
transport=DifferentiableMapping(S,S,transport_by_proj,transport_by_proj_prime,nothing)
end;

# ╔═╡ 14d42ecb-6563-4d62-94ce-a36b73ed9a78
zerotransport=DifferentiableMapping(S,S,zerodummy,zerodummy,nothing)


# ╔═╡ e2f48dcc-5c23-453d-8ff3-eb425b7b67af
"""
If no vector transport is needed, leave it away, then a zero dummy transport is used
"""
function get_Jac!(A,row_idx,col_idx,h,y,integrand)
	get_Jac!(A,row_idx,col_idx,h,y,integrand,zerotransport)
end

# ╔═╡ 86e2a93f-68a7-4b60-bcc0-a4a7cd6278f6
function assemble_local_Jac_with_connection!(A,rowidx, colidx, h, i, yl, yr, BlM, BrM, BlN, BrN,integrand, transport=zerotransport)
 dim = manifold_dimension(integrand.domain)
 idxl=dim*(i-2)
 idxr=dim*(i-1)
 ydot=(yr-yl)/h
 quadwght=0.5*h
	nA=size(A,1)
 #	Schleife über Testfunktionen
 for k in 1:dim
	Bdotlk=(0-1)*BlM[k]/h
	Bdotrk=(1-0)*BrM[k]/h
    # Schleife über Testfunktionen
	for j in 1:dim
		# Zeit-Ableitungen der Testfunktionen (=0 am jeweils anderen Rand)
		Bdotlj=(0-1)*BlN[j]/h
		Bdotrj=(1-0)*BrN[j]/h

		# y-Ableitungen der Projektionen
		Pprimel=transport.derivative(integrand.domain,yl.x[rowidx],BlM[j],BlN[k])
		Pprimer=transport.derivative(integrand.domain,yr.x[colidx],BrM[j],BrN[k])

		# Zeit- und y-Ableitungen der Projektionen
		Pprimedotl=(0-1)*Pprimel/h
		Pprimedotr=(1-0)*Pprimer/h
		
		# linke x linke Testfunktion
        if idxl>=0
		   # linker Quadraturpunkt
		   # Ableitung in der Einbettung	
         tmp=integrand.derivative(integrand,yl,ydot,BlM[j],Bdotlj,BlN[k],Bdotlk)
		   # Modifikation für Kovariante Ableitung	
		   tmp += integrand.value(integrand,yl,ydot,Pprimel,Pprimedotl)
		   # rechter Quadraturpunkt (siehe oben)
	  tmp+=integrand.derivative(integrand,yr,ydot,0.0*BlM[j],Bdotlj,0.0*BlN[k],Bdotlk)
		   tmp += integrand.value(integrand,yr,ydot,0.0*Pprimel,Pprimedotl)
           # Update des Matrixeintrags
		   A[idxl+k,idxl+j]+=quadwght*tmp
		   # TODO: Stimmt das auch bei nicht-symmetrischen Matrizen? j <-> k?
		end
		# linke x rechte Testfunktion
		if idxl>=0 && idxr<nA
		   # linker Quadraturpunkt
		   # Ableitung in der Einbettung	
			tmp=integrand.derivative(integrand, yl,ydot,0.0*BrM[j],Bdotrj,BlN[k],Bdotlk)	
		   # Modifikation für Kovariante Ableitung fällt hier weg, da Terme = 0
		   # rechter Quadraturpunkt
			tmp+=integrand.derivative(integrand, yr,ydot,BrM[j],Bdotrj,0.0*BlN[k],Bdotlk)	
           # Symmetrisches Update der Matrixeinträge
			A[idxl+k,idxr+j] += quadwght*tmp
			A[idxr+j,idxl+k] += quadwght*tmp
		 end	
		# rechte x rechte Testfunktion (siehe oben)
		 if idxr < nA
		   tmp=integrand.derivative(integrand, yl,ydot,0.0*BrM[j],Bdotrj,0.0*BrN[k],Bdotrk)
		   tmp += integrand.value(integrand, yl,ydot,0.0*Pprimer,Pprimedotr)
		   tmp+=integrand.derivative(integrand, yr,ydot,BrM[j],Bdotrj,BrN[k],Bdotrk)
		   tmp += integrand.value(integrand, yr,ydot,Pprimer,Pprimedotr)
			 
		   A[idxr+k,idxr+j]+=quadwght*tmp
			 # TODO: Stimmt das auch bei nicht-symmetrischen Matrizen?  j <-> k?
		 end
	end
 end
end

# ╔═╡ 53d912a5-8280-4bd3-980e-a49f3e9809e5
function get_Jac!(A,row_idx,col_idx,h,y,integrand,transport)
	M = integrand.domain
	N = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:length(y.x[1])-1
		yl=ArrayPartition(getindex.(y.x, (i-1...,)))
		yr=ArrayPartition(getindex.(y.x, (i...,)))

		Bcl=get_basis(M,yl.x[col_idx],DefaultOrthonormalBasis())
	    BlM = get_vectors(M,yl.x[col_idx], Bcl)
		Bcr=get_basis(M,yr.x[col_idx],DefaultOrthonormalBasis())
	    BrM = get_vectors(M,yr.x[col_idx], Bcr)

		Bcl=get_basis(N,yl.x[row_idx],DefaultOrthonormalBasis())
	    BlN = get_vectors(N,yl.x[row_idx], Bcl)
		Bcr=get_basis(N,yr.x[row_idx],DefaultOrthonormalBasis())
	    BrN = get_vectors(N,yr.x[row_idx], Bcr)

        assemble_local_Jac_with_connection!(A, row_idx,col_idx, h, i, yl, yr, BlM, BrM, BlN, BrN , integrand, transport)		
	end
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
	n = Int(manifold_dimension(M)/2)
	Ac11::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	Ac21::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	Ac12::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	Ac22::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	
	bc1 = zeros(n)
	bc2 = zeros(n)
	
	bctrial1=zeros(n)
	bctrial2=zeros(n)
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(Omega1)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(Omega2)+1))
    Oy = ArrayPartition(Oy1,Oy2);
	
	Oytrial1 = OffsetArray([y01, state.p_trial[M,1]..., yT1], 0:(length(Omega1)+1))
	Oytrial2 = OffsetArray([y02, state.p_trial[M,2]..., yT2], 0:(length(Omega2)+1))
	Oytrial = ArrayPartition(Oytrial1,Oytrial2);
	
	#S = M.manifold
	println("Assemble:")
    get_Jac!(Ac11,1,1,h,Oy,integrand1,transport)
	get_Jac!(Ac22,2,2,h,Oy,integrand2,transport)
    get_Jac!(Ac21,2,1,h,Oy,integrand12)
	get_Jac!(Ac12,1,2,h,Oy,integrand21)
    get_rhs_component!(bc1,1,h,Oy,integrand1)
	get_rhs_component!(bc2,2,h,Oy,integrand2)

	#Ac = hcat(Ac11, Ac12)
	#temp = hcat(Ac21, Ac22)
	Ac = vcat(hcat(Ac11, Ac12), hcat(Ac21, Ac22))
		
	if state.is_same == true
		bcsys = vcat(bc1, bc2)
	else
		get_rhs_simplified!(bctrial1,1,h,Oy,Oytrial,integrand1,transport)
		get_rhs_simplified!(bctrial2,2,h,Oy,Oytrial,integrand2,transport)
		bctrial = vcat(bctrial1,bctrial2)
    	bcsys=bctrial-(1.0 - state.stepsize.alpha)*vcat(bc1, bc2)
		println("alpha: ", state.stepsize.alpha)
	end
	#Asparse = sparse(Ac)
	println("Solve:")
	Xc = (Ac) \ (-bcsys)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	res_c = get_vector(M, p, Xc, B)
	println("norm:", norm(res_c))
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
	retraction_method=ProjectionRetraction(),
#stepsize=ConstantLength(1.0),
stepsize=Manopt.AffineCovariantStepsize(product),
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
n = 45
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
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.45); transparency=true)
    π1(x) = 1.02*x[1]
    π2(x) = 1.02*x[2]
    π3(x) = 1.02*x[3]
	scatter!(ax, π1.(p_res[product, 1]), π2.(p_res[product, 1]), π3.(p_res[product, 1]); markersize =8, color=:orange)
	scatter!(ax, π1.(p_res[product, 2]), π2.(p_res[product, 2]), π3.(p_res[product, 2]); markersize =8, color=:blue)
	#scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =8, color=:blue)
	scatter!(ax, π1.([y01, yT1]), π2.([y01, yT1]), π3.([y01, yT1]); markersize =8, color=:red)
	scatter!(ax, π1.([y02, yT2]), π2.([y02, yT2]), π3.([y02, yT2]); markersize =8, color=:red)
	#arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	fig
end

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╠═29043ca3-afe0-4280-a76a-7c160a117fdf
# ╠═7f6c588b-e64d-471f-9259-f3e3aeeb193a
# ╠═5c0980c5-284e-4406-bab8-9b9aff9391ba
# ╠═8cc3124c-dbb8-4286-aee3-3984d87868c1
# ╠═ba3051a0-078a-49ff-85b9-441eef4cb9fc
# ╠═0402cecb-9138-4d5b-8e09-b1a7d06e68c2
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╠═50a51e47-b6b1-4e43-b4b9-aad23f6ec390
# ╠═9cdd4289-c49d-4733-8487-f471e38fc402
# ╠═48834792-fff9-4e96-803a-b4d07e714797
# ╠═14d42ecb-6563-4d62-94ce-a36b73ed9a78
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╠═229fa902-e125-429a-852d-0668f64c7640
# ╠═ac04e6ec-61c2-475f-bb2f-83755c04bd72
# ╠═1c284f9d-f34e-435b-976d-61aaa0975fe5
# ╠═86fc6357-1106-48f9-8efe-fda152caf990
# ╠═b28a1cdc-50ad-4289-8ee9-ee4ccf4d653e
# ╠═53d912a5-8280-4bd3-980e-a49f3e9809e5
# ╠═7ee0ea56-6d68-41ae-bc10-9d1f650b3257
# ╠═e2f48dcc-5c23-453d-8ff3-eb425b7b67af
# ╠═ae6c02fd-00b1-4e15-8f26-ffeca9a91ec0
# ╠═2260abfa-86c1-48ea-acdf-b641cad0433a
# ╠═86e2a93f-68a7-4b60-bcc0-a4a7cd6278f6
# ╠═684508bd-4525-418b-b89a-85d56c01b188
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
