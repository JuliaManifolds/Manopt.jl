### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 0783b732-8574-11ef-017d-3939cfc57442
using Pkg; Pkg.activate();

# ╔═╡ b7f09653-9692-4f92-98e3-f988ed0c3d2d
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
	#using CairoMakie
	#using FileIO
end;

# ╔═╡ 1c476b4a-3ee6-4e5b-b903-abfc4d557569
begin
	# Hack fix.
	using ManifoldsBase
	using ManifoldsBase: PowerManifoldNested, get_iterator, _access_nested, _read, _write
	import ManifoldsBase: _get_vectors
	function _get_vectors(
    M::PowerManifoldNested,
    p,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:PowerBasisData},
) where {𝔽}
    zero_tv = zero_vector(M, p)
    rep_size = representation_size(M.manifold)
    vs = typeof(zero_tv)[]
    for i in get_iterator(M)
        b_i = _access_nested(M, B.data.bases, i)
        p_i = _read(M, rep_size, p, i)
        # println(get_vectors(M.manifold, p_i, b_i))
        for v in get_vectors(M.manifold, p_i, b_i) #b_i.data
            new_v = copy(M, p, zero_tv)
            copyto!(M.manifold, _write(M, rep_size, new_v, i), p_i, v)
            push!(vs, new_v)
        end
    end
    return vs
end
end

# ╔═╡ 7b3e1aa5-db29-4519-9860-09f6cc933c07
begin
	N=25
	h = 1/(N+2)*π/2
	st = 0.5
	#halt = pi - st
	halt = pi/2
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y0 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT = [sin(halt),0,cos(halt)] # endpoint of geodesic
	
	#yT = [sin(st),0,cos(st)] # startpoint of geodesic: suedpol
	#y0 = [sin(halt),0,cos(halt)] # endpoint of geodesic: nordpol

	#y0 = [cos(st),sin(st),0] # startpoint of geodesic: aequator
	#yT = [cos(halt),sin(halt),0] # endpoint of geodesic: aequator
end;

# ╔═╡ aa325d08-1990-4ef3-8205-78be6d06c711
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end;

# ╔═╡ ccf9e32c-0efd-4520-85a7-3cfb78ce9e15
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
end;

# ╔═╡ 632bb19d-02dd-4d03-bd92-e2222b26271f
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ 7b287c39-038a-4a02-b571-6cb4ee7f68d0
begin
	# force
	function w(M, p, c)
		#return [3.0*p[1]+p[2], -p[1], p[3]]
		#return c*[p[1]^2-p[2], p[1], p[3]]
		#return [0.0,3.0,0.0]
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end
end;

# ╔═╡ b59b848a-859e-4201-8f02-67e806a91551
begin
	function w_prime(M, p, c)
		#return [[3.0,1.0,0.0], [-1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return c*[[2.0*p[1],-1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return [[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
		return c*[[p[3]*2*p[1]*p[2]/(p[1]^2+p[2]^2)^2, p[3]*(-1.0/(p[1]^2+p[2]^2) + 2.0*p[2]^2/(p[1]^2+p[2]^2)^2), -p[2]/(p[1]^2+p[2]^2)], [p[3]*(1.0/(p[1]^2+p[2]^2) - 2.0*p[1]^2/(p[1]^2+p[2]^2)^2), p[3]*(-2.0*p[1]*p[2]/(p[1]^2+p[2]^2)^2), p[1]/(p[1]^2+p[2]^2)], [0.0, 0.0, 0.0]]
	end
end;

# ╔═╡ f25def4a-0733-4b46-bd48-673de0eff83e
function w_primealt(M, p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ 56dce4f9-83a9-4a50-8b91-007e4ddfeacc
function proj_prime(S, p, X, Y) # S_i*(Y)
	#return project(S, p, (- X*p' - p*X')*Y) 
	return (- X*p' - p*X')*Y
end

# ╔═╡ 55e3da7e-458f-49ce-8838-4cbcd39a97dd
function project(S,p,X)
return(X-p*p'X)
end

# ╔═╡ 483b9dc4-ff39-4c4d-86c9-ac7643752fca
function A(M, y, X, constant)
	# Include boundary points
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	S = M.manifold
	Z = zero_vector(M, y)
	for i in 1:N
		y_i = Oy[M, i]
		y_next = Oy[M, i+1]
		y_pre = Oy[M, i-1]
		X_i = X[M,i]
		
		Z[M,i] = 1/h * (2*y_i - y_next - y_pre) .+ h * w(S, y_i, constant)

		Z[M,i] = proj_prime(S, y_i, X_i, Z[M,i])
		
		Z[M,i] = Z[M, i] - h * proj_prime(S, y_i, X_i, Z[M,i])
		if i > 1
			Z[M,i] = Z[M,i] - 1/h * X[M,i-1]
		end
		Z[M,i] = Z[M,i] + 2/h * (X[M,i]) + h*X[M, i]' * w_prime(S, y_i, constant)
		if i < N
			Z[M,i] = Z[M,i] - 1/h * X[M,i+1]
		end
	end
	return Z
end

# ╔═╡ 06a99b80-7594-4e01-a2cb-b3144ea4b96c
function A0(M, y, X, constant)
	# Include boundary points
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	S = M.manifold
	Z = zero_vector(M, y)
	for i in 1:N
		y_i = Oy[M, i]
		y_next = Oy[M, i+1]
		y_pre = Oy[M, i-1]
		X_i = X[M,i]
		
		Z[M,i] = 1/h * (2*y_i - y_next - y_pre) .+ h * w(S, y_i, constant)

		Z[M,i] = 0.0*proj_prime(S, y_i, X_i, Z[M,i])
		
		Z[M,i] = Z[M, i] - 0.0*h * proj_prime(S, y_i, X_i, Z[M,i])
		if i > 1
			Z[M,i] = Z[M,i] - 1/h * X[M,i-1]
		end
		Z[M,i] = Z[M,i] + 2/h * (X[M,i]) + h*X[M, i]' * w_prime(S, y_i, constant)
		if i < N
			Z[M,i] = Z[M,i] - 1/h * X[M,i+1]
		end
	end
	return Z
end

# ╔═╡ 05c7e6fe-5335-41d5-ad31-d8ff8fe354c0
function b(M, y, constant)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		S = M.manifold
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			X[M,i]= 1/h * (2.0*y_i - y_next - y_pre) .+ h * w(S, y_i, constant)
		end
		return X
end



# ╔═╡ 764987fc-b909-47c6-a3fb-fa33865f838d
function F_at(S, y0, ydot, B, Bdot, constant)
	  return ydot'*Bdot+w(S,y0,constant)'*B
end

# ╔═╡ 38996668-cbc8-4180-8d19-c1df833227fc
function assemble_local_rhs(S, i,b, yl, yr, Bl, Br, constant)
	dim = manifold_dimension(S)
    idxl=dim*(i-1)
    idxr=dim*i
	ydot=(yr-yl)/h
	# Trapezregel
	quadwght = 0.5*h   
	for k in 1:dim
		Bdotl=-Bl[k]/h
		Bdotr= Br[k]/h
		# linke Testfunktion
        if idxl>=0 
			#linker Quadraturpunkt
			tmp =  F_at(S,yl,ydot,Bl[k],Bdotl,constant)	
			#rechter Quadraturpunkt
			tmp += F_at(S,yr,ydot,0.0*Bl[k],Bdotl,constant)	
			# Update der rechten Seite
		  	b[idxl+k]+= quadwght*tmp	
		end
		# rechte Testfunktion
		if idxr < length(b)
			tmp  = F_at(S,yl,ydot,0.0*Br[k],Bdotr,constant)	
			tmp += F_at(S,yr,ydot,Br[k],Bdotr,constant)	
            b[idxr+k]+= quadwght*tmp
		end
	end
end

# ╔═╡ 7f79b037-a17e-4886-94b3-286e73ac2bbb
function Fprime_at(S,y0,ydot,B1,B1dot,B2,B2dot,constant)
	return B1dot'*B2dot+(w_primealt(S,y0,constant)*B1)'*B2
end

# ╔═╡ 433b3483-cfce-49ee-88e2-466bf9589fa9
function assemble_local_Jac(S, i,A,b, yl, yr, Bl, Br, constant)
 dim = manifold_dimension(S)
 idxl=dim*(i-1)
 idxr=dim*i
 ydot=(yr-yl)/h
 quadwght=0.5*h
 #	Schleife über Testfunktionen
 for k in 1:dim
	Bdotlk=-Bl[k]/h
	Bdotrk=Br[k]/h
    # Schleife über Testfunktionen
	for j in 1:dim
		# Zeit-Ableitungen der Testfunktionen (=0 am jeweils anderen Rand)
		Bdotlj=(0-1)*Bl[j]/h
		Bdotrj=(1-0)*Br[j]/h

		# y-Ableitungen der Projektionen
		Pprimel=proj_prime(S,yl,Bl[j],Bl[k])
		Pprimer=proj_prime(S,yr,Br[j],Br[k])

		# Zeit- und y-Ableitungen der Projektionen
		Pprimedotl=(0-1)*Pprimel/h
		Pprimedotr=(1-0)*Pprimer/h
		
		# linke x linke Testfunktion
        if idxl>=0
		   # linker Quadraturpunkt
		   # Ableitung in der Einbettung	
		   tmp  = Fprime_at(S,yl,ydot,Bl[j],Bdotlj,Bl[k],Bdotlk,constant)
		   # Modifikation für Kovariante Ableitung	
		   tmp += F_at(S,yl,ydot,Pprimel,Pprimedotl,constant)
		   # rechter Quadraturpunkt (siehe oben)
		   tmp += Fprime_at(S,yr,ydot,0.0*Bl[j],Bdotlj,0.0*Bl[k],Bdotlk,constant)
		   tmp += F_at(S,yr,ydot,0.0*Pprimel,Pprimedotl,constant)
           # Update des Matrixeintrags
		   A[idxl+k,idxl+j]+=quadwght*tmp
		   # TODO: Stimmt das auch bei nicht-symmetrischen Matrizen? j <-> k?
		end
		# linke x rechte Testfunktion
		if idxl>=0 && idxr<length(b)
		   # linker Quadraturpunkt
		   # Ableitung in der Einbettung	
			tmp  = Fprime_at(S,yl,ydot,0.0*Br[j],Bdotrj,Bl[k],Bdotlk,constant)	
		   # Modifikation für Kovariante Ableitung fällt hier weg, da Terme = 0
		   # rechter Quadraturpunkt
			tmp += Fprime_at(S,yr,ydot,Br[j],Bdotrj,0.0*Bl[k],Bdotlk,constant)	
           # Symmetrisches Update der Matrixeinträge
			A[idxl+k,idxr+j] += quadwght*tmp
			A[idxr+j,idxl+k] += quadwght*tmp
		 end	
		# rechte x rechte Testfunktion (siehe oben)
		 if idxr < length(b)
		   tmp  = Fprime_at(S,yl,ydot,0.0*Br[j],Bdotrj,0.0*Br[k],Bdotrk,constant)
		   tmp += F_at(S,yl,ydot,0.0*Pprimer,Pprimedotr,constant)
		   tmp += Fprime_at(S,yr,ydot,Br[j],Bdotrj,Br[k],Bdotrk,constant)
		   tmp += F_at(S,yr,ydot,Pprimer,Pprimedotr,constant)
			 
		   A[idxr+k,idxr+j]+=quadwght*tmp
			 # TODO: Stimmt das auch bei nicht-symmetrischen Matrizen?  j <-> k?
		 end
	end
 end
end

# ╔═╡ 62bf2114-1551-4467-9d48-d2a3a3b8aa8e
function bundlemap(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		S = M.manifold
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			X[M,i]= 1/h * (2.0*y_i - y_next - y_pre) .+ h * w(S, y_i, 1.0)
		end
		return X
end

# ╔═╡ 48cd163d-42d1-4783-ace6-629d1ea495d4
function connection_map(E, q)
    return q
end

# ╔═╡ 0d741410-f182-4f5b-abe4-7719e627e2dc
function solve_linear_system(M, A, b, p, state, prob)
	obj = get_objective(prob)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	n = manifold_dimension(M)
	Ac = zeros(n,n);
	Ac0 = zeros(n,n);
	Acalt = zeros(n,n);
	bc = zeros(n)
	bcalt=zeros(n)
	Oy = OffsetArray([y0, p..., yT], 0:(length(Omega)+1))
	S = M.manifold
	# Schleife über Intervalle
	for i in 0:length(Omega)
		yl=Oy[i]
		yr=Oy[i+1]
		Bcl=get_basis(S,yl,DefaultOrthonormalBasis())
	    Bl = get_vectors(S, yl, Bcl)
		Bcr=get_basis(S,yr,DefaultOrthonormalBasis())
	    Br = get_vectors(S, yr, Bcr)
        assemble_local_rhs(S, i, bcalt, yl, yr, Bl, Br, obj.scaling)		
        assemble_local_Jac(S, i, Acalt,bcalt, yl, yr, Bl, Br, obj.scaling)		
	end
	e = enumerate(base)
	if state.is_same == true
		#println("Newton")
   		for (i,basis_vector) in e
      	G = A(M, p, basis_vector, obj.scaling)
      	G0 = A0(M, p, basis_vector, obj.scaling)
	  	#Ac[:,i] = get_coordinates(M, p, G, B)
		Ac[i,:] = get_coordinates(M, p, G, B)'
		Ac0[i,:] = get_coordinates(M, p, G0, B)'
		#for (j, bv) in e
			#Ac[i,j] = bv' * G
		#end
      	bc[i] = -1.0 * b(M, p, obj.scaling)'*basis_vector
		end
	else
		#println("simplified Newton")
		for (i,basis_vector) in e
      	G = A(M, p, basis_vector, obj.scaling)
	  	#Ac[:,i] = get_coordinates(M, p, G, B)
		Ac[i,:] = get_coordinates(M, p, G, B)'
		#for (j, bv) in e
			#Ac[i,j] = bv' * G
		#end
      	bc[i] = (1.0 - state.stepsize.alpha)*b(M, p, obj.scaling)'*basis_vector - b(M, state.p_trial, obj.scaling)' * vector_transport_to(M, state.p, basis_vector, state.p_trial, ProjectionTransport())
		end
	end
	#bc = get_coordinates(M, p, b(M, p), B)
	#diag_A = Diagonal([abs(Ac[i,i]) < 1e-12 ? 1.0 : 1.0/Ac[i,i] for i in 1:n])
	#println(Ac)
	#println(bc)
	println("...")
	println("b:",norm(bcalt))
	println("A:",norm(Acalt-Ac))
	println(norm(Acalt-Ac0))
	println(norm(Ac-Ac0))
	Xc = (Acalt) \ (-bcalt)
	res_c = get_vector(M, p, Xc, B)
	#println("norm =", norm(res_c))
	#println(diag(diag_A))
	#println(cond(Ac))
	#println(Xc)
	return res_c
end

# ╔═╡ 48e8395e-df79-4600-bcf9-50e318c49d58
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p, newtonstate, problem)

# ╔═╡ 00e47eab-e088-4b55-9798-8b9f28a6efe5
begin
	Random.seed!(42)
	p = rand(power)
	#y_0 = [project(S, (discretized_y[i]+0.02*p[power,i])) for i in 1:N]
	y_0 = copy(power, discretized_y)
	
end;

# ╔═╡ 0cadffa2-dc8e-432e-b198-2e519e128576
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);

it_back = 0

#ws = [-1.0*w(Manifolds.Sphere(2), p) for p in discretized_y]
#ws_res = [-1.0*w(Manifolds.Sphere(2), p) for p in iterates[length(change)-it_back]]
	
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)
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
	scatter!(ax, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y); markersize =8, color=:blue)
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	E = TangentBundle(power)
	obj = VectorbundleObjective(b, A, connection_map)
	obj.scaling = 1.0
	problem = VectorbundleManoptProblem(power, E, obj)

	increment = 0.1
    y_start = copy(power,discretized_y)
    y_current = copy(power,y_start)
    y_last = copy(power,y_start)
	for i in range(1,1)
		println(obj.scaling)
		copyto!(power,y_last,y_current)
		state = VectorbundleNewtonState(power, E, bundlemap, y_current, solve, AllocatingEvaluation(), stopping_criterion=(StopAfterIteration(50)|StopWhenChangeLess(power, 1e-10)), retraction_method=ProjectionRetraction(), stepsize=Manopt.ConstantStepsize(power,1.0))
		#retraction_method=ProjectionRetraction(), stepsize=ConstantStepsize(1.0))
		st_res = solve!(problem, state)
		println("Norm:", norm(y_last-y_current))
		if Manopt.indicates_convergence(st_res.stop)
			obj.scaling = obj.scaling + increment
			scatter!(ax, π1.(y_current), π2.(y_current), π3.(y_current); markersize =8, color=:orange)
		else
			factor=0.5
			obj.scaling = obj.scaling - increment
			global increment=increment*factor
			obj.scaling = obj.scaling +increment
			scatter!(ax, π1.(y_current), π2.(y_current), π3.(y_current); markersize =8, color=:red)
			copyto!(power,y_current,y_last)
		end
		println(Manopt.indicates_convergence(st_res.stop)) 
		println("Inc: ",increment)
		#println(Manopt.get_reason(st_res)) 

	end
	
	#st_res = vectorbundle_newton(power, TangentBundle(power), b, A, connection_map, y_0; sub_problem=solve, sub_state=AllocatingEvaluation(), stopping_criterion=(StopAfterIteration(47)|StopWhenChangeLess(1e-14)), retraction_method=ProjectionRetraction(),
#stepsize=ConstantStepsize(1.0), 
	#debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop], record=[:Iterate, :Change], return_state=true)
	#start_geodesic = deepcopy(get_solver_result(st_res))


	fig	
end

# ╔═╡ Cell order:
# ╠═0783b732-8574-11ef-017d-3939cfc57442
# ╠═b7f09653-9692-4f92-98e3-f988ed0c3d2d
# ╠═1c476b4a-3ee6-4e5b-b903-abfc4d557569
# ╠═7b3e1aa5-db29-4519-9860-09f6cc933c07
# ╠═aa325d08-1990-4ef3-8205-78be6d06c711
# ╠═ccf9e32c-0efd-4520-85a7-3cfb78ce9e15
# ╠═632bb19d-02dd-4d03-bd92-e2222b26271f
# ╠═7b287c39-038a-4a02-b571-6cb4ee7f68d0
# ╠═b59b848a-859e-4201-8f02-67e806a91551
# ╠═f25def4a-0733-4b46-bd48-673de0eff83e
# ╠═56dce4f9-83a9-4a50-8b91-007e4ddfeacc
# ╠═55e3da7e-458f-49ce-8838-4cbcd39a97dd
# ╠═483b9dc4-ff39-4c4d-86c9-ac7643752fca
# ╠═06a99b80-7594-4e01-a2cb-b3144ea4b96c
# ╠═05c7e6fe-5335-41d5-ad31-d8ff8fe354c0
# ╠═38996668-cbc8-4180-8d19-c1df833227fc
# ╠═764987fc-b909-47c6-a3fb-fa33865f838d
# ╠═7f79b037-a17e-4886-94b3-286e73ac2bbb
# ╠═433b3483-cfce-49ee-88e2-466bf9589fa9
# ╠═62bf2114-1551-4467-9d48-d2a3a3b8aa8e
# ╠═48cd163d-42d1-4783-ace6-629d1ea495d4
# ╠═48e8395e-df79-4600-bcf9-50e318c49d58
# ╠═0d741410-f182-4f5b-abe4-7719e627e2dc
# ╠═00e47eab-e088-4b55-9798-8b9f28a6efe5
# ╠═0cadffa2-dc8e-432e-b198-2e519e128576
