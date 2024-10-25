### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 85e76846-912a-11ef-294a-c717389928e4
using Pkg; Pkg.activate();

# ╔═╡ 48950604-c2c2-4310-8de5-f89db905668b
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

# ╔═╡ fe0c1524-b5f4-4afc-aee7-bd2de9d482b6
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

# ╔═╡ 6b1bca0b-f209-445e-8f41-b19372c9fffc
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

# ╔═╡ bd50d1db-7c55-4279-abe0-89f2fe986cc6
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end;

# ╔═╡ 449e9782-5bed-4642-a068-f8c9106bbe86
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
	#return [cos(halt+st - t), sin(halt+st - t), 0]
end;

# ╔═╡ a789119e-04b9-4456-acba-ec8e8702c231
begin
	# force
	function w(M, p, c)
		#return [3.0*p[1]+p[2], -p[1], p[3]]
		#return c*[p[1]^2-p[2], p[1], p[3]]
		#return [0.0,3.0,0.0]
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end
end;

# ╔═╡ a5811f04-59f8-4ea2-8616-efad7872830a
begin
	function w_prime(M, p, c)
		#return [[3.0,1.0,0.0], [-1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return c*[[2.0*p[1],-1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return [[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
		return c*[[p[3]*2*p[1]*p[2]/(p[1]^2+p[2]^2)^2, p[3]*(-1.0/(p[1]^2+p[2]^2) + 2.0*p[2]^2/(p[1]^2+p[2]^2)^2), -p[2]/(p[1]^2+p[2]^2)], [p[3]*(1.0/(p[1]^2+p[2]^2) - 2.0*p[1]^2/(p[1]^2+p[2]^2)^2), p[3]*(-2.0*p[1]*p[2]/(p[1]^2+p[2]^2)^2), p[1]/(p[1]^2+p[2]^2)], [0.0, 0.0, 0.0]]
	end
end;

# ╔═╡ 4c3bea2b-017b-4cea-b3b6-7d0efb1b0f62
function w_primealt(M, p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ 87e0b9e0-b03a-4cad-ae53-691a0aeea887
function proj_prime(S, p, X, Y) # S_i*(Y)
	#return project(S, p, (- X*p' - p*X')*Y) 
	return (- X*p' - p*X')*Y
end

# ╔═╡ 93b4873c-cc77-46a9-89eb-2ac8377f375f
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ 94e61962-2e4b-45f4-be62-cb85aee2069d
c = 1.0

# ╔═╡ 709daed1-7ef3-4fa5-b390-e18d27a7cefd
begin 
	function w_doubleprime(M, p, v)
		nenner = (p[1]^2+p[2]^2)
		w1 = 1/(nenner^2)*[(2*p[2]*p[3]*nenner-8*p[1]^2*p[2]*p[3])/nenner -p[3]*(2*p[1]*nenner^2-4*p[1]*(p[1]^4-p[2]^4))/nenner^2 2*p[1]*p[2]; (-2*p[1]*p[3]*nenner^2-4*p[1]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[2]*p[3]*nenner+8*p[1]^2*p[2]*p[3])/nenner (p[2]^2-p[1]^2); 0.0 0.0 0.0]
		
		w2 = 1/(nenner^2)*[(2*p[1]*p[3]*nenner-8*p[1]*p[2]^2*p[3])/nenner -p[3]*(-2*p[2]*nenner^2-4*p[2]*(p[1]^4-p[2]^4))/nenner^2 p[2]^2-p[1]^2; (2*p[2]*p[3]*nenner^2-4*p[2]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[1]*p[3]*nenner+8*p[1]*p[2]^2*p[3])/nenner -2*p[1]*p[2]; 0.0 0.0 0.0]
		
		w3 = 1/(nenner^2)*[2*p[1]*p[2] -p[1]^2+p[2]^2 0.0; p[2]^2-p[1]^2 -2*p[1]*p[2] 0.0; 0.0 0.0 0.0]
		return c*(v[1]*w1 + v[2]*w2 + v[3]*w3)
		#return [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
	end
end;

# ╔═╡ 794099d0-f5a4-4260-9702-326acf061686
ex = 50

# ╔═╡ 604adc83-38be-43a5-836f-330213f9ac76
function F_at(S, y0, ydot, B, Bdot, constant)
	return ex*(((ydot - w(S, y0, constant))'*(ydot - w(S, y0, constant)))^(ex/2.0 - 1))*(Bdot - w_primealt(S,y0,constant)*B)'*(ydot - w(S, y0, constant))
end

# ╔═╡ 4abeeb91-7b1a-4756-aae6-f8322e4f979c
function Fprime_at(S,y0,ydot,B1,B1dot,B2,B2dot,constant)
	return ex*(ex-2)*(((ydot - w(S, y0, constant))'*(ydot - w(S, y0, constant)))^(ex/2.0 - 2.0))*((B2dot - w_primealt(S, y0, constant)*B2)'*(ydot - w(S, y0, constant)))*((B1dot - w_primealt(S, y0, constant)*B1)'*(ydot - w(S, y0, constant))) + ex * (((ydot - w(S, y0, constant))'*(ydot - w(S, y0, constant)))^(ex/2.0 - 1.0)) * ((-1.0*w_doubleprime(S, y0, B2)*B1)'*(ydot - w(S, y0, constant)) + (B1dot - w_primealt(S, y0, constant)*B1)'*(B2dot - w_primealt(S, y0, constant)
	*B2))
end

# ╔═╡ d135a4ab-76d2-4843-9929-4301c7787a85
function b(M, y, constant)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		S = M.manifold
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			X[M,i]= 1/h * (2.0*y_i - y_next - y_pre - h*w(S, y_i, c) + h*w(S, y_next, c))
			
			diff = y_i - y_pre - (h* w(S, y_i, c))
			adj = - w_primealt(S, y_i, c)'*diff
			
			X[M,i] = X[M,i] + adj
		end
		return X
end;

# ╔═╡ 7825ad3e-02d8-4a49-a8ed-6a0d44d2b982
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
		
		Z[M,i] = 1/h * (2*y_i - y_next - y_pre - h*w(S, y_i, constant) + h*w(S, y_next, constant))

		Z[M,i] = proj_prime(S, y_i, X_i, Z[M,i])
		
		Z[M,i] = Z[M,i] + 1/h * X[M, i]

		Z[M,i] = Z[M,i] + 1/h * (X[M, i] - h*w_primealt(S, y_i, constant)*X_i)

		Z[M,i] = Z[M,i] - w_primealt(S, y_i, constant)'*(X_i - h*w_primealt(S, y_i, constant)*X_i)
		#Z[M,i] = Z[M,i] - w_prime(S, y_i)'*X_i
		#Z[M,i] = Z[M,i] - w_prime(S, y_i)'*(- h*w_prime(S, y_i)*X_i)

		Z[M,i] = Z[M,i] - w_doubleprime(S, y_i, X_i)'*(y_i - y_pre - h*w(S, y_i, constant))

		Z[M,i] = Z[M,i] - proj_prime(S, y_i, X_i, w_primealt(S, y_i, constant)'*(y_i - y_pre - h*w(S, y_i, constant))) 

		#Z[M,i] = Z[M, i] - h * proj_prime(S, y_i, X_i, Z[M,i])
		if i > 1
			Z[M,i] = Z[M,i] - 1/h * X[M,i-1]
			Z[M,i] = Z[M,i] + w_primealt(S, y_i, constant)'*X[M, i-1]
		end
		#Z[M,i] = Z[M,i] + 2/h * (X[M,i])
		if i < N
			Z[M,i] = Z[M,i] - 1/h * X[M,i+1]
			Z[M,i] = Z[M,i] + w_primealt(S, y_next, constant)*X[M,i+1]
		end
	end
	return Z
end

# ╔═╡ 5c13ce80-209c-4901-913a-3283339a11cc
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

# ╔═╡ ec1be1f3-782c-48cd-a906-05b024af727a
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

# ╔═╡ 10244e2b-9745-4f6c-b41a-a8e40b72b879
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

# ╔═╡ 86a649c8-43f8-43a0-a652-24735d28c05e
function connection_map(E, q)
    return q
end

# ╔═╡ fbf47501-4a41-4f89-940c-9c68aca26b4b
function solve_linear_system(M, A, b, p, state, prob)
	obj = get_objective(prob)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	n = manifold_dimension(M)

	Acalt = zeros(n,n);
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
        assemble_local_rhs(S, i, bcalt, yl, yr, Bl, Br, c)		
        assemble_local_Jac(S, i, Acalt,bcalt, yl, yr, Bl, Br, c)		
	end
	e = enumerate(base)
	if state.is_same == false
		#println("simplified Newton")
		for (i,basis_vector) in e
      	G = A(M, p, basis_vector, c)
		Acalt[i,:] = get_coordinates(M, p, G, B)'
      	bcalt[i] = (1.0 - state.stepsize.alpha)*b(M, p, c)'*basis_vector - b(M, state.p_trial, c)' * vector_transport_to(M, state.p, basis_vector, state.p_trial, ProjectionTransport())
		end
	end
	Xc = (Acalt) \ (-bcalt)
	res_c = get_vector(M, p, Xc, B)
	return res_c
end

# ╔═╡ b2d6b477-28d1-421c-958e-ebcfe845a6bb
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p, newtonstate, problem)

# ╔═╡ 25cdf23f-c2ff-44be-9a34-bb565c36775e
begin
	Random.seed!(42)
	p = rand(power)
	#y_0 = [project(S, (discretized_y[i]+0.02*p[power,i])) for i in 1:N]
	y_0 = copy(power, discretized_y)
	
end;

# ╔═╡ dcf6174d-0b6f-48da-b0d8-f057b2992e0b
st_res = vectorbundle_newton(power, TangentBundle(power), bundlemap, A, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
stepsize=ConstantLength(1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)

# ╔═╡ 40e63e0a-fa25-4d9d-b685-383a7957c513
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	fig_c, ax_c, plt_c = lines(1:length(change), log.(change))
	fig_c
end;

# ╔═╡ 1f575073-92cf-4fc7-b8bd-995e17e14b69
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ 48e0d10c-23c4-48c4-91fa-b2a5ae443005
p_res = get_solver_result(st_res);

# ╔═╡ 9a9d6392-94f6-4306-9add-f802e2eb70f2
begin
		Oy = OffsetArray([y0, p_res..., yT], 0:(length(Omega)+1))
		normen = zeros(N)
		normenw = zeros(N)
		normeny = zeros(N)
		sum = 0
		for i in 1:N
			y_i = Oy[power, i]
			y_next = Oy[power, i+1]
			normen[i] = norm(S, y_i, ((y_next-y_i)/h - w(S, y_next, c)))
			sum += sum + normen[i]
			normenw[i] = norm(w(S, y_next, c))
			normeny[i] = norm(S, y_i, ((y_next-y_i)/h))
		end
		#println(normen)
	plot = Figure(;)
	
    rows, cols = fldmod1(1, 2)
	
	axs = Axis(plot[rows, cols], xminorgridvisible = true, xticks = (1:length(normen)), xlabel = "time step", ylabel = "‖⋅‖")
    scatterlines!(normen, color = :blue, label="air speed")
	scatterlines!(normenw, color = :red, label="wind field")
	scatterlines!(normeny, color = :orange, label="ground speed")

	plot[1, 2] = Legend(plot, axs, "Plot of norms of the ... ", framevisible = false)
	plot
end

# ╔═╡ 88c316c6-4463-4e42-abc8-d83fc117e56f
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

ws = [1.0*w(Manifolds.Sphere(2), p, c) for p in p_res]
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
	scatter!(ax, π1.(p_res), π2.(p_res), π3.(p_res); markersize =8, color=:orange)
	scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =8, color=:blue)
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	fig
end

# ╔═╡ Cell order:
# ╠═85e76846-912a-11ef-294a-c717389928e4
# ╠═48950604-c2c2-4310-8de5-f89db905668b
# ╠═fe0c1524-b5f4-4afc-aee7-bd2de9d482b6
# ╠═6b1bca0b-f209-445e-8f41-b19372c9fffc
# ╠═bd50d1db-7c55-4279-abe0-89f2fe986cc6
# ╠═449e9782-5bed-4642-a068-f8c9106bbe86
# ╠═a789119e-04b9-4456-acba-ec8e8702c231
# ╠═a5811f04-59f8-4ea2-8616-efad7872830a
# ╠═4c3bea2b-017b-4cea-b3b6-7d0efb1b0f62
# ╠═709daed1-7ef3-4fa5-b390-e18d27a7cefd
# ╠═87e0b9e0-b03a-4cad-ae53-691a0aeea887
# ╠═93b4873c-cc77-46a9-89eb-2ac8377f375f
# ╠═94e61962-2e4b-45f4-be62-cb85aee2069d
# ╠═794099d0-f5a4-4260-9702-326acf061686
# ╠═604adc83-38be-43a5-836f-330213f9ac76
# ╠═4abeeb91-7b1a-4756-aae6-f8322e4f979c
# ╠═d135a4ab-76d2-4843-9929-4301c7787a85
# ╠═7825ad3e-02d8-4a49-a8ed-6a0d44d2b982
# ╠═5c13ce80-209c-4901-913a-3283339a11cc
# ╠═ec1be1f3-782c-48cd-a906-05b024af727a
# ╠═10244e2b-9745-4f6c-b41a-a8e40b72b879
# ╠═86a649c8-43f8-43a0-a652-24735d28c05e
# ╠═b2d6b477-28d1-421c-958e-ebcfe845a6bb
# ╠═fbf47501-4a41-4f89-940c-9c68aca26b4b
# ╠═25cdf23f-c2ff-44be-9a34-bb565c36775e
# ╠═dcf6174d-0b6f-48da-b0d8-f057b2992e0b
# ╠═40e63e0a-fa25-4d9d-b685-383a7957c513
# ╠═1f575073-92cf-4fc7-b8bd-995e17e14b69
# ╠═48e0d10c-23c4-48c4-91fa-b2a5ae443005
# ╠═9a9d6392-94f6-4306-9add-f802e2eb70f2
# ╠═88c316c6-4463-4e42-abc8-d83fc117e56f
