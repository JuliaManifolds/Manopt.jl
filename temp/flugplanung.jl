### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 14ef9e32-7986-11ef-18c9-c7c6957859f0
using Pkg; Pkg.activate();

# ╔═╡ 80420787-a807-49c9-9855-4b7001402cab
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays, RecursiveArrayTools
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ e15726ce-507c-4e16-a930-8a92afa97478
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

# ╔═╡ e59af672-f337-4b27-87d8-4675ffcb6f72
begin
	N = 100
	h = 1/(N+2)*π/2
	st = 0.2
	#halt = pi - st
	halt = pi/2
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y0 = [sin(st),0,cos(st)] # startpoint of geodesic: nordpol
	yT = [sin(halt),0,cos(halt)] # endpoint of geodesic: suedpol
	
	#yT = [sin(st),0,cos(st)] # startpoint of geodesic: suedpol
	#y0 = [sin(halt),0,cos(halt)] # endpoint of geodesic: nordpol

	#y0 = [cos(st),sin(st),0] # startpoint of geodesic: aequator
	#yT = [cos(halt),sin(halt),0] # endpoint of geodesic: aequator

	#yT = [cos(st),sin(st),0] # startpoint of geodesic: aequator
	#y0 = [cos(halt),sin(halt),0] # endpoint of geodesic: aequator
end;

# ╔═╡ d69568c4-c64f-4692-9429-597cd02beaab
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end;

# ╔═╡ 8249e92d-b611-4683-803f-591519a753fa
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
	#return [cos(halt+st - t), sin(halt+st - t), 0]
end;

# ╔═╡ 915c4cad-0df1-4469-88f2-b02aaa72675b
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ e70f0be7-e768-472f-af8f-3df37d1de880
c = 4.0

# ╔═╡ 968036d4-d75e-4456-a2f3-6747042f7389
begin
	# force
	function w(M, p)
		#return [3.0*p[1]+p[2], -p[1], p[3]]
		#return c*[p[1]^2-p[2], p[1], p[3]]
		#return [0.0,3.0,0.0]
		return c*[p[3]*(-p[2]/(p[1]^2+p[2]^2)), p[3]*(p[1]/(p[1]^2+p[2]^2)), 0.0]
	end
end;

# ╔═╡ 21150026-495a-4ff5-bfb1-a3d6362f7305
begin
	function w_prime(M, p)
		#return [[3.0,1.0,0.0], [-1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return [[2.0*p[1],-1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return [0.0 0.0 0.0; 0.0 1.0 1.0; 0.0 0.0 0.0]
		nenner = p[1]^2+p[2]^2
		#return c*[p[3]*2*p[1]*p[2]/(nenner^2) p[3]*(p[1]^2-p[2]^2)/(nenner^2) -p[2]/nenner; p[3]*(-p[1]^2+p[2]^2)/(nenner^2) p[3]*(-2*p[1]*p[2])/(nenner^2) p[1]/nenner; 0.0 0.0 0.0]
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(p[1]^2+p[2]^2)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
	end
end;

# ╔═╡ 1fef7f5f-73b2-4cbe-9330-3213719af9e1
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

# ╔═╡ 60d10fe3-a28c-487c-9dbb-0239a2fda3ca
(-w_prime(S, [1.0, 1.0, 1.0]) + w_prime(S, [1.0, 1.0, 1.0] + 0.00001*[0.0, 1.0, 0.0]))/0.00001

# ╔═╡ 86874a60-4e58-4d45-b4f7-538374eafc85
w_doubleprime(S, [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])

# ╔═╡ 63a634a4-e944-455a-9d35-d07dd0dfd8da
function proj(S, p, v)
	return v .- (p*p')*v
end;

# ╔═╡ b18bb496-3445-43f3-9c98-0e3f7c073a43
function proj_prime(S, p, X, Y) # S_i*(Y)
	#return project(S, p, (- X*p' - p*X')*Y) 
	return (- X*p' - p*X')*Y
end;

# ╔═╡ 4cec1238-369a-4fc0-ac6e-b9463e2314ac
function A(M, y, X)
	# Include boundary points
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	S = M.manifold
	Z = zero_vector(M, y)
	for i in 1:N
		y_i = Oy[M, i]
		y_next = Oy[M, i+1]
		y_pre = Oy[M, i-1]
		X_i = X[M,i]
		
		Z[M,i] = 1/h * (2*y_i - y_next - y_pre - h*w(S, y_i) + h*w(S, y_next))

		Z[M,i] = proj_prime(S, y_i, X_i, Z[M,i])
		
		Z[M,i] = Z[M,i] + 1/h * X[M, i]

		Z[M,i] = Z[M,i] + 1/h * (X[M, i] - h*w_prime(S, y_i)*X_i)

		Z[M,i] = Z[M,i] - w_prime(S, y_i)'*(X_i - h*w_prime(S, y_i)*X_i)
		#Z[M,i] = Z[M,i] - w_prime(S, y_i)'*X_i
		#Z[M,i] = Z[M,i] - w_prime(S, y_i)'*(- h*w_prime(S, y_i)*X_i)

		Z[M,i] = Z[M,i] - w_doubleprime(S, y_i, X_i)'*(y_i - y_pre - h*w(S, y_i))

		Z[M,i] = Z[M,i] - proj_prime(S, y_i, X_i, w_prime(S, y_i)'*(y_i - y_pre - h*w(S, y_i))) 

		#Z[M,i] = Z[M, i] - h * proj_prime(S, y_i, X_i, Z[M,i])
		if i > 1
			Z[M,i] = Z[M,i] - 1/h * X[M,i-1]
			Z[M,i] = Z[M,i] + w_prime(S, y_i)'*X[M, i-1]
		end
		#Z[M,i] = Z[M,i] + 2/h * (X[M,i])
		if i < N
			Z[M,i] = Z[M,i] - 1/h * X[M,i+1]
			Z[M,i] = Z[M,i] + w_prime(S, y_next)*X[M,i+1]
		end
	end
	return Z
end

# ╔═╡ 655daf7e-e9de-4fa6-827d-5617c5ebea7f
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		S = M.manifold
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			X[M,i]= 1/h * (2.0*y_i - y_next - y_pre - h*w(S, y_i) + h*w(S, y_next))
			
			diff = y_i - y_pre - (h* w(S, y_i))
			adj = - w_prime(S, y_i)'*diff
			
			X[M,i] = X[M,i] + adj
		end
		return X
end;

# ╔═╡ 756a00e8-a34e-4318-9df8-5fed5e828418
function b2(M, y, v)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		S = M.manifold
		X = zero_vector(M,y)
		rhs = zero_vector(M,y)
		ps = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			rhs[i] = 1/h * (2.0*y_i - y_next - y_pre - h*w(S, y_i) + h*w(S, y_next))
			#X[M,i] = inner(S, y_i, B, v[i])
			ps[i] = -1.0*proj_prime(S, y_i, v[i], w(S, y_i)) #- w_prime(S, y_i)
		end
		X = inner(M, y, rhs, v)
		X = X + inner(M, y, ps, v)
		return X
end;

# ╔═╡ 966f3ebf-45c1-4b44-b13a-7740a2cdd679
begin
	B = get_basis(power, discretized_y, DefaultOrthonormalBasis())
	base = get_vectors(power, discretized_y, B)
	dim = manifold_dimension(power)
	bc = zeros(dim)
	bc2 = zeros(dim)
	e = enumerate(base)
   	for (i,basis_vector) in e
		#println(basis_vector)
		#println("")
      	bc[i] = b(power, discretized_y)'*basis_vector
		bc2[i] = b2(power, discretized_y, basis_vector)
	end
end;

# ╔═╡ bf7d2ced-72d9-464f-b20b-d3540def9c1d
function connection_map(E, q)
    return q
end;

# ╔═╡ a34bb2e1-3d43-4cf3-ba82-dee9f9d59348
function solve_linear_system(M, A, b, p, state)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	n = manifold_dimension(M)
	Ac = zeros(n,n);
	bc = zeros(n)

	e = enumerate(base)
	if state.is_same == true
   		for (i,basis_vector) in e
      	G = A(M, p, basis_vector)
	  	#Ac[:,i] = get_coordinates(M, p, G, B)
		Ac[i,:] = get_coordinates(M, p, G, B)'
		#for (j, bv) in e
			#Ac[i,j] = bv' * G
		#end
      	bc[i] = -1.0 * b(M, p)'*basis_vector
		end
	else
		for (i,basis_vector) in e
      	G = A(M, p, basis_vector)
	  	#Ac[:,i] = get_coordinates(M, p, G, B)
		Ac[i,:] = get_coordinates(M, p, G, B)'
		#for (j, bv) in e
			#Ac[i,j] = bv' * G
		#end
      	bc[i] = (1.0 - state.stepsize.alpha)*b(M, p)'*basis_vector - b(M, state.p_trial)' * vector_transport_to(M, state.p, basis_vector, state.p_trial, ProjectionTransport())
		end
	end
	#bc = get_coordinates(M, p, b(M, p), B)
	#diag_A = Diagonal([abs(Ac[i,i]) < 1e-12 ? 1.0 : 1.0/Ac[i,i] for i in 1:n])
	#println(Ac)
	#println("")
	#println(norm(Ac-Ac'))
	#println(bc)
	Xc = (Ac) \ (bc)
	res_c = get_vector(M, p, Xc, B)
	#println(diag(diag_A))
	#println(cond(Ac))
	#println(Xc)
	return res_c
end;

# ╔═╡ 91014dd8-a16f-4e8d-8dd9-504bc5bcd278
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p, newtonstate);

# ╔═╡ 914e11d3-145d-46a9-a59b-9245d34e1f08
begin
	Random.seed!(42)
	p = rand(power)
	#y_0 = [project(S, (discretized_y[i]+0.02*p[power,i])) for i in 1:N]
	y_0 = discretized_y
end;

# ╔═╡ 0d5ce271-e0ac-4c75-a011-7fcb1372d973
st_res = vectorbundle_newton(power, TangentBundle(power), b, A, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
#stepsize=ConstantStepsize(1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)

# ╔═╡ 1b024dc2-ba46-4091-b178-c7ac96ef16fe
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	fig_c, ax_c, plt_c = lines(1:length(change), log.(change))
	fig_c
end;

# ╔═╡ ecd2835c-d0e8-4369-b310-3d6ae4dce914
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ 07386cfa-2d54-429d-b808-7a2d220337c9
p_res = get_solver_result(st_res);

# ╔═╡ aec51445-dc82-427a-9150-e97e6809efe2
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

ws = [-1.0*w(Manifolds.Sphere(2), p) for p in p_res]
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

# ╔═╡ 791c5c10-6d05-4021-bf2a-f58c262d7001
begin
M=power
b0 = b(M, y_0)
i = 2
ch = 1e-6
B3 = get_basis(M, y_0, DefaultOrthonormalBasis())
basis = get_vectors(M, y_0, B3)
y_1 = exp(M, y_0, ch*basis[i])
b1 = b(M, y_1)
checkA = 1/ch * (b1 - b0)
A0 = A(M, y_0, base[M,i])
#checkA = zeros(15,15)

coordA = zeros(2*N,2*N)
	en = enumerate(basis)
   	for (i,basis_vector) in en
      	G = A(M, y_0, basis_vector)
	  	#Ac[:,i] = get_coordinates(M, p, G, B)
		coordA[i,:] = get_coordinates(M, y_0, G, B3)'
	end
	coordA
end

# ╔═╡ f5f99ecc-894e-4150-ac9f-124d4bdb5f1b
typeof(StopWhenChangeLess(M,1e-13))

# ╔═╡ d59841d1-0ba6-4737-b170-be0b19f13269
A0

# ╔═╡ 08d78bb9-5a09-48e2-9d75-a244bd11fed1
checkA

# ╔═╡ Cell order:
# ╠═14ef9e32-7986-11ef-18c9-c7c6957859f0
# ╠═80420787-a807-49c9-9855-4b7001402cab
# ╠═e15726ce-507c-4e16-a930-8a92afa97478
# ╠═e59af672-f337-4b27-87d8-4675ffcb6f72
# ╠═d69568c4-c64f-4692-9429-597cd02beaab
# ╠═8249e92d-b611-4683-803f-591519a753fa
# ╠═915c4cad-0df1-4469-88f2-b02aaa72675b
# ╠═e70f0be7-e768-472f-af8f-3df37d1de880
# ╠═968036d4-d75e-4456-a2f3-6747042f7389
# ╠═21150026-495a-4ff5-bfb1-a3d6362f7305
# ╠═1fef7f5f-73b2-4cbe-9330-3213719af9e1
# ╠═60d10fe3-a28c-487c-9dbb-0239a2fda3ca
# ╠═86874a60-4e58-4d45-b4f7-538374eafc85
# ╠═63a634a4-e944-455a-9d35-d07dd0dfd8da
# ╠═b18bb496-3445-43f3-9c98-0e3f7c073a43
# ╠═4cec1238-369a-4fc0-ac6e-b9463e2314ac
# ╠═655daf7e-e9de-4fa6-827d-5617c5ebea7f
# ╠═756a00e8-a34e-4318-9df8-5fed5e828418
# ╠═966f3ebf-45c1-4b44-b13a-7740a2cdd679
# ╠═bf7d2ced-72d9-464f-b20b-d3540def9c1d
# ╠═91014dd8-a16f-4e8d-8dd9-504bc5bcd278
# ╠═a34bb2e1-3d43-4cf3-ba82-dee9f9d59348
# ╠═914e11d3-145d-46a9-a59b-9245d34e1f08
# ╠═0d5ce271-e0ac-4c75-a011-7fcb1372d973
# ╠═f5f99ecc-894e-4150-ac9f-124d4bdb5f1b
# ╠═1b024dc2-ba46-4091-b178-c7ac96ef16fe
# ╠═ecd2835c-d0e8-4369-b310-3d6ae4dce914
# ╠═07386cfa-2d54-429d-b808-7a2d220337c9
# ╠═aec51445-dc82-427a-9150-e97e6809efe2
# ╠═791c5c10-6d05-4021-bf2a-f58c262d7001
# ╠═d59841d1-0ba6-4737-b170-be0b19f13269
# ╠═08d78bb9-5a09-48e2-9d75-a244bd11fed1
