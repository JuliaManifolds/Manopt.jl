### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 11863d90-6f62-11ef-06a1-611d44587188
using Pkg; Pkg.activate();

# ╔═╡ 282efd2f-2019-4bb5-a9ff-0ba7a18b2b58
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
	#using CairoMakie
	using FileIO
end;

# ╔═╡ 344ec4fa-af2f-4f78-8de8-a7f698c4ea46
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

# ╔═╡ a0b6475f-9fe0-4f7a-b36b-b7495e477bd1
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

# ╔═╡ bfb534ea-43aa-4964-afab-5abaa0a024bf
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end;

# ╔═╡ 09932d3b-b604-424e-8c02-2f5498590098
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
end;

# ╔═╡ 4ae5e86e-77a1-411e-a732-d0273bffea6f
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ 6e66d021-38f4-4037-8d63-c8cc75978485
c = 2.5

# ╔═╡ d887274f-e198-4af7-a897-a88fd94f04e2
begin
	# force
	function w(M, p)
		#return [3.0*p[1]+p[2], -p[1], p[3]]
		#return c*[p[1]^2-p[2], p[1], p[3]]
		#return [0.0,3.0,0.0]
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0]
	end
end;

# ╔═╡ 1d4ee847-f09d-4423-91ea-aa6d60f8d1b5
begin
	function w_prime(M, p)
		#return [[3.0,1.0,0.0], [-1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return c*[[2.0*p[1],-1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]]
		#return [[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]]
		return c*[[p[3]*2*p[1]*p[2]/(p[1]^2+p[2]^2)^2, p[3]*(-1.0/(p[1]^2+p[2]^2) + 2.0*p[2]^2/(p[1]^2+p[2]^2)^2), -p[2]/(p[1]^2+p[2]^2)], [p[3]*(1.0/(p[1]^2+p[2]^2) - 2.0*p[1]^2/(p[1]^2+p[2]^2)^2), p[3]*(-2.0*p[1]*p[2]/(p[1]^2+p[2]^2)^2), p[1]/(p[1]^2+p[2]^2)], [0.0, 0.0, 0.0]]
	end
end;

# ╔═╡ 13f350e2-ba3d-4652-9358-4bc2f08d9001
function proj_prime(S, p, X, Y) # S_i*(Y)
	#return project(S, p, (- X*p' - p*X')*Y)
	return (- X*p' - p*X')*Y
end

# ╔═╡ e8abf32a-5923-48b0-b739-78e527d1a4f9
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

		Z[M,i] = 1/h * (2*y_i - y_next - y_pre) .+ h * w(S, y_i)

		Z[M,i] = proj_prime(S, y_i, X_i, Z[M,i])

		Z[M,i] = Z[M, i] - h * proj_prime(S, y_i, X_i, Z[M,i])
		if i > 1
			Z[M,i] = Z[M,i] - 1/h * X[M,i-1]
		end
		Z[M,i] = Z[M,i] + 2/h * (X[M,i]) + h*X[M, i]' * w_prime(S, y_i)
		if i < N
			Z[M,i] = Z[M,i] - 1/h * X[M,i+1]
		end
	end
	return Z
end

# ╔═╡ 4dcf87c7-8084-4b19-90da-53bca0d97124
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		S = M.manifold
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			X[M,i]= 1/h * (2.0*y_i - y_next - y_pre) .+ h * w(S, y_i)
		end
		return X
end

# ╔═╡ 1b19151f-81dc-409c-90a7-dbfef235e2aa
function connection_map(E, q)
    return q
end

# ╔═╡ 0b1f7fe3-877a-4ad6-96ac-62fa5a6c67fe
function solve_linear_system(M, A, b, p, state)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	n = manifold_dimension(M)
	Ac = zeros(n,n);
	bc = zeros(n)
	e = enumerate(base)
	if state.is_same == true
		#println("Newton")
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
		#println("simplified Newton")
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
	#println(bc)
	Xc = (Ac) \ (bc)
	res_c = get_vector(M, p, Xc, B)
	#println("norm =", norm(res_c))
	#println(diag(diag_A))
	#println(cond(Ac))
	#println(Xc)
	return res_c
end

# ╔═╡ 2fa818b0-7ae6-4e7d-b1b2-4f43216508e9
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p, newtonstate)

# ╔═╡ 46693442-c086-4aea-bc78-39392929de33
begin
	Random.seed!(42)
	p = rand(power)
	#y_0 = [project(S, (discretized_y[i]+0.02*p[power,i])) for i in 1:N]
	y_0 = discretized_y
end;

# ╔═╡ e446c5c3-88e4-459b-8987-b587b254739a
st_res = vectorbundle_newton(power, TangentBundle(power), b, A, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(47)|StopWhenChangeLess(power,1e-14)),
	retraction_method=ProjectionRetraction(),
#stepsize=ConstantStepsize(1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)

# ╔═╡ af96b519-fdef-4824-a87b-570a9b46d4fe
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	fig_c, ax_c, plt_c = lines(1:length(change), log.(change))
	fig_c
end;

# ╔═╡ 4ba82b85-11d9-49b1-a0b0-b550ea41e26c
iterates = get_record(st_res, :Iteration, :Iterate)

# ╔═╡ 96566bcd-2805-4868-a783-965d08606bd5
begin
	f = Figure(;)

    row, col = fldmod1(1, 2)

	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ 52569103-8af5-441f-9c91-4b9cb44f7e5f
p_res = get_solver_result(st_res);

# ╔═╡ 170dec4e-9f8c-4710-9637-3c1a325d4db7
p_res[1]

# ╔═╡ 6838ab6e-4a73-4e42-84ce-a11e28602fd7
y0

# ╔═╡ 43c8df22-3b57-4916-9391-b19d80efc88c
p_res[N]

# ╔═╡ 54d68b41-9a6f-4c4c-bfde-7db41939d30c
yT

# ╔═╡ 33f91dbc-fed1-4b0a-9c1b-ba2a2f54ea64
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);

it_back = 0

ws = [-1.0*w(Manifolds.Sphere(2), p) for p in discretized_y]
ws_res = [-1.0*w(Manifolds.Sphere(2), p) for p in iterates[length(change)-it_back]]

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
	scatter!(ax, π1.(p_res), π2.(p_res), π3.(p_res); markersize =8, color=:orange)
	scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =8, color=:blue)
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	#arrows!(ax, π1.(y_0), π2.(y_0), π3.(y_0), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)

	#arrows!(ax, π1.(y_0), π2.(y_0), π3.(y_0), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	arrows!(ax, π1.(iterates[length(change)-it_back]), π2.(iterates[length(change)-it_back]), π3.(iterates[length(change)-it_back]), π1.(ws_res), π2.(ws_res), π3.(ws_res); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	#scatter!(ax, π1.([p_res[1], p_res[N]]), π2.([p_res[1], p_res[N]]), π3.([p_res[1], p_res[N]]); markersize =8, color=:red)
	#arrows!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]), π1.([w(Manifolds.Sphere,y0), w(Manifolds.Sphere,yT)]), π2.([w(Manifolds.Sphere,y0), w(Manifolds.Sphere,yT)]), π3.([w(Manifolds.Sphere,y0), w(Manifolds.Sphere,yT)]); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.1)
	#scatter!(ax, π1.(iterates[length(change)-it_back]), π2.(iterates[length(change)-it_back]), π3.(iterates[length(change)-it_back]); markersize =8, color=:orange)
	fig
end

# ╔═╡ e5247033-d496-449f-b83d-8b20f9f8cd45
# homotopy

# ╔═╡ 404fafb8-8514-4f6e-ab74-9ea047fb558f
begin
	for i in range(1000)
		C = C+i/1000.0

	end
end

# ╔═╡ Cell order:
# ╠═11863d90-6f62-11ef-06a1-611d44587188
# ╠═282efd2f-2019-4bb5-a9ff-0ba7a18b2b58
# ╠═344ec4fa-af2f-4f78-8de8-a7f698c4ea46
# ╠═a0b6475f-9fe0-4f7a-b36b-b7495e477bd1
# ╠═bfb534ea-43aa-4964-afab-5abaa0a024bf
# ╠═09932d3b-b604-424e-8c02-2f5498590098
# ╠═4ae5e86e-77a1-411e-a732-d0273bffea6f
# ╠═6e66d021-38f4-4037-8d63-c8cc75978485
# ╠═d887274f-e198-4af7-a897-a88fd94f04e2
# ╠═1d4ee847-f09d-4423-91ea-aa6d60f8d1b5
# ╠═13f350e2-ba3d-4652-9358-4bc2f08d9001
# ╠═e8abf32a-5923-48b0-b739-78e527d1a4f9
# ╠═4dcf87c7-8084-4b19-90da-53bca0d97124
# ╠═1b19151f-81dc-409c-90a7-dbfef235e2aa
# ╠═2fa818b0-7ae6-4e7d-b1b2-4f43216508e9
# ╠═0b1f7fe3-877a-4ad6-96ac-62fa5a6c67fe
# ╠═46693442-c086-4aea-bc78-39392929de33
# ╠═e446c5c3-88e4-459b-8987-b587b254739a
# ╠═af96b519-fdef-4824-a87b-570a9b46d4fe
# ╠═4ba82b85-11d9-49b1-a0b0-b550ea41e26c
# ╠═96566bcd-2805-4868-a783-965d08606bd5
# ╠═52569103-8af5-441f-9c91-4b9cb44f7e5f
# ╠═170dec4e-9f8c-4710-9637-3c1a325d4db7
# ╠═6838ab6e-4a73-4e42-84ce-a11e28602fd7
# ╠═43c8df22-3b57-4916-9391-b19d80efc88c
# ╠═54d68b41-9a6f-4c4c-bfde-7db41939d30c
# ╠═33f91dbc-fed1-4b0a-9c1b-ba2a2f54ea64
# ╠═e5247033-d496-449f-b83d-8b20f9f8cd45
# ╠═404fafb8-8514-4f6e-ab74-9ea047fb558f
