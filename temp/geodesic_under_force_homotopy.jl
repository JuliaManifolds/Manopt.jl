### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 0783b732-8574-11ef-017d-3939cfc57442
using Pkg; Pkg.activate();

# ╔═╡ b7f09653-9692-4f92-98e3-f988ed0c3d2d
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays, RecursiveArrayTools
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
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

# ╔═╡ fb6a6239-d154-4d4b-ac70-16777dc54726
begin
	export_video = false
	video_file = "geodesic_under_force_animation.mp4"
end

# ╔═╡ 7b3e1aa5-db29-4519-9860-09f6cc933c07
begin
	N=30
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

# ╔═╡ 56dce4f9-83a9-4a50-8b91-007e4ddfeacc
function proj_prime(S, p, X, Y) # S_i*(Y)
	#return project(S, p, (- X*p' - p*X')*Y)
	return (- X*p' - p*X')*Y
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
	bc = zeros(n)
	e = enumerate(base)
	if state.is_same == true
		#println("Newton")
   		for (i,basis_vector) in e
      	G = A(M, p, basis_vector, obj.scaling)
	  	#Ac[:,i] = get_coordinates(M, p, G, B)
		Ac[i,:] = get_coordinates(M, p, G, B)'
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
	Xc = (Ac) \ (bc)
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

# ╔═╡ 04853afe-7032-43ef-b7b9-9836ee144073
begin
	π1(x) = 1.02*x[1]
    π2(x) = 1.02*x[2]
    π3(x) = 1.02*x[3]
end;

# ╔═╡ 0c9163c1-e8a3-408c-9ffe-550923a8c4e2
function empty_sphere_plot(n)
	u = range(0,stop=2*π,length=n);
	v = range(0,stop=π,length=n);
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
	return fig, ax, plt
end
	

# ╔═╡ 0cadffa2-dc8e-432e-b198-2e519e128576
begin
	it_back = 0
#ws = [-1.0*w(Manifolds.Sphere(2), p) for p in discretized_y]
#ws_res = [-1.0*w(Manifolds.Sphere(2), p) for p in iterates[length(change)-it_back]]
	E = TangentBundle(power)
	obj = VectorbundleObjective(b, A, connection_map)
	obj.scaling = 1.0
	problem = VectorbundleManoptProblem(power, E, obj)
	add = 1/10.0

	y_results = []
	for i in range(1,50)
		obj.scaling = obj.scaling + add
		println(obj.scaling)
		state = VectorbundleNewtonState(power, E, bundlemap, y_0, solve, AllocatingEvaluation(), stopping_criterion=(StopAfterIteration(20)|StopWhenChangeLess(power,1e-14)), retraction_method=ProjectionRetraction(), stepsize=Manopt.ConstantStepsize(power,1.0)) #stepsize= now always needs the manifold first if you use the “old” ones. They are also no longer exported.
		st_res = solve!(problem, state)
		if Manopt.indicates_convergence(st_res.stop) == true
			push!(y_results, get_solver_result(st_res))
		else
			add = add*0.5
		end
		println(Manopt.indicates_convergence(st_res.stop))
		println(Manopt.get_reason(st_res))
		#push!(y_results, get_solver_result(st_res))
	end
end

# ╔═╡ f9862529-3af6-4c38-b6f0-01bda45e97c3
begin
	n = 45
	fig, ax, plt = empty_sphere_plot(n)
	scatter!(ax, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y); markersize =8, color=:blue)
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	for (i,y) in enumerate(y_results)
		scatter!(ax, π1.(y), π2.(y), π3.(y); markersize =8, color=i, colorrange=1:length(y_results))
	end
	fig
end

# ╔═╡ 3439120b-bf39-4a26-a5fb-cc4eebb13550
begin
	# In principle this should work but there seems to be a problem on my machine
	# with WGLMakie and saving the video.
	fig_r, ax_r, plt_r = empty_sphere_plot(n)
	plot_data_x = Observable(π1.(first(y_results)))
	plot_data_y = Observable(π2.(first(y_results)))
	plot_data_z = Observable(π3.(first(y_results)))
	color_index = Observable(1)
	scatter!(ax_r, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y); markersize =8, color=:blue)
	scatter!(ax_r, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	scatter!(ax_r, plot_data_x, plot_data_y, plot_data_z; markersize =8, color=color_index, colorrange=1:length(y_results))
	#= #Does not yet work, can't get it to work to record for now
	ecord(fig_r, video_file, 1:length(y_results), framerate = 10) do i
        plot_data_x[] = π1.(y_results[i])
        plot_data_y[] = π2.(y_results[i])
        plot_data_z[] = π3.(y_results[i])
		color_index[] = i
		yield()
    end
	=#
end;

# ╔═╡ Cell order:
# ╠═0783b732-8574-11ef-017d-3939cfc57442
# ╠═b7f09653-9692-4f92-98e3-f988ed0c3d2d
# ╠═fb6a6239-d154-4d4b-ac70-16777dc54726
# ╠═1c476b4a-3ee6-4e5b-b903-abfc4d557569
# ╠═7b3e1aa5-db29-4519-9860-09f6cc933c07
# ╠═aa325d08-1990-4ef3-8205-78be6d06c711
# ╠═ccf9e32c-0efd-4520-85a7-3cfb78ce9e15
# ╠═632bb19d-02dd-4d03-bd92-e2222b26271f
# ╠═7b287c39-038a-4a02-b571-6cb4ee7f68d0
# ╠═b59b848a-859e-4201-8f02-67e806a91551
# ╠═56dce4f9-83a9-4a50-8b91-007e4ddfeacc
# ╠═483b9dc4-ff39-4c4d-86c9-ac7643752fca
# ╠═05c7e6fe-5335-41d5-ad31-d8ff8fe354c0
# ╠═62bf2114-1551-4467-9d48-d2a3a3b8aa8e
# ╠═48cd163d-42d1-4783-ace6-629d1ea495d4
# ╠═48e8395e-df79-4600-bcf9-50e318c49d58
# ╠═0d741410-f182-4f5b-abe4-7719e627e2dc
# ╠═00e47eab-e088-4b55-9798-8b9f28a6efe5
# ╠═04853afe-7032-43ef-b7b9-9836ee144073
# ╠═0c9163c1-e8a3-408c-9ffe-550923a8c4e2
# ╠═0cadffa2-dc8e-432e-b198-2e519e128576
# ╠═f9862529-3af6-4c38-b6f0-01bda45e97c3
# ╠═3439120b-bf39-4a26-a5fb-cc4eebb13550
