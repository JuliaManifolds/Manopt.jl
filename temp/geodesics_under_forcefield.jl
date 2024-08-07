### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ b7726008-53f6-11ef-216f-c1984c3e1e7b
using Pkg; Pkg.activate();

# ╔═╡ 64eb4ec9-2b11-42ed-987a-f72066adefe1
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ 668bf47a-5ea8-4fbb-beb4-087569f024c5
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

# ╔═╡ caf81526-dfb2-438e-99d2-03c6b60405af
begin
	N=300
	h = 1/(N+2)*π/2
	Omega = range(; start=0.0, stop = π/2, length=N+2)[2:end-1]
	y0 = ArrayPartition([0,0,1], [0,]) # startpoint of geodesic
	yT = ArrayPartition([1,0,0], [0,]) # endpoint of geodesic
end;

# ╔═╡ 5e585c93-44c4-48a9-a66d-d21f32ced5fd
begin
    import Manopt.get_submersion
    function get_submersion(M::Manifolds.Sphere, p)
        return p'p - 1
    end
end

# ╔═╡ 294a2d81-bff8-48f0-8cb4-9f02ad8ae9b8
begin
    import Manopt.get_submersion_derivative
    function get_submersion_derivative(M::Manifolds.Sphere, p)
        return p'
    end
end

# ╔═╡ 42c54278-2ae7-4f75-b391-9011d2154dad
M3 = PowerManifold(Euclidean(4), NestedPowerRepresentation(), N);

# ╔═╡ 89d46c32-3b24-4bf9-aa94-b5afc4b4f5bd
begin
S1 = Manifolds.Sphere(2) × ℝ^1
power = PowerManifold(S1, NestedPowerRepresentation(), N);

# zugriff auf y mit y[power, i][S, 1] für punkt auf sphäre
end;

# ╔═╡ c4664660-037f-4711-8d08-24f4c404979c
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 15611bda-f70a-4df7-ba35-2cda58870f75
discretized_ylambda = [ArrayPartition(y(Ωi),[1,]) for Ωi in Omega];

# ╔═╡ d47d15a4-022b-44d2-9df2-3a7d733b1555
discretized_ylambda[power, 3][S1, 2]

# ╔═╡ 7e6db5df-7f76-4235-94e9-e7561b0c3e06
begin
	#Random.seed!(4)
	#f = 10*rand(TangentBundle(M3))
	#f = [[0.0, 1.0, 0.0] for _ in 1:N]
	#f = [1/norm(0.1*p[TangentBundle(M3),i])*(+0.1*p[TangentBundle(M3),i]) for i in 1:N]
	function f(M, p)
		return project(M, p, [0.0, 2.0, 0.0])
		#return [0.0, 1.0, 0.0]
	end
end;

# ╔═╡ 93a41283-3a15-465a-99f4-a6fb54a8b575
S = Manifolds.Sphere(2)

# ╔═╡ 572cddcc-bc6a-4c92-b649-b109c7233f00
function A(M, y, X)
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	Ay = zero_vector(M, y)
	P = M.manifold
	#C = -1/h*Diagonal([ones(3)..., 0])
	for i in 1:N
#		E = Diagonal([1/h * (2+(-Oy[j-1][1:3]+2*Oy[j][1:3]-Oy[j+1][1:3])'*(-Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][j] - Oy[i][1:3])) for j in 1:3])
		y_i = Oy[M, i][M.manifold, 1]
		y_next = Oy[M, i][M.manifold, 1]
		y_pre = Oy[M, i][M.manifold, 1]
		E = Diagonal([1/h * (2+(-y_pre+2*y_i-y_next)'*([ - (i==j ? 2.0 : 1.0) * y_i[j] for i=1:3])) - 0.5 * (project(S, y_i, -f(S, y_i)+2*f(S, y_i)-f(S,y_next)))'*([ - (i==j ? 2.0 : 1.0) * y_i[j] for i=1:3]) for j in 1:3])
		#- h * f(Manifolds.Sphere(2), Oy[i][1:3])' * ([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3])
		#E = Diagonal([1/h * (2+(-Oy[i-1][1:3]+2*Oy[i][1:3]-Oy[i+1][1:3])'*([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3])) - 0.5 * (project(S, Oy[i][1:3], -f(S, Oy[i-1][1:3])-f(S,Oy[i+1][1:3])))'*([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3]) for j in 1:3])
		#E = vcat(E, y_i')
		#E = hcat(E, [y_i...,0])
		if i == 1
			Ay[M, i][P, 1] = E*X[M, i][P, 1] + get_submersion_derivative(P[1], y_i)' * only(X[M, i][P, 2]) - 1/h*X[M, i+1][P,1]
			Ay[M, i][P, 2] = get_submersion_derivative(P[1], y_i) * X[M, i][P, 1]
		elseif i == N
			Ay[M, i][M.manifold, 1] = - 1/h*X[M, i-1][M.manifold,1] + E*X[M, i][M.manifold, 1] + get_submersion_derivative(M.manifold[1], y_i)' * only(X[M, i][M.manifold, 2])
			Ay[M, i][M.manifold, 2] = get_submersion_derivative(M.manifold[1], y_i) * X[M, i][M.manifold, 1]
		else
			Ay[M, i][M.manifold, 1] = - 1/h*X[M, i-1][M.manifold,1] + E*X[M, i][M.manifold, 1] + get_submersion_derivative(M.manifold[1], y_i)' * only(X[M, i][M.manifold, 2]) - 1/h*X[M, i+1][M.manifold,1]
			Ay[M, i][M.manifold, 2] = get_submersion_derivative(M.manifold[1], y_i) * X[M, i][M.manifold, 1]
		end
	end
	return Ay
end

# ╔═╡ 2d418421-2558-418d-a260-225836dad049
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i][M.manifold, 1]
			y_next = Oy[M, i][M.manifold, 1]
			y_pre = Oy[M, i][M.manifold, 1]
			c = [ 1/h * (2*y_i - y_pre - y_next)'* Matrix{Float64}(I, 3, 3)[1:3,j] + y_i'*Matrix{Float64}(I, 3, 3)[1:3,j]*only(Oy[M, i][M.manifold,2]) - 0.5 * (project(S, y_i, -f(S, y_pre)+2*f(S, y_i)-f(S,y_next)))[j]
			#- h * f(Manifolds.Sphere(2), Oy[i][1:3])[j]
			for j in 1:3]
			#c = [ 1/h * (2*Oy[i][1:3] - Oy[i-1][1:3] - Oy[i+1][1:3])'* Matrix{Float64}(I, 3, 3)[1:3,j] + Oy[i][1:3]'*Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][4] - 0.5 * (project(S, Oy[i][1:3], -f(S, Oy[i-1][1:3])-f(S,Oy[i+1][1:3])))[j] for j in 1:3]
			X[M, i][M.manifold, 1] = c
			X[M, i][M.manifold, 2] = [0,]
		end
		return X
end

# ╔═╡ 0828cdb7-1a18-4059-81c6-dd7509d7de4e
function connection_map(E, q)
    return q
end

# ╔═╡ a99c6ea8-203d-4440-ac1e-2ce9c342f3e4
function solve_linear_system(M, A, b, p)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	Ac = zeros(manifold_dimension(M),manifold_dimension(M));
    for (i,basis_vector) in enumerate(base)
      G = A(M, p, basis_vector)
	  Ac[:,i] = get_coordinates(M, p, G, B)
	end
	bc = get_coordinates(M, p, b(M, p), B)
	Xc = Ac \ (-bc)
	res_c = get_vector(M, p, Xc, B)
	return res_c
end

# ╔═╡ c6c8216f-7f71-4f04-a766-edadd38000fa
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p)

# ╔═╡ be86da09-ddc7-420c-8571-ee73387b3fef
begin
	Random.seed!(42)
	p = rand(M3)
	#y_0 = [[project(S, (discretized_ylambda[i][1:3]+0.01*p[M3,i][1:3]))..., discretized_ylambda[i][4]] for i in 1:N]
	y_0 = discretized_ylambda
end;

# ╔═╡ 98100077-a040-4b7e-b832-db49913382a6
is_point(S, y_0[1][1:3])

# ╔═╡ 5596b132-af0f-4896-8617-bbe1d7c9bf41
p_res = vectorbundle_newton(power, TangentBundle(power), b, A, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(10),
	#retraction_method=ProjectionRetraction(),
	debug=[:Iteration, (:Change, "Change: %1.8e")," | ", :Stepsize, 1, "\n", :Stop]
)

# ╔═╡ b083f8e4-0dff-43d5-8bce-0f4dd85d9569
discretized_ylambda

# ╔═╡ ea267cbc-0ad4-4fb8-b378-eff8be2d7c3f
begin
n = 45
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
fig, ax, plt = surface(
  sx,sy,sz,
  color = fill(RGBA(1.,1.,1.,0.3), n, n),
  shading = Makie.automatic
)
ax.show_axis = false
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.3))
    π1(x) = 1.02*x[1]
    π2(x) = 1.02*x[2]
    π3(x) = 1.02*x[3]
	scatter!(ax, π1.(p_res), π2.(p_res), π3.(p_res); markersize =4)
	scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =3, color=:blue)
	fig
end

# ╔═╡ 9e5c02e0-15d2-4c3e-b70d-70fd9b20f65c
# ╠═╡ disabled = true
#=╠═╡
is_point(Manifolds.Sphere(2), p_res[150][1:3]; error=:info)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═b7726008-53f6-11ef-216f-c1984c3e1e7b
# ╠═64eb4ec9-2b11-42ed-987a-f72066adefe1
# ╠═668bf47a-5ea8-4fbb-beb4-087569f024c5
# ╠═caf81526-dfb2-438e-99d2-03c6b60405af
# ╠═5e585c93-44c4-48a9-a66d-d21f32ced5fd
# ╠═294a2d81-bff8-48f0-8cb4-9f02ad8ae9b8
# ╠═42c54278-2ae7-4f75-b391-9011d2154dad
# ╠═89d46c32-3b24-4bf9-aa94-b5afc4b4f5bd
# ╠═c4664660-037f-4711-8d08-24f4c404979c
# ╠═15611bda-f70a-4df7-ba35-2cda58870f75
# ╠═d47d15a4-022b-44d2-9df2-3a7d733b1555
# ╠═7e6db5df-7f76-4235-94e9-e7561b0c3e06
# ╠═93a41283-3a15-465a-99f4-a6fb54a8b575
# ╠═572cddcc-bc6a-4c92-b649-b109c7233f00
# ╠═2d418421-2558-418d-a260-225836dad049
# ╠═0828cdb7-1a18-4059-81c6-dd7509d7de4e
# ╠═c6c8216f-7f71-4f04-a766-edadd38000fa
# ╠═a99c6ea8-203d-4440-ac1e-2ce9c342f3e4
# ╠═be86da09-ddc7-420c-8571-ee73387b3fef
# ╠═98100077-a040-4b7e-b832-db49913382a6
# ╠═5596b132-af0f-4896-8617-bbe1d7c9bf41
# ╠═b083f8e4-0dff-43d5-8bce-0f4dd85d9569
# ╠═ea267cbc-0ad4-4fb8-b378-eff8be2d7c3f
# ╠═9e5c02e0-15d2-4c3e-b70d-70fd9b20f65c
