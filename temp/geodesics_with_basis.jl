### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 05150bda-555a-11ef-02cc-fd6f8ee616be
using Pkg; Pkg.activate();

# ╔═╡ 04a41fef-6fac-40b9-8923-220742eb77ac
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ 5feabae6-d646-414d-8184-3887d990bea8
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

# ╔═╡ 50d33c0d-859b-4ab3-bba6-0e3a2046a2ae
begin
	N=200
	h = 1/(N+2)*π/2
	Omega = range(; start=0.0, stop = π/2, length=N+2)[2:end-1]
	y0 = [0,0,1] # startpoint of geodesic
	yT = [1,0,0] # endpoint of geodesic
end;

# ╔═╡ fd955a44-2e0e-4f6a-9a59-8b91e69001aa
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end;

# ╔═╡ 98507869-f90b-4149-bbb4-ab59eb1597da
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 186a0713-c773-4489-8a23-4f274c1add7c
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ 507e2957-3f61-407c-b272-dac9df00eb0f
begin
	# force
	function f(M, p)
		return project(M, p, [0.0, -3.0, 0.0])
	end
end;

# ╔═╡ 359b1c09-77f0-4c88-9b04-b60fd863d81a
function A(M, y, X)
	# Include boundary points
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	Ay = zero_vector(M, y)
	for i in 1:N
		y_i = Oy[M, i]
		y_next = Oy[M, i+1]
		y_pre = Oy[M, i-1]
		E = Diagonal([1/h * (2+((-y_pre+2*y_i-y_next)'*([ - (i==j ? 2.0 : 1.0) * y_i[j] for i=1:3]))) - h * f(S, y_i)' * ([ - (k==j ? 2.0 : 1.0) * y_i[j] for k=1:3]) for j in 1:3])

		if i == 1
			Ay[M, i] = E*X[M, i] - 1/h * X[M, i+1]
		elseif i == N
			Ay[M, i] = - 1/h*X[M, i-1] + E*X[M, i]
		else
			Ay[M, i] = - 1/h*X[M, i-1]+ E*X[M, i] - 1/h*X[M, i+1]
		end
	end
	return Ay
end

# ╔═╡ eac9f22e-6a5e-4bda-9460-85e0cdd17b2c
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			y_i = Oy[M, i]
			y_next = Oy[M, i+1]
			y_pre = Oy[M, i-1]
			X[M, i] = 1/h * (2*y_i - y_pre - y_next) .- h * f(M.manifold, y_i)
		end
		return X
end

# ╔═╡ 8ab155bc-24c1-4499-a7d1-b9206af67629
function connection_map(E, q)
    return q
end

# ╔═╡ 0f5b255b-9b6e-43d6-883d-3f1759ea2f59
function solve_linear_system(M, A, b, p)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	n = manifold_dimension(M)
	Ac = zeros(n,n);
	bc = zeros(n)
    for (i,basis_vector) in enumerate(base)
      G = A(M, p, basis_vector)
	  Ac[:,i] = get_coordinates(M, p, G, B)
	  # Ac[i,:] = get_coordinates(M, p, G, B)'
      bc[i] = b(M, p)'*basis_vector
	end
	# bc = get_coordinates(M, p, b(M, p), B)
	#diag_A = Diagonal([abs(Ac[i,i]) < 1e-12 ? 1.0 : 1.0/Ac[i,i] for i in 1:n])
	Xc = (Ac) \ (-bc)
	res_c = get_vector(M, p, Xc, B)
	#println(diag(diag_A))
	#println(cond(Ac))
	#println(Ac)
	#println(bc)
	#println(Xc)
	return res_c
end

# ╔═╡ eacbdcbc-ee01-41b7-8289-85ef4c07cc2c
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p)

# ╔═╡ 5371c088-e0d6-4508-a1cd-1fefad8c333b
begin
	Random.seed!(42)
	p = rand(power)
	#y_0 = [project(S, (discretized_y[i]+0.01*p[power,i])) for i in 1:N]
	y_0 = discretized_y
end;

# ╔═╡ 2121ac73-bc7e-4084-a29f-dc9abd3a298d
st_res = vectorbundle_newton(power, TangentBundle(power), b, A, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(20),
	stepsize=ConstantStepsize(1.0),
	#retraction_method=ProjectionRetraction(),
	debug=[:Iteration, (:Change, "Change: %1.8e")," | ", :Stepsize, 1, "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)

# ╔═╡ b7a906ca-15c9-405d-9031-b797d4527f37
iterates = get_record(st_res, :Iteration, :Iterate)

# ╔═╡ f95ae6ff-a967-4600-99c4-4ba73c88b6d2
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	fig_c, ax_c, plt_c = lines(1:length(change), log.(change))
	fig_c
end

# ╔═╡ 214fe41a-c6d9-43d8-a00f-519af35214c9
p_res = get_solver_result(st_res);

# ╔═╡ 56a7fe1a-a4fd-415a-80ea-3b75deee1c13
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
fig, ax, plt = meshscatter(
  sx,sy,sz,
  color = fill(RGBA(1.,1.,1.,0.75), n, n),
  shading = Makie.automatic,
  transparency=true
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

# ╔═╡ Cell order:
# ╠═05150bda-555a-11ef-02cc-fd6f8ee616be
# ╠═04a41fef-6fac-40b9-8923-220742eb77ac
# ╠═5feabae6-d646-414d-8184-3887d990bea8
# ╠═50d33c0d-859b-4ab3-bba6-0e3a2046a2ae
# ╠═fd955a44-2e0e-4f6a-9a59-8b91e69001aa
# ╠═98507869-f90b-4149-bbb4-ab59eb1597da
# ╠═186a0713-c773-4489-8a23-4f274c1add7c
# ╠═507e2957-3f61-407c-b272-dac9df00eb0f
# ╠═359b1c09-77f0-4c88-9b04-b60fd863d81a
# ╠═eac9f22e-6a5e-4bda-9460-85e0cdd17b2c
# ╠═8ab155bc-24c1-4499-a7d1-b9206af67629
# ╠═eacbdcbc-ee01-41b7-8289-85ef4c07cc2c
# ╠═0f5b255b-9b6e-43d6-883d-3f1759ea2f59
# ╠═5371c088-e0d6-4508-a1cd-1fefad8c333b
# ╠═2121ac73-bc7e-4084-a29f-dc9abd3a298d
# ╠═b7a906ca-15c9-405d-9031-b797d4527f37
# ╠═f95ae6ff-a967-4600-99c4-4ba73c88b6d2
# ╠═214fe41a-c6d9-43d8-a00f-519af35214c9
# ╠═56a7fe1a-a4fd-415a-80ea-3b75deee1c13
