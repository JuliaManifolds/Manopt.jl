### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 0783b732-8574-11ef-017d-3939cfc57442
using Pkg; Pkg.activate();

# ╔═╡ b7f09653-9692-4f92-98e3-f988ed0c3d2d
begin
	using LinearAlgebra
	using SparseArrays
	using Manopt
	using ManoptExamples
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
	N=100
	st = 0.5
	halt = pi-0.5
	h = (halt-st)/(N+1)
	#halt = pi - st
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y0 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT = [sin(halt),0,cos(halt)] # endpoint of geodesic

	#yT = [sin(st),0,cos(st)] # startpoint of geodesic: suedpol
	#y0 = [sin(halt),0,cos(halt)] # endpoint of geodesic: nordpol

	#y0 = [cos(st),sin(st),0] # startpoint of geodesic: aequator
	#yT = [cos(halt),sin(halt),0] # endpoint of geodesic: aequator
end;

# ╔═╡ ccf9e32c-0efd-4520-85a7-3cfb78ce9e15
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
end;

# ╔═╡ 632bb19d-02dd-4d03-bd92-e2222b26271f
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ f65e7b22-8d32-4d98-9b68-7ad6791c77ee
"""
Such a structure has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""
mutable struct DifferentiableMapping{M<:AbstractManifold,F1<:Function,F2<:Function,T}
	domain::M
	value::F1
	derivative::F2
	scaling::T
end


# ╔═╡ 03253f57-72a1-499a-ab98-ad319def233c
"""
 The following two routines define the vector transport and its derivative. The second is needed to obtain covariant derivative from the ordinary derivative.

I know: the first is already implemented, but this is just for demonstration purpose
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ aa981466-5658-41b1-b07c-cc9de0c60729
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ 764987fc-b909-47c6-a3fb-fa33865f838d
"""
The following two routines define the integrand and its ordinary derivative. They use a vector field w, wich is defined, below. A scaling parameter is also employed.
"""
function F_at(Integrand, y, ydot, B, Bdot)
	  return ydot'*Bdot+w(y,Integrand.scaling)'*B
end

# ╔═╡ c16c6bf0-16bd-4863-a3e3-a9f014711222
function w(p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end

# ╔═╡ 7c6fd969-dabc-4901-a430-d9b6a22bee24
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ 7f79b037-a17e-4886-94b3-286e73ac2bbb
function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return B1dot'*B2dot+(w_prime(y,Integrand.scaling)*B1)'*B2
end

# ╔═╡ aa325d08-1990-4ef3-8205-78be6d06c711
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
integrand=DifferentiableMapping(S,F_at,F_prime_at,10.0)
transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)
end;

# ╔═╡ 62bf2114-1551-4467-9d48-d2a3a3b8aa8e
"""
Dummy
"""
function bundlemap(M, y)
		# Include boundary points
end

# ╔═╡ 48cd163d-42d1-4783-ace6-629d1ea495d4
"""
Dummy
"""
function connection_map(E, q)
    return q
end

# ╔═╡ 0d741410-f182-4f5b-abe4-7719e627e2dc
function solve_linear_system(M, p, state, prob)
	obj = get_objective(prob)
	n = manifold_dimension(M)
	Ac::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	bc = zeros(n)
	bcsys=zeros(n)
	bctrial=zeros(n)
	Oy = OffsetArray([y0, p..., yT], 0:(length(Omega)+1))
	Oytrial = OffsetArray([y0, state.p_trial..., yT], 0:(length(Omega)+1))
	S = M.manifold
	println("Assemble:")
    @time ManoptExamples.get_rhs_Jac!(bc,Ac,h,Oy,integrand,transport)
	if state.is_same == true
		bcsys=bc
	else
		@time ManoptExamples.get_rhs_simplified!(bctrial,h,Oy,Oytrial,integrand,transport)
    	bcsys=bctrial-(1.0 - state.stepsize.alpha)*bc
	end
	#Asparse = sparse(Ac)
	println("Solve:")
	@time Xc = (Ac) \ (-bcsys)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	res_c = get_vector(M, p, Xc, B)
	return res_c
end

# ╔═╡ 48e8395e-df79-4600-bcf9-50e318c49d58
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

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
	#scatter!(ax, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y); markersize =8, color=:blue)
	#scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	E = TangentBundle(power)
	obj = VectorbundleObjective(connection_map, connection_map, connection_map)
	#integrand.scaling = 1.0
	problem = VectorbundleManoptProblem(power, E, obj)
	increment = 0.1
    y_start = copy(power,discretized_y)
    y_current = copy(power,y_start)
    y_last = copy(power,y_start)
	
	scatter!(ax, π1.(y_start), π2.(y_start), π3.(y_start); markersize =8, color=:blue)
	for i in range(1,1)
		#println(integrand.scaling)
		copyto!(power,y_last,y_current)
		state = VectorbundleNewtonState(power, E, bundlemap, y_current, solve, AllocatingEvaluation(), stopping_criterion=(StopAfterIteration(50)|StopWhenChangeLess(power, 1e-8)), retraction_method=ProjectionRetraction(), stepsize=Manopt.ConstantStepsize(power,1.0))
		#stepsize=Manopt.AffineCovariantStepsize(power))
		#retraction_method=ProjectionRetraction(), stepsize=ConstantStepsize(1.0))
		st_res = solve!(problem, state)
		println("Norm:", norm(y_last-y_current))
		if Manopt.indicates_convergence(st_res.stop)
			#integrand.scaling = integrand.scaling + increment
			scatter!(ax, π1.(y_current), π2.(y_current), π3.(y_current); markersize =8, color=:orange)
			ws_res = [-1.0*w(p, integrand.scaling) for p in y_current]
			arrows!(ax, π1.(y_current), π2.(y_current), π3.(y_current), π1.(ws_res), π2.(ws_res), π3.(ws_res); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
		else
			factor=0.5
			#integrand.scaling = integrand.scaling - increment
			global increment=increment*factor
			#integrand.scaling = integrand.scaling +increment
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
# ╠═f65e7b22-8d32-4d98-9b68-7ad6791c77ee
# ╠═03253f57-72a1-499a-ab98-ad319def233c
# ╠═aa981466-5658-41b1-b07c-cc9de0c60729
# ╠═764987fc-b909-47c6-a3fb-fa33865f838d
# ╠═7f79b037-a17e-4886-94b3-286e73ac2bbb
# ╠═c16c6bf0-16bd-4863-a3e3-a9f014711222
# ╠═7c6fd969-dabc-4901-a430-d9b6a22bee24
# ╠═62bf2114-1551-4467-9d48-d2a3a3b8aa8e
# ╠═48cd163d-42d1-4783-ace6-629d1ea495d4
# ╠═48e8395e-df79-4600-bcf9-50e318c49d58
# ╠═0d741410-f182-4f5b-abe4-7719e627e2dc
# ╠═00e47eab-e088-4b55-9798-8b9f28a6efe5
# ╠═0cadffa2-dc8e-432e-b198-2e519e128576
