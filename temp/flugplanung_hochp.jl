### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 85e76846-912a-11ef-294a-c717389928e4
using Pkg; Pkg.activate();

# ╔═╡ 48950604-c2c2-4310-8de5-f89db905668b
begin
	using LinearAlgebra
	using SparseArrays
	using Manopt
	using Manifolds
	using ManoptExamples
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

# ╔═╡ bd50d1db-7c55-4279-abe0-89f2fe986cc6
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end

# ╔═╡ 449e9782-5bed-4642-a068-f8c9106bbe86
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
	#return [cos(halt+st - t), sin(halt+st - t), 0]
end;

# ╔═╡ 4ea7c8f7-80f9-482a-b535-229d2671fb4d
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ f08571fe-2e5d-417c-8330-9251233af25d
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


# ╔═╡ 1918a11e-88b9-43b9-b52d-026163580b5f
"""
 The following two routines define the vector transport and its derivative. The second is needed to obtain covariant derivative from the ordinary derivative.

I know: the first is already implemented, but this is just for demonstration purpose
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ d47accca-ebf3-4085-9d20-2e0ee6e88ef5
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ 28143183-eccb-4166-9f79-6baaa8f3f5c2
"""
Definition of a vector transport and its derivative given by the orthogonal projection
"""
transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)

# ╔═╡ be436d8c-dbc4-4d70-ae69-7e604ca76c83
ex = 50 # exponent used in the energy functional

# ╔═╡ a789119e-04b9-4456-acba-ec8e8702c231
"""
	Evaluates the wind field at a point p on the sphere (here: winding field scaled by the third component)
"""
	function w(M, p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end

# ╔═╡ a5811f04-59f8-4ea2-8616-efad7872830a
"""
	Returns the first derivative of the wind field at point p as a matrix
"""
function w_prime(M, p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ 0e5275a1-52d0-45b2-8d1c-5940b50afde8
"""
The following two routines define the integrand and its Euclidean derivative. They use a wind field w, its derivative and its second derivative, defined below. A scaling parameter is also employed.
"""
function F_at(Integrand, y, ydot, B, Bdot)
	return ex*(((ydot - w(Integrand.domain, y, Integrand.scaling))'*(ydot - w(Integrand.domain, y, Integrand.scaling)))^(ex/2.0 - 1))*(Bdot - w_prime(Integrand.domain,y,Integrand.scaling)*B)'*(ydot - w(Integrand.domain, y, Integrand.scaling))
end

# ╔═╡ 709daed1-7ef3-4fa5-b390-e18d27a7cefd
"""
	Returns the second derivative of the wind field at point p in direction v as a matrix
"""
function w_doubleprime(M, p, v, c)
	nenner = (p[1]^2+p[2]^2)
	w1 = 1/(nenner^2)*[(2*p[2]*p[3]*nenner-8*p[1]^2*p[2]*p[3])/nenner -p[3]*(2*p[1]*nenner^2-4*p[1]*(p[1]^4-p[2]^4))/nenner^2 2*p[1]*p[2]; (-2*p[1]*p[3]*nenner^2-4*p[1]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[2]*p[3]*nenner+8*p[1]^2*p[2]*p[3])/nenner (p[2]^2-p[1]^2); 0.0 0.0 0.0]
		
	w2 = 1/(nenner^2)*[(2*p[1]*p[3]*nenner-8*p[1]*p[2]^2*p[3])/nenner -p[3]*(-2*p[2]*nenner^2-4*p[2]*(p[1]^4-p[2]^4))/nenner^2 p[2]^2-p[1]^2; (2*p[2]*p[3]*nenner^2-4*p[2]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[1]*p[3]*nenner+8*p[1]*p[2]^2*p[3])/nenner -2*p[1]*p[2]; 0.0 0.0 0.0]
		
	w3 = 1/(nenner^2)*[2*p[1]*p[2] -p[1]^2+p[2]^2 0.0; p[2]^2-p[1]^2 -2*p[1]*p[2] 0.0; 0.0 0.0 0.0]
	return c*(v[1]*w1 + v[2]*w2 + v[3]*w3)
end

# ╔═╡ 8c726811-c87e-4924-9ebe-bc56e26855a9
function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	S = Integrand.domain
	constant = Integrand.scaling
	return ex*(ex-2)*(((ydot - w(S, y, constant))'*(ydot - w(S, y, constant)))^(ex/2.0 - 2.0))*((B2dot - w_prime(S, y, constant)*B2)'*(ydot - w(S, y, constant)))*((B1dot - w_prime(S, y, constant)*B1)'*(ydot - w(S, y, constant))) + ex * (((ydot - w(S, y, constant))'*(ydot - w(S, y, constant)))^(ex/2.0 - 1.0)) * ((-1.0*w_doubleprime(S, y, B2, constant)*B1)'*(ydot - w(S, y, constant)) + (B1dot - w_prime(S, y, constant)*B1)'*(B2dot - w_prime(S, y, constant)
	*B2))
end

# ╔═╡ 7f9448c8-0f0c-40de-861f-16427fd335ad
"""
	Definition of an integrand and its derivative for the simplified flight planning problem
"""
integrand=DifferentiableMapping(S,F_at,F_prime_at,1.0)

# ╔═╡ 5c13ce80-209c-4901-913a-3283339a11cc
""" 
	Dummy, necessary for calling vectorbundle_newton
"""
function bundlemap(M, y)
		return y
end

# ╔═╡ 86a649c8-43f8-43a0-a652-24735d28c05e
""" 
	Dummy, necessary for calling vectorbundle_newton
"""
function connection_map(E, q)
    return q
end

# ╔═╡ fbf47501-4a41-4f89-940c-9c68aca26b4b
"""
	Method for solving the Newton equation 
		* assembling the Newton matrix and the right hand side (using ManoptExamples.jl)
		* using a direct solver for computing the solution in base representation
	Returns the Newton direction in vector representation
"""
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
	println("Solve:")
	@time Xc = (Ac) \ (-bcsys)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	res_c = get_vector(M, p, Xc, B)
	return res_c
end

# ╔═╡ b2d6b477-28d1-421c-958e-ebcfe845a6bb
"""
	Set the solve method for solving the Newton equation in each step
"""
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

# ╔═╡ 25cdf23f-c2ff-44be-9a34-bb565c36775e
"""
	Initial geodesic
"""
y_0 = copy(power, discretized_y)

# ╔═╡ dcf6174d-0b6f-48da-b0d8-f057b2992e0b
st_res = vectorbundle_newton(power, TangentBundle(power), bundlemap, bundlemap, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
stepsize=ConstantLength(1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)
# Dass hier zweimal bundlemap übergeben werden muss, sollte noch raus. Das sind in dem Fall ja eh nur Dummies. Normalerweise sollten da F und F' übergeben werden (in welcher Art auch immer)

# ╔═╡ 40e63e0a-fa25-4d9d-b685-383a7957c513
change = get_record(st_res, :Iteration, :Change)[2:end];

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
		s = 0
		c = integrand.scaling
		for i in 1:N
			y_i = Oy[power, i]
			y_next = Oy[power, i+1]
			normen[i] = norm(S, y_i, ((y_next-y_i)/h - w(S, y_next, c)))
			s += + normen[i]
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
# ╠═4ea7c8f7-80f9-482a-b535-229d2671fb4d
# ╠═f08571fe-2e5d-417c-8330-9251233af25d
# ╠═1918a11e-88b9-43b9-b52d-026163580b5f
# ╠═d47accca-ebf3-4085-9d20-2e0ee6e88ef5
# ╠═28143183-eccb-4166-9f79-6baaa8f3f5c2
# ╠═be436d8c-dbc4-4d70-ae69-7e604ca76c83
# ╠═0e5275a1-52d0-45b2-8d1c-5940b50afde8
# ╠═8c726811-c87e-4924-9ebe-bc56e26855a9
# ╠═7f9448c8-0f0c-40de-861f-16427fd335ad
# ╠═a789119e-04b9-4456-acba-ec8e8702c231
# ╠═a5811f04-59f8-4ea2-8616-efad7872830a
# ╠═709daed1-7ef3-4fa5-b390-e18d27a7cefd
# ╠═5c13ce80-209c-4901-913a-3283339a11cc
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
