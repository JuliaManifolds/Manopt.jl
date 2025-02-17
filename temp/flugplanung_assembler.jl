### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 78895b59-4591-485c-a50d-11925f734921
using Pkg; Pkg.activate();

# ╔═╡ 293d0824-906e-11ef-2b7f-834616d0ede2
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

# ╔═╡ acc549ff-feac-4236-b4c6-743328391275
begin
	N=100
	st = 0.5
	#halt = pi-0.5
	halt = pi/2
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

# ╔═╡ 5b4ced2b-66bc-4b23-8e19-b5eedadd93d7
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
end

# ╔═╡ 708ad6f4-f8a2-437c-b75e-52031d116e06
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
	#return [cos(halt+st - t), sin(halt+st - t), 0]
end;

# ╔═╡ f0dc31e1-3b38-4b98-b649-9c626eb3e620
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ 323746d0-4962-4a7c-8db2-9c6f67f6cc96
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


# ╔═╡ abbeebcc-5cb9-424a-9c65-aa8f7e67d549
"""
 The following two routines define the vector transport and its derivative. The second is needed to obtain covariant derivative from the ordinary derivative.

I know: the first is already implemented, but this is just for demonstration purpose
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ c8279ae4-1ffd-4bde-b0b5-779410443e1e
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ dbbfc102-8694-4417-81bc-eb8a4b15e0f7
"""
Definition of a vector transport and its derivative given by the orthogonal projection
"""
transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)

# ╔═╡ ff83e7f9-cb8a-418b-a388-6c997b6a4396
"""
	Evaluates the wind field at a point p on the sphere (here: winding field scaled by the third component)
"""
	function w(M, p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end

# ╔═╡ f8df1a83-a0f6-40d8-b731-dded209b4e91
"""
	Returns the first derivative of the wind field at point p as a matrix
"""
function w_prime(M, p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ d3814468-af4d-46f4-b669-b111cc2ea27e
"""
The following two routines define the integrand and its Euclidean derivative. They use a wind field w, its derivative and its second derivative, defined below. A scaling parameter is also employed.
"""
function F_at(Integrand, y, ydot, B, Bdot)
	  #return ydot'*Bdot+w(S,y0,constant)'*B
	return (ydot - w(Integrand.domain, y, Integrand.scaling))'*(Bdot - w_prime(Integrand.domain, y, Integrand.scaling)*B)
end

# ╔═╡ 530abbb0-e541-4787-a49a-eb7366724c33
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

# ╔═╡ 2384b532-e93f-415f-857a-4b0b7662a44d
function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	#return B1dot'*B2dot+(w_primealt(S,y0,constant)*B1)'*B2
	return (-w_doubleprime(Integrand.domain, y, B1, Integrand.scaling)*B2)'*(ydot - w(Integrand.domain, y, Integrand.scaling)) + (B1dot - w_prime(Integrand.domain, y, Integrand.scaling)*B1)'*(B2dot - w_prime(Integrand.domain, y, Integrand.scaling)
	*B2)
end

# ╔═╡ 96830f5a-f767-43d1-855f-46fe3fccc825
"""
	Definition of an integrand and its derivative for the simplified flight planning problem
"""
integrand=DifferentiableMapping(S,F_at,F_prime_at,2.0)

# ╔═╡ 212e3bb5-7bb9-4828-9a89-a24411e802ef
""" 
	Dummy, necessary for calling vectorbundle_newton
"""
function bundlemap(M, y)
		return y
end

# ╔═╡ 40391fb3-27e6-4c4a-b480-a6aee2f5327f
""" 
	Dummy, necessary for calling vectorbundle_newton
"""
function connection_map(E, q)
    return q
end

# ╔═╡ 6f9b4d57-7404-4111-a195-fce2075e7c60
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

# ╔═╡ 1b10212a-d1ca-47f6-9128-4a64b13bb4a5
"""
	Set the solve method for solving the Newton equation in each step
"""
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

# ╔═╡ 5590220d-f866-4eff-840f-b6e6134a2413
"""
	Initial geodesic
"""
y_0 = copy(power, discretized_y)

# ╔═╡ c967835e-5a43-461b-8c4c-dfac573698b8
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

# ╔═╡ ca41d482-650c-46c0-9785-771767eabe1d
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ 6462ca2e-ad2a-4140-ba16-7da52b60470c
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ f8924693-8ab0-4da7-b76a-fb978280fe2c
p_res = get_solver_result(st_res);

# ╔═╡ 9e4e724a-4fdc-4aa0-a910-c80dc1725728
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
			s += normen[i]
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

# ╔═╡ b46b07e5-6fe0-4861-a0d3-c627a6814f76
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
# ╠═78895b59-4591-485c-a50d-11925f734921
# ╠═293d0824-906e-11ef-2b7f-834616d0ede2
# ╠═acc549ff-feac-4236-b4c6-743328391275
# ╠═5b4ced2b-66bc-4b23-8e19-b5eedadd93d7
# ╠═708ad6f4-f8a2-437c-b75e-52031d116e06
# ╠═f0dc31e1-3b38-4b98-b649-9c626eb3e620
# ╠═323746d0-4962-4a7c-8db2-9c6f67f6cc96
# ╠═abbeebcc-5cb9-424a-9c65-aa8f7e67d549
# ╠═c8279ae4-1ffd-4bde-b0b5-779410443e1e
# ╠═dbbfc102-8694-4417-81bc-eb8a4b15e0f7
# ╠═d3814468-af4d-46f4-b669-b111cc2ea27e
# ╠═2384b532-e93f-415f-857a-4b0b7662a44d
# ╠═96830f5a-f767-43d1-855f-46fe3fccc825
# ╠═ff83e7f9-cb8a-418b-a388-6c997b6a4396
# ╠═f8df1a83-a0f6-40d8-b731-dded209b4e91
# ╠═530abbb0-e541-4787-a49a-eb7366724c33
# ╠═212e3bb5-7bb9-4828-9a89-a24411e802ef
# ╠═40391fb3-27e6-4c4a-b480-a6aee2f5327f
# ╠═1b10212a-d1ca-47f6-9128-4a64b13bb4a5
# ╠═6f9b4d57-7404-4111-a195-fce2075e7c60
# ╠═5590220d-f866-4eff-840f-b6e6134a2413
# ╠═c967835e-5a43-461b-8c4c-dfac573698b8
# ╠═ca41d482-650c-46c0-9785-771767eabe1d
# ╠═6462ca2e-ad2a-4140-ba16-7da52b60470c
# ╠═f8924693-8ab0-4da7-b76a-fb978280fe2c
# ╠═9e4e724a-4fdc-4aa0-a910-c80dc1725728
# ╠═b46b07e5-6fe0-4861-a0d3-c627a6814f76
