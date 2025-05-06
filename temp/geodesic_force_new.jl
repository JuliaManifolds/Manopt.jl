### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ c9994bc4-b7bb-11ef-3430-8976c5eabdeb
using Pkg; Pkg.activate();

# ╔═╡ 9fb54416-3909-49c4-b1bf-cc868e580652
begin
	using LinearAlgebra
	using SparseArrays
	using OffsetArrays
	using Manopt
	using ManoptExamples
	using Manifolds
	using Random
    using WGLMakie, Makie, GeometryTypes, Colors
	#using CairoMakie
	#using FileIO
end;

# ╔═╡ 12e32b83-ae65-406c-be51-3f21935eaae5
begin
	N=20
	st = 0.5
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

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
function y(t)
	return [sin(t), 0, cos(t)]
	#return [sin(halt+st-t), 0, cos(halt+st-t)]
	#return [cos(t), sin(t), 0]
end;

# ╔═╡ 5c0980c5-284e-4406-bab8-9b9aff9391ba
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ bc449c2d-1f23-4c72-86ab-a46acbf64129
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


# ╔═╡ 50a51e47-b6b1-4e43-b4b9-aad23f6ec390
"""
 The following two routines define the vector transport and its derivative. The second is needed to obtain covariant derivative from the ordinary derivative.
"""
function transport_by_proj(S, p, X, q)
	return X - q*(q'*X)
end

# ╔═╡ 9cdd4289-c49d-4733-8487-f471e38fc402
function transport_by_proj_prime(S, p, X, dq)
	return (- dq*p' - p*dq')*X
end

# ╔═╡ 808db8aa-64f7-4b36-8c6c-929ba4fa22db
"""
Force field w and its derivative. A scaling parameter is also employed.
"""
function w(p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
"""
The following two routines define the integrand and its ordinary derivative. They use a vector field w, wich is defined, below. A scaling parameter is also employed.
"""
function F_at(Integrand, y, ydot, B, Bdot)
	  return ydot'*Bdot+w(y,Integrand.scaling)'*B
end

# ╔═╡ 288b9637-0500-40b8-a1f9-90cb9591402b
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ ac04e6ec-61c2-475f-bb2f-83755c04bd72
function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return B1dot'*B2dot+(w_prime(y,Integrand.scaling)*B1)'*B2
end

# ╔═╡ 684508bd-4525-418b-b89a-85d56c01b188
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
integrand=DifferentiableMapping(S,F_at,F_prime_at,4.0) 
transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)
end;

# ╔═╡ 98a334b1-5aa9-4e3a-a03d-f6859e77f1dc
"""
Dummy
"""
function bundlemap(M, y)
		# Include boundary points
end

# ╔═╡ 6dbfa961-639d-45af-b1d1-2622331e8455
"""
Dummy
"""
function connection_map(E, q)
    return q
end

# ╔═╡ 1abddae9-d862-4c73-a46f-7bb0a1a8f917
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

# ╔═╡ 7766ff98-f22f-4357-a2ea-040ac3b79651
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, newtonstate.p, newtonstate, problem)

# ╔═╡ 6ea95d81-5d50-453b-9077-2a8cd25e4746
norm_newtonschritt = []

# ╔═╡ 6ce088e9-1aa0-4d44-98a3-2ab8b8ba5422
begin
struct LinearizedNewtonObjective{F, T, Om}
	Manifold::AbstractManifold
	vb::AbstractManifold
	integrand::F
	transport::T
	Omega::Om
end

function LinearizedNewtonObjective(M, vb, F, VT, interval)
	return LinearizedNewtonObjective{typeof(F), typeof(VT), typeof(interval)}(M, vb, F, VT, interval)
end
	
function (lno::LinearizedNewtonObjective)(p)
	n = manifold_dimension(lno.Manifold)
	Ac::SparseMatrixCSC{Float64,Int32} =spzeros(n,n)
	bc = zeros(n)
	bcsys=zeros(n)
	bctrial=zeros(n)
	Oy = OffsetArray([y0, p..., yT], 0:(length(lno.Omega)+1))
	#Oytrial = OffsetArray([y0, state.p_trial..., yT], 0:(length(lno.Omega)+1))
	S = (lno.Manifold).manifold
	println("Assemble:")
    @time ManoptExamples.get_rhs_Jac!(bc,Ac,h,Oy,lno.integrand,lno.transport)
	bcsys=bc
	#if state.is_same == true
	#	bcsys=bc
	#else
		#@time ManoptExamples.get_rhs_simplified!(bctrial,h,Oy,Oytrial,integrand,transport)
    	#bcsys=bctrial-(1.0 - state.stepsize.alpha)*bc
	#end
	#Asparse = sparse(Ac)
	#println("Solve:")
	Xc = (Ac) \ (-bcsys)
	#println(Xc)
	#println(Ac)
	B = get_basis(lno.Manifold, p, DefaultOrthonormalBasis())
	res_c = get_vector(lno.Manifold, p, Xc, B)
	push!(norm_newtonschritt, norm(res_c))
	return res_c
end
end;

# ╔═╡ c3c46cc3-7366-4724-98c3-ba94768d472b
""" Initial geodesic """
	y_0 = copy(power, discretized_y)

# ╔═╡ 42dfd796-a6c2-43e0-a8c4-da9f4a545aa3
lno = LinearizedNewtonObjective(power, TangentBundle(power), integrand, transport, Omega)

# ╔═╡ 8c16594d-504e-4c11-bf90-174e205a2723
problem = VectorbundleManoptProblem(power, TangentBundle(power), lno)

# ╔═╡ 187488cc-30a5-417a-a7f4-51096e71e67b
#println(typeof(ConstantLength(1.0)))
state = VectorbundleNewtonState(power, TangentBundle(power), y_0; stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)), retraction_method=ProjectionRetraction())

# ╔═╡ 30f04f7e-5fdf-4e30-9858-99403bcd2193
solve!(problem, state)

# ╔═╡ a08b8946-5adb-43c7-98aa-113875c954b1
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(norm_newtonschritt)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(norm_newtonschritt[1:end], color = :blue)
	f
end

# ╔═╡ 4bd08087-470d-456a-85ce-e92e08253cb4
# ╠═╡ disabled = true
#=╠═╡
st_res = vectorbundle_newton(power, TangentBundle(power), lno, y_0;
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
stepsize=ConstantLength(1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	record=[:Iterate, :Change],
	return_state=true
)
  ╠═╡ =#

# ╔═╡ abe5c5f3-4a28-425c-afde-64b645f3a9d9
#=╠═╡
change = get_record(st_res, :Iteration, :Change)[2:end];
  ╠═╡ =#

# ╔═╡ 6451f8c5-7b4f-4792-87fd-9ed2635efa88
#=╠═╡
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end
  ╠═╡ =#

# ╔═╡ b0b8e87f-da09-4500-8aa9-e35934f7ef54
#=╠═╡
p_res = get_solver_result(st_res);
  ╠═╡ =#

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
#=╠═╡
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

ws = [-w(p, integrand.scaling) for p in p_res]
ws_start = [-w(p, integrand.scaling) for p in discretized_y]
for i in 1:n
    for j in 1:n
        sx[i,j] = cos.(u[i]) * sin(v[j]);
        sy[i,j] = sin.(u[i]) * sin(v[j]);
        sz[i,j] = cos(v[j]);
    end
end
fig, ax, plt = meshscatter(
  sx,sy,sz,
  color = fill(RGBA(1.,1.,1.,0.), n, n),
  shading = Makie.automatic,
  transparency=true
)

geodesic_start = [y0, discretized_y ...,yT]
geodesic_final = [y0, p_res ..., yT]
ax.show_axis = false
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.1); transparency=true)
    π1(x) = 1.02*x[1]
    π2(x) = 1.02*x[2]
    π3(x) = 1.02*x[3]
	
	scatterlines!(ax, π1.(geodesic_final), π2.(geodesic_final), π3.(geodesic_final); markersize =8, color=:orange, linewidth=2)
	
	scatterlines!(ax, π1.(geodesic_start), π2.(geodesic_start), π3.(geodesic_start); markersize =8, color=:blue, linewidth=2)
	
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =8, color=:red)
	
	arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=0.07)
	
	#arrows!(ax, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y), π1.(ws_start), π2.(ws_start), π3.(ws_start); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=0.15)
	fig
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╠═29043ca3-afe0-4280-a76a-7c160a117fdf
# ╠═5c0980c5-284e-4406-bab8-9b9aff9391ba
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╠═50a51e47-b6b1-4e43-b4b9-aad23f6ec390
# ╠═9cdd4289-c49d-4733-8487-f471e38fc402
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╠═ac04e6ec-61c2-475f-bb2f-83755c04bd72
# ╠═684508bd-4525-418b-b89a-85d56c01b188
# ╠═808db8aa-64f7-4b36-8c6c-929ba4fa22db
# ╠═288b9637-0500-40b8-a1f9-90cb9591402b
# ╠═98a334b1-5aa9-4e3a-a03d-f6859e77f1dc
# ╠═6dbfa961-639d-45af-b1d1-2622331e8455
# ╠═7766ff98-f22f-4357-a2ea-040ac3b79651
# ╠═1abddae9-d862-4c73-a46f-7bb0a1a8f917
# ╠═6ea95d81-5d50-453b-9077-2a8cd25e4746
# ╠═6ce088e9-1aa0-4d44-98a3-2ab8b8ba5422
# ╠═c3c46cc3-7366-4724-98c3-ba94768d472b
# ╠═42dfd796-a6c2-43e0-a8c4-da9f4a545aa3
# ╠═8c16594d-504e-4c11-bf90-174e205a2723
# ╠═187488cc-30a5-417a-a7f4-51096e71e67b
# ╠═30f04f7e-5fdf-4e30-9858-99403bcd2193
# ╠═a08b8946-5adb-43c7-98aa-113875c954b1
# ╠═4bd08087-470d-456a-85ce-e92e08253cb4
# ╠═abe5c5f3-4a28-425c-afde-64b645f3a9d9
# ╠═6451f8c5-7b4f-4792-87fd-9ed2635efa88
# ╠═b0b8e87f-da09-4500-8aa9-e35934f7ef54
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
