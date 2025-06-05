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
	N=50
	st = 0.5
	halt = pi - 0.5
	h = (halt-st)/(N+1)
	#halt = pi - st
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1]
	#Omega = range(; start=halt, stop = st, length=N+2)[2:end-1]
	
	y0 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT = [sin(halt),0,cos(halt)] # endpoint of geodesic

end;

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 5c0980c5-284e-4406-bab8-9b9aff9391ba
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ bc449c2d-1f23-4c72-86ab-a46acbf64129
"""
Such a structure has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""
mutable struct DifferentiableMapping{M<:AbstractManifold,F1<:Function,F2<:Function, T}
	domain::M
	value::F1
	derivative::F2
	scaling_wind::T
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
	Evaluates the wind field at a point p on the sphere (here: winding field scaled by the third component)
"""
	function w(p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	end

# ╔═╡ 288b9637-0500-40b8-a1f9-90cb9591402b
"""
	Returns the first derivative of the wind field at point p as a matrix
"""
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
"""
The following two routines define the integrand and its Euclidean derivative. They use a wind field w, its derivative and its second derivative, defined below. A scaling parameter is also employed.
"""
function F_at(Integrand, y, ydot, B, Bdot)
	  return (ydot - w(y, Integrand.scaling_wind))'*(Bdot - w_prime(y, Integrand.scaling_wind)*B)
end

# ╔═╡ eb3cb1db-229a-44c0-8591-90142cbb0885
"""
	Returns the second derivative of the wind field at point p in direction v as a matrix
"""
function w_doubleprime(p, v, c)
	nenner = (p[1]^2+p[2]^2)
	w1 = 1/(nenner^2)*[(2*p[2]*p[3]*nenner-8*p[1]^2*p[2]*p[3])/nenner -p[3]*(2*p[1]*nenner^2-4*p[1]*(p[1]^4-p[2]^4))/nenner^2 2*p[1]*p[2]; (-2*p[1]*p[3]*nenner^2-4*p[1]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[2]*p[3]*nenner+8*p[1]^2*p[2]*p[3])/nenner (p[2]^2-p[1]^2); 0.0 0.0 0.0]
		
	w2 = 1/(nenner^2)*[(2*p[1]*p[3]*nenner-8*p[1]*p[2]^2*p[3])/nenner -p[3]*(-2*p[2]*nenner^2-4*p[2]*(p[1]^4-p[2]^4))/nenner^2 p[2]^2-p[1]^2; (2*p[2]*p[3]*nenner^2-4*p[2]*p[3]*(p[2]^4-p[1]^4))/nenner^2 (-2*p[1]*p[3]*nenner+8*p[1]*p[2]^2*p[3])/nenner -2*p[1]*p[2]; 0.0 0.0 0.0]
		
	w3 = 1/(nenner^2)*[2*p[1]*p[2] -p[1]^2+p[2]^2 0.0; p[2]^2-p[1]^2 -2*p[1]*p[2] 0.0; 0.0 0.0 0.0]
	return c*(v[1]*w1 + v[2]*w2 + v[3]*w3)
end

# ╔═╡ ac04e6ec-61c2-475f-bb2f-83755c04bd72
function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
	return (-w_doubleprime(y, B1, Integrand.scaling_wind)*B2)'*(ydot - w(y, Integrand.scaling_wind)) + (B1dot - w_prime(y, Integrand.scaling_wind)*B1)'*(B2dot - w_prime(y, Integrand.scaling_wind)
	*B2)
end

# ╔═╡ 684508bd-4525-418b-b89a-85d56c01b188
begin
S = Manifolds.Sphere(2)
power = PowerManifold(S, NestedPowerRepresentation(), N);
integrand=DifferentiableMapping(S,F_at,F_prime_at,0.0)
transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)
end;

# ╔═╡ 1adf467b-81e8-4438-98ce-4420ad1f5bda
begin
struct NewtonEquation{F, T, Om, NM, Nrhs}
	integrand::F
	transport::T
	Omega::Om
	A::NM
	b::Nrhs
end

function NewtonEquation(M, F, VT, interval)
	n = manifold_dimension(M)
	A = spzeros(n,n)
	b = zeros(n)
	return NewtonEquation{typeof(F), typeof(VT), typeof(interval), typeof(A), typeof(b)}(F, VT, interval, A, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
	n = manifold_dimension(M)
	ne.A .= spzeros(n,n)
	ne.b .= zeros(n)
	
	Oy = OffsetArray([y0, p..., yT], 0:(length(ne.Omega)+1))
	
	println("Assemble:")
    @time ManoptExamples.get_rhs_Jac!(ne.b,ne.A,h,Oy,ne.integrand,ne.transport)

	return
end


function (ne::NewtonEquation)(M, VB, p, p_trial)
	n = manifold_dimension(M)
	bctrial=zeros(n)
	Oy = OffsetArray([y0, p..., yT], 0:(length(ne.Omega)+1))
	Oytrial = OffsetArray([y0, p_trial..., yT], 0:(length(ne.Omega)+1))

	ManoptExamples.get_rhs_simplified!(bctrial,h,Oy,Oytrial,ne.integrand,ne.transport)
	return bctrial
end
end;

# ╔═╡ 910e85fe-2db4-43f3-8cf9-a805858f3627
"""
	Computes the Newton direction by solving the linear system given by the base representation of the Newton equation directly and returns the Newton direction in vector representation
"""
function solve_in_basis_repr(problem, newtonstate) 
	Xc = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	res_c = get_vector(problem.manifold, newtonstate.p, Xc, DefaultOrthogonalBasis())
	return res_c
end

# ╔═╡ 4c26b3d0-51ed-48b1-9efe-1a4ba0949e04
begin
	y_0 = copy(power, discretized_y)

	integrand.scaling_wind = 3.0
	
	NE = NewtonEquation(power, integrand, transport, Omega)
	
	st_res = vectorbundle_newton(power, TangentBundle(power), NE, y_0; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
	stepsize=Manopt.AffineCovariantStepsize(power, theta_des=0.1),
	#stepsize=ConstantLength(power, 1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, :Change],
	return_state=true
	)
		
	p_res = get_solver_result(st_res)
	#t = 0
	#for i in range(1,50)
	#	t = i/50.0
		#integrand.scaling_wind = 0.1*t^3
	#	integrand.scaling_wind = 1.0
	#	NE = NewtonEquation(power, integrand, transport, Omega)
 
	#	st_res = vectorbundle_newton(power, TangentBundle(power), NE, y_0; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	#	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	#	retraction_method=ProjectionRetraction(),
		#stepsize=Manopt.AffineCovariantStepsize(power, theta_des=0.1),
		#stepsize=ConstantLength(power, 1.0),
	#	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
	#	record=[:Iterate, :Change],
	#	return_state=true
	#	)
	#	y_0 = copy(power, get_solver_result(st_res));
		#println("penalty-Parameter = ", pp)
	#end
 
end

# ╔═╡ ac510e7a-5d46-4c22-89aa-1e6310e076a0
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ a08b8946-5adb-43c7-98aa-113875c954b1
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change[1:end], color = :blue)
	f
end

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
begin
n = 45
u = range(0,stop=2*π,length=n);
v = range(0,stop=π,length=n);
sx = zeros(n,n); sy = zeros(n,n); sz = zeros(n,n)

ws = [w(p, integrand.scaling_wind) for p in p_res]
ws_start = [w(p, integrand.scaling_wind) for p in discretized_y]
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

ax.show_axis = false
wireframe!(ax, sx, sy, sz, color = RGBA(0.5,0.5,0.7,0.1); transparency=true)
    π1(x) = 1.0*x[1]
    π2(x) = 1.0*x[2]
    π3(x) = 1.0*x[3]
	
	scatterlines!(ax, π1.(p_res), π2.(p_res), π3.(p_res); markersize =8, color=:orange, linewidth=2)
	
	#scatterlines!(ax, π1.(geodesic_start), π2.(geodesic_start), π3.(geodesic_start); markersize =8, color=:blue, linewidth=2)
	
	scatter!(ax, π1.([y0]), π2.([y0]), π3.([y0]); markersize = 10, color=:green)
	scatter!(ax, π1.([yT]), π2.([yT]), π3.([yT]); markersize = 10, color=:red)
	arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=0.5)
	
	#arrows!(ax, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y), π1.(ws_start), π2.(ws_start), π3.(ws_start); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=0.15)
	fig
end

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
# ╠═eb3cb1db-229a-44c0-8591-90142cbb0885
# ╠═1adf467b-81e8-4438-98ce-4420ad1f5bda
# ╠═910e85fe-2db4-43f3-8cf9-a805858f3627
# ╠═4c26b3d0-51ed-48b1-9efe-1a4ba0949e04
# ╠═ac510e7a-5d46-4c22-89aa-1e6310e076a0
# ╠═a08b8946-5adb-43c7-98aa-113875c954b1
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
