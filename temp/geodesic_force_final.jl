### A Pluto.jl notebook ###
# v0.20.13

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
    using WGLMakie, Makie, GeometryTypes, Colors, ColorSchemes, NamedColors
	using GLMakie
	using CairoMakie
	using CSV, DataFrames
	using FileIO, ProgressLogging
end;

# ╔═╡ 3f3a047f-0e9d-4364-8df2-7105a104843b
md"""
In this example we compute a geodesic under a force field on the sphere by applying Newton's method on vector bundles which was introduced in \ref{paper}. This example reproduces the results from \ref{paper}.
"""

# ╔═╡ c9d93358-17a7-4c6b-8f3c-2884e65e698c
md"""
We consider the sphere $$\mathbb{S}^2$$ equipped with the Riemannian metric $\langle \cdot, \cdot \rangle$ given by the Euclidean inner product with corresponding norm $\|\cdot\|$ and an interval $\Omega \subset \mathbb R$.

Let $\mathcal X = H^1(\Omega, \mathbb S^2)$ and  $\mathcal E^* = T^*\mathcal X$ its cotangent bundle.

Our goal is to find a zero of the mapping $F \colon \mathcal X \to \mathcal E^*$ with

$$F(\gamma)\phi \coloneqq \int_\Omega \langle 	\dot{\gamma}(t), \dot{\phi}(t)\rangle + \omega(\gamma(t))\phi(t) \; dt$$

for $\gamma \in \mathcal X$ and $\phi \in T_\gamma\mathcal X$. 

Additionally, we have to take into account that boundary conditions $\gamma(0) = \gamma_0$ and $\gamma(T) = \gamma_T$ for given $\gamma_0, \gamma_T \in \mathbb S^2$ are satisfied.
This yields a geodesic under a given force field $\omega\colon \mathbb S^2 \to T^*\mathbb S^2$ connecting $\gamma_0$ and $\gamma_T$.
"""

# ╔═╡ 827fe56f-ead5-4d9a-a785-8f88ad2ad608
md"""
For our example we set
"""

# ╔═╡ 12e32b83-ae65-406c-be51-3f21935eaae5
begin
	N=50

	S = Manifolds.Sphere(2)
	power = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
	
	st = 0.4
	halt = pi - 0.4
	#halt = pi/2
	
	h = (halt-st)/(N+1)
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1] # equidistant discrete time points
	
	y0 = [sin(st),0,cos(st)] # startpoint of geodesic
	yT = [sin(halt),0,cos(halt)] # endpoint of geodesic
end;

# ╔═╡ d286999d-6324-4112-87c5-de4df7a52d93
md"""
As a starting point, we use the geodesic connecting $\gamma_0$ and $\gamma_T$:
"""

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 5c0980c5-284e-4406-bab8-9b9aff9391ba
discretized_y = [y(Ωi) for Ωi in Omega];

# ╔═╡ e00854e0-95ef-4ef5-be4c-ea19f014c8b7
md"""
In order to apply Newton's method to find a zero of $F$, we need the linear mapping $Q_{F(\gamma)}^*\circ F'(\gamma)$ (cf. \ref{paper}) which can be seen as a covariant derivative. Since the sphere is an embedded submanifold of $\mathbb R^3$, we can use the formula 

$Q_{F(\gamma)}^*\circ F'(\gamma)\delta \gamma\,\phi = F(\gamma)(\overset{\rightarrow}{V}_\gamma'(\gamma)\delta \gamma\,\phi) + F_{\mathbb R^3}'(\gamma)\delta \gamma\,\phi$

for $\delta \gamma, \, \phi \in T_\gamma \mathcal X$, where $\overset{\rightarrow}{V}_\gamma(\hat \gamma) \in L(T_\gamma \mathcal X, T_{\hat{\gamma}}\mathcal X)$ is a vector transport and 

$F_{\mathbb R^3}'(\gamma)\delta \gamma\, \phi = \int_\Omega \langle \dot{\delta \gamma}(t),\dot{\phi}(t)\rangle + \omega'(\gamma(t))\delta \gamma(t)\phi(t) \; dt$

is the euclidean derivative of $F$.


"""

# ╔═╡ 4d22e6ed-068d-4127-bf50-5b4a0f6bd9d1
md"""
We define a structure that has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""

# ╔═╡ bc449c2d-1f23-4c72-86ab-a46acbf64129
mutable struct DifferentiableMapping{M<:AbstractManifold,F1<:Function,F2<:Function,T}
	domain::M
	value::F1
	derivative::F2
	scaling::T
end;

# ╔═╡ 4d36f402-efd1-4e25-83d3-b51ba1684867
md"""
The following routines define a vector transport and its euclidean derivative. As seen above, they are needed to derive a covariant derivative of $F$.

As a vector transport we use the (pointwise) orthogonal projection onto the tangent spaces, i.e. for $p, q \in \mathbb S^2$ and $X \in T_p\mathbb S^2$ we set 

$\overset{\rightarrow}{V}_{p}(q)X = (I-q\cdot q^T)X \in T_q\mathbb S^2.$

The derivative of the vector transport is then given by 

$\left(\frac{d}{dq}\overset{\rightarrow}{V}_{p}(q)\big\vert_{q=p}\delta q\right)X = \left( - \delta q\cdot p^T - p\cdot \delta q^T\right)\cdot X.$

"""

# ╔═╡ 50a51e47-b6b1-4e43-b4b9-aad23f6ec390
begin 
	
	function transport_by_proj(S, p, X, q)
		return X - q*(q'*X)
	end

	function transport_by_proj_prime(S, p, X, dq)
		return (- dq*p' - p*dq')*X
	end

	transport=DifferentiableMapping(S,transport_by_proj,transport_by_proj_prime,nothing)
	
end;

# ╔═╡ 8a2fdd86-315d-44e1-90c7-7347304b6bc7
md"""
The following two routines define the integrand of $F$ and its euclidean derivative. They use a force field $\omega$, which is defined, below. A scaling parameter for the force is also employed.
"""

# ╔═╡ ea1f433b-b1c2-44f0-b8cf-45992946704a
md"""
In this example we consider the force field $\omega\colon \mathbb S^2 \to T^*\mathbb S^2$ given by the 1-form corresponding to a (scaled) winding field, i.e. for $C\in\mathbb R$ and $y\in \mathbb{S}^2$ we set

$\omega(y) \coloneqq \frac{C y_3}{y_1^2+y_2^2} \cdot \bigg\langle \begin{pmatrix}
        -y_2 \\ y_1 \\ 0
    \end{pmatrix}, \cdot \bigg\rangle \in (T_y\mathbb{S}^2)^*.$
"""

# ╔═╡ 808db8aa-64f7-4b36-8c6c-929ba4fa22db
function w(p, c)
		return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
end;

# ╔═╡ e6fdf785-808d-4203-a2d2-be9696b05689
md"""
Its derivative is given by 
"""

# ╔═╡ 288b9637-0500-40b8-a1f9-90cb9591402b
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
		return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
end;

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
begin 

	function F_at(Integrand, y, ydot, B, Bdot)
	  	return ydot'*Bdot+w(y,Integrand.scaling)'*B
	end

	function F_prime_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
		return B1dot'*B2dot+(w_prime(y,Integrand.scaling)*B1)'*B2
	end

	integrand=DifferentiableMapping(S,F_at,F_prime_at,3.0) 

end;

# ╔═╡ a0b939d5-40e7-4da4-baf1-8a297bb52fb7
md"""
`NewtonEquation`

In this example we implement a functor to compute the Newton matrix and the right hand side for the Newton equation \ref{paper}

$$Q^*_{F(\gamma)}\circ F'(\gamma)\delta \gamma + F(\gamma) = 0$$

by using the assembler provided in ManoptExamples.jl (cf. Referenz).
	
It returns the matrix and the right hand side in base representation.
Moreover, for the computation of the simplified Newton direction (which is necessary for affine covariant damping) a method for assembling the right hand side for the simplified Newton equation is provided.
	
"""

# ╔═╡ 6ce088e9-1aa0-4d44-98a3-2ab8b8ba5422
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

# ╔═╡ a3179c3a-cb5a-4fcb-bbdc-75ea693616d2
md"""
We compute the Newton direction $\delta \gamma$ by solving the linear system given by the base representation of the Newton equation directly and return the Newton direction in vector representation:
"""

# ╔═╡ f78557e2-363e-4803-97d7-b57df115a619
function solve_in_basis_repr(problem, newtonstate) 
	X_base = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	return get_vector(problem.manifold, newtonstate.p, X_base, DefaultOrthogonalBasis())
end;

# ╔═╡ 9a2ebb9a-74c7-4efd-b042-23263bbf4235
begin
	NE = NewtonEquation(power, integrand, transport, Omega)
		
	st_res = vectorbundle_newton(power, TangentBundle(power), NE, discretized_y; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-12; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
	#stepsize=Manopt.AffineCovariantStepsize(power, theta_des=0.5),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, :Change],
	return_state=true
)
end

# ╔═╡ 87af653d-901e-4f81-a41c-ddc613d04909
md"""
We extract the recorded values
"""

# ╔═╡ 161070b9-7953-4260-ab3b-f0f0bf8410ac
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	p_res = get_solver_result(st_res)
end;

# ╔═╡ 4c939ec9-0e1b-4194-8f22-d2639172922c
md"""
and plot the result, where we measure the norms of the Newton direction in each iteration,
"""

# ╔═╡ a08b8946-5adb-43c7-98aa-113875c954b1
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Norms of the Newton directions (semilogarithmic)"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δγ‖")
    scatterlines!(change[1:end], color = :blue)
	f	
end

# ╔═╡ 290dbe94-4686-4629-9f93-f01353aac404
md"""
and the resulting geodesic under the force field (orange). The starting geodesic (blue) is plotted as well. The force acting on each point of the geodesic is visualized by green arrows. 
"""

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
begin
n = 25
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
	color = RGBA(1.,1.,1.,0.),
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
	
	scatterlines!(ax, π1.(geodesic_final), π2.(geodesic_final), π3.(geodesic_final); markersize =5, color=:orange, linewidth=2)
	
	scatterlines!(ax, π1.(geodesic_start), π2.(geodesic_start), π3.(geodesic_start); markersize =5, color=:blue, linewidth=2)
	
	scatter!(ax, π1.([y0, yT]), π2.([y0, yT]), π3.([y0, yT]); markersize =5, color=:red)
	
	arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.05), transparency=true, lengthscale=0.1)
	

	cam = cameracontrols(ax.scene)
	cam.lookat[] =[-2.5, 2.5, 2]
	
	fig
end

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╟─3f3a047f-0e9d-4364-8df2-7105a104843b
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╟─c9d93358-17a7-4c6b-8f3c-2884e65e698c
# ╟─827fe56f-ead5-4d9a-a785-8f88ad2ad608
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╟─d286999d-6324-4112-87c5-de4df7a52d93
# ╠═29043ca3-afe0-4280-a76a-7c160a117fdf
# ╠═5c0980c5-284e-4406-bab8-9b9aff9391ba
# ╟─e00854e0-95ef-4ef5-be4c-ea19f014c8b7
# ╟─4d22e6ed-068d-4127-bf50-5b4a0f6bd9d1
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╟─4d36f402-efd1-4e25-83d3-b51ba1684867
# ╠═50a51e47-b6b1-4e43-b4b9-aad23f6ec390
# ╟─8a2fdd86-315d-44e1-90c7-7347304b6bc7
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╟─ea1f433b-b1c2-44f0-b8cf-45992946704a
# ╠═808db8aa-64f7-4b36-8c6c-929ba4fa22db
# ╟─e6fdf785-808d-4203-a2d2-be9696b05689
# ╠═288b9637-0500-40b8-a1f9-90cb9591402b
# ╟─a0b939d5-40e7-4da4-baf1-8a297bb52fb7
# ╠═6ce088e9-1aa0-4d44-98a3-2ab8b8ba5422
# ╟─a3179c3a-cb5a-4fcb-bbdc-75ea693616d2
# ╠═f78557e2-363e-4803-97d7-b57df115a619
# ╠═9a2ebb9a-74c7-4efd-b042-23263bbf4235
# ╟─87af653d-901e-4f81-a41c-ddc613d04909
# ╠═161070b9-7953-4260-ab3b-f0f0bf8410ac
# ╟─4c939ec9-0e1b-4194-8f22-d2639172922c
# ╠═a08b8946-5adb-43c7-98aa-113875c954b1
# ╟─290dbe94-4686-4629-9f93-f01353aac404
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
