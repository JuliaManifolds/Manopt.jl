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
    using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ aa62a032-baec-40ab-918a-971dace0d844
md"""
In this example we compute a geodesic on the sphere avoiding an obstacle by applying Newton's method on vector bundles which was introduced in \ref{paper}. This example reproduces the results from \ref{paper}.
"""

# ╔═╡ 0a68dba2-3b3f-4ad4-bf4a-bdc37bc28d43
md"""
We consider the sphere $$\mathbb{S}^2$$ equipped with the Riemannian metric $\langle \cdot, \cdot \rangle$ given by the Euclidean inner product with corresponding norm $\|\cdot\|$ and a time interval $\Omega =[0,T]$.

Let $\mathcal X = H^1(\Omega, \mathbb S^2)$ and  $\mathcal E^* = T^*\mathcal X$ its cotangent bundle.

Consider the minimization problem

$\min_{\gamma \in H^1(\Omega, \mathbb S^2)} \; \frac12 \int_0^T \|\dot \gamma(t)\|^2 \; dt$
under the constraint that $\gamma_3(t) \leq 1-h_{\mathrm{ref}} \; \text{ for all } t\in [0,T]$ where $\gamma_3(t)$ denotes the third component of $\gamma(t)\in \mathbb{S}^2$ and $h_{\mathrm{ref}} \in (0,1)$ is a given height. Additionally, we have to take into account that boundary conditions $\gamma(0) = \gamma_0$ and $\gamma(T) = \gamma_T$ for given $\gamma_0, \gamma_T \in \mathbb S^2$ are satisfied.

Using a penalty method, also known as Moreau-Yosida regularization (cf. \cite{HintermuellerKunisch:2006:1}), with a quadratic penalty term we can rewrite this as an unconstrained minimization problem with a penalty coefficient $p\in \mathbb R$:

$\min_{\gamma \in H^1(\Omega, \mathbb S^2)} \; \underbrace{\frac12 \int_0^T \|\dot \gamma(t)\|^2 + p \max(0, \gamma_3(t) - 1 + h_{\mathrm{ref}})^2 \; dt}_{=: f(\gamma)}$

Let $m \colon\mathbb R \to \mathbb R, \, m(x) \coloneqq \max(0, x)$. 
The objective is a semismooth function $f : \mathcal X \to \mathbb R$ with a Newton-derivative 

$f'(\gamma)\delta \gamma = \int_0^T \langle \dot\gamma(t), \dot{\delta \gamma}(t)\rangle + p \cdot m(\gamma_3(t) - 1 + h_{\mathrm{ref}})\delta \gamma_3(t)$

for $\delta \gamma \in T_\gamma \mathcal X$. 
Our goal is to find a zero of this Newton-derivative. 

This yields a geodesic avoiding the north pol cap and connecting $\gamma_0$ and $\gamma_T$.
"""

# ╔═╡ 00fe7ab7-3cb7-455c-872d-336770503c02
md"""
For our example we set
"""

# ╔═╡ 12e32b83-ae65-406c-be51-3f21935eaae5
begin
	N=200
	
	S = Manifolds.Sphere(2)
	power = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
	
	st = -pi/2 + 0.1
	halt = pi/2 - 0.1
	h = (halt-st)/(N+1)
	Omega = range(; start=st, stop = halt, length=N+2)[2:end-1] # equidistant discrete time points
	
	theta = pi/4
	y0 = [sin(theta)*cos(st),sin(theta)*sin(st),cos(theta)] # startpoint of geodesic
	yT = [sin(theta)*cos(halt),sin(theta)*sin(halt),cos(theta)] # endpoint of geodesic

	h_ref = 0.17
end;

# ╔═╡ 81c14a72-309c-470e-8c8c-bcc7dd843e43
md"""
In order to apply Newton's method to find a zero of $f'$, we need the linear mapping $Q_{f'(\gamma)}^*\circ f''(\gamma)$ (cf. \ref{paper}). Since the sphere is an embedded submanifold of $\mathbb R^3$, we can use the formular 

$Q_{f'(\gamma)}^*\circ f''(\gamma)\delta \gamma\,\phi = f'(\gamma)(\overset{\rightarrow}{V}_\gamma'(\gamma)\delta \gamma\,\phi) + f''_{\mathbb R^3}(\gamma)\delta \gamma\,\phi$

for $\delta \gamma, \, \phi \in T_\gamma \mathcal X$, where $\overset{\rightarrow}{V}_\gamma(\hat \gamma) \in L(T_\gamma \mathcal X, T_{\hat{\gamma}}\mathcal X)$ is a vector transport and 

$f_{\mathbb R^3}''(\gamma)\delta \gamma\, \phi = \int_0^T \langle \dot{\delta \gamma}(t), \dot{\phi}(t)\rangle + p \cdot m'(\gamma_3(t) - 1 + h_{\mathrm{ref}})\phi_{3}(t) \delta \gamma_{3}(t)$

is the euclidean second derivative of the objective.


"""

# ╔═╡ c61ac584-e3be-47c8-8801-684141d9e1f9
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
	scaling_penalty::T
end;

# ╔═╡ e7f10756-4ecd-4a43-a560-87676f26914a
md"""
The following routines define a vector transport and its euclidean derivative. As seen above, they are needed to derive $Q_{f'(\gamma)}^*\circ f''(\gamma)$.

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

# ╔═╡ fde5a441-9ff5-45f9-ad00-57078a16dba8
md"""
The following two routines define the integrand of $f'$ and the euclidean second derivative $f''_{\mathbb R^3}$. Here, a Newton-derivative of the maximum function given by 

$m'(x) \coloneqq \begin{cases}
        0 &: \; x<0 \\
        \text{arbitrary} &: \; x= 0\\
        1 &: \; x>0
    \end{cases}$

is used. A scaling parameter for the penalty parameter is also employed.
"""

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
begin
	
	function f_prime_at(Integrand, y, ydot, B, Bdot)
	  	return ydot'*Bdot + Integrand.scaling_penalty * max(0.0, y[3] - 1.0 + h_ref)*B[3]
	end

	function max_prime(y)
		if y[3] < 1.0 - h_ref
			return 0.0
		else
			return 1.0
		end
	end

	function f_second_at(Integrand,y,ydot,B1,B1dot,B2,B2dot)
		return B1dot'*B2dot + Integrand.scaling_penalty*max_prime(y)*B1[3]*B2[3]
	end

	integrand=DifferentiableMapping(S,f_prime_at,f_second_at,1.0)
end;

# ╔═╡ eda0a587-a19d-4f80-80bf-c4cc5a21854c
md"""
`NewtonEquation`

In this example we implement a functor to compute the Newton matrix and the right hand side for the Newton equation \ref{paper}

$$Q^*_{f''(\gamma)}\circ f''(\gamma)\delta \gamma + f'(\gamma) = 0^*_\gamma$$

by using the assembler provided in ManoptExamples.jl (cf. Referenz).
	
It returns the matrix and the right hand side in base representation.
Moreover, for the computation of the simplified Newton direction (which is necessary for affine covariant damping) a method for assembling the right hand side for the simplified Newton equation is provided.
	
"""

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
	println(norm(bctrial))
	return bctrial
end
end;

# ╔═╡ f7ba0044-ecfe-4714-9ec3-b4792d38b21c
md"""
We compute the Newton direction $\delta \gamma$ by solving the linear system given by the base representation of the Newton equation directly and return the Newton direction in vector representation:
"""

# ╔═╡ 910e85fe-2db4-43f3-8cf9-a805858f3627
function solve_in_basis_repr(problem, newtonstate) 
	Xc = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	res_c = get_vector(problem.manifold, newtonstate.p, Xc, DefaultOrthogonalBasis())
	return res_c
end;

# ╔═╡ 1798ca75-1c1a-4760-af68-87bbd60544b4
md"""
For the computation of a solution of the penalized problem we use a simple path-following method increasing the penalty parameter by a factor 1.2 in each iteration. For the first iteration we use a curve along the latitude connecting $\gamma_0$ and $\gamma_T$:
"""

# ╔═╡ 189e45a0-b64f-4d88-ad8a-d0a03f543362
begin
	function y(t)
		return [sin(theta)*cos(t), sin(theta)*sin(t), cos(theta)]
	end

	discretized_y = [y(Ωi) for Ωi in Omega];
end;

# ╔═╡ 4c26b3d0-51ed-48b1-9efe-1a4ba0949e04
begin
	y_star = copy(power, discretized_y)

	integrand.scaling_penalty = 5.0
	
	NE = NewtonEquation(power, integrand, transport, Omega)
	
	st_res = vectorbundle_newton(power, TangentBundle(power), NE, y_star; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	return_state=true
)
	y_star = copy(power, get_solver_result(st_res))
		
	for i in range(1,50)
		integrand.scaling_penalty *= 1.2

		NE = NewtonEquation(power, integrand, transport, Omega)
 
		st_res = vectorbundle_newton(power, TangentBundle(power), NE, y_star; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
		stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(power,1e-13; outer_norm=Inf)),
		retraction_method=ProjectionRetraction(),
		debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop],
		return_state=true
		)
		y_star = copy(power, get_solver_result(st_res));
	end
end;

# ╔═╡ da4fca13-23d3-4f4a-bc35-ace2b5dacaf8
md"""
This yields the geodesic shown below avoiding the north pole cap and connecting two almost antipodal points $\gamma_0$ and $\gamma_T$.
"""

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
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

x = acos(1-h_ref)

circx = zeros(n)
circy = zeros(n)
circz = zeros(n)
for i in 1:n
	circx[i] = cos.(u[i]) * sin(x);
	circy[i] = sin.(u[i]) * sin(x);
	circz[i] = cos(x);
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

	scatterlines!(ax, circx, circy, circz; markersize =2, color=:black, linewidth=2)
	
	scatterlines!(ax, π1.(y_star), π2.(y_star), π3.(y_star); markersize =8, color=:orange, linewidth=2)
	
	scatter!(ax, π1.([y0]), π2.([y0]), π3.([y0]); markersize = 10, color=:green)
	scatter!(ax, π1.([yT]), π2.([yT]), π3.([yT]); markersize = 10, color=:red)
	
	fig
end

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╟─aa62a032-baec-40ab-918a-971dace0d844
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╟─0a68dba2-3b3f-4ad4-bf4a-bdc37bc28d43
# ╟─00fe7ab7-3cb7-455c-872d-336770503c02
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╟─81c14a72-309c-470e-8c8c-bcc7dd843e43
# ╟─c61ac584-e3be-47c8-8801-684141d9e1f9
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╟─e7f10756-4ecd-4a43-a560-87676f26914a
# ╠═50a51e47-b6b1-4e43-b4b9-aad23f6ec390
# ╟─fde5a441-9ff5-45f9-ad00-57078a16dba8
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╟─eda0a587-a19d-4f80-80bf-c4cc5a21854c
# ╠═1adf467b-81e8-4438-98ce-4420ad1f5bda
# ╟─f7ba0044-ecfe-4714-9ec3-b4792d38b21c
# ╠═910e85fe-2db4-43f3-8cf9-a805858f3627
# ╟─1798ca75-1c1a-4760-af68-87bbd60544b4
# ╠═189e45a0-b64f-4d88-ad8a-d0a03f543362
# ╠═4c26b3d0-51ed-48b1-9efe-1a4ba0949e04
# ╟─da4fca13-23d3-4f4a-bc35-ace2b5dacaf8
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
