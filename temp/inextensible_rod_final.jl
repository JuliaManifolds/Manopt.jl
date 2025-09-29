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
	using Manopt
	using ManoptExamples
	using Manifolds
	using OffsetArrays
	using RecursiveArrayTools
    using WGLMakie, Makie, GeometryTypes, Colors
	using CairoMakie
end;

# ╔═╡ 116b717d-f9cf-44a0-be18-4950ea2f019a
md"""
In this example we compute equilibrium states of an inextensible elastic rod by applying Newton's method on vector bundles which was introduced in \ref{paper}. This example reproduces the results from \ref{paper}.
"""

# ╔═╡ 8153941a-39a0-4d91-9796-35cbd4c547e3
md"""
We start with the following energy minimization problem 

$\min_{y\in \mathcal M}\frac{1}{2}\int_{0}^{1} \sigma(s) \langle \ddot y(s),\ddot y(s)\rangle ds$
where $\mathcal M := \{y\mid y\in H^2([0,1];\mathbb{R}^3),\dot y(s)\in \mathbb S^2  \, \mbox{on} \, [0,1] \}.$ The quantity $\overline \sigma > \sigma(s)\ge \underline \sigma>0$ is the flexural stiffness of the rod, and $\dot y$, $\ddot y$ are the derivatives of $y$ with respect to $s\in[0,1]$. Since $\dot y(s) \in \mathbb{S}^2$ for all $s\in [0,1]$ the rod is inextensible with fixed length 1.

In addition the following boundary conditions are imposed:

$y(0)=y_a \in \mathbb{R}^3, \, \dot y(0)=v_a\in \mathbb{S}^2, \;
y(1)=y_b \in \mathbb{R}^3, \, \dot y(1)=v_b\in \mathbb{S}^2.$

Introducing $v(s):=\dot y(s)$ we reformulate the problem as a mixed problem:

$\min_{(y,v)\in Y\times \mathcal V} \frac{1}{2}\int_{0}^{1} \sigma \langle \dot v,\dot v\rangle \, ds
\quad \mbox{ s.t. } \quad \dot y-v =0$
where 

$Y=\{y\in H^2([0,1];\mathbb{R}^3)\,\, : \,\, y(0)=y_a,\,\, y(1)=y_b  \},$ 

$\mathcal V=\{v\in H^1([0,1];\mathbb{S}^2)\,\, :\,\, v(0)=v_a,\,\, v(1)=v_b  \}.$

To derive equilibrium conditions for this problem we define the Lagrangian function

$L(y,v,\lambda) =\int_{0}^{1} \frac{1}{2}  \sigma \left\langle \dot v,\dot v \right\rangle+\lambda (\dot y-v)\, ds$

using a Lagrangian multiplier $\lambda \in \Lambda \coloneqq L_2([0,1];\mathbb R^3)$.

We obtain the following equilibrium conditions via setting the derivatives of the Lagrangian to zero:

$\int_0^1 \lambda (\dot{\phi_y})\, ds=0 \quad \forall \phi_y\in Y$

$\int_{0}^{1} \sigma \left\langle  \dot v,\dot{\phi_v}\right\rangle -  \lambda(\phi_v) ds=0 \quad \forall \phi_v\in T_v \mathcal V$

$\int_{0}^{1}  \phi_\lambda(\dot y-v) ds=0 \quad \forall \phi_\lambda\in \Lambda$

Hence, have to find a zero of the mapping

$F : Y \times \mathcal V \times \Lambda \to Y^* \times T^*\mathcal V\times \Lambda^*$

defined by the equilibrium conditions.

For brevity we set $\mathcal X=Y \times \mathcal V \times \Lambda$ and $x=(y,v,\lambda)$ and obtain a mapping $F:\mathcal X \to T^*\mathcal X$.
"""

# ╔═╡ 03baac67-02bd-432e-a21f-c31c809fbd5e
md"""
For our example we set $\sigma \equiv 1$ and 
"""

# ╔═╡ 12e32b83-ae65-406c-be51-3f21935eaae5
begin
	N=50

	S = Manifolds.Sphere(2)
	R3 = Manifolds.Euclidean(3)	
	powerS = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
	powerR3 = PowerManifold(R3, NestedPowerRepresentation(), N) # power manifold of R^3
	powerR3_λ = PowerManifold(R3, NestedPowerRepresentation(), N+1) # power manifold of R^3
	product = ProductManifold(powerR3, powerS, powerR3_λ) # product manifold
	
	start_interval = 0.0
	end_interval = 1.0

	h = (end_interval-start_interval)/(N+1) 

	Omega_y = range(; start=start_interval, stop = end_interval, length=N+2)[2:end-1]
	Omega_v = range(; start=start_interval, stop = end_interval, length=N+2)[2:end-1]
	Omega_λ = range(; start=start_interval, stop = end_interval, length=N+2)[1:end-1]
	
	y0 = [0,0,0] # startpoint of rod
	y1 = [0.8,0,0] # endpoint of rod

	v0 = 1/norm([1,0,2])*[1,0,2] # start direction of rod
	v1 = 1/norm([1,0,0.8])*[1,0,0.8] # end direction of rod
end;

# ╔═╡ b52ba66d-2fc2-4ff2-9e74-cd9cc87e1b65
md"""
As a starting point, we use
"""

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
begin
	function y(t)
		return [t*0.8, 0.1*t*(1-t), 0]
	end
	
	function v(t)
		return [sin(t*pi/2+pi/4), cos(t*pi/2+pi/4), 0]
	end

	function λ(t)
		return [0.1, 0.1, 0.1]
	end

	discretized_y = [y(Ωi) for Ωi in Omega_y]
	discretized_v = [v(Ωi) for Ωi in Omega_v]
	discretized_λ = [λ(Ωi) for Ωi in Omega_λ]

	disc_point = ArrayPartition(discretized_y, discretized_v, discretized_λ)
	
end;

# ╔═╡ 7741a74f-0a73-47c8-9202-b8789782eb7b
md"""
In order to apply Newton's method to find a zero of $F$, we need the linear mapping $Q_{F(x)}^*\circ F'(x)$ (cf. \ref{paper}). Since the sphere is an embedded submanifold of $\mathbb R^3$, we can use the formula 

$Q_{F(x)}^*\circ F'(x)\delta x\,\phi = F(x)(\overset{\rightarrow}{V}_x'(x)\delta x\,\phi) + F_{\mathbb R^9}'(x)\delta x\,\phi$

for $x\in \mathcal X$ and $\delta x, \, \phi \in T_x \mathcal X$, where $\overset{\rightarrow}{V}_x(\hat x) \in L(T_x \mathcal X, T_{\hat{x}}\mathcal X)$ is a vector transport and 

$F_{\mathbb R^9}'(x)\delta x^2\, \delta x^1 = \int_0^1 \delta \lambda(\dot \phi_y) + \sigma \langle \dot{\delta v}, \dot\phi_v \rangle  -\delta \lambda(\phi_v) + \phi_\lambda(\dot{\delta y})  - \phi_\lambda(\delta v) \, ds$

is the euclidean derivative of $F$.

The part, introduced by the connection is given by 

$F(x)(\overset{\rightarrow}{V}_x'(x)\delta x\,\phi) = \int_0^1 \langle \dot v, (P'(v)\delta v \, \phi_v)\dot{} \, \rangle - \lambda(P'(v)\delta v \, \phi_v) \, ds$

where $P(v) :\mathbb{R}^3 \to T_v\mathbb{S}^2$ denotes the orthogonal projection.
"""

# ╔═╡ 6c7f20ef-e7bf-47cf-9017-8056153b5e06
md"""
We define a structure that has to be filled for two purposes:
* Definition of an integrands and their derivatives
* Definition of a vector transport and its derivative
"""

# ╔═╡ bc449c2d-1f23-4c72-86ab-a46acbf64129
mutable struct DifferentiableMapping{M<:AbstractManifold, N<:AbstractManifold,F1<:Function,F2<:Function}
	domain::M
	precodomain::N
	value::F1
	derivative::F2
end

# ╔═╡ efc9897e-cd5b-43b3-ad96-62360ccec659
md"""
The following routines define a vector transport and its euclidean derivative. As seen above, they are needed to derive a covariant derivative of $F$.

As a vector transport we use the (pointwise) orthogonal projection onto the tangent spaces, i.e. for $p, q \in \mathbb S^2$ and $X \in T_p\mathbb S^2$ we set 

$\overset{\rightarrow}{V}_{p}(q)X = (I-q\cdot q^T)X \in T_q\mathbb S^2.$

The derivative of the vector transport is then given by 

$\left(\frac{d}{dq}\overset{\rightarrow}{V}_{p}(q)\big\vert_{q=p}\delta q\right)X = \left( - \delta q\cdot p^T - p\cdot \delta q^T\right)\cdot X.$

"""

# ╔═╡ 221887ad-1a2a-4003-aae0-85672fb20c9f
begin 
	
	function transport_by_proj(S, p, X, q)
		return X - q*(q'*X)
	end

	function transport_by_proj_prime(S, p, X, dq)
		return (- dq*p' - p*dq')*X
	end

	transport = DifferentiableMapping(S,S,transport_by_proj,transport_by_proj_prime)
end;

# ╔═╡ 524ac97b-d2df-46d6-b098-ded4db69e665
md""" The following routines define the integrand of $F$ and its euclidean derivative.
"""

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
begin
	Fy_at(Integrand, y, ydot, T, Tdot) = Tdot'*y.x[3] # y component of F
	Fv_at(Integrand, y, ydot, T, Tdot) = ydot.x[2]'*Tdot-T'*y.x[3] # v component of F
	Fλ_at(Integrand, y, ydot, T, Tdot) = (ydot.x[1]-y.x[2])'*T # λ component of F
	
	F_prime_yλ_at(Integrand,y,ydot,B,Bdot,T,Tdot) = Tdot'*B # derivative of Fy_at w.r.t. λ (others are zero)

	F_prime_vv_at(Integrand,y,ydot,B,Bdot,T,Tdot) = Bdot'*Tdot # derivative of Fv_at w.r.t. v (others are zero)
	
	F_prime_vλ_at(Integrand,y,ydot,B,Bdot,T,Tdot) = -T'*B # derivative of Fv_at w.r.t. λ (others are zero)

	
	integrand_vv = DifferentiableMapping(S,S,Fv_at,F_prime_vv_at)
	integrand_yλ = DifferentiableMapping(R3,R3,Fy_at,F_prime_yλ_at)
	integrand_vλ = DifferentiableMapping(R3,S,Fv_at,F_prime_vλ_at)
	integrandb_λ = DifferentiableMapping(R3,R3,Fλ_at,F_prime_vλ_at) # needed for the third component of the right hand side, derivative is not used (thus F_prime_vλ_at is a dummy)

end;

# ╔═╡ d69da8fa-fe17-4114-84c7-651aedbc756e
md"""
If no vector transport is needed, leave it away, then the identity transport is used as dummy
"""

# ╔═╡ e2f48dcc-5c23-453d-8ff3-eb425b7b67af
begin
	identity_transport(S, p, X, q) = X
	identity_transport_prime(S, p, X, dq) = 0.0*X
	
	id_transport = DifferentiableMapping(R3,R3,identity_transport,identity_transport_prime)
		
	function get_Jac!(eval,A,row_idx,degT,col_idx,degB,h,nCells,y,integrand)			ManoptExamples.get_Jac!(eval,A,row_idx,degT,col_idx,degB,h,nCells,y,integrand,id_transport)
	end
end;

# ╔═╡ db885ad3-f53d-4b56-9428-4f00f484f37d
md"""
`NewtonEquation`

In this example we implement a functor to compute the Newton matrix and the right hand side for the Newton equation \ref{paper}

$$Q^*_{F(x)}\circ F'(x)\delta x + F(x) = 0$$

by using the assembler provided in ManoptExamples.jl (cf. Referenz).
	
It returns the matrix and the right hand side in base representation.
Moreover, for the computation of the simplified Newton direction (which is necessary for affine covariant damping) a method for assembling the right hand side for the simplified Newton equation is provided.
	
"""

# ╔═╡ d09e5081-71b8-448f-ad83-cac312f8f17d
md"""
The assembly routines need a function for evaluating the test functions at the left and right quadrature point.
"""

# ╔═╡ c9e3bf29-85af-4f97-8308-333f1472355c
function evaluate(y, i, tloc)
	return ArrayPartition(
		(1.0-tloc)*y.x[1][i-1]+tloc*y.x[1][i],
		(1.0-tloc)*y.x[2][i-1]+tloc*y.x[2][i],
		y.x[3][i]
	)
end;

# ╔═╡ ea3c49be-896c-4470-b6fe-587ebe009eab
begin
struct NewtonEquation{Fy, Fv, Fλ, Fbλ, T, Om, NM, Nrhs}
	integrand_y::Fy
	integrand_v::Fv
	integrand_λ::Fλ
	integrand_bλ::Fbλ
	vectortransport::T
	omega_y::Om
	omega_v::Om
	omega_λ::Om
	A13::NM
	A22::NM
	A23::NM
	A::NM
	b1::Nrhs
	b2::Nrhs
	b3::Nrhs
	b::Nrhs
end

function NewtonEquation(M, inty, intv, intλ, intbλ, VT, interval_y, interval_v, interval_λ)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))

	# non-zero blocks of the Newton matrix
	A13 = spzeros(n1,n3)
	A22 = spzeros(n2,n2)
	A23 = spzeros(n2,n3)
	
	A = spzeros(n1+n2+n3, n1+n2+n3)
	
	b1 = zeros(n1)
	b2 = zeros(n2)
	b3 = zeros(n3)
	b = zeros(n1+n2+n3)
	
	return NewtonEquation{typeof(inty), typeof(intv), typeof(intλ), typeof(intbλ), typeof(VT), typeof(interval_y), typeof(A13), typeof(b1)}(inty, intv, intλ, intbλ, VT, interval_y, interval_v, interval_λ, A13, A22, A23, A, b1, b2, b3, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	ne.A13 .= spzeros(n1,n3)
	ne.A22 .= spzeros(n2,n2)
	ne.A23 .= spzeros(n2,n3)
	
	ne.b1 .= zeros(n1)
	ne.b2 .= zeros(n2)
	ne.b3 .= zeros(n3)
	
	Op_y = OffsetArray([y0, p[M, 1]..., y1], 0:(length(ne.omega_y)+1))
	Op_v = OffsetArray([v0, p[M, 2]..., v1], 0:(length(ne.omega_v)+1))
	Op_λ = OffsetArray(p[M, 3], 1:length(ne.omega_λ))
	Op = ArrayPartition(Op_y,Op_v,Op_λ);
	
	println("Assemble:")
	nCells = length(ne.omega_λ)
	
	ManoptExamples.get_Jac!(evaluate,ne.A22,2,1,2,1,h,nCells,Op,ne.integrand_v,ne.vectortransport) # assemble (2,2)-block using the connection
	
    get_Jac!(evaluate,ne.A13,1,1,3,0,h,nCells,Op,ne.integrand_y) # assemble (1,3)-block without connection

	get_Jac!(evaluate,ne.A23,2,1,3,0,h,nCells,Op,ne.integrand_λ) # assemble (2,3)-block without connection

    ManoptExamples.get_rhs_row!(evaluate,ne.b1,1,1,h,nCells,Op,ne.integrand_y) 
	ManoptExamples.get_rhs_row!(evaluate,ne.b2,2,1,h,nCells,Op,ne.integrand_v)
	ManoptExamples.get_rhs_row!(evaluate,ne.b3,3,0,h,nCells,Op,ne.integrand_bλ)
	
	ne.A .= vcat(hcat(spzeros(n1,n1) , spzeros(n1,n2) , ne.A13), 
			  hcat(spzeros(n2,n1), ne.A22 , ne.A23), 
			  hcat(ne.A13', ne.A23', spzeros(n3,n3)))
	ne.b .= vcat(ne.b1, ne.b2, ne.b3)
	return
end


function (ne::NewtonEquation)(M, VB, p, p_trial)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	btrial_y = zeros(n1)
	btrial_v = zeros(n2)
	btrial_λ = zeros(n3)
	
	Op_y = OffsetArray([y0, p[M, 1]..., y1], 0:(length(ne.omega_y)+1))
	Op_v = OffsetArray([v0, p[M, 2]..., v1], 0:(length(ne.omega_v)+1))
	Op_λ = OffsetArray(p[M, 3], 1:length(ne.omega_λ))
	Op = ArrayPartition(Op_y,Op_v,Op_λ);

	
	Optrial_y = OffsetArray([y0, p_trial[M,1]..., y1], 0:(length(ne.omega_y)+1))
	Optrial_v = OffsetArray([v0, p_trial[M,2]..., v1], 0:(length(ne.omega_v)+1))
	Optrial_λ = OffsetArray(p_trial[M,3], 1:length(ne.omega_λ))
	Optrial = ArrayPartition(Optrial_y,Optrial_v,Optrial_λ);

	nCells = length(ne.omega_λ)

	ManoptExamples.get_rhs_simplified!(evaluate, btrial_y,1,1,h,nCells,Op,Optrial,ne.integrand_y, id_transport)
	ManoptExamples.get_rhs_simplified!(evaluate,btrial_v,2,1,h,nCells,Op,Optrial,ne.integrand_v,ne.vectortransport)
	ManoptExamples.get_rhs_simplified!(evaluate,btrial_λ,3,0,h,nCells,Op,Optrial,ne.integrand_bλ, id_transport)

	return vcat(btrial_y,btrial_v, btrial_λ)
end
end;

# ╔═╡ a972f348-6c12-442a-b670-f14f20fd5c77
md"""
We compute the Newton direction $\delta x$ by solving the linear system given by the base representation of the Newton equation directly and return the Newton direction in vector representation:
"""

# ╔═╡ 5fc9e70a-ff2d-44fa-8e0f-f2d235d462f3
function solve_in_basis_repr(problem, newtonstate) 
	X = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	return get_vector(problem.manifold, newtonstate.p, X, DefaultOrthogonalBasis())
end;

# ╔═╡ 3988305c-3849-4fd2-8041-172aa08ecede
begin
	# adjust norms for computation of damping factors and stopping criterion
	pr_inv = Manifolds.InverseProductRetraction(LogarithmicInverseRetraction(), ProjectionInverseRetraction(), LogarithmicInverseRetraction())
	rec = RecordChange(product;
    inverse_retraction_method=pr_inv);
end;

# ╔═╡ d903c84a-45f6-4e09-9ec2-88e248531fec
	begin
	y_0 = copy(product, disc_point)
	
	NE = NewtonEquation(product, integrand_yλ, integrand_vv, integrand_vλ, integrandb_λ, transport, Omega_y, Omega_v, Omega_λ)
		
	st_res = vectorbundle_newton(product, TangentBundle(product), NE, y_0; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(product,1e-12; outer_norm=Inf, inverse_retraction_method=pr_inv)),
	retraction_method=ProductRetraction(ExponentialRetraction(), ProjectionRetraction(), ExponentialRetraction()),
	stepsize=Manopt.AffineCovariantStepsize(product, theta_des=0.5, outer_norm=Inf),
	#stepsize=ConstantLength(product, 1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, rec => :Change, :Stepsize],
	return_state=true
)
end

# ╔═╡ 5cd1a561-291a-475e-aa34-572b7e2a6c03
md"""
We extract the recorded values
"""

# ╔═╡ abe5c5f3-4a28-425c-afde-64b645f3a9d9
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	stepsizes = get_record(st_res, :Iteration, :Stepsize)
	p_res = get_solver_result(st_res)
end;

# ╔═╡ dc20ac3f-0520-4cdb-8b6f-fc951c090679
md"""
and plot the result, where we measure the norms of the Newton direction in each iteration,
"""

# ╔═╡ 6451f8c5-7b4f-4792-87fd-9ed2635efa88
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ c3fcc368-bdc8-4a55-96a9-cd9ab471e6f6
md"""
the stepsizes computed via the affine covariant damping strategy,
"""

# ╔═╡ e429bda6-61de-4277-bd95-db0ce25e0144
begin
	f_st = Figure(;)
	
    row_st, col_st = fldmod1(1, 2)
	
	Axis(f_st[row_st, col_st], title = string("Stepsizes"), xminorgridvisible = true, xticks = (1:length(stepsizes)), xlabel = "Iteration", ylabel = "α")
    scatterlines!(stepsizes[1:end-1], color = :blue)
	f_st
end

# ╔═╡ 21d06258-7541-44a6-a5df-a1c0fa311afa
md"""
and the resulting rod (orange). The starting rod (red) is plotted as well.
"""

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
begin
fig = Figure(size = (1000, 500))
ax = Axis3(fig[1, 1], aspect = :data, viewmode = :fitzoom, azimuth=-3pi/4 + 0.3, elevation=pi/8 + 0.15) 
	#xticklabelsvisible=false, yticklabelsvisible=false, zticklabelsvisible=false, xlabelvisible=false, ylabelvisible=false, zlabelvisible=false)
#ax = Axis3(fig[1, 2], aspect = :equal)


    π1(x) = x[1]
    π2(x) = x[2]
    π3(x) = x[3]

	scatter!(ax, π1.(p_res[product, 1]), 0.3 .+ 0.0.*π2.(p_res[product, 1]), π3.(p_res[product, 1]); markersize =8, color = RGBAf(0.9, 0.7, 0.5, 0.5))

	scatter!(ax, π1.(discretized_y), 0.3 .+ 0.0.*π2.(discretized_y), π3.(discretized_y); markersize =8, color = RGBAf(0.8, 0.5, 0.5, 0.5))
	
	scatter!(ax, π1.(p_res[product, 1]), π2.(p_res[product, 1]), π3.(p_res[product, 1]); markersize =8, color=:orange)
	
	scatter!(ax, π1.([y0, y1]), π2.([y0, y1]), π3.([y0, y1]); markersize =8, color=:red)
	scatter!(ax, π1.(discretized_y), π2.(discretized_y), π3.(discretized_y); markersize =8, color=:red)

	fig
end

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╟─116b717d-f9cf-44a0-be18-4950ea2f019a
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╟─8153941a-39a0-4d91-9796-35cbd4c547e3
# ╟─03baac67-02bd-432e-a21f-c31c809fbd5e
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╟─b52ba66d-2fc2-4ff2-9e74-cd9cc87e1b65
# ╠═29043ca3-afe0-4280-a76a-7c160a117fdf
# ╟─7741a74f-0a73-47c8-9202-b8789782eb7b
# ╟─6c7f20ef-e7bf-47cf-9017-8056153b5e06
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╟─efc9897e-cd5b-43b3-ad96-62360ccec659
# ╠═221887ad-1a2a-4003-aae0-85672fb20c9f
# ╟─524ac97b-d2df-46d6-b098-ded4db69e665
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╟─d69da8fa-fe17-4114-84c7-651aedbc756e
# ╠═e2f48dcc-5c23-453d-8ff3-eb425b7b67af
# ╟─db885ad3-f53d-4b56-9428-4f00f484f37d
# ╟─d09e5081-71b8-448f-ad83-cac312f8f17d
# ╠═c9e3bf29-85af-4f97-8308-333f1472355c
# ╠═ea3c49be-896c-4470-b6fe-587ebe009eab
# ╟─a972f348-6c12-442a-b670-f14f20fd5c77
# ╠═5fc9e70a-ff2d-44fa-8e0f-f2d235d462f3
# ╠═d903c84a-45f6-4e09-9ec2-88e248531fec
# ╠═3988305c-3849-4fd2-8041-172aa08ecede
# ╟─5cd1a561-291a-475e-aa34-572b7e2a6c03
# ╠═abe5c5f3-4a28-425c-afde-64b645f3a9d9
# ╟─dc20ac3f-0520-4cdb-8b6f-fc951c090679
# ╟─6451f8c5-7b4f-4792-87fd-9ed2635efa88
# ╟─c3fcc368-bdc8-4a55-96a9-cd9ab471e6f6
# ╟─e429bda6-61de-4277-bd95-db0ce25e0144
# ╟─21d06258-7541-44a6-a5df-a1c0fa311afa
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
