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
where $\mathcal M := \{y\mid y\in H^2([0,1];\mathbb{R}^3),\dot y(s)\in \mathbb S^2  \, \mbox{on} \, [0,1] \}.$ The quantity $\overline \sigma > \sigma(s)\ge \underline \sigma>0$ is the flexural stiffness of the rod, and $\dot y$, $\ddot y$ are the derivatives of $y$ with respect to $s\in[0,1]$. 

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

$\int_0^1 \lambda (\dot{\delta  y})\, ds=0 \quad \forall \delta y\in Y$

$\int_{0}^{1} \sigma \left\langle  \dot v,\dot{\delta v}\right\rangle -  \lambda(\delta v) ds=0 \quad \forall \delta v\in T_v \mathcal V$

$\int_{0}^{1}  \delta \lambda(\dot y-v) ds=0 \quad \forall \delta \lambda\in \Lambda$

Hence, have to find a zero of the mapping

$F : Y \times \mathcal V \times \Lambda \to Y^* \times T^*\mathcal V\times \Lambda^*$

defined by the equilibrium conditions.
"""

# ╔═╡ 03baac67-02bd-432e-a21f-c31c809fbd5e
md"""
For our example we set
"""

# ╔═╡ 12e32b83-ae65-406c-be51-3f21935eaae5
begin
	N=100
	
	st1 = 0.0
	halt1 = 1.0

	windscale=1

	h = (halt1-st1)/(N+1)

	Omega1 = range(; start=st1, stop = halt1, length=N+2)[2:end-1]
	Omega2 = range(; start=st1, stop = halt1, length=N+2)[2:end-1]
	Omega3 = range(; start=st1, stop = halt1, length=N+2)[1:end-1]
	
	y01 = [0,0,0] # startpoint of rod
	yT1 = [0.8,0,0] # endpoint of rod

	y02 = 1/norm([1,0,0.5])*[1,0,0.5] # start direction of rod
	yT2 = 1/norm([1,0,0.5])*[1,0,0.5]# end direction of rod
end;

# ╔═╡ 29043ca3-afe0-4280-a76a-7c160a117fdf
function y1(t)
	return [t*0.8, 0.1*t*(1-t), 0]
end;

# ╔═╡ 7f6c588b-e64d-471f-9259-f3e3aeeb193a
function y2(t)
	return [sin(t*pi/2+pi/4), cos(t*pi/2+pi/4), 0]
end;

# ╔═╡ 5e2e2280-fe0d-443b-8824-101a138a86a0
function y3(t)
	return [0.1, 0.1, 0.1]
end;

# ╔═╡ 5c0980c5-284e-4406-bab8-9b9aff9391ba
discretized_y1 = [y1(Ωi) for Ωi in Omega1];

# ╔═╡ 8cc3124c-dbb8-4286-aee3-3984d87868c1
discretized_y2 = [y2(Ωi) for Ωi in Omega2];

# ╔═╡ 47d26c17-d422-4e6f-a5fc-7f11a2bcba0c
discretized_y3 = [y3(Ωi) for Ωi in Omega3];

# ╔═╡ ba3051a0-078a-49ff-85b9-441eef4cb9fc
disc_y = ArrayPartition(discretized_y1, discretized_y2,discretized_y3)

# ╔═╡ bc449c2d-1f23-4c72-86ab-a46acbf64129
"""
Such a structure has to be filled for two purposes:
* Definition of an integrand and its derivative
* Definition of a vector transport and its derivative
"""
mutable struct DifferentiableMapping{M<:AbstractManifold, N<:AbstractManifold,F1<:Function,F2<:Function,T}
	domain::M
	precodomain::N
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

# ╔═╡ 48834792-fff9-4e96-803a-b4d07e714797
function zerotrans_prime(S, p, X, dq)
	return 0.0*X
end

# ╔═╡ 758d34df-96ad-4295-a3a1-46acd65b26e7
function identitytrans(S, p, X, q)
	return X
end

# ╔═╡ 229fa902-e125-429a-852d-0668f64c7640
function F2_at(Integrand, y, ydot, T, Tdot)
	  return ydot.x[2]'*Tdot-T'*y.x[3]
end

# ╔═╡ 9bcaa5d0-d8de-4746-8c85-0fe24a4825e2
function F3_at(Integrand, y, ydot, T, Tdot)
	  return (ydot.x[1]-y.x[2])'*T
end

# ╔═╡ 1c284f9d-f34e-435b-976d-61aaa0975fe5
function F_prime22_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return Bdot'*Tdot
end

# ╔═╡ 86fc6357-1106-48f9-8efe-fda152caf990
function F_prime13_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return Tdot'*B
end

# ╔═╡ 03c147e6-843f-47ae-924e-86ed0260cd8e
function F_prime23_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return -T'*B
end

# ╔═╡ 808db8aa-64f7-4b36-8c6c-929ba4fa22db
"""
Force field w and its derivative. A scaling parameter is also employed.
"""
function w(p, c)
	#return c*p[3]*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
	return [0.0,0.0,0.0] #c*[-p[2]/(p[1]^2+p[2]^2), p[1]/(p[1]^2+p[2]^2), 0.0] 
end

# ╔═╡ 56ae7f53-061e-4414-90ad-85c7a12d51e2
function F1_at(Integrand, y, ydot, T, Tdot)
	  return w(y.x[1],Integrand.scaling)'*T+Tdot'*y.x[3]
end

# ╔═╡ 288b9637-0500-40b8-a1f9-90cb9591402b
function w_prime(p, c)
	nenner = p[1]^2+p[2]^2
	#return c*[p[3]*2*p[1]*p[2]/nenner^2 p[3]*(-1.0/(nenner)+2.0*p[2]^2/nenner^2) -p[2]/nenner; p[3]*(1.0/nenner-2.0*p[1]^2/(nenner^2)) p[3]*(-2.0*p[1]*p[2]/(nenner^2)) p[1]/(nenner); 0.0 0.0 0.0]
	return zeros(3,3) #c*[2*p[1]*p[2]/nenner^2 (-1.0/(nenner)+2.0*p[2]^2/nenner^2) 0.0; (1.0/nenner-2.0*p[1]^2/(nenner^2)) (-2.0*p[1]*p[2]/(nenner^2)) 0.0; 0.0 0.0 0.0]
end

# ╔═╡ ac04e6ec-61c2-475f-bb2f-83755c04bd72
function F_prime11_at(Integrand,y,ydot,B,Bdot,T,Tdot)
	return (w_prime(y.x[1],Integrand.scaling)*B)'*T
end

# ╔═╡ 684508bd-4525-418b-b89a-85d56c01b188
begin
S = Manifolds.Sphere(2)
R3 = Manifolds.Euclidean(3)	
powerS = PowerManifold(S, NestedPowerRepresentation(), N)
powerR3 = PowerManifold(R3, NestedPowerRepresentation(), N)
powerR3lambda = PowerManifold(R3, NestedPowerRepresentation(), N+1)
product = ProductManifold(powerR3, powerS, powerR3lambda)

integrand1=DifferentiableMapping(R3,R3,F1_at,F_prime11_at,windscale)
integrand2=DifferentiableMapping(S,S,F2_at,F_prime22_at,windscale)
integrand3=DifferentiableMapping(R3,R3,F3_at,F_prime11_at,windscale)
integrand13=DifferentiableMapping(R3,R3,F1_at,F_prime13_at,windscale)
integrand23=DifferentiableMapping(R3,S,F2_at,F_prime23_at,windscale)
	
transport=DifferentiableMapping(S,S,transport_by_proj,transport_by_proj_prime,nothing)
end;

# ╔═╡ 14d42ecb-6563-4d62-94ce-a36b73ed9a78
zerotransport=DifferentiableMapping(R3,R3,identitytrans,zerotrans_prime,nothing)


# ╔═╡ e2f48dcc-5c23-453d-8ff3-eb425b7b67af
"""
If no vector transport is needed, leave it away, then a zero dummy transport is used
"""
function get_Jac!(eval,A,row_idx,degT,col_idx,degB,h,nCells,y,integrand)
	ManoptExamples.get_Jac!(eval,A,row_idx,degT,col_idx,degB,h,nCells,y,integrand,zerotransport)
end

# ╔═╡ cab1527e-b7b9-4e13-8483-cba8b95c24da
function evaluate(y, i, tloc)
	return ArrayPartition(
		(1.0-tloc)*y.x[1][i-1]+tloc*y.x[1][i],
		(1.0-tloc)*y.x[2][i-1]+tloc*y.x[2][i],
		y.x[3][i]
	)
end

# ╔═╡ ea3c49be-896c-4470-b6fe-587ebe009eab
begin
struct NewtonEquation{F1, F2, F3, F13, F23, T, Om, NM, Nrhs}
	integrand1::F1
	integrand2::F2
	integrand3::F3
	integrand13::F13
	integrand23::F23
	transport::T
	Omega1::Om
	Omega2::Om
	Omega3::Om
	A11::NM
	A12::NM
	A13::NM
	A22::NM
	A23::NM
	A33::NM
	A::NM
	b1::Nrhs
	b2::Nrhs
	b3::Nrhs
	b::Nrhs
end

function NewtonEquation(M, int1, int2, int3, int13, int23, VT, interval1, interval2, interval3)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	A11 = spzeros(n1,n1)
	A12 = spzeros(n1,n2)
	A22 = spzeros(n2,n2)
	A13 = spzeros(n1,n3)
	A23 = spzeros(n2,n3)
	A33 = spzeros(n3,n3)
	A = spzeros(n1+n2+n3, n1+n2+n3)
	
	b1 = zeros(n1)
	b2 = zeros(n2)
	b3 = zeros(n3)
	b = zeros(n1+n2+n3)
	return NewtonEquation{typeof(int1), typeof(int2), typeof(int3), typeof(int13), typeof(int23), typeof(VT), typeof(interval1), typeof(A11), typeof(b1)}(int1, int2, int3, int13, int23, VT, interval1, interval2, interval3, A11, A12, A13, A22, A23, A33, A, b1, b2, b3, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	ne.A11 .= spzeros(n1,n1)
	ne.A13 .= spzeros(n1,n3)
	ne.A22 .= spzeros(n2,n2)
	ne.A23 .= spzeros(n2,n3)
	ne.A33 .= spzeros(n3,n3)
	
	ne.b1 .= zeros(n1)
	ne.b2 .= zeros(n2)
	ne.b3 .= zeros(n3)
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(Omega1)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(Omega2)+1))
	Oy3 = OffsetArray(p[M, 3], 1:length(Omega3))
	Oy = ArrayPartition(Oy1,Oy2,Oy3);
	
	println("Assemble:")
	nCells = length(ne.Omega3)
   	get_Jac!(evaluate,ne.A11,1,1,1,1,h,nCells,Oy,ne.integrand1)
	ManoptExamples.get_Jac!(evaluate,ne.A22,2,1,2,1,h,nCells,Oy,ne.integrand2,ne.transport)
    get_Jac!(evaluate,ne.A13,1,1,3,0,h,nCells,Oy,ne.integrand13)
	get_Jac!(evaluate,ne.A23,2,1,3,0,h,nCells,Oy,ne.integrand23)
	# Ac12 = 0, Ac33 = 0 
    ManoptExamples.get_rhs_row!(evaluate,ne.b1,1,1,h,nCells,Oy,ne.integrand1)
	ManoptExamples.get_rhs_row!(evaluate,ne.b2,2,1,h,nCells,Oy,ne.integrand2)
	ManoptExamples.get_rhs_row!(evaluate,ne.b3,3,0,h,nCells,Oy,ne.integrand3)
	
	ne.A .= vcat(hcat(ne.A11 , ne.A12 , ne.A13), 
			  hcat(ne.A12', ne.A22 , ne.A23), 
			  hcat(ne.A13', ne.A23', ne.A33))
	ne.b .= vcat(ne.b1, ne.b2, ne.b3)
	return
end


function (ne::NewtonEquation)(M, VB, p, p_trial)
	n1 = Int(manifold_dimension(submanifold(M, 1)))
	n2 = Int(manifold_dimension(submanifold(M, 2)))
	n3 = Int(manifold_dimension(submanifold(M, 3)))
	
	bctrial1=zeros(n1)
	bctrial2=zeros(n2)
	bctrial3=zeros(n3)
	
	Oy1 = OffsetArray([y01, p[M, 1]..., yT1], 0:(length(ne.Omega1)+1))
	Oy2 = OffsetArray([y02, p[M, 2]..., yT2], 0:(length(ne.Omega2)+1))
	Oy3 = OffsetArray(p[M, 3], 1:length(ne.Omega3))
	Oy = ArrayPartition(Oy1,Oy2,Oy3);

	
	Oytrial1 = OffsetArray([y01, p_trial[M,1]..., yT1], 0:(length(ne.Omega1)+1))
	Oytrial2 = OffsetArray([y02, p_trial[M,2]..., yT2], 0:(length(ne.Omega2)+1))
	Oytrial3 = OffsetArray(p_trial[M,3], 1:length(ne.Omega3))
	Oytrial = ArrayPartition(Oytrial1,Oytrial2,Oytrial3);

	nCells = length(ne.Omega3)

	ManoptExamples.get_rhs_simplified!(bctrial1,1,1,h,nCells,Oy,Oytrial,ne.integrand1, zerotransport)
	ManoptExamples.get_rhs_simplified!(bctrial2,2,1,h,nCells,Oy,Oytrial,ne.integrand2,ne.transport)
	ManoptExamples.get_rhs_simplified!(bctrial3,3,0,h,nCells,Oy,Oytrial,ne.integrand3, zerotransport)

	return vcat(bctrial1,bctrial2, bctrial3)
end
end;

# ╔═╡ 5fc9e70a-ff2d-44fa-8e0f-f2d235d462f3
"""
	Computes the Newton direction by solving the linear system given by the base representation of the Newton equation directly and returns the Newton direction in vector representation
"""
function solve_in_basis_repr(problem, newtonstate) 
	X = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	return get_vector(problem.manifold, newtonstate.p, X, DefaultOrthogonalBasis())
end

# ╔═╡ d903c84a-45f6-4e09-9ec2-88e248531fec
	begin
	y_0 = copy(product, disc_y)
	
	NE = NewtonEquation(product, integrand1, integrand2, integrand3, integrand13, integrand23, transport, Omega1, Omega2, Omega3)
		
	st_res = vectorbundle_newton(product, TangentBundle(product), NE, y_0; sub_problem=solve_in_basis_repr, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(product,1e-12; outer_norm=Inf)),
	#retraction_method=ProjectionRetraction(),
	#stepsize=Manopt.AffineCovariantStepsize(product, theta_des=0.5),
	#stepsize=ConstantLength(power, 1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, :Change],
	return_state=true
)
end

# ╔═╡ abe5c5f3-4a28-425c-afde-64b645f3a9d9
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ 6451f8c5-7b4f-4792-87fd-9ed2635efa88
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change, color = :blue)
	f
end

# ╔═╡ b0b8e87f-da09-4500-8aa9-e35934f7ef54
p_res = get_solver_result(st_res);

# ╔═╡ 52b11216-16d5-412c-9dc5-a7722ae19339
p_res[product,1]

# ╔═╡ 6f6eb0f9-21af-481a-a2ae-020a0ff305bf
begin
fig = Figure(size = (1000, 500))
ax = Axis3(fig[1, 2], aspect = :data, viewmode = :fitzoom, azimuth=-3pi/4 + 0.3, elevation=pi/8 + 0.15) 
	#xticklabelsvisible=false, yticklabelsvisible=false, zticklabelsvisible=false, xlabelvisible=false, ylabelvisible=false, zlabelvisible=false)
#ax = Axis3(fig[1, 2], aspect = :equal)


    π1(x) = 1.0*x[1]
    π2(x) = 1.0*x[2]
    π3(x) = 1.0*x[3]
	#scatter!(ax, π1.(p_res[product, 1]), π2.(p_res[product, 1]), -0.1.+ 0.0.*π3.(p_res[product, 1]); markersize =8, color = RGBAf(0.9, 0.7, 0.5, 0.5))

	scatter!(ax, π1.(p_res[product, 1]), 0.3 .+ 0.0.*π2.(p_res[product, 1]), π3.(p_res[product, 1]); markersize =8, color = RGBAf(0.9, 0.7, 0.5, 0.5))
	
	#scatter!(ax, π1.(discretized_y1), π2.(discretized_y1), -0.1.+ 0.0.*π3.(discretized_y1); markersize =8, color = RGBAf(0.8, 0.5, 0.5, 0.5))

	scatter!(ax, π1.(discretized_y1), 0.3 .+ 0.0.*π2.(discretized_y1), π3.(discretized_y1); markersize =8, color = RGBAf(0.8, 0.5, 0.5, 0.5))
	
	scatter!(ax, π1.(p_res[product, 1]), π2.(p_res[product, 1]), π3.(p_res[product, 1]); markersize =8, color=:orange)

	#scatter!(ax, π1.(p_res[product, 2]), π2.(p_res[product, 2]), π3.(p_res[product, 2]); markersize =8, color=:blue)
	#scatter!(ax, π1.(y_0), π2.(y_0), π3.(y_0); markersize =8, color=:blue)
	scatter!(ax, π1.([y01, yT1]), π2.([y01, yT1]), π3.([y01, yT1]); markersize =8, color=:red)
	scatter!(ax, π1.(discretized_y1), π2.(discretized_y1), π3.(discretized_y1); markersize =8, color=:red)
	#arrows!(ax, π1.(p_res), π2.(p_res), π3.(p_res), π1.(ws), π2.(ws), π3.(ws); color=:green, linewidth=0.01, arrowsize=Vec3f(0.03, 0.03, 0.13), transparency=true, lengthscale=0.15)
	fig

	#save("rod.png", fig, resolution=(1500, 800))
end

# ╔═╡ Cell order:
# ╠═c9994bc4-b7bb-11ef-3430-8976c5eabdeb
# ╟─116b717d-f9cf-44a0-be18-4950ea2f019a
# ╠═9fb54416-3909-49c4-b1bf-cc868e580652
# ╟─8153941a-39a0-4d91-9796-35cbd4c547e3
# ╟─03baac67-02bd-432e-a21f-c31c809fbd5e
# ╠═12e32b83-ae65-406c-be51-3f21935eaae5
# ╠═29043ca3-afe0-4280-a76a-7c160a117fdf
# ╠═7f6c588b-e64d-471f-9259-f3e3aeeb193a
# ╠═5e2e2280-fe0d-443b-8824-101a138a86a0
# ╠═5c0980c5-284e-4406-bab8-9b9aff9391ba
# ╠═8cc3124c-dbb8-4286-aee3-3984d87868c1
# ╠═47d26c17-d422-4e6f-a5fc-7f11a2bcba0c
# ╠═ba3051a0-078a-49ff-85b9-441eef4cb9fc
# ╠═bc449c2d-1f23-4c72-86ab-a46acbf64129
# ╠═50a51e47-b6b1-4e43-b4b9-aad23f6ec390
# ╠═9cdd4289-c49d-4733-8487-f471e38fc402
# ╠═48834792-fff9-4e96-803a-b4d07e714797
# ╠═758d34df-96ad-4295-a3a1-46acd65b26e7
# ╠═14d42ecb-6563-4d62-94ce-a36b73ed9a78
# ╠═56ae7f53-061e-4414-90ad-85c7a12d51e2
# ╠═229fa902-e125-429a-852d-0668f64c7640
# ╠═9bcaa5d0-d8de-4746-8c85-0fe24a4825e2
# ╠═ac04e6ec-61c2-475f-bb2f-83755c04bd72
# ╠═1c284f9d-f34e-435b-976d-61aaa0975fe5
# ╠═86fc6357-1106-48f9-8efe-fda152caf990
# ╠═03c147e6-843f-47ae-924e-86ed0260cd8e
# ╠═e2f48dcc-5c23-453d-8ff3-eb425b7b67af
# ╠═684508bd-4525-418b-b89a-85d56c01b188
# ╠═808db8aa-64f7-4b36-8c6c-929ba4fa22db
# ╠═288b9637-0500-40b8-a1f9-90cb9591402b
# ╠═cab1527e-b7b9-4e13-8483-cba8b95c24da
# ╠═ea3c49be-896c-4470-b6fe-587ebe009eab
# ╠═5fc9e70a-ff2d-44fa-8e0f-f2d235d462f3
# ╠═d903c84a-45f6-4e09-9ec2-88e248531fec
# ╠═abe5c5f3-4a28-425c-afde-64b645f3a9d9
# ╠═6451f8c5-7b4f-4792-87fd-9ed2635efa88
# ╠═b0b8e87f-da09-4500-8aa9-e35934f7ef54
# ╠═52b11216-16d5-412c-9dc5-a7722ae19339
# ╠═6f6eb0f9-21af-481a-a2ae-020a0ff305bf
