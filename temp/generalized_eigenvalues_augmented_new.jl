### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 3e19b659-8855-4020-9173-fe4a1ffbbc08
using Pkg; Pkg.activate();

# ╔═╡ eb35384f-0822-4953-8d34-4ed5a0df71c5
begin
	using LinearAlgebra
	using Manopt
	using ManoptExamples
	using Manifolds
	using RecursiveArrayTools
	using WGLMakie, Makie, GeometryTypes, Colors
	using DataFrames, CSV
end;

# ╔═╡ b16e5880-552b-4217-9d94-46fdfa7294c3
md"""
In this example we compute a solution of the generalized eigenvalue problem by applying Newton's method on vector bundles which was introduced in \ref{paper}. This example reproduces the results from \ref{paper}.
"""

# ╔═╡ fca6fcbc-d85c-44c5-831c-bdcd229c40d7
md"""
Let $$X = Y = \mathbb{R}^n$$ equipped with the euclidean inner product and consider two matrices $$A, \, B \in \mathbb R^{n\times n}$$. A generalized eigenvalue problem consists of finding nonzero vectors $$x\in \mathbb R^n$$ and numbers $$\mu \in \mathbb R$$ such that 

$$Ax = \mu Bx \quad \Leftrightarrow \quad Ax \in \mathrm{span}(Bx).$$ 

Consider a vector bundle $$\mathcal E$$ with base manifold $$X$$ and fibers $$E_x := \mathrm{span}(Bx)^\perp \subset Y$$, where $$\perp$$ denotes the orthogonal complement. 
Then, by taking the dual spaces $$E_x^* := L(E_x, \mathbb R)$$ as fibers, we can define a corresponding dual vector bundle $$\mathcal E^*$$. Let $\mathbb{S}^{n-1} := \{ x \in X \mid \|x\|_2 = 1\}$ be the unit sphere in $$X$$ where $$\|\cdot\|_2$$ denotes the euclidean norm. Then a zero $x\in X$ of the mapping

$$F : \mathbb{S}^X \to \mathcal E^*, \; x \mapsto \langle Ax, \cdot \rangle_2$$

satisfies

$$F(x) = 0_x^* \in E_x^* \Leftrightarrow \langle Ax, \delta y\rangle_2 = 0 \; \forall \delta y \in E_x \Leftrightarrow Ax \in (E_x)^\perp \Leftrightarrow Ax \in \mathrm{span}(Bx)$$

und thus yields a generalized eigenvector.
"""

# ╔═╡ 932d4733-4609-4612-9ade-128742ebc845
md"""
For our example we set
"""

# ╔═╡ 587cf35f-882f-4e5a-8527-a53ba7c5644f
begin
    n = 101
	M = Manifolds.Sphere(n-1)
	
    A = Matrix{Float64}(I, n, n)
    for i in 1:n
        A[i, i] = i
    end
	
    for i in 1:n
		for j in 1:n
		if i > j
			A[i,j] = 1.0
			end
			if i < j
			A[i,j] = 1.0
		end
	end
	end
	B = zeros(n,n)
	for i in 1:n
		for j in 1:n
			if i > j
				B[i,j] = -1.0
			end
		end
	end
end;

# ╔═╡ a0fe5b93-98ac-4ee9-bee1-2e53238c7b34
md"""
For the application of Newton's method to find a zeros of $F$, we need to derive a connection map $Q^*$ on $\mathcal E^*$ to define the Newton equation

$$Q_{F(x)}^*\circ F'(x)\delta x + F(x) = 0_x^*, \quad \delta x \in T_x\mathbb S^{n-1}$$

which is an equation in the dual space $E_x^*$. As we have seen in the paper, we can use the formula

$$Q^*_{F(x)}\circ F'(x)\delta x \, v = F_Y'(x)\delta x \, v + F_Y(x)P'(x)\delta x \, v, \quad v \in E_x,$$

where $F_Y : \mathbb{S}^X \to Y^*, \; x\mapsto \langle Ax, \cdot \rangle_2$ is an extension of $F$ and $P(x) : Y \to E_x$ is the orthogonal projection onto the fiber $E_x = \mathrm{span}(Bx)^\perp \subset Y$.

The Euclidean derivative of $F_Y$ is in our example given by

$$F_Y'(x)\delta x = \langle A\delta x, \cdot \rangle_2.$$

"""

# ╔═╡ 5a68a8c6-a51f-4a31-b24b-198683d49517
md"""
We define a structure that has to be filled for two purposes:
* Definition of $F_Y$ and its derivative $F'_Y$
* Definition of the orthogonal projection onto $E_x$ and its derivative
"""

# ╔═╡ 623be37d-3bf6-487d-91d4-af67fd5a9654
mutable struct DifferentiableMapping{F1<:Function,F2<:Function}
	value::F1
	derivative::F2
end;

# ╔═╡ fcbf4061-fc08-488f-a827-65cd3e51d388
md"""
The following routines define the orthogonal projection and its euclidean derivative. As seen above, they are needed to derive a covariant derivative of $F$.

The orthogonal projection onto the fibre $E_x$ is given by

$$P(x)v = v - \frac{1}{\|Bx\|_2^2} Bx\langle Bx, v \rangle_2.$$

The derivative restricted to the fibre reads

$$P'(x)\delta x \, v = -\frac{1}{\|Bx\|_2^2}Bx\langle B\delta x, v \rangle_2 \quad (v\in E_x).$$

Thus, we get 

$$F_Y(x) P'(x)\delta x \, v = -\frac{1}{\|Bx\|_2^2}\langle Ax, Bx\rangle_2\langle B\delta x, v \rangle_2 \quad (v\in E_x).$$

"""

# ╔═╡ baa217b7-637f-4720-8393-a2ac9f2309c3
begin 
	
	transport_by_projection(q, p) = I - (1/norm(B*p)^2)*(B*p)*((B*p)')

	transport_by_proj_prime(p, F) = (-1/norm(B*p)^2)*F*(B*p)*B

	transport = DifferentiableMapping(transport_by_projection,transport_by_proj_prime)
	
end;

# ╔═╡ 2bf8e95e-6f71-4e65-ac8b-087237326390
md"""
The following two routines define $F_Y$ and its euclidean derivative.
"""

# ╔═╡ 1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
begin
	F_at(p) = (A * p)'
	
	F_prime_at(p) = A

	F = DifferentiableMapping(F_at, F_prime_at)
end;

# ╔═╡ f690f7dd-a8d4-47e9-b656-76c9335f5379
md"""
`NewtonEquation`

In this example we implement a functor to compute the Newton matrix and the right hand side for the Newton equation \ref{paper}

$$Q^*_{F(x)}\circ F'(x)\delta x \, v + F(x) v = 0 \; \forall v \in E_x$$

In our example the tangent spaces and fibers can be characterized by linear functionals:

$$T_x\mathbb{S}^{n-1} = \{ \delta x\in \mathbb R^n \mid \langle x, \delta x\rangle_2 = 0\} = \ker \langle x,\cdot \rangle_2$$

$$E_{x} = \{ v \in \mathbb R^n \mid \langle Bx, v \rangle_2 = 0\} = \ker \langle Bx, \cdot\rangle_2.$$

We use the matrix representation of $Q^*_{F(x)}\circ F'(x)$ in $\mathbb R^{n\times n}$ and write the Newton equation as a saddle point system:

$$\begin{pmatrix}
                Q^*_{F(x)}\circ F'(x) & -Bx\\
                x^T & 0
            \end{pmatrix}
            \begin{pmatrix}
                \delta x \\ \delta\lambda
            \end{pmatrix}
            + \begin{pmatrix}
                Ax - [\lambda] Bx \\ 0
            \end{pmatrix}
            = \begin{pmatrix}
                0 \\ 0
            \end{pmatrix}.$$
Here, $[\lambda] :=  \frac{\langle Ax, Bx\rangle_2}{\|Bx\|_2^2}$ is an estimate for the Lagrangian multiplier $\lambda$ (which comes from the constraint $v \in E_x$).

It returns the matrix and the right hand side.
Moreover, for the computation of the simplified Newton direction (which is necessary for affine covariant damping) a method returning the right hand side for the simplified Newton equation is provided. Here, the orthogonal projection is used as a vector transport.
	
"""

# ╔═╡ e16a578c-3d64-11f0-057c-c1f978ba732a
begin
struct NewtonEquation{F, T, NM, Nrhs}
	mapping::F
	vectortransport::T
	A::NM
	b::Nrhs
end

function NewtonEquation(M, F, VT)
	N = manifold_dimension(M) + 1
	newton_matrix = zeros(N+1,N+1)
	rhs = zeros(N+1)
	return NewtonEquation{typeof(F), typeof(VT), typeof(newton_matrix), typeof(rhs)}(F, VT, newton_matrix, rhs)
end
	
function (ne::NewtonEquation)(M, VB, p)
	ne.A .= hcat(vcat(ne.mapping.derivative(p) + ne.vectortransport.derivative(p, ne.mapping.value(p)), p'), vcat(-B*p, 0))
	
	estimate = 1/(norm(B*p)^2) * (ne.mapping.value(p)*(B*p))
    ne.b .= vcat(ne.mapping.value(p)' - estimate*(B*p), 0)
end
	
function (ne::NewtonEquation)(M, VB, p, p_trial)
	estimate = 1/(norm(B*p_trial)^2) * (ne.mapping.value(p_trial)*(B*p_trial))
	rhs_p_trial = ne.mapping.value(p_trial)' - estimate*(B*p_trial)
    return vcat(ne.vectortransport.value(p_trial, p)'*rhs_p_trial, 0)
end
end;

# ╔═╡ f645101d-cbd6-434c-8363-f198c3d5e01a
md"""
We compute $(\delta x, \delta \lambda)$ by solving the augmented system directly. The direction $\delta \lambda \in \mathbb R$ is just a helper and can be thrown away. We return the Newton direction $\delta x$.
"""

# ╔═╡ 7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
function solve_augmented_system(problem, newtonstate) 
	res = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	return res[1:n]
end;

# ╔═╡ 50ac9a95-aa02-4414-8891-ea4656be4683
md""" 
As a starting point for Newton's method we choose the first unit vector
"""

# ╔═╡ 3232de6d-7b17-4f4c-9a3f-c669a1911c57
begin
	y0 = zeros(n)
	y0[1] = 1.0
end;

# ╔═╡ 576b004a-5ec0-4435-b6a8-1bcf9ea36b65
begin
	# adjust norms for computation of damping factors and stopping criterion
	inv_retr = Manifolds.ProjectionInverseRetraction()
	rec = RecordChange(M;
    inverse_retraction_method=inv_retr);
end;

# ╔═╡ 427233d0-567d-4cac-930d-7c80178f2d09
begin
	NE = NewtonEquation(M, F, transport)
	
	st_res = vectorbundle_newton(M, TangentBundle(M), NE, y0; sub_problem=solve_augmented_system, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(M,1e-12, inverse_retraction_method=ProjectionInverseRetraction())),
	retraction_method=ProjectionRetraction(),
	stepsize=Manopt.AffineCovariantStepsize(M, theta_des=0.5),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, rec => :Change, :Stepsize],
	return_state=true
)
end

# ╔═╡ 9c266539-07a1-496c-b10e-a3134582c319
md"""
We extract the recorded values
"""

# ╔═╡ 66bb735a-a41d-4742-8581-fe48907b7489
begin
	change = get_record(st_res, :Iteration, :Change)[2:end]
	stepsizes = get_record(st_res, :Iteration, :Stepsize)[1:end]
	res = get_record(st_res, :Iteration, :Iterate)[end]
end;

# ╔═╡ 2f7ba5aa-7867-4d00-8170-64bac76a4e52
# ╠═╡ disabled = true
#=╠═╡
CSV.write("results/stepsize_eigenvalues.csv", Tables.table(stepsizes[1:end-1]), writeheader=false)
  ╠═╡ =#

# ╔═╡ 37314ea0-29a1-43e1-9b5a-ba2e69fdbd37
# ╠═╡ disabled = true
#=╠═╡
CSV.write("results/norm_newton_direction_eigenvalues.csv", Tables.table(change), writeheader=false)
  ╠═╡ =#

# ╔═╡ 2175d62b-cf84-4dc8-b0cf-14b600fb89ea
md"""
and plot the results
"""

# ╔═╡ 285c7cdc-fdc7-43e8-99fa-17fd708d4328
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change[1:end], color = :blue)
	f
end

# ╔═╡ fd2bc12f-3287-405f-b42a-2696c7b30282
begin
	f_st = Figure(;)
	
    row_st, col_st = fldmod1(1, 2)
	
	Axis(f_st[row_st, col_st], title = string("Stepsizes"), xminorgridvisible = true, xticks = (1:length(stepsizes)), xlabel = "Iteration", ylabel = "α")
    scatterlines!(stepsizes[1:end-1], color = :blue)
	f_st
end

# ╔═╡ 77d9441a-1c02-4725-b219-71e0335b202c
md"""
We can check the result by computing our estimator for the result
"""

# ╔═╡ 075b9039-de7f-424d-b388-8916d3a8aa1e
estimate_res = 1/(norm(B*res)^2) * (F_at(res)*(B*res));

# ╔═╡ f127a03e-672c-41d0-a9ec-6b1905b9e658
md"""
and checking whether `(res, estimate_res)` fulfill the generalized eigenvalue equation
"""

# ╔═╡ a7a13b69-dd2c-4f24-989d-dc3356633184
println(norm(A*res - estimate_res*B*res))

# ╔═╡ Cell order:
# ╠═3e19b659-8855-4020-9173-fe4a1ffbbc08
# ╟─b16e5880-552b-4217-9d94-46fdfa7294c3
# ╠═eb35384f-0822-4953-8d34-4ed5a0df71c5
# ╟─fca6fcbc-d85c-44c5-831c-bdcd229c40d7
# ╟─932d4733-4609-4612-9ade-128742ebc845
# ╠═587cf35f-882f-4e5a-8527-a53ba7c5644f
# ╟─a0fe5b93-98ac-4ee9-bee1-2e53238c7b34
# ╟─5a68a8c6-a51f-4a31-b24b-198683d49517
# ╠═623be37d-3bf6-487d-91d4-af67fd5a9654
# ╟─fcbf4061-fc08-488f-a827-65cd3e51d388
# ╠═baa217b7-637f-4720-8393-a2ac9f2309c3
# ╟─2bf8e95e-6f71-4e65-ac8b-087237326390
# ╠═1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
# ╟─f690f7dd-a8d4-47e9-b656-76c9335f5379
# ╠═e16a578c-3d64-11f0-057c-c1f978ba732a
# ╟─f645101d-cbd6-434c-8363-f198c3d5e01a
# ╠═7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
# ╟─50ac9a95-aa02-4414-8891-ea4656be4683
# ╠═3232de6d-7b17-4f4c-9a3f-c669a1911c57
# ╠═427233d0-567d-4cac-930d-7c80178f2d09
# ╠═576b004a-5ec0-4435-b6a8-1bcf9ea36b65
# ╟─9c266539-07a1-496c-b10e-a3134582c319
# ╠═66bb735a-a41d-4742-8581-fe48907b7489
# ╠═2f7ba5aa-7867-4d00-8170-64bac76a4e52
# ╠═37314ea0-29a1-43e1-9b5a-ba2e69fdbd37
# ╟─2175d62b-cf84-4dc8-b0cf-14b600fb89ea
# ╠═285c7cdc-fdc7-43e8-99fa-17fd708d4328
# ╠═fd2bc12f-3287-405f-b42a-2696c7b30282
# ╟─77d9441a-1c02-4725-b219-71e0335b202c
# ╠═075b9039-de7f-424d-b388-8916d3a8aa1e
# ╟─f127a03e-672c-41d0-a9ec-6b1905b9e658
# ╠═a7a13b69-dd2c-4f24-989d-dc3356633184
