# [Jacobi Fields](@id JacobiFieldFunctions)

A smooth tangent vector field $J\colon [0,1] → T\mathcal M$
along a geodesic $g(\cdot;x,y)$ is called _Jacobi field_
if it fulfills the ODE

$\displaystyle 0 = \frac{D}{dt}J + R(J,\dot g)\dot g,$

where $R$ is the Riemannian curvature tensor.
Such Jacobi fields can be used to derive closed forms for the exponential map,
the logarithmic map and the geodesic, all of them with respect to both arguments:
Let $F\colon\mathcal N → \mathcal M$ be given (for the $\exp_x\cdot$
  we have $\mathcal N = T_x\mathcal M$, otherwise $\mathcal N=\mathcal M$) and denote by
$ξ_1,…,ξ_d$ an orthonormal frame along $g(\cdot;x,y)$ that diagonalizes
the curvature tensor with corresponding eigenvalues $κ_1,…,κ_d$.
Note that on symmetric manifolds such a frame always exists.

Then $DF(x)[η] = \sum_{k=1}^d \langle η,ξ_k(0)\rangle_xβ(κ_k)ξ_k(T)$ holds,
where $T$ also depends on the function $F$ as the weights $β$. The values
stem from solving the corresponding system of (decoupled) ODEs.

Note that in different references some factors might be a little different,
for example when using unit speed geodesics.

The following weights functions are available

```@autodocs
Modules = [Manopt]
Pages   = ["Jacobi_fields.jl"]
```
