# [Jacobi Fields](@id JacobiFieldFunctions)

A smooth tangent vector field $J\colon [0,1] \to T\mathcal M$
along a geodesic $g(\cdot;x,y)$ is called _Jacobi field_
if it fulfills the ODE

$\displaystyle 0 = \frac{D}{dt}J + R(J,\dot g)\dot g,$

where $R$ is the Riemannian curvature tensor.
Such Jacobi fields can be used to derive closed forms for the exponential map,
the logarithmic map and the geodesic, all of them with respect to both arguments:
Let $F\colon\mathcal N \to \mathcal M$ be given (for the $\exp_x\cdot$
  we have $\mathcal N = T_x\mathcal M$, otherwise $\mathcal N=\mathcal M$) and denote by
$\Xi_1,\ldots,\Xi_d$ an orthonormal frame along $g(\cdot;x,y)$ that diagonalizes
the curvature tensor with corresponding eigenvalues $\kappa_1,\ldots,\kappa_d$.
Note that on symmetric manifolds such a frame always exists.

Then $DF(x)[\eta] = \sum_{k=1}^d \langle \eta,\Xi_k(0)\rangle_x\beta(\kappa_k)\Xi_k(T)$ holds,
where $T$ also depends on the function $F$ as the weights $\beta$. The values
stem from solving the corresponding system of (decoupled) ODEs.

Note that in different references some factors might be a little different,
for example when using unit speed geodesics.

The following weights functions are available
```@docs
βDgx
βDexpx
βDexpξ
βDlogx
βDlogy
```
