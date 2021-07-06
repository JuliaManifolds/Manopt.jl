# [Proximal Maps](@id proximalMapFunctions)

For a function $\varphi:\mathcal M →ℝ$ the proximal map is defined
as

$\displaystyle\operatorname{prox}_{λ\varphi}(x)
= \operatorname*{argmin}_{y ∈ \mathcal M} d_{\mathcal M}^2(x,y) + \varphi(y),
\quad λ > 0,$

where $d_{\mathcal M}: \mathcal M \times \mathcal M → ℝ$ denotes
the geodesic distance on \(\mathcal M\). While it might still be difficult to
compute the minimizer, there are several proximal maps known (locally) in closed
form. Furthermore if $x^{\star} ∈ \mathcal M$ is a minimizer of $\varphi$, then

$\displaystyle\operatorname{prox}_{λ\varphi}(x^\star) = x^\star,$

i.e. a minimizer is a fixed point of the proximal map.

This page lists all proximal maps available within Manopt. To add you own, just
extend the `functions/proximal_maps.jl` file.

```@autodocs
Modules = [Manopt]
Pages   = ["proximal_maps.jl"]
```
