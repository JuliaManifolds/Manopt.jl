# [Proximal Maps](@id proximalMapFunctions)

For a function $\varphi\colon\mathcal M \to\mathbb R$ the proximal map is defined
as

$\displaystyle\operatorname{prox}_{\lambda\varphi}(x)
= \operatorname*{argmin}_{y ∈ \mathcal M} d_{\mathcal M}^2(x,y) + \varphi(y),
\quad \lambda > 0,$

where $d_{\mathcal M}\colon \mathcal M \times \mathcal M \to \mathbb R$ denotes
the geodesic distance on \(\mathcal M\). While it might still be difficult to
compute the minimizer, there are several proximal maps known (locally) in closed
form. Furthermore if $x^{\star} ∈ \mathcal M$ is a minimizer of $\varphi$, then

$\displaystyle\operatorname{prox}_{\lambda\varphi}(x^\star) = x^\star,$

i.e. a minimizer is a fixed point of the proximal map.

This page lists all proximal maps available within Manopt. To add you own, just
extend the `functions/proximal_maps.jl` file.

```@autodocs
Modules = [Manopt]
Pages   = ["proximal_maps.jl"]
```
