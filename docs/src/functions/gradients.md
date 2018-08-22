# [Gradients](@id GradientFunctions)
For a function $f\colon\mathcal M\to\mathbb R$
the Riemannian gradient $\nabla f(x)$ at $x\in\mathcal M$
is given by the unique tangent vector fulfilling

$\langle \nabla f(x), \xi\rangle_x = D_xf[\xi],\quad
\forall \xi \in T_x\mathcal M,$
where $D_xf[\xi]$ denotes the differential of $f$ at $x$ with respect to
the tangent direction (vector) $\xi$ or in other words the directional
derivative.

This page collects the available gradients.

```@autodocs
Modules = [Manopt]
Pages   = ["gradients.jl"]
```
