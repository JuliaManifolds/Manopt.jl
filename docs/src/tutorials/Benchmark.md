```@meta
EditURL = "/Users/ronnber/Repositories/Julia/Manopt.jl/docs/../tutorials/Benchmark.jl"
```
```@raw html
<style>
    table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    pre, div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "07afdc7cf4e1689adf93ba21cc038119e927b718bc56e755def2c91315540ba4"
    julia_version = "1.8.4"
-->

<div class="markdown"><h1>Illustration of how to Use Mutating Gradient Functions</h1><p>When it comes to time critital operations, a main ingredient in Julia is given by mutating functions, i.e. those that compute in place without additional memory allocations. In the following, we illustrate how to do this with <code>Manopt.jl</code>.</p><p>Let's start with the same function as in <a href="https://manoptjl.org/stable/tutorials/MeanAndMedian.html">Get Started: Optimize!</a> and compute the mean of some points, only that here we use the sphere <span class="tex">$\mathbb S^{30}$</span> and <code>n=800</code> points.</p><p>From the aforementioned example, the implementation looks like</p></div>


```
## Setup
```@raw html
<div class="markdown">
<p>If you open this notebook in Pluto locally it switches between two modes. If the tutorial is within the <code>Manopt.jl</code> repository, this notebook tries to use the local package in development mode. Otherwise, the file uses the Pluto pacakge management version.</p></div>








<div class="markdown"><p>Since the loading is a little complicated, we show, which versions of packages were installed in the following.</p></div>

<pre class='language-julia'><code class='language-julia'>with_terminal() do
    Pkg.status()
end</code></pre>
<pre id="plutouiterminal">�[32m�[1mStatus�[22m�[39m `/private/var/folders/_v/wg192lpd3mb1lp55zz7drpcw0000gn/T/jl_kzpWvD/Project.toml`
 �[90m [6e4b80f9] �[39mBenchmarkTools v1.3.2
 �[90m [1cead3c2] �[39mManifolds v0.8.42
 �[90m [0fc0a36d] �[39mManopt v0.4.0 `~/Repositories/Julia/Manopt.jl`
 �[90m [7f904dfe] �[39mPlutoUI v0.7.49
 �[90m [44cfe95a] �[39mPkg v1.8.0
 �[90m [9a3f8284] �[39mRandom
</pre>

<pre class='language-julia'><code class='language-julia'>begin
    Random.seed!(42)
    m = 30
    M = Sphere(m)
    n = 800
    σ = π / 8
    x = zeros(Float64, m + 1)
    x[2] = 1.0
    data = [exp(M, x, σ * rand(M; vector_at=x)) for i in 1:n]
end;</code></pre>



```
## Classical Definition
```@raw html
<div class="markdown">
<p>The variant from the previous tutorial defines a cost <span class="tex">$F(x)$</span> and its gradient <span class="tex">$gradF(x)$</span></p></div>

<pre class='language-julia'><code class='language-julia'>F(M, x) = sum(1 / (2 * n) * distance.(Ref(M), Ref(x), data) .^ 2)</code></pre>
<pre class="code-output documenter-example-output" id="var-F">F (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>gradF(M, x) = sum(1 / n * grad_distance.(Ref(M), data, Ref(x)))</code></pre>
<pre class="code-output documenter-example-output" id="var-gradF">gradF (generic function with 1 method)</pre>


<div class="markdown"><p>We further set the stopping criterion to be a little more strict. Then we obtain</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    sc = StopWhenGradientNormLess(1e-10)
    x0 = zeros(Float64, m + 1); x0[1] = 1/sqrt(2); x0[2] = 1/sqrt(2)
    m1 = gradient_descent(M, F, gradF, x0; stopping_criterion=sc)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-x0">31-element Vector{Float64}:
  0.0033485640684974894
 -0.9989177042558477
  0.012603956513369109
  3.724197504132531e-5
  0.0034768865059239184
  0.007393312780305253
  0.0015131386444782369
  ⋮
  0.0033263346566243146
 -0.004530004392593586
  0.0030077996887298286
 -0.01561032047036917
 -0.0016794415657848806
 -0.0025375720668704706</pre>

<pre class='language-julia'><code class='language-julia'>@benchmark gradient_descent($M, $F, $gradF, $x0; stopping_criterion=$sc)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash144613">BenchmarkTools.Trial: 81 samples with 1 evaluation.
 Range (min … max):  58.225 ms … 68.131 ms  ┊ GC (min … max): 11.41% … 17.00%
 Time  (median):     62.946 ms              ┊ GC (median):    16.09%
 Time  (mean ± σ):   62.413 ms ±  2.231 ms  ┊ GC (mean ± σ):  14.32% ±  2.45%

       ▂  ▂     ▅    ▂                  ▂▅▅▂▂▂ ▅▂█             
  ▅▁▁▅▁█▁███▅▅▅▅██▅▅███▁▅▁▅█▅▁▁▅▁▁▁██▅▁▅██████▅███▅▅▁▁▁▁▁▁▁▅▅ ▁
  58.2 ms         Histogram: frequency by time        66.4 ms &lt;

 Memory estimate: 193.85 MiB, allocs estimate: 653778.</pre>


```
## In-place Computation of the Gradient
```@raw html
<div class="markdown">
<p>We can reduce the memory allocations by implementing the gradient as a <a href="https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects">functor</a>. The motivation is twofold: on one hand, we want to avoid variables from the global scope, for example the manifold <code>M</code> or the <code>data</code>, being used within the function. Considering to do the same for more complicated cost functions might also be worth it.</p><p>Here, we store the data (as reference) and one temporary memory in order to avoid reallocation of memory per <code>grad_distance</code> computation. We have</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    struct grad!{TD,TTMP}
        data::TD
        tmp::TTMP
    end
    function (gradf!::grad!)(M, X, x)
        fill!(X, 0)
        for di in gradf!.data
            grad_distance!(M, gradf!.tmp, di, x)
            X .+= gradf!.tmp
        end
        X ./= length(gradf!.data)
        return X
    end
end</code></pre>



<div class="markdown"><p>Then we just have to initialize the gradient and perform our final benchmark. Note that we also have to interpolate all variables passed to the benchmark with a <code>$</code>.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    gradF2! = grad!(data, similar(data[1]))
    m2 = deepcopy(x0)
    gradient_descent!(
        M, F, gradF2!, m2; evaluation=InplaceEvaluation(), stopping_criterion=sc
    )
end</code></pre>
<pre class="code-output documenter-example-output" id="var-m2">31-element Vector{Float64}:
  0.00334856408243381
 -0.9989177042557205
  0.012603956512397922
  3.7241973292108026e-5
  0.0034768865018420234
  0.007393312782144477
  0.0015131386426487721
  ⋮
  0.003326334656361401
 -0.0045300043965643315
  0.003007799686052061
 -0.015610320474421206
 -0.0016794415638888209
 -0.002537572065615552</pre>

<pre class='language-julia'><code class='language-julia'>@benchmark gradient_descent!(
        $M, $F, $gradF2!, m2; evaluation=$(InplaceEvaluation()), stopping_criterion=$sc
    ) setup = (m2 = deepcopy($x0))</code></pre>
<pre class="code-output documenter-example-output" id="var-hash139609">BenchmarkTools.Trial: 154 samples with 1 evaluation.
 Range (min … max):  30.548 ms … 39.550 ms  ┊ GC (min … max): 0.00% … 17.98%
 Time  (median):     32.561 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   32.651 ms ±  1.446 ms  ┊ GC (mean ± σ):  0.96% ±  3.71%

              ▁▁█▃                                             
  ▃▃▂▂▂▃▄▂▆▅▄▄█████▇▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▂▂▁▃ ▂
  30.5 ms         Histogram: frequency by time        38.8 ms &lt;

 Memory estimate: 3.88 MiB, allocs estimate: 6286.</pre>


<div class="markdown"><p>Note that the results <code>m1</code>and <code>m2</code> are of course the same.</p></div>

<pre class='language-julia'><code class='language-julia'>distance(M, m1, m2)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash146788">0.0</pre>

<!-- PlutoStaticHTML.End -->
```


