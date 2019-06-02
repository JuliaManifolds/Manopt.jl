#
#      Graphmanifold – an array of points of one manifold,
#          where their vicinity or neighborhoods are given by a graph
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show
import SparseArrays: issparse, findnz, sparse
export Graph, GraphVertexPoint, GraphVertexTVector, GraphEdgePoint, GraphEdgeTVector
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export zeroTVector
export show, getValue
@doc doc"""
    Graph{M<:Manifold} <: Manifold

The graph manifold models manifold-valued data on a graph
$\mathcal G = (\mathcal V, \mathcal E)$, both on vertices
and edges as well as their interplay. The adjacency is stored in a matrix,
and may contain also the weights.

Since there are two possibilities in dimensions, $\lvert\mathcal V\rvert$ and
$\lvert\mathcal E\rvert$, the manifold itself will refer to the first one, while
depending on the type of `MPoint` one of them is returned.

# Fields
the default values are given in brackets
* `adjacency` – the (sparse) adjacency matrix, might also carry weights, i.e. all
  $a_{ij}>0$ refer to adjacent nodes $i$ and $j$
* `name` – (`A Graph manifold of \$Submanifold.`) name of the manifold
* `manifold` – the internal manifold present at vertices (edges) for
[`GraphVertexPoint`](@ref) ([`GraphEdgePoint`](@ref))
* `dimension` – stores the dimension of the manifold of a `GraphVertexPoint`
* `isDirected` – (`false`) indicates whether the graph is directed or not.
"""
struct Graph{mT<:Manifold} <: Manifold
  name::String
  manifold::mT
  dimension::Int
  adjacency::Mat where {Mat <: AbstractMatrix}
  isDirected::Bool
  abbreviation::String
  Graph{mT}(M::mT, adjacency::Mat, isDir::Bool=false) where {mT <: Manifold, Mat <: AbstractMatrix} = new(
    string("A Graph Manifold of ",M.name,"."),
    M,size(adjacency,1)*manifoldDimension(M),
    adjacency, isDir,
    string("GraphVertex(",M.abbreviation,",",repr(size(adjacency,1)),")") )
end
Graph(M::mT,A::Mat,dir::Bool=false) where {mT <: Manifold, Mat <: AbstractMatrix} = Graph{mT}(M,A,dir)

@doc doc"""
    GraphVertexPoint <: MPoint

A point graph vertex power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal V\rvert}$
represented by a vector of corresponding [`MPoint`](@ref)s.
"""
struct GraphVertexPoint <: MPoint
  value::Vector{P} where {P<:MPoint}
  GraphVertexPoint(v::Vector{P} where {P<:MPoint}) = new(v)
end
getValue(x::GraphVertexPoint) = x.value;

@doc doc"""
    GraphVertexTVector <: TVector

A tangent vector $\xi\in T_x\mathcal M$ to the graph vertex power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal V\rvert}$
represented by a vector of corresponding [`TVector`](@ref)s.
"""
struct GraphVertexTVector <: TVector
  value::Vector{T} where {T <: TVector}
  GraphVertexTVector(value::Vector{T} where {T <: TVector}) = new(value)
end
getValue(ξ::GraphVertexTVector) = ξ.value

@doc doc"""
    GraphEdgePoint <: MPoint

A point graph edge power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal E\rvert}$
represented by a (sparse/not completely filled matrix of corresponding
[`MPoint`](@ref)s.
"""
struct GraphEdgePoint <: MPoint
  value::V where {V <: AbstractMatrix{P} where {P<:MPoint} }
  GraphEdgePoint(v::V where { V <: AbstractMatrix{P} where {P<:MPoint} }) = new(v)
end
getValue(x::GraphEdgePoint) = x.value;

@doc doc"""
    GraphEdgeTVector <: TVector

A tangent vector $\xi\in T_x\mathcal M$ to the graph edge power manifold
$\mathcal M = \mathcal N^{\lvert\mathcal E\rvert}$
represented by a (sparse/not completely filled) matrix of corresponding
[`TVector`](@ref)s.
"""
struct GraphEdgeTVector <: TVector
  value::V where {V <: AbstractMatrix{T} where {T<:TVector}}
  GraphEdgeTVector(v::V where { V <: AbstractMatrix{T} where {T<:TVector} }) = new(v)
end
getValue(ξ::GraphEdgeTVector) = ξ.value

# Functions
# ---
"""
    distance(M,x,y)

compute a vectorized version of distance on the [`Graph`](@ref) manifold `M`
for two [`GraphVertexPoint`](@ref) `x` and `y`.
"""
distance(M::Graph, x::GraphVertexPoint, y::GraphVertexPoint) = sqrt(sum( distance.(Ref(M.manifold), getValue(x), getValue(y) ).^2 ))
"""
    distance(M,x,y)

compute a vectorized version of distance on the [`Graph`](@ref) manifold `M`
for two [`GraphEdgePoint`](@ref) `x` and `y`.
"""
distance(M::Graph, x::GraphEdgePoint, y::GraphEdgePoint) = sqrt(sum( distance.(Ref(M.manifold), getValue(x), getValue(y) ).^2 ))
"""
    dot(M,x,ξ,ν)

computes the inner product as sum of the component inner products on the
[`Graph`](@ref) vertices.
"""
dot(M::Graph, x::GraphVertexPoint, ξ::GraphVertexTVector, ν::GraphVertexTVector) = sum(dot.(Ref(M.manifold),getValue(x), getValue(ξ), getValue(ν) ))
"""
    dot(M,x,ξ,ν)

computes the inner product as sum of the component inner products on the
[`Graph`](@ref) edges.
"""
dot(M::Graph, x::GraphEdgePoint, ξ::GraphEdgeTVector, ν::GraphEdgeTVector) = sum(dot.(Ref(M.manifold),getValue(x), getValue(ξ), getValue(ν) ))
"""
    exp(M,x,ξ[, t=1.0])

computes the product exponential map on the [`Graph`](@ref) vertices and
returns the corresponding [`GraphVertexPoint`](@ref).
"""
exp(M::Graph, x::GraphVertexPoint, ξ::GraphVertexTVector, t::Float64=1.0) = GraphVertexPoint( exp.(Ref(M.manifold), getValue(x) , getValue(ξ) ))
"""
    log(M,x,y)

computes the product logarithmic map on the [`Graph`](@ref) for two
[`GraphVertexPoint`](@ref) `x` and `y` and returns the
corresponding [`GraphVertexTVector`](@ref).
"""
log(M::Graph, x::GraphVertexPoint, y::GraphVertexPoint)::GraphVertexTVector = GraphVertexTVector(log.(Ref(M.manifold), getValue(x), getValue(y) ))
"""
    log(M,x,y)

computes the product logarithmic map on the [`Graph`](@ref) for two
[`GraphEdgePoint`](@ref) `x` and `y` and returns the
corresponding [`GraphEdgeTVector`](@ref).
"""
log(M::Graph, x::GraphEdgePoint, y::GraphEdgePoint)::GraphEdgeTVector = GraphEdgeTVector(log.(Ref(M.manifold), getValue(x), getValue(y) ))
"""
    manifoldDimension(x)

returns the (product of) dimension(s) of the [`Graph`](@ref) manifold `M` the
[`GraphVertexPoint`](@ref) `x` belongs to.
"""
manifoldDimension(x::GraphVertexPoint) = sum(manifoldDimension.( getValue(x) ) )
"""
    manifoldDimension(x)

returns the (product of) dimension(s) of the [`Graph`](@ref) manifold `M` the
[`GraphEdgePoint`](@ref)`x` belongs to.
"""
manifoldDimension(x::GraphEdgePoint) = sum(manifoldDimension.( getValue(x) ) )
"""
    manifoldDimension(M)

returns the (product of) dimension(s) of the [`Graph`](@ref) manifold `M` seen
as a vertex power manifold.
"""
manifoldDimension(M::Graph) = size(M.adjacency,1) * manifoldDimension(M.manifold)
"""
    norm(M,x,ξ)

norm of the [`GraphVertexTVector`](@ref) `ξ` induced by the metric on the
manifold components of the [`Graph`](@ref) manifold `M`.
"""
norm(M::Graph, x::GraphVertexPoint, ξ::GraphVertexTVector) = sqrt(sum(dot.(Ref(M.manifold),getValue(x),getValue(ξ),getValue(ξ))))
"""
    norm(M,x,ξ)

norm of the [`GraphEdgeTVector`](@ref) `ξ` induced by the metric on the manifold
components of the [`Graph`](@ref) manifold `M`.
"""
norm(M::Graph, x::GraphEdgePoint, ξ::GraphEdgeTVector) = sqrt(sum(dot.(Ref(M.manifold),getValue(x), getValue(ξ), getValue(ξ))))

@doc doc"""
    parallelTransport(M,x,ξ)

compute the product parallelTransport map on the [`Graph`](@ref) vertex power
manifold $\mathcal M^{\lvert\mathcal V\rvert}$ and returns
the corresponding [`GraphVertexTVector`](@ref).
"""
parallelTransport(M::Graph, x::GraphVertexPoint, y::GraphVertexPoint, ξ::GraphVertexTVector) = GraphVertexTVector( parallelTransport.(Ref(M.manifold), getValue(x), getValue(y), getValue(ξ)) )
"""
    parallelTransport(M,x,ξ)

compute the product parallelTransport map on the [`Graph`](@ref) edge power
manifold and returns the corresponding [`GraphVertexTVector`](@ref).
"""
parallelTransport(M::Graph, x::GraphEdgePoint, y::GraphEdgePoint, ξ::GraphEdgeTVector) = GraphVertexTVector( parallelTransport.(Ref(M.manifold), getValue(x), getValue(y), getValue(ξ)) )
"""
    randomMPoint(M,[,:Vertex])

compute a random point on the [`Graph`](@ref) manifold, where by default a point
on the vertices is produces, use `:Edge` to generate a `GraphEdgePoint`.
Further optional parameters are passed on to the element wise random point function.
"""
randomMPoint(M::Graph,::Val{:Vertex},options...) = GraphVertexPoint(
    [ randomMPoint( M.manifold, options...) for i=1:size(M.adjacency,1) ]
)
randomMPoint(M::Graph,::Val{:Edge},options...) =  GraphEdgePoint(
    [ randomMPoint( M.manifold, options...) for i=1:sum( M.adjacency .> 0) ]
)

"""
    randomTVector(M,x)

compute a random [`GraphEdgeTVector`](@ref) to the [`GraphVertexPoint`](@ref) `x`
on the [`Graph`](@ref) manifold `M` by calling the inner random vector generation
for every edge
"""
randomTVector(M::Graph, x::GraphVertexPoint,options...) = GraphVertexTVector(
    [
        (M.adjacency[i,j] > 0) ? M.randomTVector(M.manifold, getValue(x)[i],options...) : zeroTVector(M,getValue(x)[i])
        for i = 1:size(M.adjacency,1) for j=1:size(M.adjacency,2)        
    ]
)

typicalDistance(M::Graph) = sqrt( size(M.adjacency,1) ) * typicalDistance(M.manifold);
@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`GraphVertexPoint`](@ref) $x\in\mathcal M$ on the [`Graph`](@ref) vertex manifold `M`.
"""
zeroTVector(M::Graph, x::GraphVertexPoint) = GraphVertexTVector( zeroTVector.(Ref(M.manifold), getValue(x) )  )
@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`GraphEdgePoint`](@ref) $x\in\mathcal M$ on the [`Graph`](@ref) edge manifold `M`.
"""
zeroTVector(M::Graph, x::GraphEdgePoint) = GraphEdgeTVector( zeroTVector.(Ref(M.manifold), getValue(x) )  )
"""
    startEdgePoint(M,x)

For a [`Graph`](@ref) manifold and a [`GraphVertexPoint`](@ref), this
function constructs the corresponding [`GraphEdgePoint`](@ref), such that each
edge has its start point vertex value assigned.
"""
function startEdgePoint(M::Graph, x::GraphVertexPoint)::GraphEdgePoint
  if length(getValue(x)) == 0
    throw( ErrorException("vertexToStartEdgePoint::No node given"))
  end
  if issparse(M.adjacency)
    (s,e,v) = findnz(M.adjacency)
    return GraphEdgePoint( sparse(s,e, getValue(x)[ s ]) )
  else
    sA = size(M.adjacency)
    A = Matrix(typeof(x[1]),sA)
    for i=1:sA[1]
      for j=1:sA[2]
        if M.adjacency[i,j] > 0
          A[i,j] = getValue(x)[i]
        else
          A[i,j] = missing;
        end
      end
    end
  end
end
"""
    sumEdgeTVector(M,ξ)

return the [`GraphVertexTVector`](@ref) where edge tangents are summed in their
start point.

For an [`GraphEdgeTVector`](@ref) `ξ` on a [`Graph`](@ref) manifold `M`
this function assumes that all edge tangents are attached in a tangent space
corresponding to the same point on the base manifold, i.e. all these vectors can
be summed. This sum per vectex is then returned as a
[`GraphVertexTVector`](@ref).
"""
function sumEdgeTVector(M::Graph, ξ::GraphEdgeTVector, weighted::Bool=false)
  (s,e,v) = findnz(M.adjacency)
  init = falses(size(M.adjacency,1)) # vector initialized?
  lξ = getValue(ξ)
  ν = Vector(  eltype( lξ ),size(M.adjacency,1)  )
  for i=1:length(s) # all edges
    if !init[ s[i] ]
      ν[ s[i] ] = lξ[s[i],e[i]] # initialize (i.e. also carry base if it exists)
      init[ s[i] ] = true
    else
      ν[ s[i] ] = ν[ s[i] ] + lξ[s[i], e[i]]
    end
  end
  return GraphVertexTVector(ν)
end
#
#
# Display functions for the structs
show(io::IO, M::Graph) = print(io,string("The manifold on vertices and edges of a graph of ",repr(M.manifold), " of (vertex manifold) dimension ",repr(M.dimension),".") );
show(io::IO, p::GraphVertexPoint) = print(io,string("GraphVertexV[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::GraphVertexTVector) = print(io,string("GraphVertexT[",join(repr.(ξ.value),", "),"]"));
show(io::IO, p::GraphEdgePoint) = print(io,string("GraphEdgeV[",join(repr.(p.value),", "),"]"));
show(io::IO, ξ::GraphEdgeTVector) = print(io,string("GraphEdgeT[", join(repr.(ξ.value),", "),"]"));
