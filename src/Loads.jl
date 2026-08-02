"""
    AbstractLoad

Supertype of the external load specifications [`Traction`](@ref), [`Pressure`](@ref), [`NodalForce`](@ref) and [`BodyForce`](@ref).

Loads are Neumann boundary conditions: they contribute to the external force vector and never to the tangent.
Prescribed displacements are Dirichlet boundary conditions and are not part of this hierarchy; use Ferrite's `Dirichlet` and `ConstraintHandler` for those.

Each load type is named after the units of the value its function returns.
The 2D and 3D units differ only because the entity integrated over does.
"""
abstract type AbstractLoad end

"""
    Traction(set, f)

Surface traction on a facet set, in force per unit area in 3D and force per unit length in 2D.

`set` is a facetset name or a set of `FacetIndex`.
A facet of a 2D cell is an edge, so the same type covers area loads in 3D and edge loads in 2D.

`f` returns the traction vector and may be written in any of three forms, resolved once when the [`LoadHandler`](@ref) is closed:

```julia
Traction("top", x -> Vec(0.0, -p))                   # f(x)
Traction("top", (x, t) -> Vec(0.0, -p * t))          # f(x, t)
Traction("top", (x, t, n) -> -p * t * n)             # f(x, t, n)
```

`x` is the spatial coordinate of the quadrature point, `t` is the value passed to [`external_forces!`](@ref), and `n` is the outward facet normal.
Because `f` receives `x`, any spatial profile is expressed directly, for example `(x, t) -> Vec(0.0, -p * (x[1] / L) * t)` for a load growing linearly along the x-axis.

`n` is evaluated on the undeformed geometry, so a normal-aligned traction is a dead load, fixed in direction at the start and not rotating as the surface deforms.
Follower loads, whose direction tracks the deforming surface, would contribute an extra term to the tangent and are not available.
For the common normal-aligned case, [`Pressure`](@ref) carries the sign convention.
"""
struct Traction{S, F} <: AbstractLoad
    set::S
    f::F
end

"""
    Pressure(set, f)

Normal pressure on a facet set, in the same units as [`Traction`](@ref).

`f` returns a scalar and accepts the forms `f(x)`, `f(x, t)` and `f(x, t, n)`.
The applied traction is `-f * n` with `n` the outward facet normal, so a positive value acts inward and compresses the body.

```julia
Pressure("hole", (x, t) -> p * t)
```

`Pressure` is equivalent to `Traction(set, (x, t, n) -> -f(x, t) * n)` and exists to keep that sign convention in one place.
The normal is taken on the undeformed geometry, so this is a dead load; see [`Traction`](@ref).
"""
struct Pressure{S, F} <: AbstractLoad
    set::S
    f::F
end

"""
    NodalForce(set, f; distribute=false)

Point force applied at the nodes of a node set, in force.

`set` is a nodeset name or a collection of node ids.
`f` returns the force vector and accepts the forms `f(x)` and `f(x, t)`, where `x` is the node coordinate.
The facet normal is not defined at a node, so the `f(x, t, n)` form is not available here.

`distribute=false`, the default, applies the returned value at every node of the set.
`distribute=true` treats the returned value as the resultant of the whole set and splits it equally over the nodes, so the sum over the set is the value itself.

```julia
NodalForce("tip", x -> Vec(0.0, -F))                      # F at each node
NodalForce("tip", x -> Vec(0.0, -F); distribute=true)     # F in total
```

Equal splitting matches the finite element load distribution only for first-order shape functions.
Above first order the shape function integration no longer splits the load equally, so corner and midside nodes take different shares.
To spread a resultant over a surface on such a mesh, use [`Traction`](@ref) with magnitude `F / A` instead.

Only nodes carrying degrees of freedom can receive a force.
Closing the handler throws when the set contains nodes without dofs.
"""
struct NodalForce{S, F} <: AbstractLoad
    set::S
    f::F
    distribute::Bool
end

NodalForce(set, f; distribute::Bool = false) = NodalForce(set, f, distribute)

"""
    BodyForce(f)
    BodyForce(set, f)

Body force over cells, in force per unit volume in 3D and force per unit area in 2D.

The one-argument form applies to every cell carrying degrees of freedom.
The two-argument form takes a cellset name or a collection of cell ids.
`f` returns the force vector and accepts the forms `f(x)` and `f(x, t)`, where `x` is the spatial coordinate of the quadrature point.

```julia
BodyForce(x -> Vec(0.0, -rho * g))                        # self-weight
BodyForce("rotor", (x, t) -> rho * omega(t)^2 * x)        # centrifugal, per cellset
```
"""
struct BodyForce{S, F} <: AbstractLoad
    set::S
    f::F
end

BodyForce(f) = BodyForce(nothing, f)

# Prepared integration data for one load, built by close! and reused every step.
# `values` is a FacetValues or CellValues, specialized on by the integration kernels.
# `cache` carries mutable per-entity state, so assembly stays single-threaded.
struct LoadGroup{E, V, C}
    entities::Vector{E}
    values::V
    fe::Vector{Float64}
    cache::C
end

struct PreparedIntegral
    groups::Vector{Any}
    f::Function
end

struct PreparedNodal
    nodes::Vector{Int}
    dofs::Matrix{Int} # n_comp x length(nodes)
    scale::Float64
    f::Function
end

"""
    LoadHandler(assembler; quadrature_order=nothing)
    LoadHandler(dh; thickness=1.0, quadrature_order=nothing)

External load container, built and used the way Ferrite's `ConstraintHandler` is:

```julia
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0], [1, 2]))
close!(ch)

lh = LoadHandler(assembler)
add!(lh, Traction("top", (x, t) -> Vec(0.0, -p * t)))
add!(lh, NodalForce("tip", x -> Vec(0.0, -F); distribute=true))
close!(lh)
```

[`external_forces!`](@ref) then returns the assembled external force vector for a given `t`.
The handler follows the pattern and keeps the same name as the `LoadHandler` of [FerriteAssembly.jl](https://github.com/KnutAM/FerriteAssembly.jl) (K. A. Meyer).

Constructing from a [`GenericMaterialAssembler`](@ref) takes the `DofHandler`, the `thickness` and the quadrature orders from it, so external and internal forces are integrated consistently.
Passing a `DofHandler` instead is for drivers that assemble no internal forces; give the same `thickness` the assembler uses, because facet and body integrals scale with it exactly as `ke` and `re` do.

`quadrature_order` accepts an `Int` for one order across all sub-dof-handlers, a `Vector{Int}` for one order each, and `nothing` to take the assembler's orders or, without an assembler, the interpolation default.

# Adding loads after closing

`close!` prepares the entries added since the last call and leaves earlier ones untouched, so loads may be added part-way through a simulation and the handler closed again:

```julia
add!(lh, Traction("second_stage", (x, t) -> Vec(0.0, -q * (t - t_switch))))
close!(lh)
```

Changing the magnitude, direction or profile of an existing load needs no such call, because the load function is evaluated on every `external_forces!` call and may close over computed results:

```julia
stage = Ref(1)
add!(lh, Traction("top", (x, t) -> stage[] == 1 ? Vec(0.0, -p) : zero(Vec{2})))
```

The same pattern removes a load, since a function returning zero contributes nothing to the external force vector.

# Replication under MPI

`LoadHandler` assembles serially and, under MPI, redundantly: every rank builds the complete external force vector and no reduction is performed.
This matches the replicated-data design of the assembler, whose residual is already complete on every rank after its own `Allreduce`.
Every rank must add the same loads in the same order so that the per-rank copies agree.
An `MPI.Allreduce` over the external force vector would sum those identical copies and apply the load `nranks` times.
"""
mutable struct LoadHandler
    dh::DofHandler
    thickness::Float64
    quadrature_orders::Vector{Int}
    loads::Vector{AbstractLoad}
    prepared::Vector{Any}
    f_ext::Vector{Float64}
    n_comp::Int
    _cell_to_sdh::Vector{Int}
    _node_dofs::Union{Nothing, Matrix{Int}}
    _node_has_dofs::Union{Nothing, BitVector}
end

function LoadHandler(dh::DofHandler; thickness::Real = 1.0, quadrature_order = nothing)
    n_sdh = length(dh.subdofhandlers)
    orders = _load_quadrature_orders(dh, quadrature_order, n_sdh)

    n_comp = Ferrite.n_components(get_ip(dh, 1))
    for si in 2:n_sdh
        Ferrite.n_components(get_ip(dh, si)) == n_comp || throw(ArgumentError(
            "LoadHandler requires the same number of field components in every SubDofHandler, " *
            "but SubDofHandler 1 has $n_comp and SubDofHandler $si has $(Ferrite.n_components(get_ip(dh, si)))."
        ))
    end

    cell_to_sdh = zeros(Int, getncells(dh.grid))
    for (si, sdh) in enumerate(dh.subdofhandlers)
        for cid in sdh.cellset
            cell_to_sdh[cid] = si
        end
    end

    return LoadHandler(dh, Float64(thickness), orders, AbstractLoad[], Any[],
                       zeros(ndofs(dh)), n_comp, cell_to_sdh, nothing, nothing)
end

function LoadHandler(assembler::GenericMaterialAssembler; quadrature_order = nothing)
    orders = isnothing(quadrature_order) ? copy(assembler.quadrature_orders) : quadrature_order
    return LoadHandler(assembler.dh; thickness = assembler.thickness, quadrature_order = orders)
end

function _load_quadrature_orders(dh, quadrature_order, n_sdh)
    if quadrature_order isa Int
        return fill(quadrature_order, n_sdh)
    elseif quadrature_order isa Vector{Int}
        length(quadrature_order) == n_sdh || throw(ArgumentError(
            "quadrature_order has $(length(quadrature_order)) entries but the DofHandler has $n_sdh SubDofHandlers"))
        return copy(quadrature_order)
    elseif isnothing(quadrature_order)
        return [get_default_quadrature_order_for_sdh(dh, si) for si in 1:n_sdh]
    end
    throw(ArgumentError("quadrature_order must be Int, Vector{Int}, or nothing"))
end

function Base.show(io::IO, lh::LoadHandler)
    print(io, "LoadHandler(", length(lh.loads), " loads")
    if !isempty(lh.loads)
        kinds = [nameof(typeof(load)) for load in lh.loads]
        print(io, ": ", join(string.(kinds), ", "))
    end
    n_open = length(lh.loads) - length(lh.prepared)
    n_open > 0 && print(io, ", ", n_open, " not yet closed")
    print(io, ", ndofs=", length(lh.f_ext))
    lh.thickness == 1.0 || print(io, ", thickness=", lh.thickness)
    print(io, ")")
end

"""
    add!(lh::LoadHandler, load::AbstractLoad)

Add an external load to `lh` and return `lh`.

The load is recorded here and takes effect once `close!(lh)` has been called, which [`external_forces!`](@ref) requires.
See [`LoadHandler`](@ref) for adding loads after a first `close!`.
"""
function Ferrite.add!(lh::LoadHandler, load::AbstractLoad)
    push!(lh.loads, load)
    return lh
end

"""
    close!(lh::LoadHandler)

Prepare and validate the loads added to `lh` since the last call, and return `lh`.

Closing resolves set names against the grid, builds the facet, cell and nodal integration data, resolves each load function to one of its accepted call forms, and evaluates it once to check that it returns a value of the expected shape.
Entries prepared by an earlier call are left untouched, so `close!` may be called again after further `add!` calls.
"""
function Ferrite.close!(lh::LoadHandler)
    for i in (length(lh.prepared) + 1):length(lh.loads)
        push!(lh.prepared, _prepare_load(lh, lh.loads[i], i))
    end
    return lh
end

"""
    external_forces!(lh::LoadHandler, t=0.0)

Assemble the external force vector of every load in `lh` at `t` and return it.

`t` is passed unchanged as the second argument of each load function.
The driver chooses its meaning: physical time for rate-dependent materials, or a load factor for quasi-static ramping.
Pass the same value to Ferrite's `update!(ch, t)` so that prescribed displacements and loads refer to the same instant.

Dead loads do not depend on the displacement, so call `external_forces!` once per time step, outside the Newton loop, and subtract the result from the internal force returned by [`stiffness_matrix`](@ref):

```julia
f_ext = external_forces!(lh, t)
for iter in 1:max_newton
    K, f_int = stiffness_matrix(assembler, u; dt=dt)
    residual .= f_int .- f_ext
    apply_zero!(K, residual, ch)
    norm(residual) <= tol * max(norm(f_ext), 1.0) && break
    u .-= K \\ residual
end
```

External loads contribute nothing to `K`.

Under MPI every rank assembles the complete vector and no reduction is performed; see [`LoadHandler`](@ref).

The returned vector belongs to `lh`.
Calling `external_forces!` again overwrites it, so copy the result when a step's values are still needed afterwards.

Throws when loads have been added since the last `close!(lh)`, which must be called before the added loads can be assembled.
"""
function external_forces!(lh::LoadHandler, t::Real = 0.0)
    n_open = length(lh.loads) - length(lh.prepared)
    n_open == 0 || throw(ArgumentError(
        "LoadHandler has $n_open load(s) added since the last close!. Call close!(lh) before external_forces!."))

    f_ext = lh.f_ext
    fill!(f_ext, 0.0)
    tf = Float64(t)
    for i in eachindex(lh.loads)
        _accumulate_load!(f_ext, lh, lh.loads[i], lh.prepared[i], tf)
    end
    return f_ext
end

#######################
# Load function forms #
#######################

# Resolve f(x), f(x, t) or f(x, t, n) once, at close! time, into a uniform three-argument closure
function _resolve_load_function(f, ::Type{X}, allow_normal::Bool, what::String) where {X}
    if allow_normal && hasmethod(f, Tuple{X, Float64, X})
        return (x, t, n) -> f(x, t, n)
    elseif hasmethod(f, Tuple{X, Float64})
        return (x, t, n) -> f(x, t)
    elseif hasmethod(f, Tuple{X})
        return (x, t, n) -> f(x)
    end
    forms = allow_normal ? "f(x), f(x, t) or f(x, t, n)" : "f(x) or f(x, t)"
    throw(ArgumentError("$what: the load function must be callable as $forms with x::$X, but none of these forms applies to $(typeof(f))."))
end

# Load functions may return a Vec or any indexable of matching length; both are
# converted to a concrete Vec so the integration kernels stay type stable.
@inline _to_vec(val::Vec{D, Float64}, ::Val{D}) where {D} = val
@inline _to_vec(val::Vec{D}, ::Val{D}) where {D} = Vec{D, Float64}(val)
@inline function _to_vec(val, ::Val{D}) where {D}
    length(val) == D || throw(ArgumentError(
        "load function returned a value of length $(length(val)), but the field has $D components"))
    return Vec{D, Float64}(ntuple(i -> Float64(val[i]), D))
end

function _check_vector_return(f3, x, ncomp, what)
    val = f3(x, 0.0, x)
    if val isa Number
        throw(ArgumentError("$what: the load function returned the scalar $val, but a vector of $ncomp components is required. Pressure takes a scalar function; Traction, NodalForce and BodyForce take a vector-valued one."))
    end
    length(val) == ncomp || throw(ArgumentError(
        "$what: the load function returned a value of length $(length(val)), but the field has $ncomp components."))
    return nothing
end

function _check_scalar_return(f3, x, what)
    val = f3(x, 0.0, x)
    val isa Number || throw(ArgumentError(
        "$what: the load function returned a value of length $(length(val)), but Pressure requires a scalar function. Use Traction for a vector-valued surface load."))
    return nothing
end

##################
# Set resolution #
##################

function _available(names)
    isempty(names) && return "none are defined on this grid"
    return "available: " * join(sort!(collect(String.(names))), ", ")
end

function _resolve_facetset(grid, set::AbstractString, what)
    haskey(grid.facetsets, String(set)) || throw(ArgumentError(
        "$what: the grid has no facetset \"$set\" ($(_available(keys(grid.facetsets))))."))
    return collect(FacetIndex, getfacetset(grid, String(set)))
end
_resolve_facetset(::Any, set, ::Any) = collect(FacetIndex, set)

function _resolve_nodeset(grid, set::AbstractString, what)
    haskey(grid.nodesets, String(set)) || throw(ArgumentError(
        "$what: the grid has no nodeset \"$set\" ($(_available(keys(grid.nodesets))))."))
    return sort!(collect(Int, getnodeset(grid, String(set))))
end
_resolve_nodeset(::Any, set, ::Any) = sort!(collect(Int, set))

function _resolve_cellset(grid, set::AbstractString, what)
    haskey(grid.cellsets, String(set)) || throw(ArgumentError(
        "$what: the grid has no cellset \"$set\" ($(_available(keys(grid.cellsets))))."))
    return sort!(collect(Int, getcellset(grid, String(set))))
end
_resolve_cellset(::Any, set, ::Any) = sort!(collect(Int, set))

##################
# Values objects #
##################

# FacetValues counterpart of cell_values_for_sdh in Interfaces.jl.
function facet_values_for_sdh(dh::DofHandler, sdh_idx::Int, order)
    ip = get_ip(dh, sdh_idx)
    refshape = getrefshape(ip)
    qr = FacetQuadratureRule{refshape}(order)
    sample_cid = first(dh.subdofhandlers[sdh_idx].cellset)
    geom_ip = Ferrite.geometric_interpolation(typeof(dh.grid.cells[sample_cid]))
    return FacetValues(qr, ip, geom_ip)
end

####################
# Load preparation #
####################

_load_label(load, i) = "load $i ($(nameof(typeof(load))))"

function _group_by_sdh(lh::LoadHandler, entities, cellid_of, what)
    n_sdh = length(lh.dh.subdofhandlers)
    grouped = [eltype(entities)[] for _ in 1:n_sdh]
    orphans = 0
    for e in entities
        si = lh._cell_to_sdh[cellid_of(e)]
        si == 0 ? (orphans += 1) : push!(grouped[si], e)
    end
    orphans == 0 || throw(ArgumentError(
        "$what: $orphans of $(length(entities)) entities lie on cells without degrees of freedom. " *
        "Restrict the set to the cells covered by the DofHandler."))
    return grouped
end

function _prepare_facet_load(lh::LoadHandler, set, f, i, load, allow_normal, scalar_valued)
    what = _load_label(load, i)
    grid = lh.dh.grid
    facets = _resolve_facetset(grid, set, what)
    isempty(facets) && throw(ArgumentError("$what: the facet set is empty."))

    grouped = _group_by_sdh(lh, facets, fi -> fi[1], what)
    groups = Any[]
    for (si, group_facets) in enumerate(grouped)
        isempty(group_facets) && continue
        fv = facet_values_for_sdh(lh.dh, si, lh.quadrature_orders[si])
        push!(groups, LoadGroup(group_facets, fv, zeros(getnbasefunctions(fv)), FacetCache(lh.dh)))
    end

    X = Ferrite.get_coordinate_type(grid)
    f3 = _resolve_load_function(f, X, allow_normal, what)
    probe = get_node_coordinate(grid, first(getcells(grid, facets[1][1]).nodes))
    scalar_valued ? _check_scalar_return(f3, probe, what) : _check_vector_return(f3, probe, lh.n_comp, what)

    return PreparedIntegral(groups, f3)
end

_prepare_load(lh::LoadHandler, load::Traction, i::Int) =
    _prepare_facet_load(lh, load.set, load.f, i, load, true, false)

_prepare_load(lh::LoadHandler, load::Pressure, i::Int) =
    _prepare_facet_load(lh, load.set, load.f, i, load, true, true)

function _prepare_load(lh::LoadHandler, load::BodyForce, i::Int)
    what = _load_label(load, i)
    grid = lh.dh.grid
    cells = isnothing(load.set) ? sort!(collect(Int, get_cells(lh.dh))) : _resolve_cellset(grid, load.set, what)
    isempty(cells) && throw(ArgumentError("$what: the cell set is empty."))

    grouped = _group_by_sdh(lh, cells, identity, what)
    groups = Any[]
    for (si, group_cells) in enumerate(grouped)
        isempty(group_cells) && continue
        cv = cell_values_for_sdh(lh.dh, si, lh.quadrature_orders[si])
        push!(groups, LoadGroup(group_cells, cv, zeros(getnbasefunctions(cv)), CellCache(lh.dh)))
    end

    X = Ferrite.get_coordinate_type(grid)
    f3 = _resolve_load_function(load.f, X, false, what)
    _check_vector_return(f3, get_node_coordinate(grid, first(getcells(grid, cells[1]).nodes)), lh.n_comp, what)

    return PreparedIntegral(groups, f3)
end

function _prepare_load(lh::LoadHandler, load::NodalForce, i::Int)
    what = _load_label(load, i)
    grid = lh.dh.grid
    nodes = _resolve_nodeset(grid, load.set, what)
    isempty(nodes) && throw(ArgumentError("$what: the node set is empty."))

    table, has_dofs = _node_dof_table!(lh)
    missing_dofs = count(n -> !has_dofs[n], nodes)
    missing_dofs == 0 || throw(ArgumentError(
        "$what: $missing_dofs of $(length(nodes)) nodes carry no degrees of freedom and cannot receive a force. " *
        "Restrict the set to the nodes covered by the DofHandler."))

    dofs = Matrix{Int}(undef, lh.n_comp, length(nodes))
    for (k, node) in enumerate(nodes)
        for c in 1:lh.n_comp
            dofs[c, k] = table[c, node]
        end
    end

    X = Ferrite.get_coordinate_type(grid)
    f3 = _resolve_load_function(load.f, X, false, what)
    _check_vector_return(f3, get_node_coordinate(grid, nodes[1]), lh.n_comp, what)

    scale = load.distribute ? 1.0 / length(nodes) : 1.0
    return PreparedNodal(nodes, dofs, scale, f3)
end

# Node-to-dof table, mirroring how Ferrite collects dofs for a Dirichlet condition
# on a nodeset, and reaching exactly the nodes those constraints reach.
function _node_dof_table!(lh::LoadHandler)
    isnothing(lh._node_dofs) || return lh._node_dofs, lh._node_has_dofs

    dh = lh.dh
    grid = dh.grid
    n_comp = lh.n_comp
    table = zeros(Int, n_comp, getnnodes(grid))
    has_dofs = falses(getnnodes(grid))

    for (si, sdh) in enumerate(dh.subdofhandlers)
        ip = get_ip(dh, si)
        base_ip = _base_interpolation(ip)
        interpol_points = getnbasefunctions(base_ip)
        for cc in CellIterator(dh, sdh.cellset)
            cell_dofs = celldofs(cc)
            cell_nodes = cc.nodes
            for idx in 1:min(interpol_points, length(cell_nodes))
                node = cell_nodes[idx]
                has_dofs[node] && continue
                for c in 1:n_comp
                    table[c, node] = cell_dofs[(idx - 1) * n_comp + c]
                end
                has_dofs[node] = true
            end
        end
    end

    lh._node_dofs = table
    lh._node_has_dofs = has_dofs
    return table, has_dofs
end

############
# Assembly #
############

function _accumulate_load!(f_ext, lh::LoadHandler, ::Traction, prepared::PreparedIntegral, t::Float64)
    for group in prepared.groups
        _integrate_facets!(f_ext, group, prepared.f, t, lh.thickness, Val(lh.n_comp), false)
    end
    return f_ext
end

function _accumulate_load!(f_ext, lh::LoadHandler, ::Pressure, prepared::PreparedIntegral, t::Float64)
    for group in prepared.groups
        _integrate_facets!(f_ext, group, prepared.f, t, lh.thickness, Val(lh.n_comp), true)
    end
    return f_ext
end

function _accumulate_load!(f_ext, lh::LoadHandler, ::BodyForce, prepared::PreparedIntegral, t::Float64)
    for group in prepared.groups
        _integrate_cells!(f_ext, group, prepared.f, t, lh.thickness, Val(lh.n_comp))
    end
    return f_ext
end

function _accumulate_load!(f_ext, lh::LoadHandler, ::NodalForce, prepared::PreparedNodal, t::Float64)
    return _apply_nodal!(f_ext, lh.dh.grid, prepared.nodes, prepared.dofs, prepared.scale,
                         prepared.f, t, Val(lh.n_comp))
end

# Separate method so the load function reaches the node loop concretely typed
function _apply_nodal!(f_ext, grid, nodes, dofs, scale::Float64, f, t::Float64, ncomp::Val{D}) where {D}
    for (k, node) in enumerate(nodes)
        x = get_node_coordinate(grid, node)
        value = _to_vec(f(x, t, x), ncomp) * scale
        for c in 1:D
            f_ext[dofs[c, k]] += value[c]
        end
    end
    return f_ext
end

function _integrate_facets!(f_ext, group::LoadGroup, f, t::Float64, thickness::Float64, ncomp::Val, is_pressure::Bool)
    fv = group.values
    fe = group.fe
    fc = group.cache
    nbf = getnbasefunctions(fv)

    for facet in group.entities
        reinit!(fc, facet)
        reinit!(fv, fc)
        fill!(fe, 0.0)
        coords = getcoordinates(fc)
        for qp in 1:getnquadpoints(fv)
            x = spatial_coordinate(fv, qp, coords)
            n = getnormal(fv, qp)
            value = f(x, t, n)
            traction = is_pressure ? -Float64(value) * n : _to_vec(value, ncomp)
            dGamma = getdetJdV(fv, qp) * thickness
            for i in 1:nbf
                fe[i] += (traction ⋅ shape_value(fv, qp, i)) * dGamma
            end
        end
        assemble!(f_ext, celldofs(fc), fe)
    end
    return f_ext
end

function _integrate_cells!(f_ext, group::LoadGroup, f, t::Float64, thickness::Float64, ncomp::Val)
    cv = group.values
    fe = group.fe
    cc = group.cache
    nbf = getnbasefunctions(cv)

    for cellid in group.entities
        reinit!(cc, cellid)
        reinit!(cv, cc)
        fill!(fe, 0.0)
        coords = getcoordinates(cc)
        for qp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, qp, coords)
            body = _to_vec(f(x, t, x), ncomp)
            dOmega = getdetJdV(cv, qp) * thickness
            for i in 1:nbf
                fe[i] += (body ⋅ shape_value(cv, qp, i)) * dOmega
            end
        end
        assemble!(f_ext, celldofs(cc), fe)
    end
    return f_ext
end