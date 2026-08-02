struct AssemblyWorkspace
    kes::Vector{Matrix{Float64}}   # sdh_index -> exact-size element stiffness
    res::Vector{Vector{Float64}}   # sdh_index -> exact-size element residual
    cvs::Vector{Any}
    avs::Vector{Any}
end

"""
    GenericMaterialAssembler

Stateful, preallocated Ferrite-based assembler for material models implementing the FerriteSolidMechanics material model interface.

The assembler owns the preassembled linear stiffness, the current nonlinear tangent and residual buffers, the material mapping for each active cell, and the quadrature point state vectors for nonlinear materials.
It also caches sub-dofhandler metadata, task-owned workspaces for parallel assembly, per-cell element results used during scatter, and reusable stress-output buffers.

Construct it with [`create_assembler`](@ref), then call [`stiffness_matrix`](@ref), [`compute_forces`](@ref), [`compute_stresses`](@ref), and [`update_states!`](@ref).
The Concepts page describes the assembly and state-management model.
"""
struct GenericMaterialAssembler{AH,ST}
    K_linear::SparseMatrixCSC{Float64,Int}
    K_tangent::SparseMatrixCSC{Float64,Int}
    r::Vector{Float64}

    materials::Vector{AbstractMaterial}
    cell_to_mat_idx::Vector{Int}

    linear_cells::Vector{Int}
    nonlinear_cells::Vector{Int}
    owned_nonlinear_cells::Vector{Int}

    states::Vector{Vector{AbstractMaterialState}}

    dh::DofHandler
    ah::AH
    ch::ConstraintHandler
    quadrature_orders::Vector{Int} # Per-SDH order the cell values were built with; informational
    thickness::Float64 # Scalar multiplier on ke and re; default 1.0 (no-op)

    # SDH Metadata
    _sdh_cell_to_sdh::Vector{Int}           # cell_id -> sdh_index
    _sdh_cvs::Vector{Any}                   # sdh_index -> CellValues (concrete per-SDH)
    _sdh_avs::Vector{Any}                   # sdh_index -> AlphaValues (concrete per-SDH)
    _sdh_nqp::Vector{Int}                   # sdh_index -> n_quadpoints
    _sdh_ndofs::Vector{Int}                 # sdh_index -> ndofs_per_cell
    _sdh_owned_nonlinear::Vector{Vector{Int}} # si -> indices into owned_nonlinear_cells
    _sdh_owned_all::Vector{Vector{Int}}      # si -> indices into _owned_all_cells

    # Pre-cached cell metadata (indexed by position in owned_nonlinear_cells)
    _cell_dofs::Vector{Vector{Int}}

    # Task-owned computation workspaces. A spawned OhMyThreads task receives one
    # workspace for the full batch, independent of the OS thread it runs on.
    _workspaces::Vector{AssemblyWorkspace}

    # Per-cell result caches for two-phase assembly (indexed by owned_nonlinear_cells position)
    _cell_ke::Vector{Matrix{Float64}}
    _cell_re::Vector{Vector{Float64}}

    # Pre-cached metadata for stress computations
    _owned_all_cells::Vector{Int} # all cells owned by this rank (linear + nonlinear)
    _owned_all_dofs::Vector{Vector{Int}} # DOFs for all owned cells

    # Result buffers for force/stress results
    _stresses_cache::Array{ST,2}
    _stresses_mpi_buffer::Vector{Float64}
    _mpi_flag_buffer::Vector{Int}
end

function Base.show(io::IO, fem::GenericMaterialAssembler)
    print(io, "GenericMaterialAssembler(",
        "cells=", length(fem.linear_cells) + length(fem.nonlinear_cells),
        " (", length(fem.linear_cells), " linear, ", length(fem.nonlinear_cells), " nonlinear)",
        ", ndofs=", length(fem.r),
        ", materials=", length(fem.materials),
        ", quadrature_orders=", fem.quadrature_orders,
        ", thickness=", fem.thickness)
    fem.ah === nothing || print(io, ", ah=", nameof(typeof(fem.ah)))
    mpi_size() > 1 && print(io, ", rank ", mpi_rank(), " owns ", length(fem.owned_nonlinear_cells), " nonlinear cells")
    print(io, ")")
end

function _default_workspace_count()
    # The assembler currently schedules assembly work on OhMyThreads' default
    # thread pool. We preallocate one task workspace per default worker thread
    # and cap OhMyThreads' ntasks to that count in each batch.
    return Threads.nthreads(:default)
end

function _create_workspace(dh, ah, quadrature_orders, sdh_ndofs)
    n_sdh = length(dh.subdofhandlers)
    kes = [zeros(ndofs, ndofs) for ndofs in sdh_ndofs]
    res = [zeros(ndofs) for ndofs in sdh_ndofs]
    cvs = Any[cell_values_for_sdh(dh, si, quadrature_orders[si]) for si in 1:n_sdh]
    avs = Any[create_alpha_values(ah, cv) for cv in cvs]
    return AssemblyWorkspace(kes, res, cvs, avs)
end

function _take_task_workspace!(counter::Threads.Atomic{Int}, workspaces::Vector{AssemblyWorkspace})
    workspace_id = Threads.atomic_add!(counter, 1) + 1
    if workspace_id > length(workspaces)
        error("OhMyThreads spawned more assembly tasks ($workspace_id) than preallocated workspaces ($(length(workspaces))). This would make workspace ownership ambiguous.")
    end
    return workspaces[workspace_id]
end

function create_assembler(mps::Dict{String,<:AbstractMaterial}, dh::DofHandler, ah, ch::ConstraintHandler, quadrature_order, thickness=1.0)
    return create_assembler(mps, dh, ch; ah=ah, quadrature_order=quadrature_order, thickness=thickness)
end

function create_assembler(mat::AbstractMaterial, dh::DofHandler, ah, ch::ConstraintHandler, quadrature_order, thickness=1.0)
    return create_assembler(mat, dh, ch; ah=ah, quadrature_order=quadrature_order, thickness=thickness)
end

"""
    _reinit_alpha_values!(av, cid::Int)

Reinitialize the per-cell alpha-values object `av` for cell `cid`, or return `nothing` when `av === nothing`.
Throws `ArgumentError` when `av` does not implement `Ferrite.reinit!(av, cellid::Int)`.

`_reinit_alpha_values!` runs once per owned cell in the stiffness and stress hot loops, `_stiffness_sdh_batch!` and `_stress_sdh_batch!`.
It is marked `@inline` so the `applicable`/`reinit!` call resolves at the call site.
"""
@inline function _reinit_alpha_values!(av, cid::Int)
    av === nothing && return nothing
    if applicable(reinit!, av, cid)
        reinit!(av, cid)
        return nothing
    end
    throw(ArgumentError("Alpha values object $(typeof(av)) must implement Ferrite.reinit!(av, cellid::Int)"))
end

const _KINEMATICS_FALLBACK = which(kinematics, Tuple{AbstractMaterial})

function _linear_cell_flag(mat::AbstractMaterial)
    isptr = is_linear(mat)
    # kinematics is optional for materials with a custom _assemble_element!, so ask
    # only those that declare it; the fallback method throws rather than answering
    declares_kinematics = which(kinematics, Tuple{typeof(mat)}) !== _KINEMATICS_FALLBACK
    if isptr && declares_kinematics && kinematics(mat) isa FiniteStrain
        throw(ArgumentError(
            "$(nameof(typeof(mat))) declares is_linear = true with FiniteStrain kinematics. " *
            "Linear cells are preassembled once at F = I and their residual is taken as K_linear * u, " *
            "which is only valid for a displacement-independent tangent."))
    end
    return isptr
end

"""
    create_assembler(material, dh, ch; ah=nothing, quadrature_order=nothing, thickness=1.0)
    create_assembler(materials_by_cellset, dh, ch; ah=nothing, quadrature_order=nothing, thickness=1.0)

Build a [`GenericMaterialAssembler`](@ref) from either a single material applied to all cells, or a dictionary mapping Ferrite cellset names to materials.
Create the assembler once, after the `DofHandler` and `ConstraintHandler` are closed, and reuse it for the whole simulation.

# Arguments
- `material` / `materials_by_cellset`: a single `AbstractMaterial` or a
  `Dict{String,<:AbstractMaterial}`. The dictionary form assigns different
  materials to different cellsets of the same grid, for example a stiff
  inclusion embedded in a soft matrix. The named cellsets must cover the active
  cells of `dh` exactly once. Cells outside `get_cells(dh)` are ignored.
- `dh::DofHandler`: a closed Ferrite `DofHandler`. Every sub-dof-handler of a
  mixed-element grid is supported.
- `ch::ConstraintHandler`: a closed Ferrite `ConstraintHandler`. Its Dirichlet
  pattern allocates both the linear and tangent sparse matrices.
- `ah=nothing`: optional user-supplied object. When it is non-`nothing`, the
  assembler calls [`create_alpha_values`](@ref) for a per-`CellValues`
  alpha-values object, then calls `reinit!(av, cellid)` for each cell. Material
  element routines apply `alpha_value(av, qp)` in the integration weight of
  stiffness and residual contributions, and `compute_stresses` applies the same
  factor to stress output. With `ah === nothing`, the default
  `alpha_value(::Nothing, ::Int) = 1.0` applies.
- `quadrature_order=nothing`: optional quadrature order. An `Int` sets one order
  for every sub-dof-handler, a `Vector{Int}` sets one order per sub-dof-handler
  in order, and `nothing` auto-detects from the field interpolation (typically
  `order(ip) + 1` for order > 1, else `2`). The order is keyed by
  sub-dof-handler, not by material, so two materials sharing one
  `SubDofHandler` also share its quadrature rule. To give the cellsets of a
  multi-material grid different orders, declare one `SubDofHandler` per cellset
  and pass a `Vector{Int}`.
- `thickness=1.0`: scalar multiplier applied to the element stiffness and
  residual as `ke .*= thickness; re .*= thickness`. Omit it for 3D problems,
  where the default `1.0` is a no-op. For 2D plane problems, pass the physical
  out-of-plane thickness the model requires, for example `thickness=0.1` for a
  millimetre-thick sheet. The wrappers [`PlaneStrain`](@ref) and
  [`PlaneStress`](@ref) require no thickness; their stress is
  per-unit-thickness by convention.

# Additional call shapes

Four positional call shapes are accepted, taking either a single material or a material dictionary, with or without `thickness`:

```julia
create_assembler(material, dh, ah, ch, quadrature_order)
create_assembler(material, dh, ah, ch, quadrature_order, thickness)
create_assembler(material_dict, dh, ah, ch, quadrature_order)
create_assembler(material_dict, dh, ah, ch, quadrature_order, thickness)
```

These forward to the keyword form.

# What is precomputed

- Sparse-matrix allocation matching the Dirichlet pattern of `ch`
- Per-cell mapping from `cellid` to material and to sub-dof-handler
- One `CellValues` per sub-dof-handler
- The linear stiffness of materials with `is_linear == true` (`Hooke`,
  `Hooke2D`), preassembled once into `K_linear`
- One `AbstractMaterialState` per owned nonlinear cell per quadrature point
- Per-owned-cell result buffers used by the two-phase assembly
- A stress cache of shape `(max_nquadpoints, ncells)`

See [`GenericMaterialAssembler`](@ref) for the full list of fields.
"""
function create_assembler(mps::Dict{String,<:AbstractMaterial}, dh::DofHandler, ch::ConstraintHandler; ah=nothing, quadrature_order=nothing, thickness=1.0)
    # Sorted: material indices must match across MPI ranks
    setnames = sort!(collect(keys(mps)))
    materials = AbstractMaterial[mps[name] for name in setnames]
    valid_cells = Set(get_cells(dh))

    linear_cells = Int[]
    nonlinear_cells = Int[]

    cell_to_mat_idx = zeros(Int, getncells(dh.grid))

    for (mat_idx, name) in enumerate(setnames)
        mat = materials[mat_idx]
        isptr = _linear_cell_flag(mat)

        for cellid in getcellset(dh.grid, name)
            cellid in valid_cells || continue
            if cell_to_mat_idx[cellid] != 0
                error("Cell $cellid is assigned to more than one material cellset")
            end
            cell_to_mat_idx[cellid] = mat_idx
            if isptr
                push!(linear_cells, cellid)
            else
                push!(nonlinear_cells, cellid)
            end
        end
    end
    sort!(linear_cells)
    sort!(nonlinear_cells)

    missing_cells = sort!([cellid for cellid in valid_cells if cell_to_mat_idx[cellid] == 0])
    if !isempty(missing_cells)
        shown = first(missing_cells, 10)
        more = length(missing_cells) > length(shown) ? ", ..." : ""
        error("Material cellsets do not cover all active cells of the DofHandler: " *
              "$(length(missing_cells)) of $(length(valid_cells)) cells are unassigned. " *
              "First uncovered cell ids: $(join(shown, ", "))$more. " *
              "Cellsets given: $(join(setnames, ", ")).")
    end

    return _init_generic_assembler(materials, cell_to_mat_idx, linear_cells, nonlinear_cells, dh, ah, ch, quadrature_order, thickness)
end

function create_assembler(mat::AbstractMaterial, dh::DofHandler, ch::ConstraintHandler; ah=nothing, quadrature_order=nothing, thickness=1.0)
    materials = AbstractMaterial[mat]
    valid_cells = get_cells(dh)

    cell_to_mat_idx = zeros(Int, getncells(dh.grid))

    linear_cells = Int[]
    nonlinear_cells = Int[]

    isptr = _linear_cell_flag(mat)

    for cellid in valid_cells
        cell_to_mat_idx[cellid] = 1
        if isptr
            push!(linear_cells, cellid)
        else
            push!(nonlinear_cells, cellid)
        end
    end
    sort!(linear_cells)
    sort!(nonlinear_cells)

    return _init_generic_assembler(materials, cell_to_mat_idx, linear_cells, nonlinear_cells, dh, ah, ch, quadrature_order, thickness)
end

function _init_generic_assembler(materials, cell_to_mat_idx, linear_cells, nonlinear_cells, dh, ah, ch, quadrature_order=nothing, thickness=1.0)
    n_sdh = length(dh.subdofhandlers)

    # Handle quadrature orders (Int, Vector{Int}, or nothing for auto-detect)
    quadrature_orders = if quadrature_order isa Int
        fill(quadrature_order, n_sdh)
    elseif quadrature_order isa Vector{Int}
        length(quadrature_order) == n_sdh || throw(ArgumentError("quadrature_order has $(length(quadrature_order)) entries but the DofHandler has $n_sdh SubDofHandlers"))
        quadrature_order
    elseif isnothing(quadrature_order)
        [get_default_quadrature_order_for_sdh(dh, si) for si in 1:n_sdh]
    else
        error("quadrature_order must be Int, Vector{Int}, or nothing")
    end

    # With every cell nonlinear, K_linear stays all-zero and only its dimensions
    # are used, so the sparsity pattern is skipped.
    K_linear = isempty(linear_cells) ? spzeros(Float64, Int, ndofs(dh), ndofs(dh)) :
               allocate_matrix(dh, ch)
    K_tangent = allocate_matrix(dh, ch)
    r = zeros(ndofs(dh))

    rank = mpi_rank()
    nranks = mpi_size()

    # SDH Metadata
    _sdh_cell_to_sdh = zeros(Int, getncells(dh.grid))
    _sdh_cvs = Any[]
    _sdh_avs = Any[]
    _sdh_nqp = Int[]
    _sdh_ndofs = Int[]

    for (si, sdh) in enumerate(dh.subdofhandlers)
        for cid in sdh.cellset
            _sdh_cell_to_sdh[cid] = si
        end
        cv = cell_values_for_sdh(dh, si, quadrature_orders[si])
        push!(_sdh_cvs, cv)
        push!(_sdh_avs, create_alpha_values(ah, cv))
        push!(_sdh_nqp, getnquadpoints(cv))
        push!(_sdh_ndofs, ndofs_per_cell(dh, first(sdh.cellset)))
    end

    # Pre-assemble Linear Part
    assembler = start_assemble(K_linear)
    owned_linear_cells = filter(cid -> (mod(cid - 1, nranks) == rank), linear_cells)

    if !isempty(owned_linear_cells)
        owned_linear_by_sdh = [Int[] for _ in 1:n_sdh]
        for cid in owned_linear_cells
            push!(owned_linear_by_sdh[_sdh_cell_to_sdh[cid]], cid)
        end

        for si in 1:n_sdh
            owned_linear_sdh = owned_linear_by_sdh[si]
            isempty(owned_linear_sdh) && continue

            cv = _sdh_cvs[si]
            av = _sdh_avs[si]

            for cell in CellIterator(dh, owned_linear_sdh)
                cid = cellid(cell)

                reinit!(cv, cell)
                _reinit_alpha_values!(av, cid)

                ke = zeros(_sdh_ndofs[si], _sdh_ndofs[si])
                mat = materials[cell_to_mat_idx[cid]]
                _assemble_element!(ke, nothing, NoState(), mat, cv, av, nothing, 0.0)
                ke .*= thickness
                assemble!(assembler, celldofs(cell), ke)
            end
        end
    end
    !isempty(linear_cells) && mpi_allreduce!(K_linear.nzval)

    # Compute owned nonlinear cells
    owned_nonlinear_cells = filter(cid -> (mod(cid - 1, nranks) == rank), nonlinear_cells)

    # Precompute SDH index sets for hot loops
    _sdh_owned_nonlinear = [Int[] for _ in 1:n_sdh]
    for (idx, cid) in enumerate(owned_nonlinear_cells)
        si = _sdh_cell_to_sdh[cid]
        push!(_sdh_owned_nonlinear[si], idx)
    end

    # States
    states = Vector{Vector{AbstractMaterialState}}(undef, getncells(dh.grid))
    for cid in owned_nonlinear_cells
        mat = materials[cell_to_mat_idx[cid]]
        si = _sdh_cell_to_sdh[cid]
        n_qp = _sdh_nqp[si]
        states[cid] = AbstractMaterialState[create_state(mat) for _ in 1:n_qp]
    end

    _cell_dofs = [celldofs(dh, cid) for cid in owned_nonlinear_cells]

    nworkspaces = max(_default_workspace_count(), 1)
    _workspaces = [_create_workspace(dh, ah, quadrature_orders, _sdh_ndofs) for _ in 1:nworkspaces]

    all_cells = sort!(collect(get_cells(dh)))

    _owned_all_cells = filter(cid -> (mod(cid - 1, nranks) == rank), all_cells)
    _owned_all_dofs = [celldofs(dh, cid) for cid in _owned_all_cells]

    _sdh_owned_all = [Int[] for _ in 1:n_sdh]
    for (idx, cid) in enumerate(_owned_all_cells)
        si = _sdh_cell_to_sdh[cid]
        push!(_sdh_owned_all[si], idx)
    end

    sdim = Ferrite.getspatialdim(dh.grid)
    ST = sdim == 2 ? Tensor{2,2,Float64,4} : Tensor{2,3,Float64,9}
    num_components = sdim == 2 ? 4 : 9

    max_nqp = maximum(_sdh_nqp)
    _stresses_cache = zeros(ST, max_nqp, getncells(dh.grid))
    # Only `compute_stresses` under MPI reads this; sizing it eagerly would double the stress cache on every serial run; resizes on demand
    _stresses_mpi_buffer = zeros(Float64, nranks > 1 ? length(_stresses_cache) * num_components : 0)
    _mpi_flag_buffer = zeros(Int, 1)

    _cell_ke = Matrix{Float64}[]
    _cell_re = Vector{Float64}[]
    for (idx, cid) in enumerate(owned_nonlinear_cells)
        si = _sdh_cell_to_sdh[cid]
        push!(_cell_ke, zeros(_sdh_ndofs[si], _sdh_ndofs[si]))
        push!(_cell_re, zeros(_sdh_ndofs[si]))
    end

    return GenericMaterialAssembler{typeof(ah),ST}(
        K_linear, K_tangent, r,
        materials, cell_to_mat_idx,
        linear_cells, nonlinear_cells, owned_nonlinear_cells,
        states,
        dh, ah, ch,
        quadrature_orders,
        thickness,
        _sdh_cell_to_sdh, _sdh_cvs, _sdh_avs, _sdh_nqp, _sdh_ndofs, _sdh_owned_nonlinear, _sdh_owned_all,
        _cell_dofs,
        _workspaces,
        _cell_ke, _cell_re,
        _owned_all_cells, _owned_all_dofs,
        _stresses_cache, _stresses_mpi_buffer, _mpi_flag_buffer
    )
end

_is_purely_linear(fem::GenericMaterialAssembler) =
    isempty(fem.nonlinear_cells) && !isempty(fem.linear_cells)

function _assemble_stiffness_local!(fem::GenericMaterialAssembler, u::AbstractVector, dt)
    # Nothing to accumulate; the collective step overwrites both buffers from K_linear
    _is_purely_linear(fem) && return nothing

    fill!(fem.K_tangent.nzval, 0.0)
    fill!(fem.r, 0.0)

    if !isempty(fem.nonlinear_cells)
        n_sdh = length(fem.dh.subdofhandlers)

        for si in 1:n_sdh
            indices = fem._sdh_owned_nonlinear[si]
            isempty(indices) && continue

            # Fetch concrete CV type for this batch to trigger specialization
            cv_ref = fem._sdh_cvs[si]
            av_ref = fem._sdh_avs[si]
            _stiffness_sdh_batch!(fem, si, indices, u, dt, cv_ref, av_ref)
        end

        # Both buffers were zeroed above
        assembler = start_assemble(fem.K_tangent, fem.r; fillzero=false)
        for idx in 1:length(fem.owned_nonlinear_cells)
            assemble!(assembler, fem._cell_dofs[idx], fem._cell_ke[idx], fem._cell_re[idx])
        end
    end

    return nothing
end

function _finish_stiffness_collective!(fem::GenericMaterialAssembler, u::AbstractVector)
    if _is_purely_linear(fem)
        copyto!(fem.K_tangent.nzval, fem.K_linear.nzval)
        mul!(fem.r, fem.K_linear, u)
        return fem.K_tangent, fem.r
    end

    if !isempty(fem.nonlinear_cells)
        mpi_allreduce!(fem.K_tangent.nzval)
        mpi_allreduce!(fem.r)
    end

    # Skipped when every cell is nonlinear: K_linear is all-zero there, and the
    # add plus mul! are ~6% of each assembly.
    if !isempty(fem.linear_cells)
        fem.K_tangent.nzval .+= fem.K_linear.nzval
        mul!(fem.r, fem.K_linear, u, 1.0, 1.0)
    end

    return fem.K_tangent, fem.r
end

"""
    stiffness_matrix(assembler, u; dt=0.0)

Assemble and return `(K, r)`, the current tangent stiffness matrix and the internal residual vector for the displacement vector `u`.

`K` is the *algorithmic* tangent.
For materials that obtain their tangent via automatic differentiation (e.g. `NeoHooke`, `VEPD_Detrez2010`, `VEVP_Zhao2021_AD`, `VEVP_MOAMMM`) it is exact up to AD numerics, and for materials with an analytic tangent it is the linearized form returned by the model.

`dt` is the time step size of the current load step and is forwarded to every material's element routine.
Rate-independent models (`Hooke`, `Hooke2D`, `NeoHooke`, `J2Plasticity`) typically ignore it, while viscoelastic and viscoplastic models use it to evolve internal variables.
For viscous models, `dt=0.0` freezes the viscous evolution rather than producing rate-independent behaviour.

Assembly runs in two passes:

1. Each owned nonlinear cell computes its element `ke` and `re` in parallel
   (`OhMyThreads.@tasks`, greedy scheduler)
2. Per-cell results are scattered into the global matrix and vector, and
   reduced across MPI ranks when `MPI.COMM_WORLD` has more than one process

Linear cells, those whose material has `is_linear == true`, are preassembled once when the assembler is created and are added to the tangent on every call without re-evaluation.

Under MPI this is a collective call that every rank must reach.
A local failure on one rank is made collective before it propagates, so peer ranks throw instead of blocking in the reduction.
Use [`try_stiffness_matrix`](@ref) when the outer solver should recover from local constitutive failures rather than abort.

`K` and `r` are the assembler's reusable buffers and are overwritten by the next assembly call.
`copy` them to keep a previous tangent or residual.
"""
function stiffness_matrix(fem::GenericMaterialAssembler, u::AbstractVector; dt=0.0)
    _run_local_phase!(fem, _assemble_stiffness_local!, false, u, dt)
    return _finish_stiffness_collective!(fem, u)
end

"""
    _run_local_phase!(fem, phase, recover, args...)

Run `phase(fem, args...)` and make any failure collective before it propagates, so
peer ranks cannot block in the reduction that follows.

With `recover = true`, `_run_local_phase!` returns a [`LocalAssemblyFailure`](@ref) instead of throwing it, and returns [`RemoteAssemblyFailure`](@ref) on ranks where another rank failed.
With `recover = false` it always throws.
It returns `nothing` when every rank succeeded.

In serial runs `mpi_any!` skips the reduction at the single rank, and an exception that is not returned to the caller is rethrown in place so its backtrace survives.
"""
function _run_local_phase!(fem::GenericMaterialAssembler, phase::F, recover::Bool, args...) where {F}
    local_error = nothing
    fatal_error = nothing
    try
        phase(fem, args...)
    catch err
        local_error = recover ? _recoverable_local_assembly_failure(err) : nothing
        if local_error === nothing
            mpi_size() == 1 && rethrow()
            fatal_error = err
        end
    end

    if mpi_size() > 1 && mpi_any!(fem._mpi_flag_buffer, fatal_error !== nothing)
        fatal_error === nothing && error("unrecoverable assembly error occurred on another MPI rank")
        throw(fatal_error)
    end

    recover || return nothing
    mpi_any!(fem._mpi_flag_buffer, local_error !== nothing) || return nothing
    return isnothing(local_error) ? RemoteAssemblyFailure() : local_error
end

"""
    try_stiffness_matrix(assembler, u; dt=0.0)

Non-throwing wrapper around [`stiffness_matrix`](@ref) for adaptive outer solvers that reject the current load or time step and retry with a smaller `dt` when a local constitutive update fails.
Returns a named tuple:

- `(converged = true, K = K, r = r, error = nothing)` on success
- `(converged = false, K = nothing, r = nothing, error = err)` when a
  recoverable [`LocalAssemblyFailure`](@ref) occurs

Only `LocalAssemblyFailure` is treated as recoverable.
Every other exception is rethrown, including a `MethodError` in a material implementation and a generic `DomainError` that does not subtype `LocalAssemblyFailure`.
A material whose local constitutive failure should be retried by an adaptive outer solver must therefore throw an exception subtyping `LocalAssemblyFailure`.
Threaded assembly may wrap failures in task exceptions; a wrapped failure is recoverable only when every wrapped leaf exception is a `LocalAssemblyFailure`.

On failure, reject the current outer step, restore the quadrature states with [`revert_states!`](@ref), restore the displacement vector to the last accepted step, reduce `dt`, and assemble again.
Do not call [`update_states!`](@ref) for a failed attempt; commit only after the outer Newton solve has converged.
The assembler's matrix and residual buffers are meaningless after a failed attempt, so use only the returned `error`.

In MPI runs, unrecoverable local errors are made collective before they are thrown, so peer ranks do not wait in a later reduction.
Recoverable failure flags are also reduced across `MPI.COMM_WORLD` before the global tangent and residual reductions.
If one rank fails recoverably, every rank receives `converged = false`, and ranks without the original exception receive [`RemoteAssemblyFailure`](@ref) as `error`.
"""
function try_stiffness_matrix(fem::GenericMaterialAssembler, u::AbstractVector; dt=0.0)
    err = _run_local_phase!(fem, _assemble_stiffness_local!, true, u, dt)
    err === nothing || return (converged=false, K=nothing, r=nothing, error=err)

    K, r = _finish_stiffness_collective!(fem, u)
    return (converged=true, K=K, r=r, error=nothing)
end

# `_assemble_stiffness_local!` runs per-cell work inside `OhMyThreads.@tasks`,
# which wraps thrown exceptions in `TaskFailedException` or `CompositeException`.
# A wrapped failure is recoverable only when every leaf exception is a
# `LocalAssemblyFailure`; mixed failures must still rethrow the original error.
function _recoverable_local_assembly_failure(err)
    first_failure = Ref{Union{Nothing,LocalAssemblyFailure}}(nothing)
    _all_failures_are_recoverable(err, first_failure) || return nothing
    return first_failure[]
end

function _all_failures_are_recoverable(err, first_failure)
    if err isa TaskFailedException
        if isdefined(err, :task) && isdefined(err.task, :result)
            return _all_failures_are_recoverable(err.task.result, first_failure)
        end
        return false
    elseif err isa CompositeException
        isempty(err.exceptions) && return false
        all_recoverable = true
        for wrapped_err in err.exceptions
            all_recoverable &= _all_failures_are_recoverable(wrapped_err, first_failure)
        end
        return all_recoverable
    elseif err isa LocalAssemblyFailure
        first_failure[] === nothing && (first_failure[] = err)
        return true
    else
        return false
    end
end

# Function barrier for stiffness assembly
function _stiffness_sdh_batch!(fem, si, indices, u, dt, cv_ref::CV, av_ref::AV) where {CV,AV}
    grid = fem.dh.grid
    workspaces = fem._workspaces
    n_tasks = min(length(workspaces), length(indices))
    workspace_counter = Threads.Atomic{Int}(0)

    OhMyThreads.@tasks for idx in indices
        @set begin
            scheduler = :greedy
            ntasks = n_tasks
        end
        # One preallocated workspace per spawned task. This is independent of
        # the OS thread running the task, so task migration cannot alias buffers.
        @local workspace = _take_task_workspace!(workspace_counter, workspaces)
        cid = fem.owned_nonlinear_cells[idx]
        mat = fem.materials[fem.cell_to_mat_idx[cid]]

        ke = workspace.kes[si]
        re = workspace.res[si]

        # Zero-cost type assertion inside loop
        cv = workspace.cvs[si]::CV
        av = workspace.avs[si]::AV

        reinit!(cv, getcoordinates(grid, cid))
        _reinit_alpha_values!(av, cid)
        fill!(ke, 0)
        fill!(re, 0)

        celldofs_vals = u[fem._cell_dofs[idx]]
        cell_state = fem.states[cid]

        _assemble_element!(ke, re, cell_state, mat, cv, av, celldofs_vals, dt)

        if fem.thickness != 1.0
            ke .*= fem.thickness
            re .*= fem.thickness
        end
        copyto!(fem._cell_ke[idx], ke)
        copyto!(fem._cell_re[idx], re)
    end
    return nothing
end

"""
    compute_forces(assembler, u; dt=0.0)

Return only the internal force (residual) vector for displacement `u`.

`compute_forces` wraps `stiffness_matrix` and discards the tangent matrix.
It runs the same assembly path, so the per-cell element loops still compute both `ke` and `re`.
Inside a Newton solve, call `stiffness_matrix` directly and ignore the matrix instead, which avoids a second assembly.

`dt` has the same meaning as in `stiffness_matrix`.

The returned vector is the assembler's reusable residual buffer and is overwritten by the next assembly call.
`copy` it to keep a previous residual.
"""
function compute_forces(fem::GenericMaterialAssembler, u::AbstractVector; dt=0.0)
    _, f_int = stiffness_matrix(fem, u; dt=dt)
    return f_int
end

"""
    compute_stresses(assembler, u; dt=0.0)

Evaluate quadrature point stresses for the current displacement vector `u` and return them as an array of `Tensor` values, specifically `Tensor{2,dim}`.

The returned array has shape `(max_nquadpoints, getncells(grid))` and is indexed `σ[qp, cellid]` by Ferrite cell id.
It is sized for the whole grid, so when the `DofHandler` covers only a subset of cells, the columns of inactive cells stay zero and the active cells keep their own ids as column indices; they are not compacted to `1:n_active`.
For mixed sub-dof-handlers, entries above a cell's own quadrature count remain zero.
The array is the assembler's reusable internal stress cache, and a later `compute_stresses` call on the same assembler overwrites it.
`copy` it to keep a previous stress field.

Each quadrature point stress is multiplied by `alpha_value(av, qp)` when the assembler was built with a non-`nothing` `ah` keyword, and by `1.0` otherwise.
See [`alpha_value`](@ref).

`dt` defaults to `0.0` and is passed to the material's stress routine for API consistency.
For postprocessing, call `compute_stresses(assembler, u)` and leave that default in place.
Stress postprocessing is expected to be non-evolving: bundled stateful materials report the current trial or the committed stress without advancing history or committing state.
`compute_stresses` is not a substitute for `stiffness_matrix`, `compute_forces`, or `update_states!`.
Perform the time step update with assembly, then call `update_states!` after the outer solve has converged.

`compute_stresses` loops over all owned cells, linear and nonlinear, and runs in parallel with `OhMyThreads.@tasks`.
When `MPI.COMM_WORLD` has more than one process the per-rank results are summed across ranks, so every rank ends up with the same global stress field.
It is therefore a collective call that every rank must reach.
A local failure on one rank is made collective before it propagates, so peer ranks throw instead of blocking in the reduction.
"""
function compute_stresses(fem::GenericMaterialAssembler, u::AbstractVector; dt=0.0)
    _run_local_phase!(fem, _compute_stresses_local!, false, u, dt)

    stresses = fem._stresses_cache
    nranks = mpi_size()

    if nranks > 1
        v_stresses = vec(reinterpret(Float64, stresses))
        buffer = fem._stresses_mpi_buffer
        length(buffer) == length(v_stresses) || resize!(buffer, length(v_stresses))
        copyto!(buffer, v_stresses)
        mpi_allreduce!(buffer)
        copyto!(v_stresses, buffer)
    end

    return stresses
end

function _compute_stresses_local!(fem::GenericMaterialAssembler, u::AbstractVector, dt)
    stresses = fem._stresses_cache
    fill!(stresses, zero(eltype(stresses)))

    for si in 1:length(fem.dh.subdofhandlers)
        owned_indices = fem._sdh_owned_all[si]
        isempty(owned_indices) && continue

        cv_ref = fem._sdh_cvs[si]
        av_ref = fem._sdh_avs[si]
        _stress_sdh_batch!(fem, si, owned_indices, u, dt, cv_ref, av_ref)
    end

    return nothing
end

# Function barrier for stress computation
function _stress_sdh_batch!(fem, si, indices, u, dt, cv_ref::CV, av_ref::AV) where {CV,AV}
    grid = fem.dh.grid
    nqp = fem._sdh_nqp[si]
    stresses = fem._stresses_cache
    workspaces = fem._workspaces
    n_tasks = min(length(workspaces), length(indices))
    workspace_counter = Threads.Atomic{Int}(0)

    OhMyThreads.@tasks for idx in indices
        @set begin
            scheduler = :greedy
            ntasks = n_tasks
        end
        # One preallocated workspace per spawned task. This is independent of
        # the OS thread running the task, so task migration cannot alias buffers.
        @local workspace = _take_task_workspace!(workspace_counter, workspaces)
        cid = fem._owned_all_cells[idx]
        mat_idx = fem.cell_to_mat_idx[cid]
        if mat_idx != 0
            mat = fem.materials[mat_idx]
            cv = workspace.cvs[si]::CV
            av = workspace.avs[si]::AV

            reinit!(cv, getcoordinates(grid, cid))
            _reinit_alpha_values!(av, cid)

            celldofs_vals = u[fem._owned_all_dofs[idx]]
            cell_state = isassigned(fem.states, cid) ? fem.states[cid] : nothing
            cell_num = cid

            for qp in 1:nqp
                α = alpha_value(av, qp)
                qp_state = isnothing(cell_state) ? nothing : cell_state[qp]
                σ = _compute_stress_qp(mat, cv, av, qp, celldofs_vals, qp_state, dt)
                stresses[qp, cell_num] = α * σ
            end
        end
    end
end

"""
    update_states!(assembler)

Commit the trial states of all owned nonlinear quadrature points after a converged Newton step.

`update_states!` calls `update_state!` on every quadrature point state of every owned nonlinear cell, copying the trial values into the committed slot.
Call it once per load step, after Newton has converged, and never inside the Newton loop.
If the step is rejected instead, call [`revert_states!`](@ref) to discard the trial values.

`update_state!` either mutates the state in place and returns `nothing`, or returns a replacement `AbstractMaterialState`.
A returned state is stored back into the quadrature point state vector.

Cells with a linear material hold no history and are skipped, as are cells owned by other MPI ranks.
"""
function update_states!(fem::GenericMaterialAssembler)
    # `Threads.@threads`, not the `OhMyThreads.@tasks` used for assembly
    # too cheap for a task scheduler / tested faster than every OhMyThreads scheduler
    isempty(fem.owned_nonlinear_cells) && return
    Threads.@threads for cid in fem.owned_nonlinear_cells
        if isassigned(fem.states, cid)
            cell_states = fem.states[cid]
            for qp in eachindex(cell_states)
                updated = update_state!(cell_states[qp])
                if updated isa AbstractMaterialState
                    cell_states[qp] = updated
                end
            end
        end
    end
end

"""
    revert_states!(assembler)

Discard the trial states of all owned nonlinear quadrature points and restore their last committed values.

`revert_states!` calls `revert_state!` on every quadrature point state of every owned nonlinear cell, copying the committed values back into the trial slot.
Call it when a trial step is rejected, typically before retrying with a smaller time step.

`revert_state!` either mutates the state in place and returns `nothing`, or returns a replacement `AbstractMaterialState`.
A returned state is stored back into the quadrature point state vector.

Cells with a linear material hold no history and are skipped, as are cells owned by other MPI ranks.
"""
function revert_states!(fem::GenericMaterialAssembler)
    isempty(fem.owned_nonlinear_cells) && return
    Threads.@threads for cid in fem.owned_nonlinear_cells
        if isassigned(fem.states, cid)
            cell_states = fem.states[cid]
            for qp in eachindex(cell_states)
                reverted = revert_state!(cell_states[qp])
                if reverted isa AbstractMaterialState
                    cell_states[qp] = reverted
                end
            end
        end
    end
end