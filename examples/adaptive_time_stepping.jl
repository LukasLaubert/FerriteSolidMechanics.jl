using Ferrite
using FerriteSolidMechanics
using LinearAlgebra

# Reuse the quarter-plate-with-hole mesh helper from the plate tutorial.
# Guard the include so this file can be loaded together
# with the plate example without redefining methods.
if !isdefined(@__MODULE__, :quarter_plate_with_hole_grid)
    include(joinpath(@__DIR__, "plate_with_hole_planestress.jl"))
end
const _quarter_plate_with_hole_grid = getfield(@__MODULE__, :quarter_plate_with_hole_grid)

# Default material: VEPD_Detrez2010 with parameters chosen so the local plastic
# return can raise a recoverable `VEPD_Detrez2010ConvergenceError`
# (<: `LocalAssemblyFailure`) when `dt` is too coarse. Swap to any other
# material by passing `material=` to `run_adaptive_time_stepping`.
#
# (E, ν, R0, Q, b, α, β, n_ab, μ_ab, G, τ)
const _DETREZ_PARAMETERS = (
    167.0, 0.27, 3.24, 28.6, 20.8,
    0.59, 41.7, 4.5, 3.58,
    [101.0, 21.0], [10.0, 100.0],
)
_default_material() = PlaneStrain(VEPD_Detrez2010(_DETREZ_PARAMETERS...))


"""
    run_adaptive_time_stepping(; material=_default_material(), kwargs...)

Quarter plate-with-hole (plane strain) driven by an adaptive outer loop: a step is
retried with a smaller `dt` when a local update raises a recoverable
`LocalAssemblyFailure` or the Newton loop fails to converge, and `dt` grows back
toward `dt_max` after accepted steps. Works with any `AbstractMaterial`.
"""
function run_adaptive_time_stepping(;
    material::AbstractMaterial=_default_material(), # any AbstractMaterial
    nr::Int=4,                   # radial mesh resolution of the quarter-plate-with-hole grid
    ntheta::Int=12,              # circumferential mesh resolution
    radius::Float64=0.25,        # hole radius
    plate_size::Float64=1.0,     # outer plate size
    displacement::Float64=0.12,  # full imposed right-edge displacement at t = t_end (default provokes a few recoverable rejections with the default material)
    t_end::Float64=1.0,          # end time of the ramp
    dt::Float64=0.25,            # starting trial step size
    dt_min::Float64=1e-4,        # lower bound checked when shrinking rejected steps
    dt_max::Float64=0.25,        # upper bound applied when growing after accepted steps
    dt_shrink::Float64=0.5,      # step size factor on a rejected step
    dt_grow::Float64=2.0,        # step size factor on an accepted step
    max_rejections::Int=20,      # max consecutive rejected trial steps before aborting
    max_newton::Int=30,          # max Newton iterations per accepted-step attempt
    newton_tol::Float64=1e-6,    # absolute residual tolerance
    vtk::Union{String,Nothing}=nothing, # nothing to disable VTK output, or a PVD basename
)
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    controller = TimeStepController(; dt_min, dt_max, shrink=dt_shrink, grow=dt_grow, max_rejections)

    grid = _quarter_plate_with_hole_grid(; nr, ntheta, radius, plate_size)

    interpolation = Lagrange{RefQuadrilateral,1}()^2
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)

    # Symmetry BCs: bottom edge u_y = 0, left edge u_x = 0.
    # Right edge is displaced outward, scaled by the load parameter t.
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "symmetry_x"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "symmetry_y"), (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [displacement * t], [1]))
    close!(ch)

    assembler = create_assembler(material, dh, ch; quadrature_order=2)

    u = zeros(ndofs(dh))
    apply!(u, ch)

    # Optional VTK output
    pvd = nothing
    if vtk !== nothing
        Base.invokelatest(_adaptive_ensure_writevtk_loaded)
        mkpath(dirname(vtk))
        pvd = Base.invokelatest(_adaptive_open_pvd, vtk)
    end

    # Adaptive outer loop: dt shrinks on failure, grows on acceptance
    t = 0.0
    step = 0

    # Histories (one entry per *accepted* step)
    history_t = Float64[]
    history_dt = Float64[]
    history_res = Float64[]
    history_iters = Int[]
    accepted_steps = 0
    rejected_steps = 0
    # Count successful controller updates: accept_step! may saturate at dt_max;
    # reject_step! either decreases dt or throws below dt_min
    dt_decreases = 0
    dt_increases = 0

    _write_output_step!(pvd, dh, u, t, step, vtk)

    while t < t_end - 1e-12
        accepted = false
        u_start = copy(u)
        t_start = t

        while !accepted
            t_trial = min(t_start + dt, t_end)
            dt_step = t_trial - t_start
            update!(ch, t_trial)
            u .= u_start
            apply!(u, ch)

            newton_converged = false
            step_rejected = false
            final_res = Inf
            final_iter = 0

            for iter in 1:max_newton
                result = try_stiffness_matrix(assembler, u; dt=dt_step)
                if !result.converged
                    # Recoverable local failure: roll back state and shrink the attempted step
                    revert_states!(assembler)
                    u .= u_start
                    dt = reject_step!(controller, dt_step)
                    dt_decreases += 1
                    step_rejected = true
                    rejected_steps += 1
                    break
                end

                K, r = result.K, result.r
                apply_zero!(K, r, ch)
                final_res = norm(r)
                final_iter = iter
                if final_res <= newton_tol
                    newton_converged = true
                    break
                end
                u .-= K \ r
            end

            if newton_converged
                update_states!(assembler)
                t = t_trial
                step += 1
                accepted_steps += 1
                accepted = true
                push!(history_t, t)
                push!(history_dt, dt_step)
                push!(history_res, final_res)
                push!(history_iters, final_iter)
                old_dt = dt_step
                dt = accept_step!(controller, dt_step)
                dt > old_dt && (dt_increases += 1)
            elseif !step_rejected
                # Newton failed to converge: also shrink and retry.
                revert_states!(assembler)
                u .= u_start
                dt = reject_step!(controller, dt_step)
                dt_decreases += 1
                rejected_steps += 1
            end
        end

        _write_output_step!(pvd, dh, u, t, step, vtk)
    end

    pvd !== nothing && Base.invokelatest(_adaptive_close_pvd, pvd)

    stresses = compute_stresses(assembler, u)
    return (u=u, stresses=stresses, grid=grid, dh=dh, ch=ch, assembler=assembler,
        accepted_steps=accepted_steps, rejected_steps=rejected_steps,
        dt_decreases=dt_decreases, dt_increases=dt_increases, history_t=history_t,
        history_dt=history_dt, history_res=history_res, history_iters=history_iters)
end

function _adaptive_ensure_writevtk_loaded()
    try
        @eval using WriteVTK
    catch
        throw(ArgumentError("VTK output requires WriteVTK.jl in the active environment"))
    end
    return nothing
end

function _write_output_step!(pvd, dh, u, step_time, step, vtk)
    pvd === nothing && return
    vtk === nothing && return
    Base.invokelatest(_adaptive_write_vtk_step!, pvd, dh, u, step_time, "$(vtk)_$(lpad(step, 5, '0'))")
    return nothing
end

# Internal helpers: called via invokelatest because WriteVTK is loaded at runtime.
_adaptive_open_pvd(vtk) = paraview_collection(vtk)
_adaptive_close_pvd(pvd) = vtk_save(pvd)

function _adaptive_write_vtk_step!(pvd, dh, u, step_time, path)
    VTKGridFile(path, dh) do vtk_file
        write_solution(vtk_file, dh, u)
        pvd[step_time] = vtk_file
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_adaptive_time_stepping()
    println("Solved adaptive time stepping quarter plate")
    println("  cells: $(getncells(result.grid))")
    println("  dofs: $(ndofs(result.dh))")
    println("  accepted steps: $(result.accepted_steps)")
    println("  rejected/retried steps: $(result.rejected_steps)")
    println("  dt decreases: $(result.dt_decreases)")
    println("  dt increases: $(result.dt_increases)")
    println("  peak displacement: $(maximum(abs, result.u))")
    if isempty(result.history_dt)
        println("  (no accepted-step history)")
    else
        min_dt = minimum(result.history_dt)
        max_dt = maximum(result.history_dt)
        println("  dt range over accepted steps: [$min_dt, $max_dt]")
    end
end