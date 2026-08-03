using Ferrite
using FerriteSolidMechanics
using LinearAlgebra
using MPI

"""
    run_mpi_four_point_bending(; kwargs...)

Run a 2D plane strain four-point bending example.

The same file can be launched directly with `julia --project=.` or through
MPI/SLURM with `srun julia --project=.`. All ranks assemble and solve the
same global system; FerriteSolidMechanics partitions element work internally once
MPI is initialized. If MPI has not been initialized, the function runs as a
single sequential rank. Rank 0 writes the PVD/VTU output.
"""
function run_mpi_four_point_bending(;
    density::Int=4,              # mesh density: nx=10*density, ny=density
    beam_length::Float64=10.0,   # beam length L
    beam_height::Float64=1.0,    # beam height H
    load_pressure::Float64=0.4,  # downward traction magnitude on each loaded top segment
    load_steps::Int=120,         # ramp steps from zero to full load
    hold_steps::Int=80,          # steps held at full load
    recover_steps::Int=80,       # zero-load recovery steps after immediate unload
    dt::Float64=0.05,            # time step size passed to the material
    max_newton::Int=25,          # maximum Newton iterations per step
    newton_tol::Float64=1e-7,    # relative Newton residual tolerance
    output_every::Int=1,         # write every Nth converged step plus the final step
    vtk::Union{String,Nothing}=joinpath(@__DIR__, "results", "mpi_four_point_bending", "four_point_bending"), # nothing to disable VTK
)
    density >= 1 || throw(ArgumentError("density must be >= 1"))
    beam_length > 0 || throw(ArgumentError("beam_length must be > 0"))
    beam_height > 0 || throw(ArgumentError("beam_height must be > 0"))
    load_pressure >= 0 || throw(ArgumentError("load_pressure must be >= 0"))
    load_steps >= 1 || throw(ArgumentError("load_steps must be >= 1"))
    hold_steps >= 0 || throw(ArgumentError("hold_steps must be >= 0"))
    recover_steps >= 0 || throw(ArgumentError("recover_steps must be >= 0"))
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    max_newton >= 1 || throw(ArgumentError("max_newton must be >= 1"))
    newton_tol > 0 || throw(ArgumentError("newton_tol must be > 0"))
    output_every >= 1 || throw(ArgumentError("output_every must be >= 1"))

    rank = _example_mpi_rank()
    nranks = _example_mpi_size()
    _example_mpi_barrier()
    elapsed_ns = time_ns()

    nx = 10 * density
    ny = density
    grid = generate_grid(Quadrilateral, (nx, ny), Vec(0.0, 0.0), Vec(beam_length, beam_height))

    hx = beam_length / nx
    half_segment = max(1.01 * hx, 0.025 * beam_length)
    x_support_left = 0.10 * beam_length
    x_support_right = 0.90 * beam_length
    x_load_left = 0.30 * beam_length
    x_load_right = 0.70 * beam_length

    on_bottom_segment(x, x0) = isapprox(x[2], 0.0; atol=1e-10) && abs(x[1] - x0) <= half_segment
    on_top_segment(x, x0) = isapprox(x[2], beam_height; atol=1e-10) && abs(x[1] - x0) <= half_segment
    addfacetset!(grid, "support_left", x -> on_bottom_segment(x, x_support_left))
    addfacetset!(grid, "support_right", x -> on_bottom_segment(x, x_support_right))
    addfacetset!(grid, "load_left", x -> on_top_segment(x, x_load_left))
    addfacetset!(grid, "load_right", x -> on_top_segment(x, x_load_right))

    for name in ("support_left", "support_right")
        isempty(getfacetset(grid, name)) && error("facetset '$name' is empty; increase support segment width or adjust mesh density")
    end

    interpolation = Lagrange{RefQuadrilateral,1}()^2
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "support_left"), (x, t) -> [0.0, 0.0], [1, 2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "support_right"), (x, t) -> [0.0], [2]))
    close!(ch)

    material = PlaneStrain(VEVP_Zhao2021_AD(5.0, 20.0, 6, 0.0, 0.0, 283.0, 18.51, 0.1421, 0.6152, 3.0, 3.0, 0.0, 0.0, 283.0))
    assembler = create_assembler(material, dh, ch; quadrature_order=2)

    # The load function's second argument is the value passed to external_forces!,
    # either a load factor or physical time; x is the spatial coordinate.
    lh = LoadHandler(assembler)
    for name in ("load_left", "load_right")
        add!(lh, Traction(name, (x, factor) -> Vec(0.0, -load_pressure * factor)))
    end
    close!(lh)

    u = zeros(ndofs(dh))
    apply!(u, ch)

    f_ext = external_forces!(lh, 0.0)
    f_int = zeros(ndofs(dh))
    residual = zeros(ndofs(dh))

    output_dir = vtk === nothing ? nothing : dirname(vtk)
    pvd = nothing
    history_io = nothing
    if rank == 0 && vtk !== nothing
        Base.invokelatest(_fpb_ensure_writevtk_loaded)
        mkpath(output_dir)
        pvd = Base.invokelatest(_fpb_open_pvd, vtk)
        history_io = open(joinpath(output_dir, "history.txt"), "w")
        println(history_io, "step,time,load_factor,min_uy,max_uy,residual_norm,newton_iterations")
    end

    write_output_step!(pvd, history_io, rank, 0, 0.0, 0.0, dh, grid, assembler, u, f_int, f_ext, 0.0, 0, vtk)

    load_profile = vcat(
        collect(range(1.0 / load_steps, 1.0; length=load_steps)),
        fill(1.0, hold_steps),
        [0.0],
        fill(0.0, recover_steps),
    )

    rank == 0 && println(
        "MPI four-point bending: density=$density, cells=$(getncells(grid)), dofs=$(ndofs(dh)), ",
        "ranks=$nranks, steps=$(length(load_profile))",
    )

    min_uy0, max_uy0 = nodal_uy_extrema(dh, u)
    min_uy_history = Float64[min_uy0]
    max_uy_history = Float64[max_uy0]
    residual_history = Float64[0.0]
    iteration_history = Int[0]

    # Give the replicated K \ residual solve its own BLAS threads; returns 1 in 2D (no-op), ncores in 3D.
    LinearAlgebra.BLAS.set_num_threads(recommended_blas_threads(dh))

    for (step, load_factor) in enumerate(load_profile)
        # External loads do not depend on u, so they are assembled only initially per step
        f_ext = external_forces!(lh, load_factor)

        converged = false
        final_residual_norm = Inf
        final_iter = 0

        for iter in 1:max_newton
            K, f_int_step = stiffness_matrix(assembler, u; dt=dt)
            copyto!(f_int, f_int_step)
            residual .= f_int .- f_ext
            apply_zero!(K, residual, ch)
            final_residual_norm = norm(residual)
            final_iter = iter

            threshold = newton_tol * max(norm(f_ext), 1.0)
            if final_residual_norm <= threshold
                converged = true
                break
            end

            u .-= K \ residual
            # Large 3D systems only (Unix/cluster): split the factorization across ranks.
            # Add MUMPS.jl to the env, `import MUMPS` once at the top, and swap the line above for:
            #   u .-= distributed_solve(K, residual, MPI.COMM_WORLD)
            # Use instead of the BLAS tuning above, not with it. See the tutorial's distributed solve section.
        end

        converged || error("Newton did not converge at step $step, load_factor=$load_factor, norm=$final_residual_norm")
        update_states!(assembler)

        min_uy, max_uy = nodal_uy_extrema(dh, u)
        push!(min_uy_history, min_uy)
        push!(max_uy_history, max_uy)
        push!(residual_history, final_residual_norm)
        push!(iteration_history, final_iter)

        if iszero(step % output_every) || step == length(load_profile)
            write_output_step!(pvd, history_io, rank, step, step * dt, load_factor, dh, grid, assembler, u, f_int, f_ext, final_residual_norm, final_iter, vtk)
        end

        rank == 0 && println(
            "  step $(lpad(step, 4)): load=$(round(load_factor; digits=4)), ",
            "iters=$final_iter, |r|=$(round(final_residual_norm; sigdigits=4)), ",
            "min uy=$(round(min_uy_history[end]; sigdigits=5))",
        )
    end

    if rank == 0 && vtk !== nothing
        Base.invokelatest(_fpb_close_pvd, pvd)
        close(history_io)
        println("Wrote $(vtk).pvd")
        println("Wrote $(joinpath(output_dir, "history.txt"))")
    end

    _example_mpi_barrier()
    elapsed_s = (time_ns() - elapsed_ns) / 1.0e9
    if rank == 0
        println("Peak downward displacement: $(minimum(min_uy_history))")
        println("Residual downward displacement after recovery: $(min_uy_history[end])")
        println("Elapsed wall time: $(round(elapsed_s; digits=3)) s (ranks=$nranks, threads=$(Threads.nthreads()))")
    end

    return (u=u, grid=grid, dh=dh, ch=ch, assembler=assembler, load_profile=load_profile,
    min_uy_history=min_uy_history, max_uy_history=max_uy_history,
    residual_history=residual_history, iteration_history=iteration_history, elapsed_s=elapsed_s)
end

function _fpb_ensure_writevtk_loaded()
    try
        @eval using WriteVTK
    catch
        throw(ArgumentError("VTK output requires WriteVTK.jl in the active environment"))
    end
    return nothing
end

function _example_mpi_rank()
    return MPI.Initialized() && !MPI.Finalized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
end

function _example_mpi_size()
    return MPI.Initialized() && !MPI.Finalized() ? MPI.Comm_size(MPI.COMM_WORLD) : 1
end

function _example_mpi_barrier()
    MPI.Initialized() && !MPI.Finalized() && MPI.Barrier(MPI.COMM_WORLD)
    return nothing
end

function write_output_step!(
    pvd, history_io, rank::Int, step::Int, step_time::Float64, load_factor::Float64,
    dh, grid, assembler, u, f_int, f_ext, residual_norm::Float64, newton_iterations::Int,
    vtk::Union{String,Nothing},
)
    vtk === nothing && return
    # compute_stresses is collective: every rank must call it before the rank-0 guard.
    stresses = compute_stresses(assembler, u)
    rank == 0 || return

    stress_components = cell_average_stress_components(stresses, getncells(grid))
    vtk_path = "$(vtk)_$(lpad(step, 5, '0'))"
    Base.invokelatest(
        _fpb_write_vtk_step!, pvd, dh, grid, u, f_int, f_ext, stress_components, load_factor, step_time, vtk_path,
    )

    min_uy, max_uy = nodal_uy_extrema(dh, u)
    println(
        history_io,
        step, ",", step_time, ",", load_factor, ",",
        min_uy, ",", max_uy, ",",
        residual_norm, ",", newton_iterations,
    )
    flush(history_io)
    return nothing
end

# Internal helpers: called via invokelatest because WriteVTK is loaded at runtime.
_fpb_open_pvd(vtk) = paraview_collection(vtk)
_fpb_close_pvd(pvd) = vtk_save(pvd)

function _fpb_write_vtk_step!(pvd, dh, grid, u, f_int, f_ext, stress_components, load_factor, step_time, path)
    sigma_xx, sigma_yy, sigma_xy, sigma_vm = stress_components
    VTKGridFile(path, dh) do vtk_file
        write_solution(vtk_file, dh, u)
        write_solution(vtk_file, dh, f_int, "f_int")
        write_solution(vtk_file, dh, f_ext, "f_ext")
        write_cell_data(vtk_file, sigma_xx, "sigma_xx_cellavg")
        write_cell_data(vtk_file, sigma_yy, "sigma_yy_cellavg")
        write_cell_data(vtk_file, sigma_xy, "sigma_xy_cellavg")
        write_cell_data(vtk_file, sigma_vm, "sigma_vm_2d_cellavg")
        write_cell_data(vtk_file, fill(load_factor, getncells(grid)), "load_factor")
        for name in ("support_left", "support_right", "load_left", "load_right")
            Ferrite.write_facetset(vtk_file, grid, name)
        end
        pvd[step_time] = vtk_file
    end
    return nothing
end

function cell_average_stress_components(stresses, ncells::Int)
    sigma_xx = Vector{Float64}(undef, ncells)
    sigma_yy = Vector{Float64}(undef, ncells)
    sigma_xy = Vector{Float64}(undef, ncells)
    sigma_vm = Vector{Float64}(undef, ncells)

    for cell in 1:ncells
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for qp in axes(stresses, 1)
            sigma = stresses[qp, cell]
            sxx += sigma[1, 1]
            syy += sigma[2, 2]
            sxy += sigma[1, 2]
        end
        nqp = size(stresses, 1)
        sxx /= nqp
        syy /= nqp
        sxy /= nqp

        sigma_xx[cell] = sxx
        sigma_yy[cell] = syy
        sigma_xy[cell] = sxy
        sigma_vm[cell] = sqrt(max(sxx^2 - sxx * syy + syy^2 + 3.0 * sxy^2, 0.0))
    end

    return sigma_xx, sigma_yy, sigma_xy, sigma_vm
end

nodal_uy_extrema(dh, u) = extrema(v[2] for v in evaluate_at_grid_nodes(dh, u, :u))

function main(; kwargs...)
    started_mpi = false
    if !MPI.Initialized()
        MPI.Init()
        started_mpi = true
    end

    try
        run_mpi_four_point_bending(; kwargs...)
    finally
        if MPI.Initialized() && !MPI.Finalized()
            MPI.Barrier(MPI.COMM_WORLD)
        end
        started_mpi && !MPI.Finalized() && MPI.Finalize()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end