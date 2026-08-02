# Package extension providing `distributed_solve` via the MUMPS.jl distributed sparse
# direct solver. Activated by loading MUMPS via `import MUMPS`.
#
# A fresh Mumps instance is created per call: reusing a handle across Newton iterations
# corrupts the second solve, and the rebuild cost is negligible next to the factorization.
# `finalize(m)` calls `Base.finalize`, which triggers MUMPS' destructor.
module FerriteSolidMechanicsMUMPSExt

using FerriteSolidMechanics: FerriteSolidMechanics
using LinearAlgebra: mul!, norm
using MPI
import MUMPS

# Extra room for MUMPS' internal workarray, as ICNTL(14) percentages. Too little fails
# the factorization with INFO(1) = -9, so 20 (the MUMPS default) escalates on retry.
const _ICNTL14_SCHEDULE = (20, 60, 150, 400)

function FerriteSolidMechanics.distributed_solve(K, residual, comm; verify_rtol::Real=1.0e-8,
                                            workspace_pct::Union{Integer,Nothing}=nothing)
    # An explicit percentage is a cap the caller chose, so it is used without retrying.
    workspace_pct === nothing ||
        return _solve_once(K, residual, comm, workspace_pct, verify_rtol)
    for workspace_pct in _ICNTL14_SCHEDULE[1:end-1]
        try
            return _solve_once(K, residual, comm, workspace_pct, verify_rtol)
        catch err
            # Only -9 is worth more room. Every rank raised it from the same reduced code,
            # so they all retry together and the next call stays collective.
            (err isa FerriteSolidMechanics.DistributedSolveError && err.infog1 == -9) || rethrow()
        end
    end
    return _solve_once(K, residual, comm, last(_ICNTL14_SCHEDULE), verify_rtol)
end

function _solve_once(K, residual, comm, workspace_pct, verify_rtol)
    icntl = MUMPS.default_icntl[:]
    icntl[3] = 0   # no global-info stream
    icntl[4] = 1   # errors only, no diagnostics
    icntl[14] = workspace_pct
    m = MUMPS.Mumps{Float64}(MUMPS.mumps_unsymmetric, icntl, MUMPS.default_cntl64)
    du = Vector{Float64}(undef, length(residual))
    # `finally` because the status checks throw: an instance left for the GC is finalized
    # after MPI_Finalize, and MUMPS' destructor then aborts the run. Collective either way.
    try
        MUMPS.associate_matrix!(m, K)
        MUMPS.factorize!(m)
        _check_status(m, :factorize, comm)
        MUMPS.associate_rhs!(m, copy(residual))   # copy: MUMPS overwrites the rhs in place
        MUMPS.solve!(m)
        _check_status(m, :solve, comm)
        MPI.Comm_rank(comm) == 0 && (du .= MUMPS.get_solution(m))
    finally
        finalize(m)
    end
    MPI.Bcast!(du, 0, comm)
    verify_rtol > 0 && _check_backward_error(K, du, residual, verify_rtol)
    return du
end

# Both arrays: INFOG (global) reports a singular factorization, but a workspace failure
# reaches only the failing rank's INFO. Reduced, so one rank's failure stops all of them.
function _check_status(m, phase::Symbol, comm)
    code = 0
    for field in (:infog, :info)
        arr = hasproperty(m, field) ? getproperty(m, field) : nothing
        (arr === nothing || isempty(arr)) && continue
        code = min(code, Int(first(arr)))
    end
    # -1 only says "some other rank failed"; prefer a rank's own diagnostic code.
    worst = MPI.Comm_size(comm) > 1 ? MPI.Allreduce(code, MPI.MIN, comm) : code
    worst < 0 && throw(FerriteSolidMechanics.DistributedSolveError(phase, worst))
    return nothing
end

# A direct solve is backward stable, so `K * du` must reproduce `residual`. One matvec,
# and the only check that does not depend on MUMPS reporting its own failure.
function _check_backward_error(K, du, residual, rtol)
    scale = norm(residual)
    scale == 0 && return nothing
    w = similar(du)
    mul!(w, K, du)
    w .-= residual
    rel = norm(w) / scale
    isfinite(rel) && rel <= rtol ||
        throw(FerriteSolidMechanics.DistributedSolveError(:verify, 0, rel))
    return nothing
end

end
