using Test
using MPI
using Serialization
using LinearAlgebra

# Rank count is fixed at launch, so this spawns `mpiexec -n N` and compares 1
# rank against 2. K and r need a tolerance because the Allreduce reassociates
# the summation; a dropped or double-counted cell would be an O(1) difference,
# far outside it. Stresses stay bitwise: each rank writes only its own cells,
# so the reduction adds exact zeros. External loads are replicated, never reduced

const MPI_WORKER = joinpath(@__DIR__, "mpi_worker.jl")
const MPI_PROJECT = dirname(@__DIR__)

function mpiexec_available()
    try
        return mpiexec() do exe
            success(pipeline(`$exe -n 1 $(Base.julia_cmd()) --startup-file=no -e "exit(0)"`;
                             stdout=devnull, stderr=devnull))
        end
    catch
        return false
    end
end

function run_mpi_worker(nranks::Int, dir::String)
    outfile = joinpath(dir, "mpi_result_$nranks.jls")
    ok = mpiexec() do exe
        cmd = `$exe -n $nranks $(Base.julia_cmd()) --project=$MPI_PROJECT --startup-file=no $MPI_WORKER $outfile`
        success(pipeline(cmd; stdout=devnull, stderr=devnull))
    end
    ok || error("mpi worker failed at -n $nranks")
    return deserialize(outfile)
end

if !mpiexec_available()
    @warn "Skipping MPI tests: no usable mpiexec (MPI.jl library = $(MPI.MPI_LIBRARY))"
else
    @testset "MPI assembly agrees across rank counts" begin
        mktempdir() do dir
            r1 = run_mpi_worker(1, dir)
            r2 = run_mpi_worker(2, dir)

            @test r1["__nranks__"] == 1
            @test r2["__nranks__"] == 2
            @test keys(r1) == keys(r2)

            for case in ("neo_hooke_3d", "hooke_3d_linear", "j2_3d", "j2_3d_step2", "try_stiffness_ok")
                nz1, rr1 = r1[case]
                nz2, rr2 = r2[case]
                @test length(nz1) == length(nz2)
                @test isapprox(nz1, nz2; rtol=1e-10, atol=1e-12)
                @test isapprox(rr1, rr2; rtol=1e-10, atol=1e-12)
                @test norm(nz1 - nz2) <= 1e-8 * max(norm(nz1), 1.0)
            end

            @test r1["neo_hooke_3d_stress"] == r2["neo_hooke_3d_stress"]

            # External loads are replicated with no reduction, so nothing reassociates
            @test r1["external_forces"] == r2["external_forces"]
            @test norm(r1["external_forces"]) > 0.0

            # Rank 0 fails recoverably; the worker asserts every rank agrees and that
            # non-failing ranks get RemoteAssemblyFailure. Rank 0 keeps its own error.
            @test r1["recoverable_failure"] == (0, 0)
            @test r2["recoverable_failure"] == (0, 0)
        end
    end
end
