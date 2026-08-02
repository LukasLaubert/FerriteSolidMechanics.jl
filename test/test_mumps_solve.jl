# Tests the MUMPS extension's `distributed_solve`. Needs MUMPS.jl installed and a
# usable mpiexec; runtests.jl includes this only on Unix with MUMPS present, and
# the mpiexec_available() guard below skips if no launcher works. Spawns the
# worker at 1 and 2 ranks (rank count is fixed at launch) and checks the update
# solves the system and agrees across rank counts within MUMPS's reassociation
# tolerance. Run standalone with e.g.
#   julia --project=<env with MUMPS + FerriteSolidMechanics> test/test_mumps_solve.jl
using Test
using MPI
using Serialization
using LinearAlgebra

const MUMPS_WORKER = joinpath(@__DIR__, "mumps_solve_worker.jl")
const MUMPS_PROJECT = dirname(@__DIR__)

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

function run_mumps_worker(nranks::Int, dir::String)
    outfile = joinpath(dir, "mumps_result_$nranks.jls")
    ok = mpiexec() do exe
        cmd = `$exe -n $nranks $(Base.julia_cmd()) --project=$MUMPS_PROJECT --startup-file=no $MUMPS_WORKER $outfile`
        success(pipeline(cmd; stdout=devnull, stderr=devnull))
    end
    ok || error("mumps worker failed at -n $nranks")
    return deserialize(outfile)
end

if !mpiexec_available()
    @warn "Skipping MUMPS distributed_solve tests: no usable mpiexec (MPI.jl library = $(MPI.MPI_LIBRARY))"
else
    @testset "distributed_solve (MUMPS) matches the direct solve" begin
        mktempdir() do dir
            r1 = run_mumps_worker(1, dir)
            r2 = run_mumps_worker(2, dir)

            @test r1["nranks"] == 1
            @test r2["nranks"] == 2
            @test r1["maxresid"] < 1.0e-8
            @test r2["maxresid"] < 1.0e-8
            # 1 vs 2 ranks: MUMPS reassociates, so agreement is tolerance, not bitwise
            @test norm(r1["du"] - r2["du"]) / norm(r1["du"]) < 1.0e-8
        end
    end
end
