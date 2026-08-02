using Test

# Thread count is fixed at startup, so the workload runs as subprocesses
# The worker's grids are sized to spawn several tasks; the 1-2 cell grids
# elsewhere in this suite stay single-task even at -t8

const WORKER = joinpath(@__DIR__, "thread_determinism_worker.jl")
const PROJECT = dirname(@__DIR__)

function run_worker(nthreads::Int)
    cmd = `$(Base.julia_cmd()) --project=$PROJECT --threads=$nthreads --startup-file=no $WORKER`
    out = read(pipeline(cmd; stderr=devnull), String)
    hashes = Dict{String,String}()
    reported = Ref(0)
    for line in eachline(IOBuffer(out))
        parts = split(line)
        if length(parts) == 3 && parts[1] == "HASH"
            hashes[parts[2]] = parts[3]
        elseif length(parts) == 2 && parts[1] == "NTHREADS"
            reported[] = parse(Int, parts[2])
        end
    end
    return hashes, reported[]
end

@testset "assembly is bitwise independent of thread count" begin
    # 4 and 8 oversubscribe typical CI runners on purpose: task migration
    # between workspaces gets more likely
    thread_counts = [1, 4, 8]

    results = Dict{Int,Dict{String,String}}()
    for n in thread_counts
        hashes, reported = run_worker(n)
        @test reported == n            # the subprocess really got n threads
        @test !isempty(hashes)         # the worker ran to completion
        results[n] = hashes
    end

    reference = results[1]
    @test length(reference) == 10      # every case in the worker reported

    for n in thread_counts[2:end]
        @test keys(results[n]) == keys(reference)
        for (case, h) in reference
            @test results[n][case] == h
        end
    end
end
