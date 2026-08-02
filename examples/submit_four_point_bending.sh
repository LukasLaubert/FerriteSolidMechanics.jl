#!/bin/bash -l

#SBATCH --partition=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --job-name=fm_4pb_mpi
#SBATCH --output=examples/results/mpi_four_point_bending/slurm-%j.out
#SBATCH --error=examples/results/mpi_four_point_bending/slurm-%j.err

set -euo pipefail

module load openmpi hwloc

export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
SRUN_MPI_TYPE="${SRUN_MPI_TYPE:-pmix_v3}"

mkdir -p examples/results/mpi_four_point_bending

UCX_NET_DEVICES=none julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'

srun --mpi="${SRUN_MPI_TYPE}" julia --project=. --threads="${SLURM_CPUS_PER_TASK:-1}" -e 'include(ARGS[1]); main(; vtk=nothing)' examples/mpi_four_point_bending.jl
