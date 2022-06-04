#!/bin/bash
#SBATCH --job-name=pacasp
#SBATCH --output=pacasp.out
#SBATCH --error=pacasp.err
#SBATCH --time=50:00:00
#SBATCH --ntasks=2207
#SBATCH -p standard

TMPDIR=/tmp/lustre_shared/$USER/$SLURM_JOBID
mkdir -p $TMPDIR
cd $TMPDIR
mkdir -p instances
mkdir -p out
cp $SLURM_SUBMIT_DIR/sp sp
cp $SLURM_SUBMIT_DIR/gen gen
cp $SLURM_SUBMIT_DIR/instances/* instances/
module load mpich
time mpiexec -n 2207 ./sp run
# zip -r $SLURM_JOBID.zip out
# cp -r $SLURM_JOBID.zip $SLURM_SUBMIT_DIR/
# rm -r $TMPDIR