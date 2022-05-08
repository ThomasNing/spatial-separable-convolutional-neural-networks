# MobileNets

Implementation of the MobileNets CNN model in Cuda

Paper: https://arxiv.org/abs/1704.04861

## Installation Procedures
1. Copy the entire repository to your workspace. 
2. Run the following commands in exact order.
```
module load nvidia-hpc-sdk
nvcc MobileNets_host.cu -o MobileNets_host
sbatch job.sl
cat gpujob.out
```

