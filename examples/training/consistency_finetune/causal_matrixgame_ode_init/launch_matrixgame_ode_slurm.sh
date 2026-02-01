#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p preprocess_output

# Launch 8 jobs, one for each node (Total 64 GPUs)
for node_id in {0..7}; do
    # Calculate the starting file number for this node (0, 8, 16...)
    start_file=$((node_id * 8))
    
    echo "Launching node $node_id with files merge_${start_file}.txt to merge_$((start_file + 7)).txt"
    
    sbatch --job-name=mg-ode-${node_id} \
           --output=preprocess_output/mg-ode-node-${node_id}.out \
           --error=preprocess_output/mg-ode-node-${node_id}.err \
           $(pwd)/examples/training/consistency_finetune/causal_matrixgame_ode_init/matrixgame_ode_worker.slurm $start_file $node_id
done

echo "All 8 nodes (64 GPUs) for ODE preprocessing launched successfully!"
