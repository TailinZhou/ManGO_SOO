# set up machine parameters
NUM_CPUS=4
NUM_GPUS=1

for TASK_NAME in \
    gfp ; do
    
  for ALGORITHM_NAME in \
      cbas ; do
  
    # launch several model-based optimization algorithms using the command line interface
    # for example: 
    # (design-baselines) name@computer:~/$ cbas gfp \
    #                                        --local-dir ~/db-results/cbas-gfp \
    #                                        --cpus 32 \
    #                                        --gpus 8 \
    #                                        --num-parallel 8 \
    #                                        --num-samples 8
    $ALGORITHM_NAME $TASK_NAME \
      --local-dir ~/db-results/$ALGORITHM_NAME-$TASK_NAME \
      --cpus $NUM_CPUS \
      --gpus $NUM_GPUS \
      --num-parallel 8 \
      --num-samples 8
    
  done
  
done

# generate the main performance table of the paper
design-baselines make-table --dir ~/db-results/ --percentile 100th

# generate the performance tables in the appendix
design-baselines make-table --dir ~/db-results/ --percentile 50th
design-baselines make-table --dir ~/db-results/ --percentile 100th --no-normalize