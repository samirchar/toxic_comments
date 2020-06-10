while true;
do nvidia-smi --query-gpu=memory.used --format=csv >> gpu_mem_usage.log; sleep 1;
done
