while true; do
    rm -rf /var/tmp/neuron-compile-cache
    NEURON_CC_FLAGS="--retry_failed_compilation --num-parallel-jobs=8" \
    torchrun --nproc_per_node=8 \
        ./gemm_nki.py --m_start=0 --m_end=16384 --k_start=0 --k_end=8192 \
        --m_step=1024 --k_step=1024
    exit_code=$? 
    echo "exited ($exit_code). restarting..." >&2
    sleep 2
    if [ $exit_code -eq 0 ]; then
        break
    fi
done
