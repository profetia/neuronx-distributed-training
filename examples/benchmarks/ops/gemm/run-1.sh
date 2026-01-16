task_list=(
    # "gemm_benchmark_k16-8192-32_m16-1024-32_rank0.csv"
    "gemm_benchmark_k16-8192-32_m1040-2048-32_rank1.csv"
    # "gemm_benchmark_k16-8192-32_m2064-3072-32_rank2.csv"
    # "gemm_benchmark_k16-8192-32_m3088-4096-32_rank3.csv"
    # "gemm_benchmark_k16-8192-32_m4112-5120-32_rank4.csv"
    "gemm_benchmark_k16-8192-32_m5136-6144-32_rank5.csv"
    # "gemm_benchmark_k16-8192-32_m6160-7168-32_rank6.csv"
    # "gemm_benchmark_k16-8192-32_m7184-8192-32_rank7.csv"
    # "gemm_benchmark_k16-8192-32_m8208-9216-32_rank8.csv"
    "gemm_benchmark_k16-8192-32_m9240-10240-32_rank9.csv"
    # "gemm_benchmark_k16-8192-32_m10272-11264-32_rank10.csv"
    # "gemm_benchmark_k16-8192-32_m11296-12288-32_rank11.csv"
    # "gemm_benchmark_k16-8192-32_m12320-13312-32_rank12.csv"
    "gemm_benchmark_k16-8192-32_m13344-14336-32_rank13.csv"
    # "gemm_benchmark_k16-8192-32_m14368-15360-32_rank14.csv"
    # "gemm_benchmark_k16-8192-32_m15392-16384-32_rank15.csv"
)
m_start_list=(1040 5136 9240 13344)
m_end_list=(2048 6144 10240 14336)

for i in ${!task_list[@]}; do
    task=${task_list[$i]}
    m_start=${m_start_list[$i]}
    m_end=${m_end_list[$i]}
    echo "Processing $task"
    while true; do
        rm -rf /var/tmp/neuron-compile-cache
        NEURON_CC_FLAGS="--retry_failed_compilation --num-parallel-jobs=4" \
        torchrun --nproc_per_node=32 \
            ./gemm.py --m_start=$m_start --m_end=$m_end --k_start=16 --k_end=8192 \
            --partition_unfinished $task
        exit_code=$? 
        echo "exited ($exit_code). restarting..." >&2
        sleep 2
        echo "Finished attempt for $task"
        if [ $exit_code -eq 0 ]; then
            break
        fi
    done
done


# python upscale.py gemm_benchmark_k16-8192-16_m16-1024-16_rank0.csv gemm_benchmark_k32-8192-32_m16-1024-32_rank0.csv
# python upscale.py gemm_benchmark_k16-8192-16_m1024-2048-16_rank1.csv gemm_benchmark_k32-8192-32_m1024-2048-32_rank1.csv
# python upscale.py gemm_benchmark_k16-8192-16_m2048-3072-16_rank2.csv gemm_benchmark_k32-8192-32_m2048-3072-32_rank2.csv
# python upscale.py gemm_benchmark_k16-8192-16_m3072-4096-16_rank3.csv gemm_benchmark_k32-8192-32_m3072-4096-32_rank3.csv
# python upscale.py gemm_benchmark_k16-8192-16_m4096-5120-16_rank4.csv gemm_benchmark_k32-8192-32_m4096-5120-32_rank4.csv
# python upscale.py gemm_benchmark_k16-8192-16_m5120-6144-16_rank5.csv gemm_benchmark_k32-8192-32_m5120-6144-32_rank5.csv
# python upscale.py gemm_benchmark_k16-8192-16_m6144-7168-16_rank6.csv gemm_benchmark_k32-8192-32_m6144-7168-32_rank6.csv
# python upscale.py gemm_benchmark_k16-8192-16_m7168-8192-16_rank7.csv gemm_benchmark_k32-8192-32_m7168-8192-32_rank7.csv
# python upscale.py gemm_benchmark_k16-8192-16_m8192-9216-16_rank8.csv gemm_benchmark_k32-8192-32_m8192-9216-32_rank8.csv
# python upscale.py gemm_benchmark_k16-8192-16_m9216-10240-16_rank9.csv gemm_benchmark_k32-8192-32_m9216-10240-32_rank9.csv
# python upscale.py gemm_benchmark_k16-8192-16_m10240-11264-16_rank10.csv gemm_benchmark_k32-8192-32_m10240-11264-32_rank10.csv
# python upscale.py gemm_benchmark_k16-8192-16_m11264-12288-16_rank11.csv gemm_benchmark_k32-8192-32_m11264-12288-32_rank11.csv
# python upscale.py gemm_benchmark_k16-8192-16_m12288-13312-16_rank12.csv gemm_benchmark_k32-8192-32_m12288-13312-32_rank12.csv
# python upscale.py gemm_benchmark_k16-8192-16_m13312-14336-16_rank13.csv gemm_benchmark_k32-8192-32_m13312-14336-32_rank13.csv
# python upscale.py gemm_benchmark_k16-8192-16_m14336-15360-16_rank14.csv gemm_benchmark_k32-8192-32_m14336-15360-32_rank14.csv
# python upscale.py gemm_benchmark_k16-8192-16_m15360-16384-16_rank15.csv gemm_benchmark_k32-8192-32_m15360-16384-32_rank15.csv