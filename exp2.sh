#!/bin/bash

# 定义参数范围
# num_micros=(8 16 32)
# sp_splits=(1 4)
# seq_lens=(8096 16384)
# model_configs=('3.5b' '7b')
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
PP_SP_STRS=('average' 'uniform_comp')
num_micros=(32)
sp_splits=(4 1)
# sp_splits=(1)
seq_lens=(16384)
model_configs=('7b')
VPP_SIZES=(1 2)
TP_SIZE=2
PP_SIZE=4
# Distributed
MICRO_BATCH=1
# WORLD_SIZE=1
WORLD_SIZE=1
TRAIN_ITER=20
GPUS_PER_NODE=8
NGPUS=$((WORLD_SIZE * GPUS_PER_NODE))
# MASTER_ADDR=g4006
MASTER_ADDR=localhost

MASTER_PORT=12306
export TP_SIZE PP_SIZE MICRO_BATCH WORLD_SIZE GPUS_PER_NODE MASTER_ADDR MASTER_PORT PP_SP_STR

# 结果文件配置
declare -A config_1_3b=(
    [num_layers]=24
    [hidden]=2048
    [num_attn_heads]=16
)

declare -A config_2_7b=(
    [num_layers]=32
    [hidden]=2560
    [num_attn_heads]=32
)

declare -A config_7b=(
    [num_layers]=32
    [hidden]=4096
    [num_attn_heads]=32
)

declare -A config_13b=(
    [num_layers]=40
    [hidden]=5120
    [num_attn_heads]=40
)

declare -A config_30b=(
    [num_layers]=64
    [hidden]=6144
    [num_attn_heads]=64
)

# 创建一个映射从模型配置名称到相应的数组名称
declare -A model_config_map=(
    ["2.7b"]="config_2_7b"
    ["1.3b"]="config_1_3b"
    ["30b"]="config_30b"
    ["7b"]="config_7b"
    ["13b"]="config_13b"
)

# 设置变量

# Debug 开关，设置为1以启用屏幕输出
DEBUG=1
EXP_OUTPUT_FILE="./exp_logs/exps/${DATETIME}_exp.log"
echo "micro_bsz, num_of_batch, seq_length, vpp_size, tp_size, pp_size, sp_splits, model_size, split_strategy, dp_size, throughput, memory_array, one_step_time, Ngpus, tflops" > $EXP_OUTPUT_FILE

for VPP_SIZE in "${VPP_SIZES[@]}"; do
    for num_micro in "${num_micros[@]}"; do
        for SEQ_LENGTH in "${seq_lens[@]}"; do
            for model_config in "${model_configs[@]}"; do
                for PP_SP in "${sp_splits[@]}"; do
                    if [ $PP_SP -eq 1 ]; then
                        PP_SP_STRS_RES=('average')
                    else
                        PP_SP_STRS_RES=("${PP_SP_STRS[@]}")
                    fi

                    for PP_SP_STR in "${PP_SP_STRS_RES[@]}"; do
                        echo $PP_SP_STR
                        DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
                        DP_SIZE=$((NGPUS / TP_SIZE / PP_SIZE))
                        GLOBAL_BATCH=$((MICRO_BATCH * num_micro * DP_SIZE))

                        # 根据模型配置名动态获取配置
                        config_name="${model_config_map[$model_config]}"
                        declare -n config=$config_name
                        NUM_LAYERS="${config[num_layers]}"
                        HIDDEN="${config[hidden]}"
                        NUM_ATTN_HEADS="${config[num_attn_heads]}"
                        NUM_LAYERS_PER_VSTAGE=$((NUM_LAYERS / PP_SIZE / VPP_SIZE))

                        echo $NUM_LAYERS
                        echo $NUM_LAYERS_PER_VSTAGE

                        # 导出变量以便 run.sh 脚本可以使用
                        export NUM_LAYERS HIDDEN NUM_ATTN_HEADS PP_SP SEQ_LENGTH GLOBAL_BATCH NUM_LAYERS_PER_VSTAGE VPP_SIZE TRAIN_ITER

                        echo "****************MODEL_CONFIG*****************"
                        echo "number of layers: $NUM_LAYERS"
                        echo "hidden size: $HIDDEN"
                        echo "number of attention heads: $NUM_ATTN_HEADS"
                        echo "Running experiment with TP_SIZE=${TP_SIZE}, PP_SIZE=${PP_SIZE}, MICRO_BATCH=${MICRO_BATCH}, WORLD_SIZE=${WORLD_SIZE}, num_micro=${num_micro}, PP_SP=${PP_SP}, SEQ_LENGTH=${SEQ_LENGTH}, model_config=${model_config}"

                        # 执行 run.sh 并根据 DEBUG 重定向输出到日志和/或屏幕
                        if [ $PP_SP -eq 1 ]; then
                            PP_SP_STR_RES="none"
                        else
                            PP_SP_STR_RES=$PP_SP_STR
                        fi
                        logname="./exp_logs/TP${TP_SIZE}_PP${PP_SIZE}_VPP${VPP_SIZE}_MICRO${MICRO_BATCH}_WORLD${WORLD_SIZE}_num_micro${num_micro}_sp${PP_SP}_str${PP_SP_STR_RES}_seqlen${SEQ_LENGTH}_model${model_config}.log"
                        
                        echo "**********Log file***********"
                        echo "File path: $logname"

                        if [ "$DEBUG" -eq 1 ]; then
                            set -o pipefail
                            ./run.sh 2>&1 | tee $logname
                        else
                            ./run.sh > $logname 2>&1
                        fi

                        return_code=$?

                        if [ $return_code -ne 0 ]; then
                            echo "Error: Process terminated abnormally."
                            echo "$MICRO_BATCH,$num_micro,$SEQ_LENGTH,$VPP_SIZE,$TP_SIZE,$PP_SIZE,$PP_SP,$model_config,$PP_SP_STR_RES,$DP_SIZE,OOM,OOM,OOM,$NGPUS,OOM" >> $EXP_OUTPUT_FILE
                        else
                            echo "*********Output*********"
                            output=$(tail -n 4 $logname)
                            echo "$output"
                            declare -A data
                            while IFS=: read -r key value; do
                                data["$key"]=$(echo $value | xargs)  # 使用 xargs 去除多余的空格
                            done <<< "$output"

                            echo "$MICRO_BATCH,$num_micro,$SEQ_LENGTH,$VPP_SIZE,$TP_SIZE,$PP_SIZE,$PP_SP,$model_config,$PP_SP_STR_RES,$DP_SIZE,${data[toks]},${data[mem_arr]},${data[time]},$NGPUS,${data[tflops]}" >> $EXP_OUTPUT_FILE
                        fi
                    done
                done
            done
        done
    done
done

echo "All experiments completed."
echo "Final Log file path: $EXP_OUTPUT_FILE"
