#!/bin/bash

# Runs the "175B" parameter model

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

splits=$1
# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## env config
GPUS_PER_NODE=8
# HOST-10-140-60-[33-70,82-84,86,88-91,94-97,102-106,108-109,113,124-129]
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
NNODES=$SLURM_NNODES

CHECKPOINT_PATH=./tmp #<Specify path>
TENSORBOARD_LOGS_PATH=./tmp #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_document

VOCAB_FILE=/mnt/petrelfs/wangguoteng.p/chenqiaoling/chenqiaoling/NewFSTP/HT-Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=/mnt/petrelfs/wangguoteng.p/chenqiaoling/chenqiaoling/NewFSTP/HT-Megatron-DeepSpeed/data/gpt2-merges.txt
DATA_PATH=/mnt/petrelfs/wangguoteng.p/chenqiaoling/chenqiaoling/NewFSTP/HT-Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document


TENSOR_MODEL_PARALLEL_SIZE=4
NUM_LAYERS=16
HIDDEN_SIZE=8192
NUM_ATTENTION_HEADS=64
GLOBAL_BATCH_SIZE=8
SEQ_LEN=65536
TRAIN_SAMPLES=73242188  # 300B tokens / 4096
LR_WARMUP_SAMPLES=50000
LR_DECAY_SAMPLES=73192188 # TRAIN_SAMPLES - LR_WARMUP_SAMPLES
    #    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
        #    --sequence-parallel \
            #    --untie-embeddings-and-output-weights \
CHECKPOINT_DIR="./checkpoints"
DATACACHE_DIR="./data-cache"
TENSORBOARD_DIR="./tensorboard"
options=" \
       --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
       --sequence-parallel \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
       --pipeline-model-parallel-size 8 \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       --use-distributed-optimizer \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type rope \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --group-query-attention \
       --num-query-groups 8 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --train-samples ${TRAIN_SAMPLES} \
       --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
       --lr-decay-samples ${LR_DECAY_SAMPLES} \
       --split 99,1,0 \
       --tokenizer-type GPT2BPETokenizer \
       --distributed-backend nccl \
       --micro-batch-size 1 \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --swiglu \
       --lr 2.5e-4 \
       --min-lr 2.5e-5 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval 2000 \
       --eval-interval 2000 \
       --eval-iters 32 \
       --bf16 \
       --pipe-sp-splits 64 \
       --pipe-sp-strategy uniform_comp \
       --tensorboard-dir ${TENSORBOARD_DIR}"
    #    --pipe-sp-splits 64 \

torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
/mnt/petrelfs/wangguoteng.p/chenqiaoling/chenqiaoling/Seq1F1B/pretrain_gpt.py ${options}