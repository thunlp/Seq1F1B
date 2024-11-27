#!/bin/bash
MAX_RESTARTS=0
export NCCL_IB_QPS_PER_CONNECTION=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --max_restarts=${MAX_RESTARTS}"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs
DATASET="codeparrot_content_document"
if [ $VPP_SIZE == 1 ]; then
        VPP_STR=""
else
        VPP_STR="--num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VSTAGE"
fi
        # --ffn-hidden-size ${FFN_HIDDEN} \

options=" \
        --tensor-model-parallel-size $TP_SIZE \
        --timing-log-level 2 \
        --pipe-sp-strategy ${PP_SP_STR} \
        --pipe-sp-splits ${PP_SP} \
        --pipeline-model-parallel-size $PP_SIZE \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN}\
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings ${SEQ_LENGTH} \
        --micro-batch-size ${MICRO_BATCH} \
        --global-batch-size ${GLOBAL_BATCH} \
        --lr 6.0e-5 \
        --min-lr 6.0e-6 \
        --lr-decay-style cosine \
                --train-iters ${TRAIN_ITER} \
        --log-interval 1 \
        --eval-iters 0 \
        --eval-interval 1000 \
                --use-flash-attn \
        --data-path $DATA_PATH/data/codeparrot_content_document \
        --vocab-file $DATA_PATH/data/vocab.json \
        --merge-file $DATA_PATH/data/merges.txt \
        --initial-loss-scale 65536 \
        --save-interval 1000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --bf16 \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --use-distributed-optimizer \
        --hidden-dropout 0 \
        --attention-dropout 0 \
        --sequence-parallel \
        --no-async-tensor-model-parallel-allreduce \
        $VPP_STR
        "
        # --no-scatter-gather-tensors-in-pipeline \
        # --no-gradient-accumulation-fusion \
        # --no-masked-softmax-fusion \
        #--use-distributed-optimizer \
        # --recompute-method block \
        # --recompute-num-layers 48 \
        # --distribute-saved-activations \
if [ "$PROFILE" = "true" ]; then
        options="${options}
        --profile \
        --profile-step-start 3 \
        --profile-step-end 5 \
        --profile-ranks 0 \
        "
fi
if [ $RECOMPUTE -eq 1]; then
        options="${options}
        --recompute-method uniform \
        --recompute-granularity full \
        "
fi

run_cmd="torchrun $DISTRIBUTED_ARGS ${DIR}/pretrain_gpt.py ${options}"
echo $run_cmd
$run_cmd
