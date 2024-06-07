#!/bin/bash
SEED=413

# dataset parameters
MAX_SEQ_LENGTH=64
VIRTUAL_PIECE_STEP_RATIO=1
FLATTEN_VIRTUAL_PIECES=true
PERMUTE_MPS=false
PERMUTE_TRACK_NUMBER=false
PITCH_AUGMENTATION_RANGE=0

# model parameter
USE_LINEAR_ATTENTION=false
LAYERS_NUMBER=1
ATTN_HEADS_NUMBER=8
EMBEDDING_DIM=32
NOT_USE_MPS_NUMBER=true

# training parameter
BATCH_SIZE=8
MAX_UPDATES=200
VALIDATION_INTERVAL=100
LOSS_PADDING="ignore"
MAX_GRAD_NORM=1.0
LEARNING_RATE_PEAK=0.0001
LEARNING_RATE_WARMUP_UPDATES=100
LEARNING_RATE_DECAY_END_UPDATES=200
LEARNING_RATE_DECAY_END_RATIO=0.5
EARLY_STOP=0

# do eval with valid set?
VALID_EVAL_SAMPLE_NUMBER=0

# training device
USE_DEVICE="cpu"

# generation & evaluation setting
EVAL_CONFIG_NAME="test"
