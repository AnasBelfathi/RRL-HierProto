#!/usr/bin/env bash
# Interactive smoke test for the pipeline (baseline + PCM + PBR + Mind-proto)
# on whichever task $PCM_TASK is set to (see env.sh).
# Run this INSIDE an interactive srun allocation:
#
#   srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 \
#        --hint=nomultithread -A your_account@v100 bash
#   cd $WORK/rrl-prototype-methods
#   bash slurms/jeanzay/pipeline/smoke_test.sh
#
# Modeled after masking-discursive-injection/discourse-legal-masking/scripts/local/smoke.sh:
# same idea (run the whole pipeline once, small-scale, just to check it doesn't
# crash), adapted here to Jean Zay since this codebase needs allennlp + a GPU.
#
# Small tasks can run steps 1-2 at full scale directly -- only the training
# steps (3-6) are truncated via --mini_data True, to catch code errors fast
# before the real array jobs (00/10/20/21/22/25).
# No -u: env.sh's module/conda loading is not nounset-safe (see env.sh).
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"   # sets PCM_*, cd's to PCM_HOME (does NOT set -e in interactive shells)

echo "════════════════════ [1/6] build_centroids (PCM prototypes) ════════════════════"
load_extraction_modules
python context-extraction/build_centroids.py \
    --datasets "$PCM_TASK" --emb_type none --strategy mean --use_labels

echo "════════════════════ [2/6] match_centroids (sentence -> prototype) ════════════════════"
python matching-context/match_centroids.py \
    --dataset "$PCM_TASK" \
    --centroids_dir "$(pcm_centroids_dir none)" \
    --emb_type none --strategy mean \
    --out_root "$PCM_MATCH_OUT_ROOT"

echo "════════════════════ [3/6] baseline_run.py — baseline (mini_data, 2 docs/split, 1 epoch) ════════════════════"
load_training_modules
python baseline_run.py \
    --task "$PCM_TASK" --strategy baseline --seed 1 \
    --tokenized_folder "$PCM_TOKENIZED_DIR" --output_dir smoke-test-output/baseline \
    --mini_data True \
    --emb_type decoder --centroid_strategy mean \
    --ctx_fusion concat_proj --ctx_position "" \
    --use_crf True --use_sentence_lstm True --use_word_lstm True --use_attention_pooling True \
    --unique_name full

echo "════════════════════ [4/6] baseline_run.py — PCM (mini_data, 2 docs/split, 1 epoch) ════════════════════"
python baseline_run.py \
    --task "$PCM_TASK" --strategy baseline --seed 1 \
    --tokenized_folder "$PCM_TOKENIZED_DIR" --output_dir smoke-test-output/pcm \
    --mini_data True \
    --emb_type none --centroid_strategy mean \
    --ctx_fusion concat_proj --ctx_position pre \
    --use_crf True --use_sentence_lstm True --use_word_lstm True --use_attention_pooling True

echo "════════════════════ [5/6] baseline_run.py — PBR (mini_data, 2 docs/split, 1 epoch) ════════════════════"
python baseline_run.py \
    --task "$PCM_TASK" --strategy baseline --seed 1 \
    --tokenized_folder "$PCM_TOKENIZED_DIR" --output_dir smoke-test-output/pbr \
    --mini_data True \
    --emb_type decoder --centroid_strategy mean \
    --ctx_fusion concat_proj --ctx_position "" \
    --use_crf True --use_sentence_lstm True --use_word_lstm True --use_attention_pooling True \
    --unique_name pbr \
    --use_prototypes True --proto_training joint --n_prototypes 8 \
    --lambda_c 0.9 --lambda_s 0.9 --proto_dist euclidean

echo "════════════════════ [6/6] mind_neighbours_run.py — Mind-proto (mini_data, 2 docs/split, 1 epoch) ════════════════════"
python mind_neighbours_run.py \
    --task "$PCM_TASK" --seed 1 \
    --tokenized_folder "$PCM_TOKENIZED_DIR" --output_dir smoke-test-output/mind_proto \
    --mini_data True \
    --lambda_step 0.5 --tau_grid 1.0

echo "✅ Smoke test OK — vous pouvez lancer le pipeline complet : bash slurms/jeanzay/pipeline/launch.sh <task> <seed> [seed...]"
