#!/usr/bin/env bash
# --------------------------------------------------------------------
TASKS=("legal-eval")
EMB_TYPES=("no_cluster")          # ou encoder / none
CENT_STRATS=("mean")           # mean / median / …
CTX_FUSIONS=("concat_proj")    # concat_proj / gated_add / none
CTX_POSITIONS=("pre")
SEEDS=(1)

TOKENIZED="processed-datasets"
OUT_DIR="output-training-testing"
MINI_DATA="TRUE"

mkdir -p "${OUT_DIR}"

for TASK in "${TASKS[@]}"; do
  for EMB in "${EMB_TYPES[@]}"; do
    for CSTR in "${CENT_STRATS[@]}"; do
      for FUS in "${CTX_FUSIONS[@]}"; do
        for POS in "${CTX_POSITIONS[@]}"; do
          for SEED in "${SEEDS[@]}"; do

            echo "▶ $TASK | $EMB/$CSTR | $FUS-$POS | seed=$SEED"

            # unique_name uniquement si POS est vide
            EXTRA_UNAME=""
            if [[ -z "${POS}" ]]; then
              EXTRA_UNAME="--unique_name full"
            fi

            python baseline_run.py \
              --task "$TASK" \
              --strategy baseline \
              --seed "$SEED" \
              --tokenized_folder "$TOKENIZED" \
              --output_dir "$OUT_DIR" \
              --mini_data "$MINI_DATA" \
              --emb_type "$EMB" \
              --centroid_strategy "$CSTR" \
              --ctx_fusion "$FUS" \
              --ctx_position "$POS" \
              --use_crf True \
              --use_sentence_lstm True \
              --use_word_lstm True \
              --use_attention_pooling True \
              $EXTRA_UNAME

            if [[ $? -ne 0 ]]; then
              echo "❌ erreur pour combo précédente – arrêt"
              exit 1
            fi
          done
        done
      done
    done
  done
done

echo "✅ Tous les runs terminés"
