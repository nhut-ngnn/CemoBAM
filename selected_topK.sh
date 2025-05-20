TRAIN_PATH="feature/MELD_BERT_WAV2VEC_train.pkl"
VALID_PATH="feature/MELD_BERT_WAV2VEC_val.pkl"
TEST_PATH="feature/MELD_BERT_WAV2VEC_test.pkl"

SAVE_DIR="saved_models"

python trainer/selected_topK.py \
  --train_path "$TRAIN_PATH" \
  --valid_path "$VALID_PATH" \
  --test_path "$TEST_PATH" \
  --fusion_type "min" \
  --hidden_dim 512 \
  --num_classes 7 \
  --num_layers 3 \
  --epochs 100 \
  --lr 0.0001 \
  --weight_decay 0.0005 \
  --save_dir "$SAVE_DIR"
