
TEST_PATH="feature/IEMOCAP_BERT_WAV2VEC_test.pkl"
MODEL_PATH="saved_models/IEMOCAP_HemoGAT_min.pt"
FUSION_TYPE="min"
HIDDEN_DIM=512
NUM_CLASSES=4
NUM_LAYERS=3
K_TEXT=7
K_AUDIO=8

python trainer/predict.py \
  --test_path "$TEST_PATH" \
  --model_path "$MODEL_PATH" \
  --fusion_type "$FUSION_TYPE" \
  --hidden_dim $HIDDEN_DIM \
  --num_classes $NUM_CLASSES \
  --num_layers $NUM_LAYERS \
  --k_text $K_TEXT \
  --k_audio $K_AUDIO
