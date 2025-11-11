#!/bin/bash
# MIND 데이터셋용 프롬프트 파일 생성 스크립트

echo "=========================================="
echo "Step 1: Creating test user history..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml \
  --gpu_id=1 \
  --dump_to_chat=True \
  --test_v=test \
  --chat_hislen=10

if [ $? -eq 0 ] || [ $? -eq 130 ]; then  # 130 is KeyboardInterrupt
    echo ""
    echo "✓ Step 1 completed: mind_uid_dict.pkl created"
    echo ""
    echo "=========================================="
    echo "Step 2: Creating user/item profiles..."
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=1 python run_recbole.py \
      --dataset=mind \
      --model=SASRec \
      --config_files=dataset/mind/mind.yaml \
      --gpu_id=1 \
      --dump_profile=True \
      --test_v=test
    
    if [ $? -eq 0 ] || [ $? -eq 130 ]; then
        echo ""
        echo "✓ Step 2 completed: mind_chat.pkl created"
        echo ""
        echo "=========================================="
        echo "All prompt files created successfully!"
        echo "=========================================="
    fi
fi

