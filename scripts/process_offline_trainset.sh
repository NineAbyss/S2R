acc_min=0.1
acc_max=0.7
fp_in="xxxxx/your_sample_result.jsonl"
fp_save="xxxxxx.json"
use_data_balance=true

cd S2R
python3 ./tools/process_offline_trainset.py \
    --acc_min $acc_min \
    --acc_max $acc_max \
    --fp_in $fp_in \
    --fp_save $fp_save \
    --use_data_balance $use_data_balance