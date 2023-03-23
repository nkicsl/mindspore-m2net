
python3 tools/eval.py --config_file='configs/baseline.yml' \
MODEL.DEVICE_ID     "('0')" \
MODEL.NAME          "('densenet121')" \
MODEL.CROSSMODEITY  "(True)" \
MODEL.PRETRAIN_CHOICE "('self')" \
DATASETS.NAMES      "('nkupv2')" \
TEST.WEIGHT         "('./logs/path/to/weights')" \
TEST.EVALUATE_ONLY  "('on')" \
OUTPUT_DIR          "('./logs/path/output/')"