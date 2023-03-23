python3 tools/train.py --config_file='configs/baseline.yml' \
MODEL.CROSSMODEITY          "(False)" \
MODEL.APP_CE_LOSS           "('off')" \
MODEL.TRIPLET_LOSS          "('normal')" \
DATASETS.NAMES              "('nkupv2')" \
DATALOADER.SAMPLER          "('off')" \
DATALOADER.NUM_INSTANCE     "(8)" \
OUTPUT_DIR                  "('./logs/path/output/')"