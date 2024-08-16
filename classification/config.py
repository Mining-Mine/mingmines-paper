### ===================================================== Configurations of Classification =====================================================================

class LABEL():
    MINE = 1
    NOT_MINE = 0

LABELED_DATASET_PATH = "/mnt/mining/latest_labled_dataset"
INCLUDE_NOT_SURE = False  # whether include not sure into the classification

MODEL = "CNN"

LOAD_SAVED_MODEL = True

BATCH_SIZE = 1
TRAIN_EPOCH = 10
MAX_SAMPLES = 100000

INPUT_SHAPE = (1734, 2422, 3)