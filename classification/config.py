### ===================================================== Configurations of Classification =====================================================================

class LABEL():
    MINE = 1
    NOT_MINE = 0

LABELED_DATASET_PATH = "/mnt/mining/labeled_mine_dataset"
INCLUDE_NOT_SURE = False  # whether include not sure into the classification

MODEL = "CNN"

BATCH_SIZE = 1
TRAIN_EPOCH = 10

INPUT_SHAPE = (1734, 2422, 3)

# LABLERS = ["jiale", "peiran", "yuxuan", "huiyu"]
LABLERS = ["peiran"]  # TODO: there is bug when reading huiyu's dataset