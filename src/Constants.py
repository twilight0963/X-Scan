import os

USE_AMP=True
IMG_SIZE=240
BATCH_SIZE=64
INIT_EPOCH=0
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR=os.path.join(BASE_DIR,"Datasets","MergedData","train")
TEST_DIR=os.path.join(BASE_DIR,"Datasets","MergedData","test")
EPOCHS=20