from cvpods.configs.fcos_config import FCOSConfig

from augmentations import WeakAug,StrongAug
from dataset import PartialCOCO
    
_config_dict = dict(
    DATASETS=dict(
        SUPERVISED=[
            (PartialCOCO,dict(
                percentage=10,
                seed=1,
                supervised=True,
                sup_file='../COCO_Division/COCO_supervision.txt'
            )),
        ],
        UNSUPERVISED=[
            (PartialCOCO,dict(
                percentage=10,
                seed=1,
                supervised=False,
                sup_file='../COCO_Division/COCO_supervision.txt'
            )),
        ],
        TEST=("coco_2017_val",),
    ),
    MODEL=dict(
        WEIGHTS='detectron2://ImageNetPretrained/MSRA/R-50.pkl',
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            QUALITY_BRANCH='iou',
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
        ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=4,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_ITER=180000,
            STEPS=(179995, ),
            WARMUP_ITERS=1000,
            WARMUP_FACTOR=1.0 / 1000,
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
        ),
        IMS_PER_BATCH=16,
        CHECKPOINT_PERIOD=5000,
        CLIP_GRADIENTS=dict(ENABLED=True)
    ),
    TRAINER=dict(
        NAME="SemiRunner",
        EMA=dict(
            DECAY_FACTOR=0.9996,
            UPDATE_STEPS=1,
            START_STEPS=3000,
            FAKE=False
        ),
        SSL=dict(
            BURN_IN_STEPS=5000,
        ),
        DISTILL=dict(
            RATIO=0.01,
            SUP_WEIGHT=1,
            UNSUP_WEIGHT=1,
            SUPPRESS='linear',
            WEIGHTS=dict(
                LOGITS=4.,
                DELTAS=1.,
                QUALITY=1.,    
            ),
            GAMMA=2.
        ),
        # WINDOW_SIZE=1,
    ),
    TEST=dict(
        EVAL_PERIOD=2000,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                SUPERVISED=(WeakAug,dict(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice")),
                UNSUPERVISED=(StrongAug,)
            ),
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR='outputs',
    GLOBAL=dict(
        LOG_INTERVAL=10,
    )
)

class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
