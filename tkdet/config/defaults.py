from fvcore.common.config import CfgNode as CN

_C = CN()


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.META_ARCHITECTURE = ""

_C.MODEL.DEVICE = "cuda"

_C.MODEL.WEIGHTS = ""

_C.MODEL.MASK_ON = False

_C.MODEL.KEYPOINT_ON = False

_C.MODEL.NUM_CLASSES = -1

_C.MODEL.SCORE_THRESHOLD = 0.05

_C.MODEL.NMS_THRESHOLD = 0.45

_C.MODEL.LOAD_PROPOSALS = False


# -----------------------------------------------------------------------------
# Backbone options
# -----------------------------------------------------------------------------
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = ""

_C.MODEL.BACKBONE.OUT_FEATURES = []

_C.MODEL.BACKBONE.FREEZE_AT = 2


# -----------------------------------------------------------------------------
# Neck options
# -----------------------------------------------------------------------------
_C.MODEL.NECK = CN()

_C.MODEL.NECK.ENABLE = True

_C.MODEL.NECK.NAME = "FPN"


# -----------------------------------------------------------------------------
# RPN options
# -----------------------------------------------------------------------------
_C.MODEL.RPN = CN()

_C.MODEL.RPN.NAME = "RPN"

_C.MODEL.RPN.IN_FEATURES = []

_C.MODEL.RPN.NMS_THRESH = 0.7

_C.MODEL.RPN.BOUNDARY_THRESH = -1

_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"

_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]

_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]

_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256

_C.MODEL.RPN.POSITIVE_FRACTION = 0.5

_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

_C.MODEL.RPN.LOSS_WEIGHT = 1.0

_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000

_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000

_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000

_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

_C.MODEL.RPN.NMS_THRESH = 0.7

_C.MODEL.RPN.MIN_SIZE = 0


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()

_C.MODEL.ROI_HEADS.NAME = "StandardROIHeads"

_C.MODEL.ROI_HEADS.IN_FEATURES = []

_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]

_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]

_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True


# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()

_C.MODEL.ROI_BOX_HEAD.NAME = ""

_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)

_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0

_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14

_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0

_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0

_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024

_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0

_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256

_C.MODEL.ROI_BOX_HEAD.NORM = ""

_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False

_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False


# ---------------------------------------------------------------------------- #
# Cascaded Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()

_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)


# ---------------------------------------------------------------------------- #
# Mask Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()

_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"

_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0

_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0

_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256

_C.MODEL.ROI_MASK_HEAD.NORM = ""

_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False

_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"


# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD = CN()

_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"

_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14

_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0

_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))

_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17

_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1

_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True

_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"


# -----------------------------------------------------------------------------
# Anchor options
# -----------------------------------------------------------------------------
_C.MODEL.ANCHOR = CN()

_C.MODEL.ANCHOR.NAME = "DefaultAnchorGenerator"

_C.MODEL.ANCHOR.SIZES = [[32, 64, 128, 256, 512]]

_C.MODEL.ANCHOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

_C.MODEL.ANCHOR.OFFSET = 0.0


# -----------------------------------------------------------------------------
# RetinaNet options
# -----------------------------------------------------------------------------
_C.RETINANET = CN()

_C.RETINANET.NUM_CONVS = 4

_C.RETINANET.TOPK_CANDIDATES = 1000

_C.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]

_C.RETINANET.IOU_LABELS = [0, -1, 1]

_C.RETINANET.PRIOR_PROB = 0.01

_C.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


# -----------------------------------------------------------------------------
# FCOS options
# -----------------------------------------------------------------------------
_C.FCOS = CN()

_C.FCOS.NORM_REG_TARGETS = False

_C.FCOS.CENTERNESS_ON_REG = False

_C.FCOS.NUM_CONVS = 4

_C.FCOS.PRIOR_PROB = 0.01

_C.FCOS.CENTER_SAMPLING_RADIUS = 0.0

_C.FCOS.TOPK_CANDIDATES = 1000


# -----------------------------------------------------------------------------
# SSD options
# -----------------------------------------------------------------------------
_C.SSD = CN()

_C.SSD.SIZE = 300

_C.SSD.STRIDES = [8, 16, 32, 64, 100, 300]

_C.SSD.BBOX_REG_WEIGHTS = [10, 10, 5, 5]

_C.SSD.IOU_THRESHOLDS = [0.4, 0.5]

_C.SSD.IOU_LABELS = [0, -1, 1]

_C.SSD.HEAD = CN()

_C.SSD.HEAD.NAME = "SSDHead"

_C.SSD.HEAD.NORM = "BN"


# -----------------------------------------------------------------------------
# ResNet and ResNeXt
# -----------------------------------------------------------------------------
_C.RESNET = CN()

_C.RESNET.NORM = "BN"

_C.RESNET.ZERO_INIT_RESIDUAL = False

_C.RESNET.REPLACE_STRIDE_WITH_DILATION = [False, False, False]


# -----------------------------------------------------------------------------
# ShuffleNet v2
# -----------------------------------------------------------------------------
_C.SHUFFLENET_V2 = CN()

_C.SHUFFLENET_V2.NORM = "BN"


# -----------------------------------------------------------------------------
# MobileNet v2
# -----------------------------------------------------------------------------
_C.MOBILENET_V2 = CN()

_C.MOBILENET_V2.NORM = "BN"


# -----------------------------------------------------------------------------
# EfficientNet
# -----------------------------------------------------------------------------
_C.EFFICIENTNET = CN()

_C.EFFICIENTNET.NORM = "BN"

_C.EFFICIENTNET.ACTIVATION = "Swish"


# -----------------------------------------------------------------------------
# DarkNet
# -----------------------------------------------------------------------------
_C.DARKNET = CN()

_C.DARKNET.NORM = "BN"

_C.DARKNET.ACTIVATION = "LeakyReLU"

_C.DARKNET.STEM_CHANNELS = 32


# -----------------------------------------------------------------------------
# VGG
# -----------------------------------------------------------------------------
_C.VGG = CN()

_C.VGG.NORM = ""


# -----------------------------------------------------------------------------
# FPN options
# -----------------------------------------------------------------------------
_C.FPN = CN()

_C.FPN.NORM = ""

_C.FPN.OUT_CHANNELS = 256

_C.FPN.IN_FEATURES = _C.MODEL.BACKBONE.OUT_FEATURES

_C.FPN.FUSE_TYPE = "sum"

_C.FPN.TOP_BLOCK = CN()

_C.FPN.TOP_BLOCK.NAME = "LastLevelMaxPool"

_C.FPN.TOP_BLOCK.IN_FEATURE = "p5"


# -----------------------------------------------------------------------------
# Input options
# -----------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.PIXEL_MEAN = [123.675, 116.280, 103.530]

_C.INPUT.PIXEL_STD = [58.395, 57.120, 57.375]

_C.INPUT.FORMAT = "RGB"

_C.INPUT.TRANSFORM = "DefaultTransform"

_C.INPUT.MIN_SIZE_TRAIN = (800,)

_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

_C.INPUT.MAX_SIZE_TRAIN = 1333

_C.INPUT.MIN_SIZE_TEST = 800

_C.INPUT.MAX_SIZE_TEST = 1333

_C.INPUT.CROP = CN({"ENABLED": False})

_C.INPUT.CROP.TYPE = "relative_range"

_C.INPUT.CROP.SIZE = [0.9, 0.9]

_C.INPUT.MASK_FORMAT = "polygon"


# -----------------------------------------------------------------------------
# Datasets options
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

_C.DATASETS.TRAIN = ()

_C.DATASETS.PROPOSAL_FILES_TRAIN = ()

_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000

_C.DATASETS.TEST = ()

_C.DATASETS.PROPOSAL_FILES_TEST = ()

_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000


# -----------------------------------------------------------------------------
# DataLoader options
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.ASPECT_RATIO_GROUPING = True

_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

_C.DATALOADER.REPEAT_THRESHOLD = 0.0

_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True


# -----------------------------------------------------------------------------
# Solver options
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000

_C.SOLVER.WARMUP_ITERS = 1000

_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.IMS_PER_BATCH = 16

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})

_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"

_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0


# -----------------------------------------------------------------------------
# Loss options
# -----------------------------------------------------------------------------
_C.LOSS = CN()

_C.LOSS.FOCAL_LOSS = CN()

_C.LOSS.FOCAL_LOSS.ALPHA = 0.25

_C.LOSS.FOCAL_LOSS.GAMMA = 2

_C.LOSS.SMOOTH_L1_LOSS = CN()

_C.LOSS.SMOOTH_L1_LOSS.BETA = 0.11


# -----------------------------------------------------------------------------
# Test options
# -----------------------------------------------------------------------------
_C.TEST = CN()

_C.TEST.EXPECTED_RESULTS = []

_C.TEST.EVAL_PERIOD = 0

_C.TEST.KEYPOINT_OKS_SIGMAS = []

_C.TEST.DETECTIONS_PER_IMAGE = 100

_C.TEST.PRECISE_BN = CN({"ENABLED": False})

_C.TEST.PRECISE_BN.NUM_ITER = 200


# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
_C.VIS_PERIOD = 0

_C.OUTPUT_DIR = "./output"

_C.SEED = -1

_C.CUDNN_BENCHMARK = False
