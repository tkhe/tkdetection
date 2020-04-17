
For a few datasets that tkdetection natively supports,
the datasets are assumed to exist in a directory specified by the environment variable
`TKDET_DATASETS` (default is `./datasets` relative to your current working directory).
Under this directory, tkdetection expects to find datasets in the following structure:

## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

## Expected dataset structure for LVIS instance segmentation:
```
coco/
  {train,val,test}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_image_info_test.json
```

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

## Expected dataset structure for cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note:
labelTrainIds.png are created by `cityscapesscripts/preparation/createTrainIdLabelImgs.py`.
They are not needed for instance segmentation.

## Expected dataset structure for Pascal VOC:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
```
