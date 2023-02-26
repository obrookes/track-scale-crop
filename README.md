# The Pan-African Dataset

A collection of utils for processing raw detections relating to the PanAf Dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/43569179/163564388-531c34f7-8ac0-4620-a2ff-3d2dc34fa324.jpg" width=400>
  <img src="https://user-images.githubusercontent.com/43569179/163564476-27c96484-c084-4247-b1ac-8652f294df50.jpg" width=400>
  <br>
  <img src="https://user-images.githubusercontent.com/43569179/163564509-48cec8eb-f7e5-49f4-a0ad-04ec473b0733.jpg" width=400>
  <img src="https://user-images.githubusercontent.com/43569179/163564522-a67f9c57-16f8-4c4a-b058-60fe174afa72.jpg" width=400>
</p>

## Generate tracklets from detections

```bash
python track.py 
    --detection_path=path/to/country/site
    --video_path=path/to/coutry/site 
    --l_confidence=0.25 
    --h_confidence=0.85 
    --iou=0.5 
    --length=72 
    --outpath=path/to/country/site
```

## Scaling tracklets

```bash
python scale_bboxes.py 
    --annotation_path=path/to/country/site 
    --video_path=path/to/country/site 
    --outpath=path/to/country/site 
    --bbox_scaler=0.1
```

## PanAf to Kinetics annotation conversion

```bash
python panaf_to_kinetics.py \
    --data_path=videos/original/guineabissau/boe \
    --ann_path=detections/scaled/convnext_cascade_rcnn/guineabissau/boe \
    --outdir=videos/cropped/guineabissau/boe
```
