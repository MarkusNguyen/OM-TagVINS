"""Rear-camera constants shared by the front-end modules..

Extracted from the original monolithic module; see optimize.py / frontend.py
for the pipeline entry points that call these helpers.
"""

REAR_CAM = "cam1"
ACTIVE_CAMS = (REAR_CAM,)
CAMERA_IMAGE_KEYS = {REAR_CAM: "rear_imgs"}
