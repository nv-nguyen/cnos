export LIGHTING_ITENSITY=1.0 # lighting intensity
export RADIUS=0.4 # distance to camera
python -m src.poses.pyrender $CAD_PATH ./src/poses/predefined_poses/obj_poses_level0.npy $OUTPUT_DIR 0 False $LIGHTING_ITENSITY $RADIUS