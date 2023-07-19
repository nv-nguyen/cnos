from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.visualizer import _create_text_labels, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, BitMasks, Instances
import numpy as np


# overide the default visualizer
class CustomVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = (
            predictions.pred_classes.tolist()
            if predictions.has("pred_classes")
            else None
        )
        labels = _create_text_labels(
            classes, scores, self.metadata.get("thing_classes", None)
        )
        keypoints = (
            predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        )

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [
                GenericMask(x, self.output.height, self.output.width) for x in masks
            ]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                for c in classes
            ]
            alpha = 0.5
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


class CNOSVisualizer(object):
    def __init__(self, obj_names, img_size):
        super(CNOSVisualizer, self).__init__()
        metadata = self.build_metadata(obj_names)
        self.visualizer = CustomVisualizer(
            np.zeros((img_size[0], img_size[1], 3)),
            metadata=metadata,
            scale=1,
            instance_mode=ColorMode.SEGMENTATION,
        )

    def build_metadata(self, obj_names):
        # Define your custom metadata
        custom_metadata = MetadataCatalog.get(
            obj_names[0]
        )  # use the first obj name as the dataset name
        if not hasattr(custom_metadata, "thing_classes"):
            custom_metadata.thing_classes = obj_names
            custom_metadata.thing_colors = np.random.randint(
                0, 255, (len(obj_names), 3)
            ).tolist()
        return custom_metadata

    def convert_to_instances(self, masks, bboxes, scores, labels):
        # Create an Instances object
        instances = Instances(masks.shape[1:])
        instances.set("pred_boxes", bboxes)
        instances.set("pred_masks", masks)
        if scores is not None:
            instances.set("scores", scores)
        if labels is not None:
            instances.set("pred_classes", labels)
        return instances

    def forward(self, rgb, masks, bboxes, scores, labels, save_path):
        self.visualizer.output.reset_image(rgb)
        instances = self.convert_to_instances(masks, bboxes, scores, labels)
        self.visualizer.draw_instance_predictions(instances)
        output = self.visualizer.get_output()
        output.save(save_path)
