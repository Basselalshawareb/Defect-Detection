from mmdet.datasets.transforms import RandomAffine
import cv2

from mmdet.structures.bbox import autocast_box_type
from mmdet.registry import TRANSFORMS



@TRANSFORMS.register_module()
class RotateCenter(RandomAffine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _center_coordinates(self, height, width):
        trans_x = width/2
        trans_y = height/2
        translate_matrix_pre = self._get_translation_matrix(trans_x, trans_y)
        translate_matrix_post = self._get_translation_matrix(-trans_x, -trans_y)
        return translate_matrix_pre, translate_matrix_post

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)
        translate_matrix_pre, translate_matrix_post = self._center_coordinates(height, width)
        warp_matrix_centered = translate_matrix_pre@warp_matrix@translate_matrix_post
        img = cv2.warpPerspective(
            img,
            warp_matrix_centered,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes.project_(warp_matrix_centered)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

            if 'gt_masks' in results:
                raise NotImplementedError('RandomAffine only supports bbox.')
        return results