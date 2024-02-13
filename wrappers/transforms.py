from mmdet.datasets.transforms import RandomAffine
import cv2

from mmdet.structures.bbox import autocast_box_type
from mmdet.registry import TRANSFORMS
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from numpy import random


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

    
    @cache_randomness
    def _get_random_homography_matrix(self, height, width):
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        translate_matrix_pre, translate_matrix_post = self._center_coordinates(height, width)

        # Sandwishing transformations with translations
        warp_matrix = shear_matrix @ rotation_matrix @ scaling_matrix
        warp_matrix_centered = (
            translate_matrix_pre @ warp_matrix @ translate_matrix_post)
        
        return warp_matrix_centered