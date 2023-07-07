# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module to construct DELF feature extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

from delf import feature_extractor


# Minimum dimensions below which features are not extracted (empty
# features are returned). This applies after any resizing is performed.
_MIN_HEIGHT = 10
_MIN_WIDTH = 10



def ExtractorFn(model, image, resize_factor=1.0):
    new_shape = (224, 224)
    pil_image = Image.fromarray(image)
    resized_image = np.array(pil_image.resize(new_shape, resample=Image.BILINEAR))
    height, width, channels = pil_image.shape
    scale_factors = np.array([new_shape[1] / height, new_shape[0] / width],
                             dtype=float)

    # Input tensors.
    image_tensor = tf.convert_to_tensor(resized_image)

    # Extracted features.
    extracted_features = {}
    output = None

    predict = model.signatures['serving_default']
    output_dict = predict(
        input_image=image_tensor,
        input_scales=image_scales_tensor,
        input_max_feature_num=max_feature_num_tensor,
        input_abs_thres=score_threshold_tensor)
    output = [
        output_dict['boxes'], output_dict['features'],
        output_dict['scales'], output_dict['scores']
    ]

    # Post-process extracted features: normalize, PCA (optional), pooling.
    if config.use_global_features:
      raw_global_descriptors = output[-1]
      global_descriptors_per_scale = feature_extractor.PostProcessDescriptors(
          raw_global_descriptors, config.delf_global_config.use_pca,
          global_pca_parameters)
      unnormalized_global_descriptor = tf.reduce_sum(
          global_descriptors_per_scale, axis=0, name='sum_pooling')
      global_descriptor = tf.nn.l2_normalize(
          unnormalized_global_descriptor, axis=0, name='final_l2_normalization')
      extracted_features.update({
          'global_descriptor': global_descriptor.numpy(),
      })

    if config.use_local_features:
      boxes = output[0]
      raw_local_descriptors = output[1]
      feature_scales = output[2]
      attention_with_extra_dim = output[3]

      attention = tf.reshape(attention_with_extra_dim,
                             [tf.shape(attention_with_extra_dim)[0]])
      locations, local_descriptors = (
          feature_extractor.DelfFeaturePostProcessing(
              boxes, raw_local_descriptors, config.delf_local_config.use_pca,
              local_pca_parameters))
      if not config.delf_local_config.use_resized_coordinates:
        locations /= scale_factors

      extracted_features.update({
          'local_features': {
              'locations': locations.numpy(),
              'descriptors': local_descriptors.numpy(),
              'scales': feature_scales.numpy(),
              'attention': attention.numpy(),
          }
      })

    return extracted_features