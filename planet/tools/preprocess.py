# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def preprocess(image, bits, noise=None, return_noise=False):
    bins = 2 ** bits
    image = tf.to_float(image)
    if bits < 8:
        image = tf.floor(image / 2 ** (8 - bits))
    image = image / bins
    if noise==None:
        noise = tf.random_uniform(tf.shape(image), 0, 1.0 / bins)
    image = image + noise
    image = image - 0.5
    if return_noise:
        return image, noise
    return image


def augment(sequence, aug='drq', phase='plan', same=False, simclr=False):
    '''
    :param sequence:
    :param aug:  use rad style translation or drq style
    :param phase: train or test
    :param same: same augmentation for the whole trajectory
    :return:
    '''
    images = sequence['image']
    epi_len, IMG_SIZE, _, _ = images.get_shape().as_list()
    pad = 3

    target_s = IMG_SIZE - 2 * pad

    # RAD style augmentation
    # Central crop + random translate
    # render 108 ->  central crop to 100 -> random pad to 108
    def rad_translate(ori):
        image = tf.image.crop_to_bounding_box(ori, pad, pad, target_s, target_s)
        if phase == 'train':
            offset_h = tf.random_uniform(shape=[], minval=0, maxval=2 * pad, dtype=tf.int32)
            offset_w = tf.random_uniform(shape=[], minval=0, maxval=2 * pad, dtype=tf.int32)
        else:
            print('test/plan augment')
            offset_w, offset_h = pad, pad
        # offset_w, offset_h = pad, pad
        image = tf.image.pad_to_bounding_box(image, offset_h, offset_w, IMG_SIZE, IMG_SIZE)
        # return image
        return image, tf.convert_to_tensor([offset_w, offset_h], dtype=tf.float32)

    # DrQ style augmentation
    # Symmetric pad + crop it to original size
    # render 84 -> pad 4 at boundries + crop to 84
    def drq_translate(ori):
        if phase == 'test':
            return ori
        if same:
            image = tf.pad(ori, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='SYMMETRIC')
        else:
            image = tf.pad(ori, paddings=[[pad, pad], [pad, pad], [0, 0]], mode='SYMMETRIC')
        offset_h = tf.random_uniform(shape=[], minval=0, maxval=2 * pad, dtype=tf.int32)
        offset_w = tf.random_uniform(shape=[], minval=0, maxval=2 * pad, dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, IMG_SIZE, IMG_SIZE)
        return image

    func = rad_translate if aug == 'rad' else drq_translate
    # if not same:
    #     sequence['ori_img'] = images
    # result = func(images) if same else tf.map_fn(lambda img: func(img),
    #                                              images, parallel_iterations=10,
    #                                              back_prop=False)

    result = func(images) if same else tf.map_fn(lambda img: func(img),
                                                            images, parallel_iterations=10,
                                                            back_prop=False, dtype=(tf.uint8, tf.float32))
    if simclr and phase!='plan':
        print('no kidding bro')
        result2 = func(images) if same else tf.map_fn(lambda img: func(img),
                                        images, parallel_iterations=10, back_prop=False, dtype=(tf.uint8, tf.float32))
        for k, v in sequence.items():
            if len(v.shape) > 0:
                sequence[k] = tf.concat([v, v], 0)
            else:
                sequence[k] = tf.stack([v, v], 0)

        if isinstance(result, tuple):
            sequence['image'] = tf.concat([result[0], result2[0]], 0)
            sequence['aug'] = tf.concat([result[1], result2[1]], 0)
        else:
            sequence['image'] = tf.concat([result, result2], 0)
        # if sequence['action'] is not None:
        #     sequence['action'] = tf.concat([sequence['action'], sequence['action']], 0)
        # if sequence['reward'] is not None:
        #     sequence['reward'] = tf.concat([sequence['reward'], sequence['reward']], 0)
    else:
        sequence['image'] = result[0]
        sequence['aug'] = result[1]
    # print(sequence['image'])

    return sequence

def postprocess(image, bits, dtype=tf.float32):
    bins = 2 ** bits
    if dtype == tf.float32:
        image = tf.floor(bins * (image + 0.5)) / bins
    elif dtype == tf.uint8:
        image = image + 0.5
        image = tf.floor(bins * image)
        image = image * (256.0 / bins)
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    else:
        raise NotImplementedError(dtype)
    return image
