import tensorflow as tf
import numpy as np
import math

from typeguard import typechecked

from tensorflow_addons.utils import types
from tensorflow.python.keras.utils import conv_utils

from tensorflow_addons.layers.deformable_conv2d import _deformable_conv2d


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def equi_coord(pano_W, pano_H, k_W, k_H, u, v):
    fov_w = k_W * np.deg2rad(360.0 / float(pano_W))
    focal = (float(k_W) / 2) / np.tan(fov_w / 2)
    c_x = 0
    c_y = 0

    u_r, v_r = u, v
    u_r, v_r = u_r - float(pano_W) / 2.0, v_r - float(pano_H) / 2.0
    phi, theta = u_r / (pano_W) * (np.pi) * 2, -v_r / (pano_H) * (np.pi)

    ROT = rotation_matrix((0, 1, 0), phi)
    ROT = np.matmul(ROT, rotation_matrix((1, 0, 0), theta))  # np.eye(3)

    h_range = np.array(range(k_H))
    w_range = np.array(range(k_W))
    w_ones = np.ones(k_W)
    h_ones = np.ones(k_H)
    h_grid = (
        np.matmul(np.expand_dims(h_range, -1), np.expand_dims(w_ones, 0))
        + 0.5
        - float(k_H) / 2
    )
    w_grid = (
        np.matmul(np.expand_dims(h_ones, -1), np.expand_dims(w_range, 0))
        + 0.5
        - float(k_W) / 2
    )

    K = np.array([[focal, 0, c_x], [0, focal, c_y], [0.0, 0.0, 1.0]])
    inv_K = np.linalg.inv(K)
    rays = np.stack([w_grid, h_grid, np.ones(h_grid.shape)], 0)
    rays = np.matmul(inv_K, rays.reshape(3, k_H * k_W))
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rays = np.matmul(ROT, rays)
    rays = rays.reshape((3, k_H, k_W))

    phi = np.arctan2(rays[0, ...], rays[2, ...])
    theta = np.arcsin(np.clip(rays[1, ...], -1, 1))
    x = (pano_W) / (2.0 * np.pi) * phi + float(pano_W) / 2.0
    y = (pano_H) / (np.pi) * theta + float(pano_H) / 2.0

    roi_y = h_grid + v_r + float(pano_H) / 2.0
    roi_x = w_grid + u_r + float(pano_W) / 2.0

    new_roi_y = y
    new_roi_x = x

    offsets_x = new_roi_x - roi_x
    offsets_y = new_roi_y - roi_y

    return offsets_x, offsets_y


def equi_coord_fixed_resoltuion(pano_W, pano_H, k_W, k_H, u, v, pano_Hf=-1, pano_Wf=-1):
    pano_Hf = pano_H if pano_Hf <= 0 else pano_H / pano_Hf
    pano_Wf = pano_W if pano_Wf <= 0 else pano_W / pano_Wf
    fov_w = k_W * np.deg2rad(360.0 / float(pano_Wf))
    focal = (float(k_W) / 2) / np.tan(fov_w / 2)
    c_x = 0
    c_y = 0

    u_r, v_r = u, v
    u_r, v_r = u_r - float(pano_W) / 2.0, v_r - float(pano_H) / 2.0
    phi, theta = u_r / (pano_W) * (np.pi) * 2, -v_r / (pano_H) * (np.pi)

    ROT = rotation_matrix((0, 1, 0), phi)
    ROT = np.matmul(ROT, rotation_matrix((1, 0, 0), theta))  # np.eye(3)

    h_range = np.array(range(k_H))
    w_range = np.array(range(k_W))
    w_ones = np.ones(k_W)
    h_ones = np.ones(k_H)
    h_grid = (
        np.matmul(np.expand_dims(h_range, -1), np.expand_dims(w_ones, 0))
        + 0.5
        - float(k_H) / 2
    )
    w_grid = (
        np.matmul(np.expand_dims(h_ones, -1), np.expand_dims(w_range, 0))
        + 0.5
        - float(k_W) / 2
    )

    K = np.array([[focal, 0, c_x], [0, focal, c_y], [0.0, 0.0, 1.0]])
    inv_K = np.linalg.inv(K)
    rays = np.stack([w_grid, h_grid, np.ones(h_grid.shape)], 0)
    rays = np.matmul(inv_K, rays.reshape(3, k_H * k_W))
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rays = np.matmul(ROT, rays)
    rays = rays.reshape((3, k_H, k_W))

    phi = np.arctan2(rays[0, ...], rays[2, ...])
    theta = np.arcsin(np.clip(rays[1, ...], -1, 1))
    x = (pano_W) / (2.0 * np.pi) * phi + float(pano_W) / 2.0
    y = (pano_H) / (np.pi) * theta + float(pano_H) / 2.0

    roi_y = h_grid + v_r + float(pano_H) / 2.0
    roi_x = w_grid + u_r + float(pano_W) / 2.0

    new_roi_y = y
    new_roi_x = x

    offsets_x = new_roi_x - roi_x
    offsets_y = new_roi_y - roi_y

    return offsets_x, offsets_y


def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width=1, s_height=1, bs=16):
    offset = np.zeros(shape=[pano_H, pano_W, k_H * k_W * 2])

    for v in range(0, pano_H, s_height):
        for u in range(0, pano_W, s_width):
            offsets_x, offsets_y = equi_coord_fixed_resoltuion(
                pano_W, pano_H, k_W, k_H, u, v, 1, 1
            )
            offsets = np.concatenate(
                (np.expand_dims(offsets_y, -1), np.expand_dims(offsets_x, -1)), axis=-1
            )
            total_offsets = offsets.flatten().astype("float32")
            offset[v, u, :] = total_offsets

    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, 0)
    offset = tf.tile(offset, multiples=[bs, 1, 1, 1])
    offset = tf.cast(offset, tf.float32)

    return offset


class EquiConv(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        filters: int,
        kernel_size: tuple = (3, 3),
        num_groups: int = 1,
        deformable_groups: int = 1,
        strides: tuple = (1, 1),
        im2col: int = 1,
        use_bias: bool = False,
        padding: str = "valid",
        data_format: str = "channels_last",
        dilations: tuple = (1, 1),
        use_relu: bool = False,
        kernel_initializer: types.Initializer = None,
        kernel_regularizer: types.Regularizer = None,
        kernel_constraint: types.Constraint = None,
        **kwargs
    ):
        super(EquiConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.deformable_groups = deformable_groups
        self.strides = strides
        self.im2col = im2col
        self.use_bias = use_bias
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations
        self.use_relu = use_relu
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        if self.padding == "valid":
            self.tf_pad = "VALID"
        else:
            self.tf_pad = "SAME"

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel = int(input_shape[-1])
        else:
            channel = int(input_shape[1])
        self.kernel = self.add_weight(
            shape=[self.filters, channel, self.kernel_size[0], self.kernel_size[1]],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[1, self.filters, 1, 1],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters] + new_space)

    def call(self, inputs, **kwargs):
        if self.data_format == "channels_first":
            data = tf.transpose(inputs, [0, 2, 3, 1])
        else:
            data = inputs
        n, h, w, c_i = tuple(data.get_shape().as_list())
        data_shape = tf.shape(data)
        """
        The original implement in paper here bs is set as self.batch_size, here wo use data_shape[0],
        because self.batch_size if constant value and can't changed, but actually image batch_size can
        change in train and test period, so we use tf.shape to get actual dynamic batch_size.
        """
        offset = tf.stop_gradient(
            distortion_aware_map(
                w,
                h,
                self.kernel_size[0],
                self.kernel_size[1],
                s_width=self.strides[0],
                s_height=self.strides[1],
                bs=data_shape[0],
            )
        )
        mask = tf.stop_gradient(
            tf.zeros(
                shape=[
                    data_shape[0],
                    data_shape[1],
                    data_shape[2],
                    self.kernel_size[0] * self.kernel_size[1],
                ]
            )
        )
        data = tf.transpose(data, [0, 3, 1, 2])
        offset = tf.transpose(offset, [0, 3, 1, 2])
        mask = tf.transpose(mask, [0, 3, 1, 2])
        res = _deformable_conv2d(
            data,
            self.kernel,
            offset,
            mask,
            [1, 1, self.strides[0], self.strides[1]],
            num_groups=self.num_groups,
            deformable_groups=self.deformable_groups,
            padding=self.tf_pad,
            data_format="NCHW",
        )
        if self.use_bias:
            res = tf.add(res, self.bias)
        if self.use_relu:
            res = tf.nn.relu(res)
        if self.data_format == "channels_last":
            return tf.transpose(res, [0, 2, 3, 1])
        else:
            return res

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "num_groups": self.num_groups,
            "deformable_groups": self.deformable_groups,
            "strides": self.strides,
            "im2col": self.im2col,
            "use_bias": self.use_bias,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilations": self.dilations,
            "use_relu": self.use_relu,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "tf_pad": self.tf_pad,
        }
        base_config = super().get_config()
        return {**base_config, **config}
