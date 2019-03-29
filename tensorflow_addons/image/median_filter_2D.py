from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.contrib.distributions.python.ops import sample_stats



@tf_export('image.median_filter_2D')
def median_filter_2D(input, filter_shape=(3, 3)):
    """This method performs Median Filtering on image.Filter shape can be user given.
       This method takes both kind of images where pixel values lie between 0 to 255 and where it lies between 0.0 and 1.0
       Args:
           input: A 3D `Tensor` of type `float32` or 'int32' or 'float64' or 'int64 and of shape`[rows, columns, channels]`

           filter_shape: Optional Argument. A tuple of 2 integers .(R,C).R is the first value is the number of rows in the
           filter and C is the second value in the filter is the number of columns in the filter. This creates a filter of
           shape (R,C) or RxC filter. Default value = (3,3)

        Returns:
            A 3D median filtered image tensor of shape [rows,columns,channels] and type 'int32'. Pixel value of returned
            tensor ranges between 0 to 255
    """

    if not isinstance(filter_shape, tuple):
        raise TypeError('Filter shape must be a tuple')
    if len(filter_shape) != 2:
        raise ValueError('Filter shape must be a tuple of 2 integers .Got %s values in tuple' % len(filter_shape))
    filter_shapex = filter_shape[0]
    filter_shapey = filter_shape[1]
    if isinstance(filter_shapex, int) and isinstance(filter_shapey,
            int):
        pass
    else:
        raise TypeError('Size of the filter must be Integers')
    input = image_ops_impl._Assert3DImage(input)
    (m, no, ch) = (input.shape[0].value, input.shape[1].value, input.shape[2].value)
    if  m != None and  no != None and ch != None:
        (m, no, ch) = (int(m), int(no), int(ch))
    else:
        raise TypeError('All the Dimensions of the input image tensor must be Integers.')
    if m < filter_shapex or no < filter_shapey:
        raise ValueError('No of Pixels in each dimension of the image should be more than the filter size. Got filter_shape (%sx'
                          % filter_shape[0] + '%s).' % filter_shape[1]
                         + ' Image Shape (%s)' % input.shape)
    if filter_shapex % 2 == 0 or filter_shapey % 2 == 0:
        raise ValueError('Filter size should be odd. Got filter_shape (%sx'
                          % filter_shape[0] + '%s)' % filter_shape[1])
    input = math_ops.cast(input, dtypes.float32)
    tf_i = array_ops.reshape(input, [m * no * ch])
    ma = math_ops.reduce_max(tf_i)

    def normalize(li):
        one = ops.convert_to_tensor(1.0)
        two = ops.convert_to_tensor(255.0)

        def func1():
            return li

        def func2():
            return math_ops.truediv(li, two)

        return control_flow_ops.cond(gen_math_ops.greater(ma, one),
                func2, func1)

    input = normalize(input)

    # k and l is the Zero-padding size

    listi = []
    for a in range(ch):
        img = input[:, :, a:a + 1]
        img = array_ops.reshape(img, [1, m, no, 1])
        slic = gen_array_ops.extract_image_patches(img, [1,
                filter_shapex, filter_shapey, 1], [1, 1, 1, 1], [1, 1,
                1, 1], padding='SAME')
        li = sample_stats.percentile(slic, 50, axis=3)
        li = array_ops.reshape(li, [m, no, 1])
        listi.append(li)
    y = array_ops.concat(listi[0], 2)

    for i in range(len(listi) - 1):
        y = array_ops.concat([y, listi[i + 1]], 2)

    y *= 255
    y = math_ops.cast(y, dtypes.int32)

    return y
