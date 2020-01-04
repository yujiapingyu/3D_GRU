"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid


## Eidetic LSTM recurrent network cell
class GRU3DCell(object):
    """
    The basic class of Eidetic LSTM recurrent network cell.
    """

    def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               layer_norm=True,
               norm_scale=True,
               norm_shift=True,
               forget_bias=1.0,
               name="eidetic_lstm_cell"):

    # Construct GRU3DCell. #
        if conv_ndims != len(input_shape) - 2:
            raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(input_shape, conv_ndims))

        self._conv_ndims = conv_ndims
        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._layer_norm = layer_norm
        self._norm_scale = norm_scale
        self._norm_shift = norm_shift
        self._forget_bias = forget_bias
        self._layer_name = name

        self._state_size = self._input_shape[:-1] + [self._output_channels]
        self._output_size = self._input_shape[:-1] + [self._output_channels]

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    # layer normalization
    def _norm(self, input, scale=True, shift=True, name=None):
        normalized = fluid.layers.layer_norm(input=input, begin_norm_axis=1, scale=scale, shift=shift, name=self._layer_name + "_" + name)
        return normalized

    # attention
    def _attn(self, in_query, in_keys, in_values):
        q_shape = list(in_query.shape)
        if len(q_shape) == 4:
            batch = q_shape[0]
            width = q_shape[1]
            height = q_shape[2]
            num_channels = q_shape[3]
        elif len(q_shape) == 5:
            batch = q_shape[0]
            width = q_shape[2]
            height = q_shape[3]
            num_channels = q_shape[4]
        else:
            raise ValueError("Invalid input_shape {} for the query".format(q_shape))

        k_shape = list(in_keys.shape)
        if len(k_shape) != 5:
            raise ValueError("Invalid input_shape {} for the keys".format(k_shape))

        v_shape = list(in_values.shape)
        if len(v_shape) != 5:
            raise ValueError("Invalid input_shape {} for the values".format(v_shape))

        if width != k_shape[2] or height != k_shape[3] or num_channels != k_shape[4]:
            raise ValueError("Invalid input_shape {} and {}, not compatible.".format(q_shape, k_shape))
        if width != v_shape[2] or height != v_shape[3] or num_channels != v_shape[4]:
            raise ValueError("Invalid input_shape {} and {}, not compatible.".format(q_shape, v_shape))
        if k_shape[2] != v_shape[2] or k_shape[3] != v_shape[3] or k_shape[4] != v_shape[4]:
            raise ValueError("Invalid input_shape {} and {}, not compatible.".format(k_shape, v_shape))

        query = fluid.layers.reshape(in_query, shape=[batch, -1, num_channels])
        keys = fluid.layers.reshape(in_keys, shape=[batch, -1, num_channels])
        values = fluid.layers.reshape(in_values, shape=[batch, -1, num_channels])
        attn = fluid.layers.matmul(query, keys, False, True)
        attn = fluid.layers.softmax(attn, axis=2)

        attn = fluid.layers.matmul(attn, values, False, False)
        if len(q_shape) == 4:
          attn = fluid.layers.reshape(attn, shape=[batch, width, height, num_channels])
        else:
          attn = fluid.layers.reshape(attn, shape=[batch, -1, width, height, num_channels])
        return attn

    # convolution
    def _conv(self, inputs, output_channels, kernel_shape, name=None):
        if self._conv_ndims == 2:
            return fluid.layers.conv2d(inputs, output_channels, kernel_shape, 1, padding="SAME", name=self._layer_norm + "_" + name)
        elif self._conv_ndims == 3:
            return fluid.layers.conv3d(inputs, output_channels, kernel_shape, 1, padding="SAME", data_format='NDHWC', name=self._layer_name + "_" + name)

    def __call__(self, inputs, hidden, global_memory):
        new_hidden = self._conv(hidden, 4 * self._output_channels, self._kernel_shape, name="hidden")
        if self._layer_norm:
            new_hidden = self._norm(new_hidden, name="nh")
        z_h, r_h, h_hat_h, h_h = fluid.layers.split(new_hidden, 4, -1)

        new_inputs = self._conv(inputs, 7 * self._output_channels, self._kernel_shape, name="input")
        if self._layer_norm:
            new_inputs = self._norm(new_inputs, name="ni")
        z_x, r_x, h_hat_x, a_x, f_x, i_x, g_x = fluid.layers.split(new_inputs, 7, -1)

        new_global_memory = self._conv(global_memory, 5 * self._output_channels, self._kernel_shape, name="gm")
        if self._layer_norm:
            new_global_memory = self._norm(new_global_memory, name="ngm")
        a_m, f_m, i_m, g_m, m_m = fluid.layers.split(new_global_memory, 5, -1)

        z_t = fluid.layers.sigmoid(z_x + z_h)
        r_t = fluid.layers.sigmoid(r_x + r_h)
        a_t = fluid.layers.sigmoid(a_x + a_m)
        f_t = fluid.layers.sigmoid(f_x + f_m + self._forget_bias)
        i_t = fluid.layers.sigmoid(i_x + i_m)

        recall = self._attn(a_t, g_m, g_m)
        g_t = fluid.layers.tanh(g_x + recall)
        new_global_memory = g_t * i_t + f_t * global_memory

        h_hat_t = fluid.layers.tanh(r_t*h_hat_h + h_hat_x + new_global_memory)
        output = (1-z_t)*h_h + z_t*h_hat_t

        return output, new_global_memory


class Eidetic2DLSTMCell(GRU3DCell):
  """
  2D Eidetic LSTM recurrent network cell.
  """

  def __init__(self, name="eidetic_2d_lstm_cell", **kwargs):
    super(Eidetic2DLSTMCell, self).__init__(conv_ndims=2, name=name, **kwargs)


class Eidetic3DLSTMCell(GRU3DCell):
  """
  3D Eidetic LSTM recurrent network cell.
  """

  def __init__(self, name="eidetic_3d_lstm_cell", **kwargs):
    super(Eidetic3DLSTMCell, self).__init__(conv_ndims=3, name=name, **kwargs)
