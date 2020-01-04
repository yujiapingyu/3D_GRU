"""Builds an GRU_3d RNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.layers.rnn_cell import GRU3DCell as gru_3d
import paddle.fluid as fluid

"""Builds an GRU_3d RNN."""
# Stack the list of tensors x by axis(The origin function 'stack' of paddle has bug...)
def stack(x, axis=1):
    for i in range(len(x)):
        x[i] = fluid.layers.unsqueeze(x[i], axes=[axis])
    return fluid.layers.concat(x, axis=axis)


# The discriminator to distinguish whether a image is artificial
def discriminator(data):
    conv1 = fluid.layers.conv2d(input=data, num_filters=12,stride=2,padding="SAME",filter_size=3,act="relu", data_format="NHWC", name="dis_conv1")
    bn1 = fluid.layers.batch_norm(input=conv1)
    fc1 = fluid.layers.fc(input=bn1, size=128, act=None, name="dis_fc1")
    predict = fluid.layers.fc(input=fc1, size=1, act=None, name="dis_fc2")
    return predict


# Calculate loss of discriminator and generator
def cal_loss(gen_images, images):
    b, l, w, h, c = gen_images.shape
    num = b * l

    gen_images_2d = fluid.layers.reshape(x=gen_images, shape=[-1, w, h, c])
    images_2d = fluid.layers.reshape(x=images, shape=[-1, w, h, c])

    predict1 = discriminator(gen_images_2d)
    predict2 = discriminator(images_2d)

    zero_label = fluid.layers.zeros([num, 1], dtype="float32")
    one_label = fluid.layers.ones([num, 1], dtype="float32")

    d_loss_gen = fluid.layers.sigmoid_cross_entropy_with_logits(x=predict1, label=zero_label)
    d_loss_true = fluid.layers.sigmoid_cross_entropy_with_logits(x=predict2, label=one_label)
    g_loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=predict1, label=one_label)

    d_loss_mean = fluid.layers.mean(d_loss_gen + d_loss_true)
    g_loss_mean = fluid.layers.mean(g_loss)
    return d_loss_mean, g_loss_mean


# 3D rnn
def rnn(images, real_input_flag, num_layers, num_hidden, configs):
    """Builds a RNN according to the config."""

    gen_images, lstm_layer, hidden = [], [], []
    shape = images.shape    #(2, 30, 16, 16, 64)
    batch_size = shape[0]
    # seq_length = shape[1]
    ims_width = shape[2]
    ims_height = shape[3]
    output_channels = shape[-1]
    # filter_size = configs.filter_size
    total_length = configs.total_length
    input_length = configs.input_length

    window_length = 2
    window_stride = 1

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = output_channels # 64
        else:
            num_hidden_in = num_hidden[i - 1]   # 64, 64, 64
        new_lstm = gru_3d(
            name='e3d' + str(i),
            input_shape=[batch_size, window_length, ims_width, ims_height, num_hidden_in],
            output_channels=num_hidden[i],  # 64, 64, 64
            kernel_shape=[2, 5, 5])
        lstm_layer.append(new_lstm)
        zero_state = fluid.layers.zeros(
            [batch_size, window_length, ims_width, ims_height, num_hidden[i]], dtype="float32")
        hidden.append(zero_state)

    memory = zero_state

    input_list = []
    for time_step in range(window_length - 1):
        input_list.append(fluid.layers.zeros([batch_size, ims_width, ims_height, output_channels], dtype = 'float32'))
    for time_step in range(total_length - 1):
        if time_step < input_length:
            input_frm = images[:, time_step]
        else:
            time_diff = time_step - input_length
            input_frm = real_input_flag[:, time_diff] * images[:, time_step]  + (1 - real_input_flag[:, time_diff]) * x_gen # x_gen是上一张生成的图片
        input_list.append(input_frm)

        if time_step % (window_length - window_stride) == 0:
            input_frm = stack(input_list[time_step:], axis=1)  # （2,2,16,16,64）
            for i in range(num_layers):
                if i == 0:
                    inputs = input_frm
                else:
                    inputs = hidden[i - 1]

                hidden[i], memory = lstm_layer[i](inputs, hidden[i], memory)

        # hidden[num_layers-1].shape: (2, 2, 16, 16, 64)
        x_gen = fluid.layers.conv3d(hidden[num_layers-1], output_channels,
                                    [window_length, 1, 1], [window_length, 1, 1],
                                    padding='SAME', data_format='NDHWC', name="generator")
        # x_gen.shape: (2, 1, 16, 16, 64)

        x_gen = fluid.layers.squeeze(x_gen, axes=[])    #（2, 16, 16, 64）
        gen_images.append(x_gen)


    gen_images = stack(gen_images, axis=1) # gen_images (2, 29, 16, 16, 64)

    loss1 = fluid.layers.reduce_sum(fluid.layers.square(gen_images - images[:, 1:])) / 2.0
    loss2 = fluid.layers.reduce_sum(fluid.layers.abs(gen_images - images[:, 1:]))

    # d_loss, g_loss = cal_loss(gen_images, images[:, 1:])

    loss = loss1 + loss2

    out_len = total_length - input_length
    out_ims = gen_images[:, -out_len:] # (2, 20, 16, 16, 24)

    return [out_ims, loss]
