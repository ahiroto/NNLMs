# coding:utf-8
import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear


class RNNBase(link.Chain):

    def __init__(self, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0):
        super(RNNBase, self).__init__(
            upward=linear.Linear(in_size, out_size,
                                 initialW=0),
            lateral=linear.Linear(out_size, out_size,
                                  initialW=0, nobias=True),
        )
        self.state_size = out_size
        self.lateral_init = lateral_init
        self.upward_init = upward_init
        self.bias_init = bias_init

        if in_size is not None:
            self._initialize_params()

    def _initialize_params(self):
        initializers.init_weight(
            self.lateral.W.data[:, :], self.lateral_init)
        initializers.init_weight(
            self.upward.W.data[:, :], self.upward_init)


class RNN(RNNBase):
    """
    Simple Recurrent Neurarl Network (SRN)
    """

    def __init__(self, in_size, out_size, **kwargs):
        super(RNN, self).__init__(in_size, out_size, **kwargs)
        self.reset_state()

    def to_cpu(self):
        super(RNN, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(RNN, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        """Sets the internal state.

        It sets the :attr:`h` attributes.

        Args:
            h (~chainer.Variable): A new output at the previous time step.
        """
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`h` attributes.

        """
        self.h = None

    def __call__(self, x):

        batch = x.shape[0]
        rnn_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than the '
                       'size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                rnn_in += self.lateral(h_update)
            else:
                rnn_in += self.lateral(self.h)

        y = sigmoid.sigmoid(rnn_in)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
