"""
Weighted Sparse Threshold Implementation.

This extends the `SparseThreshold` Operator by multiplying the
to-be-thresholded data by provided weights, across a specific dimension.
"""
import numpy as np

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class WeightedSparseThreshold(SparseThreshold):
    """Weighted version of `SparseThreshold` in ModOpt.

    When chosen `scale_based`, it allows the users to specify an array of
    weights W[i] and each weight is assigen to respective scale `i`.
    Also, custom weights can be defined.
    Note that the weights on coarse scale is always set to 0

    Parameters
    ----------
    weights : numpy.ndarray
        Input array of weights or a tuple holding base weight W and power P
    coeffs_shape: tuple
        The shape of linear coefficients
    weight_type : string 'custom' | 'scale_based' | 'custom_scale',
        default 'scale_based'
        Mode of operation of proximity:
        custom      -> custom array of weights
        scale_based -> custom weights applied per scale
    zero_weight_coarse: bool, default True

    linear: object, default `Identity()`
        Linear operator, to be used in cost function evaluation

    See Also
    --------
    SparseThreshold : parent class
    """

    def __init__(self, weights, coeffs_shape, weight_type='scale_based',
                 zero_weight_coarse=True, linear=Identity(), **kwargs):
        self.cf_shape = coeffs_shape
        self.weight_type = weight_type
        available_weight_type = ('scale_based', 'custom')
        if self.weight_type not in available_weight_type:
            raise ValueError('Weight type must be one of ' +
                             ' '.join(available_weight_type))
        self.zero_weight_coarse = zero_weight_coarse
        self.mu = weights
        super().__init__(
            weights=self.mu,
            linear=linear,
            **kwargs
        )

    @property
    def mu(self):
        """`mu` is the weights used for thresholding."""
        return self.weights

    @mu.setter
    def mu(self, input_weights):
        """Update `mu`, based on `coeffs_shape` and `weight_type`."""
        weights_init = np.zeros(np.sum(np.prod(self.cf_shape, axis=-1)))
        start = 0
        if self.weight_type == 'scale_based':
            scale_shapes = np.unique(self.cf_shape, axis=0)
            num_scales = len(scale_shapes)
            if isinstance(input_weights, (float, int, np.float64)):
                weights = input_weights * np.ones(num_scales)
            else:
                if len(input_weights) != num_scales:
                    raise ValueError('The number of weights dont match '
                                     'the number of scales')
                weights = input_weights
            for i, scale_shape in enumerate(np.unique(self.cf_shape, axis=0)):
                scale_sz = np.prod(scale_shape)
                stop = start + scale_sz * np.sum(scale_shape == self.cf_shape)
                weights_init[start:stop] = weights[i]
                start = stop
        elif self.weight_type == 'custom':
            if isinstance(input_weights, (float, int, np.float64)):
                input_weights = input_weights * np.ones(weights_init.shape[0])
            weights_init = input_weights
        if self.zero_weight_coarse:
            weights_init[:np.prod(self.cf_shape[0])] = 0
        self.weights = weights_init
