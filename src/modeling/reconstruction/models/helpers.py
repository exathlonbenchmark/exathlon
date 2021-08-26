"""Reconstruction helpers module.

Gathers function for helping building reconstruction models.
"""
import math


def get_shape_and_cropping(window_size, n_features, reduction_factor):
    """Returns the reduced window shape and cropping corresponding to `reduction_factor`.

    => Convolutional generators typically multiply spatial dimensions by a constant
    factor at specific layers. In the case the original window dimensions are not multiples
    of their total reduction factor, windows will be generated as slightly larger and then
    cropped to the right dimensions.

    This function returns the shape to start from when generating windows from a
    `reduction_factor`-fold reduction of its spatial dimensions, as well as the final
    dimension-wise cropping to apply at the end of the generation process.

    Args:
        window_size (int): size of input samples in number of records.
        n_features (int): number of input features.
        reduction_factor (float): total reduction factor of spatial dimensions (e.g. 4.0).

    Returns:
        int, [[int, int], [int, int]]: reduced window shape and cropping to apply.
    """
    reduced_shape = math.ceil(window_size / reduction_factor), math.ceil(n_features / reduction_factor)
    cropping = [[0, 0], [0, 0]]
    for i, dim in enumerate([window_size, n_features]):
        # crop the excess in shape when multiplying back by the factor
        dim_crop = reduction_factor - (dim % reduction_factor)
        if dim_crop != reduction_factor:
            # crop more at the end if uneven (arbitrary)
            half_crop = dim_crop / 2.
            cropping[i] = [math.floor(half_crop), math.ceil(half_crop)]
    return reduced_shape, cropping
