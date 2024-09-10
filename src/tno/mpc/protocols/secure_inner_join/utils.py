"""
Utilities
"""

import numpy as np
import numpy.typing as npt


def randomize_ndarray(arr: npt.NDArray[np.object_]) -> None:
    """
    Randomize all elements in a numpy array with ciphertexts.

    This function calls 'RandomizableCiphertext.randomize' 'arr.size' times,
    as expected. Note that this contrasts with the 'arr.size+1' calls made by
    'np.vectorize(lambda _: _.randomize())(arr)'.

    :param arr: array to be randomized
    """
    for data in np.nditer(arr, flags=["refs_ok"]):
        data[()].randomize()  # type: ignore[call-overload]
