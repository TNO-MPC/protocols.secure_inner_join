"""
Validates proper functioning of LSH hashing
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
from bitarray import bitarray

from tno.mpc.protocols.secure_inner_join.lsh import (
    get_hyper_planes,
    lsh_hash,
    weighted_hamming_distance,
)

# Test cases in the form (day, month, year, zip4_code) twice, expected output in the same order (once)
test_cases = [
    ((1, 1, 2000, 1234), (1, 1, 2000, 1234), (0, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 1999, 1234), (0, 0, 1, 0)),
    ((1, 1, 2000, 1234), (1, 1, 2001, 1234), (0, 0, 1, 0)),
    ((1, 1, 2000, 1234), (1, 1, 1995, 1234), (0, 0, 5, 0)),
    ((1, 1, 2000, 1234), (1, 2, 2000, 1234), (0, 1, 0, 0)),
    ((1, 1, 2000, 1234), (1, 12, 2000, 1234), (0, 1, 0, 0)),
    ((1, 1, 2000, 1234), (1, 8, 2000, 1234), (0, 5, 0, 0)),
    ((1, 1, 2000, 1234), (31, 1, 2000, 1234), (1, 0, 0, 0)),
    ((1, 1, 2000, 1234), (2, 1, 2000, 1234), (1, 0, 0, 0)),
    ((1, 1, 2000, 1234), (27, 1, 2000, 1234), (5, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 1334), (0, 0, 0, 1)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 1134), (0, 0, 0, 1)),
    ((1, 1, 2000, 5634), (1, 1, 2000, 5734), (0, 0, 0, 1)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 1734), (0, 0, 0, 5)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 9234), (0, 0, 0, 80)),
    ((1, 1, 2000, 1034), (1, 1, 2000, 9934), (0, 0, 0, 90)),
    ((1, 1, 2000, 9834), (1, 1, 2000, 9934), (0, 0, 0, 1)),
    ((1, 1, 2000, 1234), (2, 2, 2000, 1234), (1, 1, 0, 0)),
    ((1, 1, 2000, 1234), (1, 2, 2001, 1234), (0, 1, 1, 0)),
    ((1, 1, 2000, 1234), (1, 1, 2001, 1134), (0, 0, 1, 1)),
    ((1, 1, 2000, 1234), (2, 1, 2001, 1234), (1, 0, 1, 0)),
    ((1, 1, 2000, 1234), (2, 1, 2000, 1334), (1, 0, 0, 1)),
    ((1, 1, 2000, 1234), (2, 2, 2001, 1234), (1, 1, 1, 0)),
    ((1, 1, 2000, 1234), (1, 2, 2001, 1334), (0, 1, 1, 1)),
    ((1, 1, 2000, 1234), (2, 1, 2001, 1334), (1, 0, 1, 1)),
    ((1, 1, 2000, 1234), (2, 2, 2001, 1234), (1, 1, 1, 0)),
    ((1, 1, 2000, 1234), (2, 2, 2001, 1134), (1, 1, 1, 1)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 1243), (0, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 1284), (0, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 2000, 1299), (0, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 1900, 1234), (0, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 300, 1234), (0, 0, 0, 0)),
    ((1, 1, 2000, 1234), (1, 1, 1800, 1234), (0, 0, 0, 0)),
    ((1, 1, 301, 1234), (1, 1, 300, 1234), (0, 0, 1, 0)),
    ((1, 1, 2000, 1234), (15, 6, 1950, 6034), (14, 5, 50, 48)),
    ((1, 1, 2000, 5034), (15, 6, 1950, 6034), (14, 5, 50, 10)),
]


@pytest.fixture(scope="module", name="hyper_planes_mask")
def fixture_hyper_planes_mask() -> Tuple[npt.NDArray[np.int_], bitarray]:
    """
    Fixture of hyper planes with masking

    :return: pair of hyper planes and masking
    """
    return get_hyper_planes(amount=10000, mask=True)


@pytest.mark.parametrize("value_pair_1, value_pair_2, _expected_outcome", test_cases)
def test_masking(
    value_pair_1: Tuple[int, int, int, int],
    value_pair_2: Tuple[int, int, int, int],
    _expected_outcome: Tuple[int, int, int, int],
    hyper_planes_mask: Tuple[npt.NDArray[np.int_], bitarray],
) -> None:
    """
    Validates correctness of mask usage

    :param value_pair_1: first pairs of date, zip in (dd, mm, yyyy, zip4)-format to encode
    :param value_pair_2: second pairs of date, zip in (dd, mm, yyyy, zip4)-format to encode
    :param hyper_planes_mask: hyper planes to use
    """
    hyper_planes = hyper_planes_mask[0]
    mask = hyper_planes_mask[1]
    encoded_value_1 = lsh_hash(*value_pair_1, hyper_planes)
    encoded_value_2 = lsh_hash(*value_pair_2, hyper_planes)
    total_score, separate_scores = weighted_hamming_distance(
        encoded_value_1, encoded_value_2
    )
    encoded_value_1_mask = lsh_hash(*value_pair_1, hyper_planes, mask)
    encoded_value_2_mask = lsh_hash(*value_pair_2, hyper_planes, mask)
    total_score_mask, separate_scores_mask = weighted_hamming_distance(
        encoded_value_1_mask, encoded_value_2_mask
    )
    assert total_score == total_score_mask
    assert separate_scores == separate_scores_mask


@pytest.mark.parametrize("value_pair_1, value_pair_2, expected_outcome", test_cases)
def test_distance(
    value_pair_1: Tuple[int, int, int, int],
    value_pair_2: Tuple[int, int, int, int],
    expected_outcome: Tuple[int, int, int, int],
    hyper_planes_mask: Tuple[npt.NDArray[np.int_], bitarray],
) -> None:
    """
    Validates correctness of computed distance

    :param value_pair_1: first pairs of date, zip in (dd, mm, yyyy, zip4)-format to encode
    :param value_pair_2: second pairs of date, zip in (dd, mm, yyyy, zip4)-format to encode
    :param expected_outcome: expected (approximate) outcome of distance computation
    :param hyper_planes_mask: hyper planes-mask pair to use
    """
    hyper_planes = hyper_planes_mask[0]
    encoded_value_1 = lsh_hash(*value_pair_1, hyper_planes)
    encoded_value_2 = lsh_hash(*value_pair_2, hyper_planes)
    total_score, separate_scores = weighted_hamming_distance(
        encoded_value_1, encoded_value_2
    )

    expected_deviations = (0.06, 0.03, 0.1, 0.2)
    assert len(separate_scores) == len(expected_outcome)
    expected_total_deviation = 0.0
    expected_total = 0.0
    for score, expected, expected_deviation in zip(
        separate_scores, expected_outcome, expected_deviations
    ):
        if expected == 0:
            assert score == expected
        else:
            assert score == pytest.approx(expected, expected_deviation)
            expected_total += expected
            expected_total_deviation += expected_deviation

    if expected_total == 0:
        assert total_score == expected_total
    else:
        assert total_score == pytest.approx(expected_total, expected_total_deviation)
