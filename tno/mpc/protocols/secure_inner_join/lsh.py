"""
This implements Locality-Sensitive Hashing for dates and zip2-codes.
"""

from typing import Optional, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from numpy.random import Generator
from randomgen import PCG32  # type: ignore[import]
from typing_extensions import Literal


@overload
def get_hyper_planes(
    amount: int = ..., seed: int = ..., mask: Literal[True] = ...
) -> Tuple[npt.NDArray[np.int_], bitarray]:
    ...


@overload
def get_hyper_planes(
    amount: int, seed: int, mask: Literal[False]
) -> npt.NDArray[np.int_]:
    ...


@overload
def get_hyper_planes(
    amount: int = ..., seed: int = ..., mask: bool = ...
) -> Union[npt.NDArray[np.int_], Tuple[npt.NDArray[np.int_], bitarray]]:
    ...


def get_hyper_planes(
    amount: int = 2000, seed: int = 42, mask: bool = False
) -> Union[npt.NDArray[np.int_], Tuple[npt.NDArray[np.int_], bitarray]]:
    """
    Construct a specified number of hyper planes with a set seed.
    We assume the following order: (day, month, year, zip2-code).

    :param amount: number of hyper planes to construct
    :param seed: seed to use for the random generator
    :param mask: set to true to generate a bit mask to use for masking
    :return: array containing the random hyper planes
    """
    random_generator = Generator(PCG32(seed))

    # get random values
    hyper_planes: npt.NDArray[np.int_] = np.array(
        [
            random_generator.integers(0, 62, amount),
            random_generator.integers(0, 12, amount),
            random_generator.integers(0, 100, amount),
            random_generator.integers(10, 100, amount),
        ]
    ).T

    if mask:
        bit_mask = bitarray(
            iter(random_generator.integers(0, 2, amount * 4, dtype=np.uint8))
        )
        return hyper_planes, bit_mask

    return hyper_planes


def encode(
    day: int, month: int, year: int, zip4_code: int
) -> Tuple[int, int, int, int]:
    """
    Encodes day, month, year and zip2 to a Tuple.

    :param day: day of birth
    :param month: month of birth
    :param year: year of birth
    :param zip4_code: the four digits of the postal code
    :return: encoded representation
    """
    assert 1000 <= zip4_code < 10000, "Wrong zip4_code provided"
    assert 1 <= day <= 31, "Wrong day provided"
    assert 1 <= month <= 12, "Wrong month provided"
    day = 2 * day
    year = year % 100
    zip2_code = zip4_code // 100
    return day, month, year, zip2_code


def lsh_hash(
    day: int,
    month: int,
    year: int,
    zip4_code: int,
    hyper_planes: npt.NDArray[np.int_],
    bit_mask: Optional[bitarray] = None,
) -> bitarray:
    """
    Computes a hash encoding for a given encoded input, given a collection of hyperplanes

    :param day: day of birth
    :param month: month of birth
    :param year: year of birth
    :param zip4_code: the four digits of the postal code
    :param hyper_planes: $n$ hyperplanes sampled from
        $[0,62)\times[0,12)\times[0,100)\times[10,100)$
    :param bit_mask: masking to apply to the hashing
    :return: an encode hash, first for $n$ bits belong to day, second $n$
        bits belong to month, etc.
    """
    encoding = encode(day, month, year, zip4_code)
    diff = encoding - hyper_planes
    diff[:, 0:3] = diff[:, 0:3] % [62, 12, 100]

    comparison = np.less_equal(diff, [31, 6, 50, 0])

    combined_hash = bitarray(comparison.astype(np.uint8).T.flat)
    if bit_mask is not None:
        combined_hash ^= bit_mask

    return combined_hash


def weighted_hamming_distance(
    hash_1: bitarray, hash_2: bitarray
) -> Tuple[float, Tuple[float, float, float, float]]:
    """
    if score ~= 1 than we expect at most one element to be one-off

    The score represents the actual distance between two encodings if the number of buckets is large enough
    :param hash_1: first hash
    :param hash_2: second hash
    :return: an x-off distance score, and a tuple of x-off distances per (day, month, year, zip2)
    """
    assert len(hash_1) == len(
        hash_2
    ), f"Mismatch in length of encoding, {len(hash_1)} != {len(hash_2)}, hash_1: {hash_1}, hash_2: {hash_2}"
    difference = hash_1 ^ hash_2

    buckets = len(difference) // 4

    # Transform hash distance to approximate original distance
    days_distance = difference[0:buckets].count() * 15.5 / buckets
    month_distance = difference[buckets : 2 * buckets].count() * 6 / buckets
    year_distance = difference[2 * buckets : 3 * buckets].count() * 50 / buckets
    zip2_distance = difference[3 * buckets : 4 * buckets].count() * 90 / buckets

    distance_tuple = (days_distance, month_distance, year_distance, zip2_distance)
    score = sum(distance_tuple)
    return score, distance_tuple
