"""
Tests that can be ran using pytest to test the secure inner join functionality
"""

import asyncio
from typing import Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper
from tno.mpc.protocols.secure_inner_join.test.test_fixtures import (  # pylint: disable=unused-import
    event_loop,
    fixture_alice,
    fixture_bob,
    fixture_charlie,
    fixture_dave,
    fixture_henri,
    fixture_parties,
    fixture_pool_http,
    fixture_pool_http_3p,
    fixture_pool_http_4p,
    fixture_pool_http_5p,
)

data_alice: npt.NDArray[np.object_] = np.array(
    [
        ["Michiel", "Marcus", "06-05-1874", "1234AB", "M", -1, 31.232],
        ["Thomas", "Rooijakkers", "09-01-1874", "8972ZX", "M", 6, 13.5],
        ["Thomas", "Rooijakkers", "09-01-1874", "1234AB", "M", 2, 12.5],
        ["Thomas", "Rooijakkers", "07-12-1874", "1234AB", "M", 7, 14.3],
        ["Thomas", "Rooijakkers", "17-02-1876", "5634AB", "M", 8, 15.5],
        ["Thomas", "Rooijakkers", "09-01-1874", "7534CD", "M", 9, 11.5],
        ["Bart", "Kamphorst", "06-06-1872", "3412CD", "M", 3, 23.11],
        ["Bart", "Who", "06-12-1875", "3231CD", "M", 5, 2.12],
        ["Nicole", "Gervasoni", "30-01-1877", "3411AS", "F", 1, 8.3],
    ],
    dtype=object,
)

data_bob: npt.NDArray[np.object_] = np.array(
    [
        ["Tomas", "Roijackers", "09-01-1874", "1234AB", "M", 6, 3],
        ["Tomas", "Rooiakkers", "07-12-1874", "2232XY", "M", 7, 1],
        ["Tomas", "Rooijackers", "16-02-1875", "5712DX", "M", 11, 7],
        ["Tomas", "Roijackers", "09-01-1874", "7521LS", "M", 12, 13],
        ["Thomas", "Rooijakkers", "09-01-1874", "1234AB", "M", 5, 10],
        ["Thomas", "Rooijakkers", "06-12-1874", "1234AB", "F", 37, 14],
        ["Thomas", "Rooijakkers", "09-01-1830", "1234AB", "M", 17, 4],
        ["Thomas", "Someone-else", "01-01-1873", "6789CD", "M", 6, 4],
        ["Victor", "Li", "09-01-1823", "6231LI", "M", 231, 2],
        ["Bart", "Kamphoorst", "07-06-1872", "3412CD", "M", 30, 1],
        ["Michiel", "Marcus", "06-05-1874", "1482SH", "M", 40, 8],
        ["Tariq", "Bontekoe", "24-12-1873", "8394HG", "M", 42, 6],
    ],
    dtype=object,
)


data_charlie: npt.NDArray[np.object_] = np.array(
    [
        ["Tomas", "Roijackers", "09-01-1874", "1234AB", "M", 6, 3],
        ["Tomas", "Rooijackers", "16-02-1875", "5712DX", "M", 11, 7],
        ["Tomas", "Roijackers", "09-01-1874", "7521LS", "M", 12, 13],
        ["Thomas", "Rooijakkers", "09-01-1874", "1234AB", "M", 5, 10],
        ["Thomas", "Rooijakkers", "06-12-1874", "1234AB", "F", 37, 14],
        ["Thomas", "Rooijakkers", "09-01-1830", "1234AB", "M", 17, 4],
        ["Thomas", "Someone-else", "01-01-1873", "6789CD", "M", 6, 4],
        ["Victor", "Li", "09-01-1823", "6231LI", "M", 231, 2],
        ["Bart", "Kamphoorst", "07-06-1872", "3412CD", "M", 30, 1],
        ["Michiel", "Marcus", "06-05-1874", "1482SH", "M", 40, 8],
        ["Tariq", "Bontekoe", "24-12-1873", "8394HG", "M", 42, 6],
    ],
    dtype=object,
)


data_dave: npt.NDArray[np.object_] = np.array(
    [
        ["Tariq", "Bontekoe", "24-12-1873", "8394HG", "M", 42, 6],
    ],
    dtype=object,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob,feature_names_charlie,feature_names_dave",
    [
        (
            ("feature_alice_1", "feature_alice_2"),
            ("feature_bob_1", "feature_bob_2"),
            ("feature_charlie_1", "feature_charlie_2"),
            ("feature_dave_1", "feature_dave_2"),
        ),
        (
            ("feature_alice_1", "feature_alice_2"),
            ("feature_bob_1", "feature_bob_2"),
            ("feature_charlie_1", "feature_charlie_2"),
            (),
        ),
        (("feature_alice_1", "feature_alice_2"), (), (), ()),
        ((), (), (), ()),
    ],
)
@pytest.mark.parametrize(
    "data_alice,data_bob,data_charlie,data_dave",
    (((data_alice[:, 5:], data_bob[:, 5:], data_charlie[:, 5:], data_dave[:, 5:]),)),
)
@pytest.mark.parametrize(
    "identifiers_alice,identifiers_bob,identifiers_charlie,identifiers_dave",
    (
        (
            (
                data_alice[:, 0:5],
                data_bob[:, 0:5],
                data_charlie[:, 0:5],
                data_dave[:, 0:5],
            ),
        )
    ),
)
@pytest.mark.parametrize(
    "identifiers_phonetic_alice,identifiers_phonetic_bob,identifiers_phonetic_charlie,identifiers_phonetic_dave",
    (
        (
            (
                data_alice[:, 0:2],
                data_bob[:, 0:2],
                data_charlie[:, 0:2],
                data_dave[:, 0:2],
            ),
        )
    ),
)
@pytest.mark.parametrize(
    "identifiers_phonetic_exact_alice,identifiers_phonetic_exact_bob,identifiers_phonetic_exact_charlie,identifiers_phonetic_exact_dave",
    (
        (
            (
                data_alice[:, 4],
                data_bob[:, 4],
                data_charlie[:, 4],
                data_dave[:, 4],
            ),
        )
    ),
)
@pytest.mark.parametrize(
    "identifier_date_alice,identifier_date_bob,identifier_date_charlie,identifier_date_dave",
    (
        (
            (
                data_alice[:, 2],
                data_bob[:, 2],
                data_charlie[:, 2],
                data_dave[:, 2],
            ),
        )
    ),
)
@pytest.mark.parametrize(
    "identifier_zip6_alice,identifier_zip6_bob,identifier_zip6_charlie,identifier_zip6_dave",
    (
        (
            (
                data_alice[:, 3],
                data_bob[:, 3],
                data_charlie[:, 3],
                data_dave[:, 3],
            ),
        )
    ),
)
async def test_secure_matching(
    parties: Tuple[Tuple[DatabaseOwner, ...], Helper]
) -> None:
    """
    Tests entire protocol

    :param parties: all parties involved in this secure inner join iteration
    """
    all_parties = parties[0] + (parties[1],)
    await asyncio.gather(*[party.run_protocol() for party in all_parties])
    correct_outcome: npt.NDArray[np.object_]
    if len(parties[0]) == 2:
        correct_outcome = np.array(
            [
                [
                    2,
                    12.5,
                    5,
                    10,
                ],
                [
                    -1,
                    31.232,
                    40,
                    8,
                ],
                [
                    3,
                    23.11,
                    30,
                    1,
                ],
                [
                    9,
                    11.5,
                    12,
                    13,
                ],
                [
                    8,
                    15.5,
                    11,
                    7,
                ],
                [
                    7,
                    14.3,
                    6,
                    3,
                ],
            ]
        )
    elif len(parties[0]) == 3:
        correct_outcome = np.array(
            [
                [
                    2,
                    12.5,
                    5,
                    10,
                    5,
                    10,
                ],
                [
                    -1,
                    31.232,
                    40,
                    8,
                    40,
                    8,
                ],
                [
                    3,
                    23.11,
                    30,
                    1,
                    30,
                    1,
                ],
                [
                    9,
                    11.5,
                    12,
                    13,
                    12,
                    13,
                ],
                [
                    8,
                    15.5,
                    11,
                    7,
                    11,
                    7,
                ],
                [
                    7,
                    14.3,
                    6,
                    3,
                    6,
                    3,
                ],
            ]
        )
    else:  # len(parties[0]) == 4
        correct_outcome = np.empty(
            [0, len(parties[0][0].feature_names)], dtype=np.object_
        )
    actual_outcome = cast(
        npt.NDArray[np.object_], sum(map(lambda party: party.shares, parties[0]))
    )
    np.testing.assert_array_equal(
        actual_outcome[np.argsort(actual_outcome[:, 0]), :],
        correct_outcome[np.argsort(correct_outcome[:, 0]), :],
    )
