"""
Tests that can be ran using pytest to test the secure inner join functionality
"""

import asyncio
from typing import Tuple

import numpy as np
import pytest

from tno.mpc.communication import Pool
from tno.mpc.communication.test import (  # pylint: disable=unused-import
    fixture_pool_http_3p,
)

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper


@pytest.fixture(name="alice")
@pytest.mark.asyncio
async def fixture_alice(
    pool_http_3p: Tuple[Pool, Pool, Pool], feature_names_alice: Tuple[str, ...]
) -> DatabaseOwner:
    """
    Constructs player Alice

    :param pool_http_3p: collection of (three) communication pools
    :param feature_names_alice: feature names of Alice's data
    :return: an initialized database owner
    """
    data = np.array(
        [
            ["Thomas", 2, 12.5],
            ["Michiel", -1, 31.232],
            ["Bart", 3, 23.11],
            ["Nicole", 1, 8.3],
        ],
        dtype=object,
    )
    return DatabaseOwner(
        identifier=list(pool_http_3p[2].pool_handlers)[0],
        data_parties=list(pool_http_3p[2].pool_handlers),
        helper=list(pool_http_3p[1].pool_handlers)[1],
        data=data,
        feature_names=feature_names_alice,
        pool=pool_http_3p[0],
    )


@pytest.fixture(name="bob")
@pytest.mark.asyncio
async def fixture_bob(
    pool_http_3p: Tuple[Pool, Pool, Pool], feature_names_bob: Tuple[str, ...]
) -> DatabaseOwner:
    """
    Constructs player Bob

    :param pool_http_3p: collection of (three) communication pools
    :param feature_names_bob: feature names of Bob's data
    :return: an initialized database owner
    """
    data = np.array(
        [
            ["Thomas", 5, 10],
            ["Victor", 231, 2],
            ["Bart", 30, 1],
            ["Michiel", 40, 8],
            ["Tariq", 42, 6],
        ],
        dtype=object,
    )
    return DatabaseOwner(
        identifier=list(pool_http_3p[2].pool_handlers)[1],
        data_parties=list(pool_http_3p[2].pool_handlers),
        helper=list(pool_http_3p[1].pool_handlers)[1],
        data=data,
        feature_names=feature_names_bob,
        pool=pool_http_3p[1],
    )


@pytest.fixture(name="henri")
@pytest.mark.asyncio
async def fixture_henri(pool_http_3p: Tuple[Pool, Pool, Pool]) -> Helper:
    """
    Constructs player henri

    :param pool_http_3p: collection of (three) communication pools
    :return: an initialized helper party
    """
    return Helper(
        identifier=list(pool_http_3p[1].pool_handlers)[1],
        data_parties=list(pool_http_3p[2].pool_handlers),
        helper=list(pool_http_3p[1].pool_handlers)[1],
        pool=pool_http_3p[2],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        (("feature_alice_1", "feature_alice_2"), ("feature_bob_1", "feature_bob_2")),
        (("feature_alice_1", "feature_alice_2"), ()),
        ((), ("feature_bob_1", "feature_bob_2")),
        ((), ()),
    ],
)
async def test_secure_inner_join(
    alice: DatabaseOwner, bob: DatabaseOwner, henri: Helper
) -> None:
    """
    Tests entire protocol

    :param alice: first database owner
    :param bob: second database owner
    :param henri: helper party
    """
    await asyncio.gather(
        *[alice.run_protocol(), bob.run_protocol(), henri.run_protocol()]
    )
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
        ]
    )
    actual_outcome = alice.shares + bob.shares
    np.testing.assert_array_equal(
        actual_outcome[np.argsort(actual_outcome[:, 0]), :],
        correct_outcome[np.argsort(correct_outcome[:, 0]), :],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        (("feature_alice_1", "feature_alice_2"), ("feature_bob_1", "feature_bob_2")),
        (("feature_alice_1", "feature_alice_2"), ()),
        ((), ("feature_bob_1", "feature_bob_2")),
        ((), ()),
    ],
)
async def test_features_send_receive(alice: DatabaseOwner, bob: DatabaseOwner) -> None:
    """
    Tests sending and receiving of feature names

    :param alice: first database owner
    :param bob: second database owner
    """
    # pylint: disable=protected-access
    await asyncio.gather(
        *[
            alice.send_feature_names(),
            alice.receive_feature_names(),
            bob.send_feature_names(),
            bob.receive_feature_names(),
        ]
    )
    assert alice._received_feature_names
    assert bob._received_feature_names
    assert alice.feature_names == bob.feature_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        ((), ()),
    ],
)
async def test_randomness_send_receive(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of randomness

    :param alice: first database owner
    :param bob: second database owner
    """
    await asyncio.gather(
        *[
            alice.send_randomness(),
            alice.receive_randomness(),
            bob.send_randomness(),
            bob.receive_randomness(),
        ]
    )
    assert alice.shared_randomness == bob.shared_randomness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        ((), ()),
    ],
)
async def test_paillier_send_receive_single_direction_bob(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of paillier schemes to Bob

    :param alice: first database owner
    :param bob: second database owner
    """
    await asyncio.gather(*[alice.send_paillier_scheme(), bob.receive_paillier_scheme()])
    assert (
        alice.paillier_scheme.public_key.n == bob.received_paillier_scheme.public_key.n
    )
    assert alice.paillier_scheme.precision == bob.received_paillier_scheme.precision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        ((), ()),
    ],
)
async def test_paillier_send_receive_single_direction_alice(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of paillier schemes to Alice

    :param alice: first database owner
    :param bob: second database owner
    """
    await asyncio.gather(*[bob.send_paillier_scheme(), alice.receive_paillier_scheme()])
    assert (
        bob.paillier_scheme.public_key.n == alice.received_paillier_scheme.public_key.n
    )
    assert bob.paillier_scheme.precision == alice.received_paillier_scheme.precision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        (("feature_alice_1", "feature_alice_2"), ("feature_bob_1", "feature_bob_2")),
        (("feature_alice_1", "feature_alice_2"), ()),
        ((), ("feature_bob_1", "feature_bob_2")),
        ((), ()),
    ],
)
async def test_features_send_receive_single_direction_bob(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of feature names to Bob

    :param alice: first database owner
    :param bob: second database owner
    """
    # pylint: disable=protected-access
    await asyncio.gather(*[alice.send_feature_names(), bob.receive_feature_names()])
    assert alice._own_feature_names == bob._received_feature_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        (("feature_alice_1", "feature_alice_2"), ("feature_bob_1", "feature_bob_2")),
        (("feature_alice_1", "feature_alice_2"), ()),
        ((), ("feature_bob_1", "feature_bob_2")),
        ((), ()),
    ],
)
async def test_features_send_receive_single_direction_alice(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of feature names to Alice

    :param alice: first database owner
    :param bob: second database owner
    """
    # pylint: disable=protected-access
    await asyncio.gather(*[bob.send_feature_names(), alice.receive_feature_names()])
    assert bob._own_feature_names == alice._received_feature_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        ((), ()),
    ],
)
async def test_randomness_send_receive_single_direction_bob(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of randomness to Bob

    :param alice: first database owner
    :param bob: second database owner
    """
    # pylint: disable=protected-access
    await asyncio.gather(*[alice.send_randomness(), bob.receive_randomness()])
    assert alice._own_randomness == bob._received_randomness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob",
    [
        ((), ()),
    ],
)
async def test_randomness_send_receive_single_direction_alice(
    alice: DatabaseOwner, bob: DatabaseOwner
) -> None:
    """
    Tests sending and receiving of randomness to Alice

    :param alice: first database owner
    :param bob: second database owner
    """
    # pylint: disable=protected-access
    await asyncio.gather(*[bob.send_randomness(), alice.receive_randomness()])
    assert bob._own_randomness == alice._received_randomness
