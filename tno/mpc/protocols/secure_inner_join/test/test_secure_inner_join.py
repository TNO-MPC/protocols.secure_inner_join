"""
Tests that can be ran using pytest to test the secure inner join functionality
"""

import asyncio
from hashlib import blake2b
from typing import Callable, Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest

from tno.mpc.communication import Pool

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper
from tno.mpc.protocols.secure_inner_join.test.test_fixtures import (  # pylint: disable=unused-import
    compute_regular_intersection,
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

NONE = ((None, None, None, None),)

data_alice: npt.NDArray[np.object_] = np.array(
    [
        ["Thomas", 2, 12.5],
        ["Michiel", -1, 31.232],
        ["Bart", 3, 23.11],
        ["Nicole", 1, 8.3],
        ["Robert", -5, 12.4],
        ["Alex", -7, 3.5],
        ["Daniel", 23, 1.15],
    ],
    dtype=object,
)
data_bob: npt.NDArray[np.object_] = np.array(
    [
        ["Thomas", 5, 10],
        ["Victor", 231, 2],
        ["Michiel", 40, 8],
        ["Tariq", 42, 6],
        ["Alex", 133, 5],
        ["Daniel", 1000, 11],
    ],
    dtype=object,
)
data_charlie: npt.NDArray[np.object_] = np.array(
    [
        ["Thomas", 3, 1],
        ["Victor", 4.5, 2],
        ["Bart", 100, 1],
        ["Tariq", 22, 0],
        ["Alex", -33, 6],
        ["Daniel", -1.2, 100],
    ],
    dtype=object,
)
data_dave: npt.NDArray[np.object_] = np.array(
    [
        ["Thomas", 5, 10],
        ["Bart", 30, 1],
        ["Tariq", 42, 6],
        ["Daniel", 1000, 11],
        ["Alessandro", -1000, -10],
    ],
    dtype=object,
)

intersection_two_party = compute_regular_intersection((data_alice, data_bob))
intersection_three_party = compute_regular_intersection(
    (data_alice, data_bob, data_charlie)
)
intersection_four_party = compute_regular_intersection(
    (data_alice, data_bob, data_charlie, data_dave)
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
    (((data_alice[:, 1:], data_bob[:, 1:], data_charlie[:, 1:], data_dave[:, 1:]),)),
)
@pytest.mark.parametrize(
    "identifiers_alice,identifiers_bob,identifiers_charlie,identifiers_dave",
    (((data_alice[:, 0], data_bob[:, 0], data_charlie[:, 0], data_dave[:, 0]),)),
)
@pytest.mark.parametrize(
    "identifiers_phonetic_alice,identifiers_phonetic_bob,identifiers_phonetic_charlie,identifiers_phonetic_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifiers_phonetic_exact_alice,identifiers_phonetic_exact_bob,identifiers_phonetic_exact_charlie,identifiers_phonetic_exact_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_date_alice,identifier_date_bob,identifier_date_charlie,identifier_date_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_zip6_alice,identifier_zip6_bob,identifier_zip6_charlie,identifier_zip6_dave",
    NONE,
)
async def test_secure_inner_join(
    parties: Tuple[Tuple[DatabaseOwner, ...], Helper]
) -> None:
    """
    Tests entire protocol

    :param parties: all parties involved in this secure inner join iteration
    """
    all_parties = parties[0] + (parties[1],)
    await asyncio.gather(*[party.run_protocol() for party in all_parties])
    if len(parties[0]) == 2:
        correct_outcome = intersection_two_party
    elif len(parties[0]) == 3:
        correct_outcome = intersection_three_party
    else:  # len(parties[0]) == 4
        correct_outcome = intersection_four_party

    actual_outcome = cast(
        npt.NDArray[np.object_], sum(map(lambda party: party.shares, parties[0]))
    )
    np.testing.assert_array_equal(
        actual_outcome[np.argsort(actual_outcome[:, 0]), :],
        correct_outcome[np.argsort(correct_outcome[:, 1]), 1:],
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
    (((data_alice[:, 1:], data_bob[:, 1:], data_charlie[:, 1:], data_dave[:, 1:]),)),
)
@pytest.mark.parametrize(
    "identifiers_alice,identifiers_bob,identifiers_charlie,identifiers_dave",
    (((data_alice[:, 0], data_bob[:, 0], data_charlie[:, 0], data_dave[:, 0]),)),
)
@pytest.mark.parametrize(
    "identifiers_phonetic_alice,identifiers_phonetic_bob,identifiers_phonetic_charlie,identifiers_phonetic_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifiers_phonetic_exact_alice,identifiers_phonetic_exact_bob,identifiers_phonetic_exact_charlie,identifiers_phonetic_exact_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_date_alice,identifier_date_bob,identifier_date_charlie,identifier_date_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_zip6_alice,identifier_zip6_bob,identifier_zip6_charlie,identifier_zip6_dave",
    NONE,
)
async def test_features_send_receive(
    parties: Tuple[Tuple[DatabaseOwner, ...], Helper]
) -> None:
    """
    Tests sending and receiving of feature names

    :param parties: all parties involved in this secure inner join iteration
    """
    # pylint: disable=protected-access
    await asyncio.gather(
        *(
            [party.send_feature_names_to_all() for party in parties[0]]
            + [party.receive_all_feature_names() for party in parties[0]]
        )
    )
    for party in parties[0]:
        assert party._received_feature_names
        assert parties[0][0].feature_names == party.feature_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_parties_alice",
    [
        ("local1", "local2", "local3", "local4", "local5", "local6"),
        ("local1",),
        ("alice", "bob"),
    ],
)
@pytest.mark.asyncio
async def test_verify_data_parties_incorrect(
    pool_http: Tuple[Pool, ...],
    data_parties_alice: Tuple[str, ...],
) -> None:
    """
    Test the verification of the data_parties, when the number of data_parties is too small or too large.
    This should raise an error either on construction or in verifying by sending.

    :param pool_http: collection of (at least three) communication pools
    :param data_parties_alice: Tuple containing the data parties that Alice learns.
    """
    henri = Helper(
        identifier=list(pool_http[1].pool_handlers)[0],
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        pool=pool_http[0],
    )
    with pytest.raises(ValueError):
        alice = DatabaseOwner(
            identifier=list(pool_http[0].pool_handlers)[0],
            data_parties=data_parties_alice,
            helper=list(pool_http[1].pool_handlers)[0],
            identifiers=np.zeros(5, dtype=object),
            data=np.zeros((5, 5), dtype=object),
            feature_names=tuple(),
            pool=pool_http[1],
        )
        await asyncio.gather(
            alice.receive_and_verify_data_parties(), henri.send_data_parties()
        )


@pytest.mark.asyncio
async def test_verify_data_parties_incorrect_sort(
    pool_http: Tuple[Pool, ...],
) -> None:
    """
    Test that data parties get sorted correctly, if they are not initially.

    :param pool_http: collection of (at least three) communication pools
    """
    henri = Helper(
        identifier=list(pool_http[1].pool_handlers)[0],
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        pool=pool_http[0],
    )
    data_parties_alice = list(pool_http[0].pool_handlers)
    data_parties_alice.append(data_parties_alice.pop(0))
    alice = DatabaseOwner(
        identifier=list(pool_http[0].pool_handlers)[0],
        data_parties=data_parties_alice,
        helper=list(pool_http[1].pool_handlers)[0],
        identifiers=np.zeros(5, dtype=object),
        data=np.zeros((5, 5), dtype=object),
        feature_names=tuple(),
        pool=pool_http[1],
    )
    await asyncio.gather(
        alice.receive_and_verify_data_parties(), henri.send_data_parties()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob,feature_names_charlie,feature_names_dave",
    [
        ((), (), (), ()),
    ],
)
@pytest.mark.parametrize(
    "data_alice,data_bob,data_charlie,data_dave",
    (((data_alice[:, 1:], data_bob[:, 1:], data_charlie[:, 1:], data_dave[:, 1:]),)),
)
@pytest.mark.parametrize(
    "identifiers_alice,identifiers_bob,identifiers_charlie,identifiers_dave",
    (((data_alice[:, 0], data_bob[:, 0], data_charlie[:, 0], data_dave[:, 0]),)),
)
@pytest.mark.parametrize(
    "identifiers_phonetic_alice,identifiers_phonetic_bob,identifiers_phonetic_charlie,identifiers_phonetic_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifiers_phonetic_exact_alice,identifiers_phonetic_exact_bob,identifiers_phonetic_exact_charlie,identifiers_phonetic_exact_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_date_alice,identifier_date_bob,identifier_date_charlie,identifier_date_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_zip6_alice,identifier_zip6_bob,identifier_zip6_charlie,identifier_zip6_dave",
    NONE,
)
async def test_randomness_send_receive(
    parties: Tuple[Tuple[DatabaseOwner, ...], Helper]
) -> None:
    """
    Tests sending and receiving of randomness

    :param parties: all parties involved in this secure inner join iteration
    """
    await asyncio.gather(
        *(
            [party.send_randomness_to_all() for party in parties[0]]
            + [party.receive_all_randomness() for party in parties[0]]
        )
    )
    for party in parties[0]:
        assert party.shared_randomness
        assert parties[0][0].shared_randomness == party.shared_randomness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob,feature_names_charlie,feature_names_dave",
    [
        ((), (), (), ()),
    ],
)
@pytest.mark.parametrize(
    "data_alice,data_bob,data_charlie,data_dave",
    (((data_alice[:, 1:], data_bob[:, 1:], data_charlie[:, 1:], data_dave[:, 1:]),)),
)
@pytest.mark.parametrize(
    "identifiers_alice,identifiers_bob,identifiers_charlie,identifiers_dave",
    (((data_alice[:, 0], data_bob[:, 0], data_charlie[:, 0], data_dave[:, 0]),)),
)
@pytest.mark.parametrize(
    "identifiers_phonetic_alice,identifiers_phonetic_bob,identifiers_phonetic_charlie,identifiers_phonetic_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifiers_phonetic_exact_alice,identifiers_phonetic_exact_bob,identifiers_phonetic_exact_charlie,identifiers_phonetic_exact_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_date_alice,identifier_date_bob,identifier_date_charlie,identifier_date_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_zip6_alice,identifier_zip6_bob,identifier_zip6_charlie,identifier_zip6_dave",
    NONE,
)
async def test_paillier_send_receive(
    parties: Tuple[Tuple[DatabaseOwner, ...], Helper]
) -> None:
    """
    Tests sending and receiving of paillier schemes

    :param parties: all parties involved in this secure inner join iteration
    """
    await asyncio.gather(
        *(
            [party.send_paillier_scheme_to_all() for party in parties[0]]
            + [party.receive_all_paillier_schemes() for party in parties[0]]
        )
    )
    for party1 in parties[0]:
        for party2 in parties[0]:
            if party1 == party2:
                continue
            assert (
                party1.paillier_scheme.public_key.n
                == party2.received_paillier_schemes[party1.identifier].public_key.n
            )
            assert (
                party1.paillier_scheme.precision
                == party2.received_paillier_schemes[party1.identifier].precision
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_alice,data_bob,data_charlie,data_dave",
    (((data_alice[:, 1:], data_bob[:, 1:], data_charlie[:, 1:], data_dave[:, 1:]),)),
)
@pytest.mark.parametrize(
    "identifiers_alice,identifiers_bob,identifiers_charlie,identifiers_dave",
    (((data_alice[:, 0], data_bob[:, 0], data_charlie[:, 0], data_dave[:, 0]),)),
)
@pytest.mark.parametrize(
    "feature_names_alice,feature_names_bob,feature_names_charlie,feature_names_dave",
    [
        ((), (), (), ()),
    ],
)
@pytest.mark.parametrize(
    "identifiers_phonetic_alice,identifiers_phonetic_bob,identifiers_phonetic_charlie,identifiers_phonetic_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifiers_phonetic_exact_alice,identifiers_phonetic_exact_bob,identifiers_phonetic_exact_charlie,identifiers_phonetic_exact_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_date_alice,identifier_date_bob,identifier_date_charlie,identifier_date_dave",
    NONE,
)
@pytest.mark.parametrize(
    "identifier_zip6_alice,identifier_zip6_bob,identifier_zip6_charlie,identifier_zip6_dave",
    NONE,
)
async def test_hash_functions(
    parties: Tuple[Tuple[DatabaseOwner, ...], Helper]
) -> None:
    """
    Test to see if setting the hash function behaves as expected.

    :param parties: all parties involved in this secure inner join iteration
    """
    await asyncio.gather(
        *(
            [party.send_randomness_to_all() for party in parties[0]]
            + [party.receive_all_randomness() for party in parties[0]]
        )
    )

    party = parties[0][0]
    out_default_hash = party._hash_entry("heyhoi")
    again_default_hash = party._hash_entry("heyhoi")

    party.hash_fun = cast(Callable[[bytes], bytes], lambda x: blake2b(x).digest())

    out_blake2b = party._hash_entry("heyhoi")

    assert out_default_hash == again_default_hash
    assert out_default_hash != out_blake2b
