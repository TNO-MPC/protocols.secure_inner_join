"""
Fixtures that can be used for defining test cases.
"""

# pylint: disable=redefined-outer-name
from __future__ import annotations

from functools import partial, reduce
from typing import Callable, Iterator

import numpy as np
import numpy.typing as npt
import pytest
from _pytest.fixtures import FixtureRequest

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import Paillier

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper


def compute_regular_intersection(
    datasets: tuple[npt.NDArray[np.object_], ...]
) -> npt.NDArray[np.object_]:
    """
    Computes an intersection the regular way (i.e. without fancy MPC) of a tuple of numpy arrays. Each numpy array
    should have identifiers (from the same universal set) in the first column. Features go in the subsequent columns.

    :param datasets: Tuple of numpy arrays that is to be intersected.
    :return: Intersected dataset, with the identifiers in the first column and features (sorted as they were in the
        input argument in the subsequent columns).
    """
    common_identifiers = reduce(
        lambda arr1, arr2: np.intersect1d(arr1, arr2, assume_unique=True),
        (dataset[:, 0] for dataset in datasets),
    )
    all_features: list[npt.NDArray[np.object_]] = [
        np.atleast_2d(np.array(common_identifiers)).transpose()
    ]

    compute_intersection = lambda common_identifier, dataset: np.where(
        dataset[:, 0] == common_identifier
    )[0][0]

    for dataset in datasets:
        indices = list(
            map(
                partial(
                    compute_intersection,
                    dataset=dataset,
                ),
                common_identifiers,
            )
        )
        all_features.append(dataset[indices, 1:])
    return np.hstack(all_features)


@pytest.fixture(name="parties")
def fixture_parties(
    pool_http: tuple[Pool, ...],
    alice: DatabaseOwner,
    bob: DatabaseOwner,
    charlie: DatabaseOwner,
    dave: DatabaseOwner,
    henri: Helper,
) -> tuple[tuple[DatabaseOwner, ...], Helper]:
    """
    Get all parties given a HTTP of size (3,4,5). The parties consist of the one helper pool and the remainder is data
    parties.

    :param pool_http: HTTP pools to be used by the parties.
    :param alice: Data owner Alice.
    :param bob: Data owner Bob.
    :param charlie: Data owner Charlie.
    :param dave: Data owner Dave.
    :param henri: Helper party Henri.
    :return: A tuple consisting of two elements. The first element is a tuple of (2,3,4) data owners. The second
        element is the helper party.
    """
    if len(pool_http) == 3:
        return (alice, bob), henri
    if len(pool_http) == 4:
        return (alice, bob, charlie), henri
    # if len(pool_http) == 5:
    return (alice, bob, charlie, dave), henri


@pytest.fixture(
    name="pool_http",
    params=[3, 4, 5],
    ids=["2-party", "3-party", "4-party"],
    scope="module",
)
def fixture_pool_http(
    request: FixtureRequest,
    http_pool_group_factory: Callable[[int], tuple[Pool, ...]],
) -> tuple[Pool, ...]:
    """
    Creates a collection of 3, 4 and 5 communication pools

    :param http_pool_group_factory: Factory for creating a HTTP pool group.
    :param request: A fixture request used to indirectly parametrize.
    :return: a collection of communication pools
    """
    return http_pool_group_factory(request.param)


@pytest.fixture
def paillier_scheme() -> Iterator[Paillier]:
    """
    Yield a paillier scheme and properly shut it down afterwards.

    :return: Paillier scheme.
    """
    scheme = Paillier.from_security_parameter(key_length=2048, precision=8)
    yield scheme
    scheme.shut_down()


@pytest.fixture(name="alice")
def fixture_alice(
    pool_http: tuple[Pool, ...],
    paillier_scheme: Paillier,
    feature_names_alice: tuple[str, ...],
    identifiers_alice: npt.NDArray[np.object_],
    data_alice: npt.NDArray[np.object_],
    identifiers_phonetic_alice: npt.NDArray[np.object_],
    identifiers_phonetic_exact_alice: npt.NDArray[np.object_],
    identifier_date_alice: npt.NDArray[np.object_],
    identifier_zip6_alice: npt.NDArray[np.object_],
) -> DatabaseOwner:
    """
    Constructs player Alice

    :param pool_http: collection of (at least three) communication pools
    :param paillier_scheme: paillier scheme
    :param feature_names_alice: feature names of Alice's data
    :param identifiers_alice: identifiers of alice
    :param data_alice: data of alice
    :param identifiers_phonetic_alice: phonetic identifiers alice
    :param identifiers_phonetic_exact_alice: exact phonetic identifiers alice
    :param identifier_date_alice: date identifiers alice
    :param identifier_zip6_alice: zip6 identifiers alice
    :return: an initialized database owner
    """
    return create_database_owner(
        pool_http=pool_http,
        paillier=paillier_scheme,
        identifiers=identifiers_alice,
        data=data_alice,
        identifiers_phonetic=identifiers_phonetic_alice,
        identifiers_phonetic_exact=identifiers_phonetic_exact_alice,
        identifier_date=identifier_date_alice,
        identifier_zip6=identifier_zip6_alice,
        feature_names=feature_names_alice,
        database_nr=0,
    )


@pytest.fixture(name="bob")
def fixture_bob(
    pool_http: tuple[Pool, ...],
    paillier_scheme: Paillier,
    feature_names_bob: tuple[str, ...],
    identifiers_bob: npt.NDArray[np.object_],
    data_bob: npt.NDArray[np.object_],
    identifiers_phonetic_bob: npt.NDArray[np.object_],
    identifiers_phonetic_exact_bob: npt.NDArray[np.object_],
    identifier_date_bob: npt.NDArray[np.object_],
    identifier_zip6_bob: npt.NDArray[np.object_],
) -> DatabaseOwner:
    """
    Constructs player Bob

    :param pool_http: collection of (at least three) communication pools
    :param paillier_scheme: paillier scheme
    :param feature_names_bob: feature names of Bob's data
    :param identifiers_bob: identifiers of bob
    :param data_bob: data of bob
    :param identifiers_phonetic_bob: phonetic identifiers bob
    :param identifiers_phonetic_exact_bob: exact phonetic identifiers bob
    :param identifier_date_bob: date identifiers bob
    :param identifier_zip6_bob: zip6 identifiers bob
    :param :return: an initialized database owner
    """
    return create_database_owner(
        pool_http=pool_http,
        paillier=paillier_scheme,
        identifiers=identifiers_bob,
        data=data_bob,
        identifiers_phonetic=identifiers_phonetic_bob,
        identifiers_phonetic_exact=identifiers_phonetic_exact_bob,
        identifier_date=identifier_date_bob,
        identifier_zip6=identifier_zip6_bob,
        feature_names=feature_names_bob,
        database_nr=1,
    )


@pytest.fixture(name="charlie")
def fixture_charlie(
    pool_http: tuple[Pool, ...],
    paillier_scheme: Paillier,
    feature_names_charlie: tuple[str, ...],
    identifiers_charlie: npt.NDArray[np.object_],
    data_charlie: npt.NDArray[np.object_],
    identifiers_phonetic_charlie: npt.NDArray[np.object_],
    identifiers_phonetic_exact_charlie: npt.NDArray[np.object_],
    identifier_date_charlie: npt.NDArray[np.object_],
    identifier_zip6_charlie: npt.NDArray[np.object_],
) -> DatabaseOwner | None:
    """
    Constructs player Charlie

    :param pool_http: collection of (at least four) communication pools
    :param paillier_scheme: paillier scheme
    :param feature_names_charlie: feature names of Charlie's data
    :param identifiers_charlie: identifiers of charlie
    :param data_charlie: data of charlie
    :param identifiers_phonetic_charlie: phonetic identifiers charlie
    :param identifiers_phonetic_exact_charlie: exact phonetic identifiers charlie
    :param identifier_date_charlie: date identifiers charlie
    :param identifier_zip6_charlie: zip6 identifiers charlie
    :return: an initialized database owner
    """
    if len(pool_http) < 4:
        return None
    return create_database_owner(
        pool_http=pool_http,
        paillier=paillier_scheme,
        identifiers=identifiers_charlie,
        data=data_charlie,
        identifiers_phonetic=identifiers_phonetic_charlie,
        identifiers_phonetic_exact=identifiers_phonetic_exact_charlie,
        identifier_date=identifier_date_charlie,
        identifier_zip6=identifier_zip6_charlie,
        feature_names=feature_names_charlie,
        database_nr=2,
    )


@pytest.fixture(name="dave")
def fixture_dave(
    pool_http: tuple[Pool, ...],
    paillier_scheme: Paillier,
    feature_names_dave: tuple[str, ...],
    identifiers_dave: npt.NDArray[np.object_],
    data_dave: npt.NDArray[np.object_],
    identifiers_phonetic_dave: npt.NDArray[np.object_],
    identifiers_phonetic_exact_dave: npt.NDArray[np.object_],
    identifier_date_dave: npt.NDArray[np.object_],
    identifier_zip6_dave: npt.NDArray[np.object_],
) -> DatabaseOwner | None:
    """
    Constructs player Dave

    :param pool_http: collection of (at least five) communication pools
    :param paillier_scheme: paillier scheme
    :param feature_names_dave: feature names of Dave's data
    :param identifiers_dave: identifiers of dave
    :param data_dave: data of dave
    :param identifiers_phonetic_dave: phonetic identifiers dave
    :param identifiers_phonetic_exact_dave: exact phonetic identifiers dave
    :param identifier_date_dave: date identifiers dave
    :param identifier_zip6_dave: zip6 identifiers dave
    :return: an initialized database owner
    """
    if len(pool_http) < 5:
        return None
    return create_database_owner(
        pool_http=pool_http,
        paillier=paillier_scheme,
        identifiers=identifiers_dave,
        data=data_dave,
        identifiers_phonetic=identifiers_phonetic_dave,
        identifiers_phonetic_exact=identifiers_phonetic_exact_dave,
        identifier_date=identifier_date_dave,
        identifier_zip6=identifier_zip6_dave,
        feature_names=feature_names_dave,
        database_nr=3,
    )


def create_database_owner(
    pool_http: tuple[Pool, ...],
    paillier: Paillier,
    feature_names: tuple[str, ...],
    identifiers: npt.NDArray[np.object_],
    data: npt.NDArray[np.object_],
    identifiers_phonetic: npt.NDArray[np.object_],
    identifiers_phonetic_exact: npt.NDArray[np.object_],
    identifier_date: npt.NDArray[np.object_],
    identifier_zip6: npt.NDArray[np.object_],
    database_nr: int,
) -> DatabaseOwner:
    """
    Constructs DatabaseOwner

    :param pool_http: collection of communication pools
    :param paillier: paillier scheme
    :param feature_names: feature names of this database owner's data
    :param identifiers: identifiers of this database owner
    :param data: data of this database owner
    :param identifiers_phonetic: phonetic identifiers
    :param identifiers_phonetic_exact: exact phonetic identifiers
    :param identifier_date: date identifiers
    :param identifier_zip6: zip6 identifiers
    :param database_nr: instance count of this DatabaseOwner
    :return: an initialized database owner
    """
    return DatabaseOwner(
        identifier=list(pool_http[0].pool_handlers)[database_nr],
        data_parties=list(pool_http[0].pool_handlers),
        paillier_scheme=paillier,
        helper=list(pool_http[1].pool_handlers)[0],
        identifiers=identifiers,
        data=data,
        identifiers_phonetic=identifiers_phonetic,
        identifiers_phonetic_exact=identifiers_phonetic_exact,
        identifier_date=identifier_date,
        identifier_zip6=identifier_zip6,
        feature_names=feature_names,
        pool=pool_http[database_nr + 1],
    )


def threshold_function(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    lookup_table: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[float, tuple[float, float, float, float]],
    ],
) -> bool:
    """
    Example of a threshold function implementation

    :param pairs: pairs to compare
    :param lookup_table: lookup_table of scores for all pairs
    :return: True if threshold function is satisfied, else False
    """
    return all(lookup_table[pair][0] <= 4.5 for pair in pairs)


@pytest.fixture(name="henri")
def fixture_henri(pool_http: tuple[Pool, ...]) -> Helper:
    """
    Constructs player henri

    :param pool_http: collection of (at least three) communication pools
    :return: an initialized helper party
    """
    return Helper(
        identifier=list(pool_http[1].pool_handlers)[0],
        lsh_threshold_function=threshold_function,
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        pool=pool_http[0],
    )
