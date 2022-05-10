"""
Fixtures that can be used for defining test cases.
"""
from functools import partial, reduce
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from _pytest.fixtures import FixtureRequest

from tno.mpc.communication import Pool
from tno.mpc.communication.test import event_loop as event_loop
from tno.mpc.communication.test import fixture_pool_http_3p as fixture_pool_http_3p
from tno.mpc.communication.test import fixture_pool_http_4p as fixture_pool_http_4p
from tno.mpc.communication.test import fixture_pool_http_5p as fixture_pool_http_5p

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper


def compute_regular_intersection(
    datasets: Tuple[npt.NDArray[np.object_], ...]
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
    all_features: List[npt.NDArray[np.object_]] = [
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
    pool_http: Tuple[Pool, ...],
    alice: DatabaseOwner,
    bob: DatabaseOwner,
    charlie: DatabaseOwner,
    dave: DatabaseOwner,
    henri: Helper,
) -> Tuple[Tuple[DatabaseOwner, ...], Helper]:
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
async def fixture_pool_http(
    request: FixtureRequest,
    pool_http_3p: AsyncGenerator[Tuple[Pool, ...], None],
    pool_http_4p: AsyncGenerator[Tuple[Pool, ...], None],
    pool_http_5p: AsyncGenerator[Tuple[Pool, ...], None],
) -> AsyncGenerator[Tuple[Pool, ...], None]:
    """
    Creates a collection of 3, 4 and 5 communication pools

    :param pool_http_3p: Pool of 3 HTTP clients.
    :param pool_http_4p: Pool of 4 HTTP clients.
    :param pool_http_5p: Pool of 5 HTTP clients.
    :param request: A fixture request used to indirectly parametrize.
    :raise NotImplementedError: raised when based on the given param, no fixture can be created
    :return: a collection of communication pools
    """
    if request.param == 3:  # type: ignore[attr-defined]
        return pool_http_3p
    if request.param == 4:  # type: ignore[attr-defined]
        return pool_http_4p
    if request.param == 5:  # type: ignore[attr-defined]
        return pool_http_5p
    raise NotImplementedError("This has not been implemented")


@pytest.fixture(name="alice")
def fixture_alice(
    pool_http: Tuple[Pool, ...],
    feature_names_alice: Tuple[str, ...],
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
    :param feature_names_alice: feature names of Alice's data
    :param identifiers_alice: identifiers of alice
    :param data_alice: data of alice
    :param identifiers_phonetic_alice: phonetic identifiers alice
    :param identifiers_phonetic_exact_alice: exact phonetic identifiers alice
    :param identifier_date_alice: date identifiers alice
    :param identifier_zip6_alice: zip6 identifiers alice
    :return: an initialized database owner
    """
    return DatabaseOwner(
        identifier=list(pool_http[0].pool_handlers)[0],
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        identifiers=identifiers_alice,
        data=data_alice,
        identifiers_phonetic=identifiers_phonetic_alice,
        identifiers_phonetic_exact=identifiers_phonetic_exact_alice,
        identifier_date=identifier_date_alice,
        identifier_zip6=identifier_zip6_alice,
        feature_names=feature_names_alice,
        pool=pool_http[1],
    )


@pytest.fixture(name="bob")
def fixture_bob(
    pool_http: Tuple[Pool, ...],
    feature_names_bob: Tuple[str, ...],
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
    :param feature_names_bob: feature names of Bob's data
    :param identifiers_bob: identifiers of bob
    :param data_bob: data of bob
    :param identifiers_phonetic_bob: phonetic identifiers bob
    :param identifiers_phonetic_exact_bob: exact phonetic identifiers bob
    :param identifier_date_bob: date identifiers bob
    :param identifier_zip6_bob: zip6 identifiers bob
    :param :return: an initialized database owner
    """
    return DatabaseOwner(
        identifier=list(pool_http[0].pool_handlers)[1],
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        identifiers=identifiers_bob,
        data=data_bob,
        identifiers_phonetic=identifiers_phonetic_bob,
        identifiers_phonetic_exact=identifiers_phonetic_exact_bob,
        identifier_date=identifier_date_bob,
        identifier_zip6=identifier_zip6_bob,
        feature_names=feature_names_bob,
        pool=pool_http[2],
    )


@pytest.fixture(name="charlie")
def fixture_charlie(
    pool_http: Tuple[Pool, ...],
    feature_names_charlie: Tuple[str, ...],
    identifiers_charlie: npt.NDArray[np.object_],
    data_charlie: npt.NDArray[np.object_],
    identifiers_phonetic_charlie: npt.NDArray[np.object_],
    identifiers_phonetic_exact_charlie: npt.NDArray[np.object_],
    identifier_date_charlie: npt.NDArray[np.object_],
    identifier_zip6_charlie: npt.NDArray[np.object_],
) -> Optional[DatabaseOwner]:
    """
    Constructs player Charlie

    :param pool_http: collection of (at least four) communication pools
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
    return DatabaseOwner(
        identifier=list(pool_http[0].pool_handlers)[2],
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        identifiers=identifiers_charlie,
        data=data_charlie,
        identifiers_phonetic=identifiers_phonetic_charlie,
        identifiers_phonetic_exact=identifiers_phonetic_exact_charlie,
        identifier_date=identifier_date_charlie,
        identifier_zip6=identifier_zip6_charlie,
        feature_names=feature_names_charlie,
        pool=pool_http[3],
    )


@pytest.fixture(name="dave")
def fixture_dave(
    pool_http: Tuple[Pool, ...],
    feature_names_dave: Tuple[str, ...],
    identifiers_dave: npt.NDArray[np.object_],
    data_dave: npt.NDArray[np.object_],
    identifiers_phonetic_dave: npt.NDArray[np.object_],
    identifiers_phonetic_exact_dave: npt.NDArray[np.object_],
    identifier_date_dave: npt.NDArray[np.object_],
    identifier_zip6_dave: npt.NDArray[np.object_],
) -> Optional[DatabaseOwner]:
    """
    Constructs player Dave

    :param pool_http: collection of (at least five) communication pools
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
    return DatabaseOwner(
        identifier=list(pool_http[0].pool_handlers)[3],
        data_parties=list(pool_http[0].pool_handlers),
        helper=list(pool_http[1].pool_handlers)[0],
        identifiers=identifiers_dave,
        data=data_dave,
        identifiers_phonetic=identifiers_phonetic_dave,
        identifiers_phonetic_exact=identifiers_phonetic_exact_dave,
        identifier_date=identifier_date_dave,
        identifier_zip6=identifier_zip6_dave,
        feature_names=feature_names_dave,
        pool=pool_http[4],
    )


def threshold_function(
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lookup_table: Dict[
        Tuple[Tuple[int, int], Tuple[int, int]],
        Tuple[float, Tuple[float, float, float, float]],
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
def fixture_henri(pool_http: Tuple[Pool, ...]) -> Helper:
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
