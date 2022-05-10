"""
Module contains Helper class (Henri) for performing secure set intersection
"""

import asyncio
import itertools
import math
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from .lsh import weighted_hamming_distance
from .player import Player

# for better readability
PlayerStr = str


class Helper(Player):
    """
    Class for a helper party
    """

    def __init__(
        self,
        *args: Any,
        lsh_threshold_function: Optional[
            Callable[
                [
                    List[Tuple[Tuple[int, int], Tuple[int, int]]],
                    Dict[
                        Tuple[Tuple[int, int], Tuple[int, int]],
                        Tuple[float, Tuple[float, float, float, float]],
                    ],
                ],
                bool,
            ]
        ] = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Initializes a helper instance

        :param \*args: passed on to base class
        :param lsh_thresholds_function: threshold function for the LSH distance computation,
            if None defaults to Helper.default\_threshold\_function
        :param \**kwargs: passed on to base class
        :raise ValueError: raised when (at least) one of data parties is not in the pool.
        """
        super().__init__(*args, **kwargs)

        if not all(
            data_owner in self._pool.pool_handlers.keys()
            for data_owner in self.data_parties
        ):
            raise ValueError("A data owner is missing in the communication pool.")

        self._threshold_function = (
            lsh_threshold_function or self.default_threshold_function
        )

        # To be filled
        self._databases: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._filtered_databases: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._identifiers: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._ph_identifiers: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._lsh_identifiers: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._intersection: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._approx_intersection: Dict[PlayerStr, npt.NDArray[np.object_]] = {}
        self._feature_columns: Optional[Dict[PlayerStr, List[int]]] = None
        self._lookup_table: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]],
            Tuple[float, Tuple[float, float, float, float]],
        ] = {}
        self.shares: Dict[PlayerStr, npt.NDArray[np.object_]] = {}

    @property
    def intersection_size(self) -> int:
        """
        The size of the intersection between the identifier columns of all data parties.

        :return: The intersection size.
        :raise ValueError: In case the intersection size cannot be determined (yet).
        """
        if len(self._intersection) != len(self.data_parties) or len(
            self._approx_intersection
        ) != len(self.data_parties):
            raise ValueError("Intersection size can not been determined (yet).")
        return cast(
            int,
            self._intersection[self.data_parties[0]].shape[0]
            + self._approx_intersection[self.data_parties[0]].shape[0],
        )

    async def combine_and_send_to_all(self) -> None:
        """
        Computes the intersection size and sends the result to all data parties.

        :raise ValueError: In case not all encrypted databases have been received (yet).
        """
        if len(self._databases) != len(self.data_parties):
            raise ValueError("Not all encrypted databases have been received (yet).")
        self._compute_intersection()
        if all(len(_) > 0 for _ in self._ph_identifiers.values()) and len(
            self._ph_identifiers
        ) == len(self.data_parties):
            self._compute_approx_intersection()
        await asyncio.gather(
            *[
                self.send_message(party, self.intersection_size, "intersection_size")
                for party in self.data_parties
            ]
        )
        self._logger.info(
            f"Computed intersection size {self.intersection_size} and sent to all parties"
        )

    async def obtain_and_process_all_shares(self) -> None:
        """
        Receive the random shares from all data parties in an asynchronous manner and process them to determine the
        remainder of the shares from the data and the received shares.

        :raise ValueError: In case the intersection has nog been computed yet.
        """
        if len(self._intersection) != len(self.data_parties):
            raise ValueError("Did not compute intersection yet.")
        await asyncio.gather(
            *[self._obtain_and_process_shares(party) for party in self.data_parties]
        )
        self._logger.info(
            "Obtained shares from all parties. Also subtracted shares from overlap."
        )

    async def receive_identifiers(self, party: str) -> None:
        """
        Receive hashed identifiers from party

        :param party: name of the party to receive data from
        """
        self._identifiers[party] = await self.receive_message(
            party, msg_id="hashed_identifiers"
        )
        self._logger.info(f"Stored hashed identifiers from {party}")

    async def receive_lsh_identifiers(self, party: str) -> None:
        """
        Receive encoded Locality-Sensitive Hashing (LSH) identifiers from party

        :param party: name of the party to receive data from
        """
        self._lsh_identifiers[party] = await self.receive_message(
            party, msg_id="hashed_lsh_identifiers"
        )
        self._logger.info(f"Stored LSH identifiers from {party}")

    async def receive_ph_identifiers(self, party: str) -> None:
        """
        Receive hashed phonetic identifiers from party

        :param party: name of the party to receive data from
        """
        self._ph_identifiers[party] = await self.receive_message(
            party, msg_id="hashed_ph_identifiers"
        )
        self._logger.info(f"Stored hashed phonetic identifiers from {party}")

    async def run_protocol(self) -> None:
        """
        Run the entire protocol, start to end, in an asynchronous manner
        """
        self._logger.info("Ready to roll, starting now!")
        await self.send_data_parties()
        await self.store_data()
        await self.combine_and_send_to_all()
        if self.intersection_size > 0:
            await self.obtain_and_process_all_shares()
            await self.send_shares_to_all()
        self._logger.info("All done")

    async def send_data_parties(self) -> None:
        """
        Send the data parties with accompanying addresses and ports (that have been sorted alphabetically) to all
        players, so they can check that there data parties tuple is exactly the same.
        """
        await asyncio.gather(
            *[
                self.send_message(
                    party, self.data_parties_and_addresses, "data_parties"
                )
                for party in self.data_parties
            ]
        )

    async def send_shares_to_all(self) -> None:
        """
        Send the final encrypted shares to all parties.
        """
        await asyncio.gather(
            *[
                self.send_message(party, self.shares[party], "real_share")
                for party in self.data_parties
            ]
        )

        self._logger.info("Sent all encrypted real shares.")

    async def store_data(self) -> None:
        """
        Receive and store the data from all data parties
        """
        tasks = []
        for party in self.data_parties:
            tasks.append(self.receive_identifiers(party))
            tasks.append(self.receive_ph_identifiers(party))
            tasks.append(self.receive_lsh_identifiers(party))
            tasks.append(self._receive_data(party))
        await asyncio.gather(*tasks)
        self._logger.info(
            "Stored hashed identifiers and encrypted data from all parties"
        )

    @staticmethod
    def _intersection_ids(
        x: Union[
            npt.NDArray[np.object_],
            Tuple[npt.NDArray[np.bool_], npt.NDArray[np.object_]],
        ],
        y: npt.NDArray[np.object_],
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.object_]]:
        """
        This method determines the intersection and intersection indices of either two 1D-numpy arrays. Or of the
        previous result of this method and a new 1D-numpy array. The result of this method is a tuple containing the
        intersection values (that can be used for a subsequent intersection) and as the second element a numpy array
        containing on every row the indices of the intersection of the previous and current arrays in that order.

        :param x: 1D-numpy array for use in the intersection or previous result of this method (tuple of 1D-numpy
            array and 2D-numpy array with intersection indices). The indices get update to only contain the ones
            that follow from this new intersection.
        :param y: 1D-numpy array for use in the intersection. Intersection indices will be appended to the second
            tuple entry of the result.
        :return: (tuple of 1D-numpy array with intersection values and 2D-numpy array with intersection indices)
        """
        if isinstance(x, tuple):
            intersection, x_indices, y_indices = np.intersect1d(
                x[0], y, assume_unique=True, return_indices=True
            )
            return intersection, np.vstack((x[1][:, x_indices], y_indices))
        # else
        intersection, x_indices, y_indices = np.intersect1d(
            x, y, assume_unique=True, return_indices=True
        )
        return intersection, np.vstack((x_indices, y_indices))

    def _compute_intersection(self) -> None:
        """
        Compute the intersection between the hashed identifier columns of all data parties and get all data entries with
        these identifiers.
        """
        identifiers = {party: self._identifiers[party] for party in self._data_parties}
        # Determine intersecting indices
        _, intersection_indices = reduce(
            self._intersection_ids,  # type: ignore[arg-type]
            identifiers.values(),
        )

        # determine the complete intersection for every party.
        for party_index, party in enumerate(self._data_parties):
            intersection = intersection_indices[party_index, :]
            self._intersection[party] = self._databases[party][intersection, :]

            # Empty data and identifiers for already intersecting data
            self._filtered_databases[party] = np.delete(
                self._databases[party], intersection, axis=0
            )
            self._approx_intersection[party] = np.empty(
                [0, self._databases[party].shape[1]], dtype=np.object_
            )
            if (
                isinstance(self._ph_identifiers[party], np.ndarray)
                and len(self._ph_identifiers[party]) > 0
            ):
                self._ph_identifiers[party] = np.delete(
                    self._ph_identifiers[party], intersection
                )
            if (
                isinstance(self._lsh_identifiers[party], np.ndarray)
                and len(self._lsh_identifiers[party]) > 0
            ):
                self._lsh_identifiers[party] = np.delete(
                    self._lsh_identifiers[party], intersection
                )

    def _compute_approx_intersection(self) -> None:
        """
        Compute the approximate intersection between the hashed phonetic identifier columns of the data parties
        Best matches are selected in a greedy manner, based on LSH distance.
        """
        ph_identifiers = {
            party: self._ph_identifiers[party] for party in self._data_parties
        }
        unique_identifiers: Dict[str, npt.NDArray[np.object_]] = {}
        unique_indices: Dict[str, List[npt.NDArray[np.int_]]] = {}
        for party, identifiers in ph_identifiers.items():
            unique_identifiers[party], unique_indices[party] = self._unique_indices(
                identifiers
            )

        # Determine intersecting indices
        _, intersection_indices = reduce(
            self._intersection_ids,  # type: ignore[arg-type]
            unique_identifiers.values(),
        )

        # Determine index combinations of phonetic matches for all parties, e.g. [[[1, 2], [2]], [[0, 3], [1,6]]]
        # record 1 and 2 of data set A have the same phonetic encoding as record 2 from B, etc.
        intersection_indices_complete = []
        for intersection_tuple in zip(*intersection_indices):
            intersection_indices_complete.append(
                [
                    indices[intersection_tuple[party_index]]
                    for party_index, indices in enumerate(unique_indices.values())
                ]
            )

        # Constructs a set of possible phonetic overlaps to check LSH distance for
        combinations: Set[Tuple[int, ...]] = set()
        match_count: npt.NDArray[np.int_] = np.zeros(len(self.data_parties), dtype=int)
        for indices in intersection_indices_complete:
            combinations.update(itertools.product(*indices))
            match_count += [len(index) for index in indices]

        # Annotate all pairs with index origin, e.g. (2, 3) -> ((0, 2), (1, 3))
        combinations_annotated = cast(
            Set[Tuple[Tuple[int, int], ...]],
            set(tuple(enumerate(_)) for _ in combinations),
        )

        # Generate all possibile unique pairs of indices
        unique_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        for combination in combinations_annotated:
            unique_pairs.update(itertools.combinations(combination, 2))

        # Compute distance for all pairs
        for pair in unique_pairs:
            self._lookup_table[pair] = self.lsh_distance(pair[0], pair[1])

        # Sum distances (in case of more than two datasets)
        distances: List[Tuple[Tuple[Tuple[int, int], ...], float]] = [
            (_, self.summed_lsh_distance(_)) for _ in combinations_annotated
        ]

        distances.sort(key=lambda _: _[1])

        matches: List[Tuple[Tuple[int, int], ...]] = []
        max_matches = min(match_count)
        selected: List[Set[int]] = [set() for _ in range(len(self.data_parties))]

        # Pick best matches in a greedy way, closest first
        for distance in distances:
            if math.isinf(distance[1]) or len(matches) >= max_matches:
                break
            if all(pair[1] not in selected[pair[0]] for pair in distance[0]):
                matches.append(distance[0])
                for pair in distance[0]:  # type: ignore[assignment]
                    selected[cast(int, pair[0])].add(cast(int, pair[1]))

        # Add matches to approximate intersection
        if matches:
            for party_index, party in enumerate(self._data_parties):
                match_indices = [_[party_index][1] for _ in matches]
                self._approx_intersection[party] = self._filtered_databases[party][
                    match_indices, :
                ]

    @staticmethod
    def _unique_indices(
        array: npt.NDArray[np.object_],
    ) -> Tuple[npt.NDArray[np.object_], List[npt.NDArray[np.int_]]]:
        """
        Determine unique indices together with their locations (indices)

        :param array: array of which we wish to find the unique indices
        :return: unique indices and the locations of those indices in the original array
        """
        # sort indices by unique element
        idx_sort = np.argsort(array, kind="mergesort")
        sorted_array = array[idx_sort]
        # find first index
        unique_array: npt.NDArray[np.object_]
        unique_array, idx_start = np.unique(sorted_array, return_index=True)

        return unique_array, np.split(idx_sort, idx_start[1:])

    @staticmethod
    def default_threshold_function(
        pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        lookup_table: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]],
            Tuple[float, Tuple[float, float, float, float]],
        ],
    ) -> bool:
        """
        Default threshold function implementation for validating whether two LSH hashes are near enough

        Default is to allow the overall difference score to be <= 4.5 and the element-wise difference score to be <= 1.5
        for all elements (day, month, year, zip2)

        :param pairs: pairs to compare
        :param lookup_table: lookup_table of scores for all pairs
        :return: True if threshold function is satisfied, else False
        """
        return all(lookup_table[pair][0] <= 4.5 for pair in pairs) and all(
            all(score <= 1.5 for score in lookup_table[pair][1]) for pair in pairs
        )

    def lsh_distance(
        self, index_1: Tuple[int, int], index_2: Tuple[int, int]
    ) -> Tuple[float, Tuple[float, float, float, float]]:
        """
        Computes LSH distance between two indices. Every index is of the form (x, y) where
        x represents the party_index and y represents the index of the lsh hash.

        :param index_1: of the form (party_index, lsh_hash_index)
        :param index_2: of the form (party_index, lsh_hash_index)
        :return: overall and individual distance
        """
        value_1 = self._lsh_identifiers[self.data_parties[index_1[0]]][index_1[1]]
        value_2 = self._lsh_identifiers[self.data_parties[index_2[0]]][index_2[1]]
        distance, individual_distances = weighted_hamming_distance(
            value_1,
            value_2,
        )
        return distance, individual_distances

    def summed_lsh_distance(self, combination: Tuple[Tuple[int, int], ...]) -> float:
        """
        Computes the summed LSH distance of a set of pairs, returns "inf" if (at least)
        one of the distances exceeds a set threshold. As the number of pairs per combination
        is constant, it suffices to sum, no need to compute the mean.

        :param combination: a combination of an index pair per party.
        :return: sum of distances of all overall pair combinations or "inf".
        """
        pairs = list(itertools.combinations(combination, 2))
        if self._threshold_function(pairs, self._lookup_table):
            return sum(self._lookup_table[pair][0] for pair in pairs)
        return float("inf")

    async def _obtain_and_process_shares(self, party: str) -> None:
        """
        Receive the random shares from the given data party and process them. In processing the received shares get
        subtracted from the encrypted feature values of the corresponding data party.

        :param party: Identifier of the party that sent the shares.
        """
        random_shares = await self.receive_message(party, "random_share")

        for other_party in self.data_parties:
            if other_party == party:
                continue
            if other_party not in self.shares:
                self.shares[other_party] = np.vstack(
                    [
                        self._intersection[other_party],
                        self._approx_intersection[other_party],
                    ]
                )
            self.shares[other_party] = np.subtract(
                self.shares[other_party], random_shares[other_party]
            )

        self._logger.info(f"Obtained share from {party}, subtracted from overlap.")

    async def _receive_data(self, party: str) -> None:
        """
        Receive encrypted attributes and hashed identifiers from the party with the given identifier.

        :param party: Identifier of the party to receive data from.
        """
        self._databases[party] = await self.receive_message(
            party, msg_id="encrypted_data"
        )
        self._logger.info(f"Stored encrypted data from {party}")
