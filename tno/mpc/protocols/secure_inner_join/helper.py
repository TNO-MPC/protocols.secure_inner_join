"""
Module contains Helper class (Henri) for performing secure set intersection
"""
import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from .player import Player


class Helper(Player):
    """
    Class for a helper party
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes a helper instance

        :raise ValueError: raised when (at least) one of data parties is not in the pool.
        """
        super().__init__(*args, **kwargs)

        if not all(
            data_owner in self.pool.pool_handlers.keys()
            for data_owner in self.data_parties
        ):
            raise ValueError("A data owner is missing in the communication pool.")

        # To be filled
        self._databases: Dict[str, np.ndarray] = {}
        self._intersection: Optional[np.ndarray] = None
        self._feature_columns: Optional[Dict[str, List[int]]] = None

    @property
    def intersection_size(self) -> int:
        """
        The size of the intersection between the identifier columns of the data parties

        :return: the intersection size
        :raise AttributeError: raised when the intersection size cannot be determined yet
        """
        if isinstance(self._intersection, np.ndarray):
            return int(self._intersection.shape[0])
        raise AttributeError("Intersection size cannot be determined (yet)")

    async def combine_and_send(self) -> None:
        """
        Computes the intersection size and sends the result to the data parties
        """
        self._compute_intersection()
        await asyncio.gather(
            *[
                self.send_message(party, self.intersection_size, "intersection_size")
                for party in self.data_parties
            ]
        )
        self._logger.info("Computed intersection size and sent to all parties")

    async def obtain_shares(self) -> None:
        """
        Receive the random shares from the data parties and processes them
        """
        await asyncio.gather(
            *[
                self._process_share(self.data_parties[0], self.data_parties[1]),
                self._process_share(self.data_parties[1], self.data_parties[0]),
            ]
        )

    async def receive_data(self, party: str) -> None:
        """
        Receive encrypted attributes and hashed identifiers from party

        :param party: name of the party to receive data from
        """
        self._databases[party] = await self.receive_message(
            party, msg_id="encrypted_data"
        )
        self._logger.info(f"Stored encrypted data from {party}")

    async def run_protocol(self) -> None:
        """
        Run the entire protocol, start to end, in an asynchronous manner
        """
        self._logger.info("Ready to roll, starting now!")
        await self.store_data()
        await self.combine_and_send()
        await self.obtain_shares()
        self._logger.info("All done")

    async def store_data(self) -> None:
        """
        Receive and store the data from all data parties
        """
        tasks = []
        for party in self.data_parties:
            tasks.append(self.receive_data(party))
        await asyncio.gather(*tasks)
        self._logger.info("Stored encrypted data from all parties")

    def _compute_intersection(self) -> None:
        """
        Compute the intersection between the hashed identifier columns of the data parties
        """
        # Get the dataframes with encrypted data
        data0 = np.asarray(self._databases[self.party_alice])[:, 1:]
        data1 = np.asarray(self._databases[self.party_bob])[:, 1:]

        identifiers_alice = self._databases[self.party_alice][:, 0]
        identifiers_bob = self._databases[self.party_bob][:, 0]

        # Determine intersecting indices
        _, alice_indices, bob_indices = np.intersect1d(
            identifiers_alice,
            identifiers_bob,
            assume_unique=True,
            return_indices=True,
        )

        # Mask respective arrays and hstack (concatenate) for final o/p
        self._feature_columns = {
            self.party_alice: list(range(0, data0.shape[1])),
            self.party_bob: list(
                range(data0.shape[1], data0.shape[1] + data1.shape[1])
            ),
        }
        self._intersection = np.hstack([data0[alice_indices, :], data1[bob_indices, :]])

    async def _process_share(self, party_1: str, party_2: str) -> None:
        """
        Receive a random share from party_1, add these to the encrypted
        attributes of party_2, and send the results to party_2

        :param party_1: party to receive from
        :param party_2: party to send results to
        :raise ValueError: raised when there are no feature columns determined,
            or when the is no known intersection yet.
        """
        random_share = await self.receive_message(party_1, "random_share")
        if self._feature_columns is None:
            raise ValueError("Did not determine feature columns yet.")
        features = self._feature_columns[party_2]
        if self._intersection is None:
            raise ValueError("Did not compute intersection yet.")
        real_share = np.subtract(self._intersection[:, features], random_share)
        await self.send_message(party_2, real_share, "real_share")
        self._logger.info(
            f"Obtained share from {party_2}, subtracted from overlap,"
            f"and sent resulting share to {party_1}"
        )
