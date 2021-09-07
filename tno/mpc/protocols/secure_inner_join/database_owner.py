"""
Module contains DatabaseOwner class (either Alice or Bob) for performing secure set intersection
"""
import asyncio
import secrets  # "True" (secure) randomness
from dataclasses import dataclass
from hashlib import sha256 as hash_fun  # replace sha256 with your favorite hash
from typing import Any, AnyStr, Optional, SupportsInt, Tuple

import numpy as np

from tno.mpc.encryption_schemes.paillier import Paillier

from .player import Player


class DatabaseOwner(Player):
    """
    Class for a database owner
    """

    @dataclass
    class Collection:
        """
        Nested data class to store received data
        """

        feature_names: Optional[Tuple[str, ...]] = None
        intersection_size: Optional[int] = None
        paillier_scheme: Optional[Paillier] = None
        randomness: Optional[int] = None
        share: Optional[np.ndarray] = None

    def __init__(
        self,
        *args: Any,
        data: np.ndarray,
        feature_names: Tuple[str, ...] = (),
        paillier_scheme: Paillier = Paillier.from_security_parameter(
            key_length=2048, precision=8
        ),
        ran_length: int = 64,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a database owner instance

        :param data: data of the database owner,
                            first column should contain the identifiers
        :param feature_names: optional names of the shared features
        :param paillier_scheme: a tno.mpc.encryption_schemes.paillier.Paillier instance
        :param ran_length: number of bits for shared randomness salt
        :raise ValueError: raised when helper or data parties are not in the pool.
        """
        super().__init__(*args, **kwargs)

        if self._other_party not in self.pool.pool_handlers.keys():
            raise ValueError(
                f"Data owner {self._other_party} is not in the communication pool."
            )
        if self.helper not in self.pool.pool_handlers.keys():
            raise ValueError(
                f"Helper {self._other_party} is not in the communication pool."
            )

        self.paillier_scheme = paillier_scheme
        self.__randomness = secrets.randbits(ran_length)
        self.__data = data
        self.__feature_names = feature_names or tuple(
            "" for _ in range(self.__data.shape[1] - 1)
        )

        # To be filled
        self._scrambled_data: Optional[np.ndarray] = None
        self._received: DatabaseOwner.Collection = DatabaseOwner.Collection()
        self._share: Optional[np.ndarray] = None

    @property
    def feature_names(self) -> Optional[Tuple[str, ...]]:
        """
        The feature names of the inner join in the same order for both database owners

        :return: feature names
        """
        if self.identifier == self.party_alice:
            return self._own_feature_names + self._received_feature_names
        return self._received_feature_names + self._own_feature_names

    @property
    def intersection_size(self) -> int:
        """
        The intersection size determined by the helper

        :return: size of intersection
        :raise ValueError: raised when there is no intersection size available yet
        """
        if self._received.intersection_size is None:
            raise ValueError("Did not receive intersection size yet.")
        return self._received.intersection_size

    @property
    def received_paillier_scheme(self) -> Paillier:
        """
        The received Paillier scheme

        :return: the received Paillier scheme
        :raise ValueError: raised when there is no paillier scheme received yet
        """
        if self._received.paillier_scheme is None:
            raise ValueError("Did not receive paillier scheme yet.")
        return self._received.paillier_scheme

    @property
    def shared_randomness(self) -> int:
        """
        The shared randomness (sum of own randomness and that of the other party)

        :return: shared randomness
        """
        return self._own_randomness + self._received_randomness

    @property
    def shares(self) -> np.ndarray:
        """
        The shares of the secure inner join

        :return: the secure inner join shares
        :raise ValueError: raised when not all shares are available
        """
        if self._share is None or self._received.share is None:
            raise ValueError("Not all shares are available (yet).")
        if self.identifier == self.party_alice:
            return np.hstack((self._received.share, self._share))
        return np.hstack((self._share, self._received.share))

    @property
    def _other_party(self) -> str:
        """
        :return: the data party that is not you
        :raise ValueError: raised when we are not Alice nor Bob.
        """
        if self.identifier not in (self.party_alice, self.party_bob):
            raise ValueError(
                f"We should be either {self.party_alice} or {self.party_bob}"
            )
        if self.identifier == self.party_alice:
            return self.party_bob
        return self.party_alice

    @property
    def _own_feature_names(self) -> Tuple[str, ...]:
        """
        :return: the feature names belonging to your own data set
        """
        return self.__feature_names

    @property
    def _own_randomness(self) -> int:
        """
        :return: the initialised randomness
        """
        return self.__randomness

    @property
    def _received_feature_names(self) -> Tuple[str, ...]:
        """
        :return: the received feature names
        :raise ValueError: raised when there are no feature names received yet
        """
        if self._received.feature_names is None:
            raise ValueError("Did not receive feature names yet.")
        return self._received.feature_names or tuple()

    @property
    def _received_randomness(self) -> int:
        """
        :return: the received randomness
        :raise ValueError: raised when there is no randomness received yet
        """
        if self._received.randomness is None:
            raise ValueError("Did not receive randomness yet.")
        return self._received.randomness

    def encrypt_data(self) -> None:
        """
        Encrypts the data
        """
        self._encrypt_attributes()
        self._logger.info("Encrypted data")

    def generate_share(self) -> None:
        """
        Generates a random additive share
        """
        # self._share refers to features belonging to the *other* party
        self._share = np.ndarray(
            [self.intersection_size, len(self._received_feature_names)]
        )
        self._share = np.vectorize(self._signed_randomness)(self._share)
        self._logger.info("Generated share")

    def hash_data(self) -> None:
        """
        Hashes the identifiers of the dataset
        """
        self._hash_identifiers()
        self._logger.info("Hashed data")

    async def receive_feature_names(self) -> None:
        """
        Receive the feature names of the other data party
        """
        self._received.feature_names = await self.receive_message(
            self._other_party, msg_id="feature_names"
        )
        self._logger.info("Received feature names")

    async def receive_intersection_size(self) -> None:
        """
        Receive the computed intersection size
        """
        self._received.intersection_size = await self.receive_message(
            self.helper, msg_id="intersection_size"
        )
        self._logger.info(f"Received intersection: {self.intersection_size}")

    async def receive_paillier_scheme(self) -> None:
        """
        Receive the Paillier scheme of the other party,
        this enables encryption with their public key.
        """
        self._received.paillier_scheme = await self.receive_message(
            self._other_party, msg_id="paillier_scheme"
        )
        self._logger.info("Received Paillier scheme")

    async def receive_randomness(self) -> None:
        """
        Receive randomness from other data_owner to be used in the salted hash
        """
        self._received.randomness = await self.receive_message(
            self._other_party, msg_id="randomness"
        )
        self._logger.info("Received randomness")

    async def receive_share(self) -> None:
        """
        Receive an additive share of your own attributes (columns)
        """
        # self._share refers to *own* features
        encrypted_share = await self.receive_message(self.helper, msg_id="real_share")
        # Decrypt
        self._received.share = np.vectorize(self.paillier_scheme.decrypt)(
            encrypted_share
        )
        self._logger.info("Stored share")

    async def run_protocol(self) -> None:
        """
        Run the entire protocol, start to end, in an asynchronous manner
        """
        self._logger.info("Ready to roll, starting now!")
        await asyncio.gather(
            *[
                self.send_paillier_scheme(),
                self.send_randomness(),
                self.send_feature_names(),
                self.receive_paillier_scheme(),
                self.receive_randomness(),
                self.receive_feature_names(),
            ]
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.hash_data)
        await loop.run_in_executor(None, self.encrypt_data)
        await self.send_encrypted_data()
        await self.receive_intersection_size()
        await loop.run_in_executor(None, self.generate_share)
        await asyncio.gather(
            *[
                self.send_share(),
                self.receive_share(),
            ]
        )

    async def send_encrypted_data(self) -> None:
        """
        Send the encrypted data to the helper
        """
        self._logger.info(f"Sending encrypted data to {self.helper}")
        await self.send_message(self.helper, self._scrambled_data, "encrypted_data")

    async def send_feature_names(self) -> None:
        """
        Send the feature names of the data set to the other data party
        """
        self._logger.info(f"Sending feature names to {self._other_party}")
        await self.send_message(
            self._other_party, self._own_feature_names, msg_id="feature_names"
        )

    async def send_paillier_scheme(self) -> None:
        """
        Send the Paillier scheme to the other party,
        this enables encryption with your public key.
        The private key is NOT sent.
        """
        self._logger.info(f"Sending Paillier Scheme to {self._other_party}")
        await self.send_message(
            self._other_party, self.paillier_scheme, msg_id="paillier_scheme"
        )

    async def send_randomness(self) -> None:
        """
        Send randomness to other data_owner to be used in the salted hash
        """
        self._logger.info(f"Sending randomness to {self._other_party}")
        await self.send_message(
            self._other_party, self._own_randomness, msg_id="randomness"
        )

    async def send_share(self) -> None:
        """
        Send a random generated share to the helper party
        """
        loop = asyncio.get_event_loop()
        encrypted_share = await loop.run_in_executor(None, self._encrypt_share)
        await self.send_message(self.helper, encrypted_share, msg_id="random_share")
        self._logger.info("Sent share")

    def _encrypt_attributes(self) -> None:
        """
        Encrypts attributes stored in self.__data
        and stores the encryption in self._scrambled_data

        :raise ValueError: raised when there is no data to hash available yet
        """
        if self._scrambled_data is None:
            raise ValueError(
                "Did not initialise scrambled data yet."
                "Please hash identifiers first."
            )
        self._scrambled_data[:, 1:] = np.vectorize(self.paillier_scheme.encrypt)(
            self.__data[:, 1:]
        )

    def _encrypt_share(self) -> np.ndarray:
        """
        Encrypt the share with the public key of the other party

        :return: an encrypted share
        """
        return np.asarray(
            np.vectorize(self.received_paillier_scheme.encrypt)(self._share)
        )

    def _hash_entry(self, entry: AnyStr) -> bytes:
        """
        Returns the SHA256 hash digest of byte length 32 (256 bits) using the
        provided entry and the shared randomness as input

        :param entry: entry to hash
        :return: the hash digest
        """
        string = str(entry) + str(self.shared_randomness)
        bytes_string = string.encode("utf-8")
        return hash_fun(bytes_string).digest()

    def _hash_identifiers(self) -> None:
        """
        Hash the identifiers of the data set, assumed to be the first column
        """
        self._scrambled_data = np.ndarray(self.__data.shape, dtype=object)
        self._scrambled_data[:, 0] = np.vectorize(self._hash_entry)(self.__data[:, 0])

    def _signed_randomness(self, *_args: Any, **_kwargs: Any) -> SupportsInt:
        """
        Return a signed random plaintext, satisfying security restraints for an additive masking

        :return: signed randomness
        """
        return self.received_paillier_scheme.random_plaintext()
