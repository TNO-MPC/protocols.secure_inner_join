"""
Module contains DatabaseOwner class (either Alice or Bob) for performing secure set intersection
"""

import asyncio
import secrets  # "True" (secure) randomness
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, AnyStr, Callable, Dict, Optional, SupportsInt, Tuple, overload

import numpy as np
import numpy.typing as npt
from bitarray import bitarray
from mypy_extensions import KwArg, VarArg

from tno.mpc.encryption_schemes.paillier import Paillier, PaillierCiphertext

from .lsh import get_hyper_planes, lsh_hash
from .phonem import phonem_encode
from .player import Player


def sha256_hash_digest(bytes_string: bytes) -> bytes:
    """
    Returns the sha256 hash digest of byte length 32 (256 bits).

    :param bytes_string: bytes string to be hashed.
    :return: The hash digest.
    """
    return sha256(bytes_string).digest()


class DatabaseOwner(Player):
    """
    Class for a database owner
    """

    @dataclass
    class Collection:
        """
        Nested data class to store received data
        """

        feature_names: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
        intersection_size: Optional[int] = None
        paillier_scheme: Dict[str, Paillier] = field(default_factory=dict)
        randomness: Dict[str, int] = field(default_factory=dict)
        share: Optional[
            npt.NDArray[np.object_]
        ] = None  # this contains the shares belonging to the own features

    def __init__(
        self,
        *args: Any,
        identifiers: npt.NDArray[Any],
        data: npt.NDArray[Any],
        identifiers_phonetic: Optional[npt.NDArray[Any]] = None,
        identifiers_phonetic_exact: Optional[npt.NDArray[Any]] = None,
        identifier_date: Optional[npt.NDArray[Any]] = None,
        identifier_zip6: Optional[npt.NDArray[Any]] = None,
        feature_names: Tuple[str, ...] = (),
        paillier_scheme: Paillier = Paillier.from_security_parameter(
            key_length=2048, precision=8
        ),
        randomness_length: int = 64,
        phonetic_algorithm: Callable[[str], str] = phonem_encode,
        lsh_slices: int = 1000,
        hash_fun: Callable[[bytes], bytes] = sha256_hash_digest,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a database owner instance

        :param identifiers: identifiers to find exactly matching data for
        :param data: attributes (feature values) that will end up in the
                                     secure inner join
        :param identifiers_phonetic: identifiers to find matching data for
                                     that can contain phonetic errors
        :param identifiers_phonetic_exact: exact identifiers to append to
                                     phonetic encoding
        :param identifier_date: identifier to find matching data for
                                     that can contain erroneous date (of birth).
                                     Should be of the form dd-mm-yyyy
        :param identifier_zip6: identifier to find matching data for
                                     that can contain erroneous zip6 code.
                                     Should be of the form 1234AB
        :param feature_names: optional names of the shared features
        :param paillier_scheme: Instance of a Paillier scheme.
        :param randomness_length: number of bits for shared randomness salt
        :param phonetic_algorithm: phonetic algorithm (function) to use for phonetic matching
        :param lsh_slices: number of slices/hyperplanes to construct for LSH hashing, higher
                                     number results in higher accuracy
        :param hash_fun: hash function used (default sha256).
        :raise ValueError: raised when helper or data parties are not in the pool.
        """
        super().__init__(*args, **kwargs)

        if any(
            map(
                lambda other_party: other_party not in self._pool.pool_handlers.keys(),
                self._other_parties,
            )
        ):
            raise ValueError(
                f"Not all data owners '{self._other_parties}' are present in the communication pool '{tuple(self._pool.pool_handlers.keys())}'."
            )
        if self.helper not in self._pool.pool_handlers.keys():
            raise ValueError(f"Helper {self.helper} is not in the communication pool.")

        # initialize attributes
        self.paillier_scheme = paillier_scheme
        self.phonetic_algorithm = phonetic_algorithm
        self.hash_fun = hash_fun
        self.__randomness = secrets.randbits(randomness_length)
        self.__identifiers = DatabaseOwner._to_object_array(identifiers)
        self.__identifiers_phonetic = DatabaseOwner._to_object_array(
            identifiers_phonetic
        )
        self.__identifiers_phonetic_exact = DatabaseOwner._to_object_array(
            identifiers_phonetic_exact
        )
        self.__identifier_date = DatabaseOwner._to_object_array(identifier_date)
        self.__identifier_zip6 = DatabaseOwner._to_object_array(identifier_zip6)
        self.__data = DatabaseOwner._to_object_array(data)
        self.__feature_names = feature_names or tuple(
            "" for _ in range(self.__data.shape[1])
        )
        self.lsh_slices = lsh_slices

        # To be filled
        self._scrambled_identifiers: Optional[npt.NDArray[np.object_]] = None
        self._scrambled_data: Optional[npt.NDArray[np.object_]] = None
        self._scrambled_ph_identifiers: Optional[npt.NDArray[np.object_]] = None
        self._scrambled_lsh_identifiers: Optional[npt.NDArray[np.object_]] = None
        self._lsh_hyperplanes: Optional[npt.NDArray[np.int_]] = None
        self._lsh_bitmask: Optional[bitarray] = None
        self._received: DatabaseOwner.Collection = DatabaseOwner.Collection()
        self._share: Dict[
            str, npt.NDArray[np.object_]
        ] = {}  # this contains the shares belonging to the features of other parties

    @overload
    @staticmethod
    def _to_object_array(array: npt.NDArray[Any]) -> npt.NDArray[np.object_]:
        ...

    @overload
    @staticmethod
    def _to_object_array(array: None) -> None:
        ...

    @staticmethod
    def _to_object_array(
        array: Optional[npt.NDArray[Any]],
    ) -> Optional[npt.NDArray[np.object_]]:
        """
        Converts any array to an object array

        :param array: array to convert
        :return: None if array is None else object array
        """
        if array is None:
            return None
        return np.asarray(array, dtype=np.object_)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """
        The feature names of the inner join (same order for all data parties).

        :return: Tuple of feature names.
        """
        all_feature_names: Tuple[str, ...] = ()
        for data_party in self.data_parties:
            if data_party == self.identifier:
                all_feature_names += self._own_feature_names
            else:
                all_feature_names += self._received_feature_names[data_party]
        return all_feature_names

    @property
    def intersection_size(self) -> int:
        """
        The intersection size as was determined by the helper.

        :return: Intersection size.
        :raise ValueError: raised when there is no intersection size available yet.
        """
        if self._received.intersection_size is None:
            raise ValueError("Did not receive intersection size yet.")
        return self._received.intersection_size

    @property
    def received_paillier_schemes(self) -> Dict[str, Paillier]:
        """
        The received Paillier schemes of all data parties.

        :return: A dictionary mapping data party identifiers to Paillier schemes.
        :raise ValueError: Raised when all Paillier schemes have not yet been received.
        """
        if len(self._received.paillier_scheme) != len(self._other_parties):
            raise ValueError("Did not receive all paillier schemes yet.")
        return self._received.paillier_scheme

    @property
    def shared_randomness(self) -> int:
        """
        The shared randomness (sum of own randomness and that of the other parties).

        :return: Shared randomness.
        """
        return self._own_randomness + sum(self._received_randomness.values())

    @property
    def shares(self) -> npt.NDArray[np.object_]:
        """
        The shares of the complete secure inner join.

        :return: All secure inner join shares.
        :raise ValueError: Raised when not all shares are available.
        """
        if self.intersection_size == 0:
            return np.empty([0, len(self.feature_names)], dtype=np.object_)
        if self._received.share is None or len(self._share) != len(self._other_parties):
            raise ValueError("Not all shares are available (yet).")

        return np.hstack(
            tuple(
                map(
                    lambda party: self._received.share
                    if party == self.identifier
                    else self._share[party],
                    self.data_parties,
                )
            )
        )

    @property
    def _other_parties(self) -> Tuple[str, ...]:
        """
        The identifiers of all other data parties.

        :return: A tuple containing the identifiers of all data parties that are not you.
        :raise ValueError: Raised when we are not listed among all data parties.
        """
        if self.identifier not in self.data_parties:
            raise ValueError("We should be among the data parties.")
        data_parties_list = list(self.data_parties)
        data_parties_list.remove(self.identifier)
        return tuple(data_parties_list)

    @property
    def _own_feature_names(self) -> Tuple[str, ...]:
        """
        The feature names (columns) of the own dataset.

        :return: A tuple containing the feature names belonging to the own data set.
        """
        return self.__feature_names

    @property
    def _own_randomness(self) -> int:
        """
        The own initialised randomness value.

        :return: Randomness value.
        """
        return self.__randomness

    @property
    def _received_feature_names(self) -> Dict[str, Tuple[str, ...]]:
        """
        The features names of all other data parties.

        :return: A dictionary mapping the other data parties to their respective feature names.
        :raise ValueError: In case not all feature names have been received yet.
        """
        if len(self._received.feature_names) != len(self._other_parties):
            raise ValueError("Did not receive all feature names yet.")
        return self._received.feature_names

    @property
    def _received_randomness(self) -> Dict[str, int]:
        """
        The randomness values of all other data parties.

        :return: A dictionary mapping the other data parties to their respective randomness values.
        :raise ValueError: Raised when not all randomness values have been received yet.
        """
        if len(self._received.randomness) != len(self._other_parties):
            raise ValueError("Did not receive all randomness yet.")
        return self._received.randomness

    def encode_lsh_data(self) -> None:
        """
        Encode the Locality-Sensitive Hashing identifiers of the dataset
        """
        self._lsh_encoding()
        self._logger.info("Encoded LSH identifiers data")

    def encode_phonetic_data(self) -> None:
        """
        Encode and hash the phonetic identifiers of the dataset
        """
        self._phonetic_encoding()
        self._logger.info("Encoded and hashed phonetic identifiers data")

    def encrypt_data(self) -> None:
        """
        Encrypts the own data, by hashing the identifier column using the shared randomness, and by Paillier encrypting
        the feature values.
        """
        self._encrypt_attributes()
        self._logger.info("Encrypted data")

    def generate_shares(self) -> None:
        """
        Generates random additive shares for all other data parties.
        """
        # Note that self._share stores to the shares belonging to the features of all other parties
        for other_party in self._other_parties:
            self._share[other_party] = np.vectorize(
                self._signed_randomness_for_party(other_party)
            )(
                npt.NDArray[np.object_](
                    (
                        self.intersection_size,
                        len(self._received_feature_names[other_party]),
                    )
                )
            )

        self._logger.info("Generated shares for all other parties")

    def hash_data(self) -> None:
        """
        Hash the identifiers of the dataset using the shared randomness.
        """
        self._hash_identifiers()
        self._logger.info("Hashed data")

    async def receive_all_feature_names(self) -> None:
        """
        Receive the feature names of all other data parties.
        """
        await asyncio.gather(
            *[self._receive_feature_names(party) for party in self._other_parties]
        )

        self._logger.info("Received feature names from all parties")

    async def receive_intersection_size(self) -> None:
        """
        Receive the computed intersection size from the helper party.
        """
        self._received.intersection_size = await self.receive_message(
            self.helper, msg_id="intersection_size"
        )
        self._logger.info(f"Received intersection: {self.intersection_size}")

    async def receive_all_paillier_schemes(self) -> None:
        """
        Receive the Paillier schemes of all other parties, thereby making encryption with their public keys possible.
        """
        await asyncio.gather(
            *[self._receive_paillier_scheme(party) for party in self._other_parties]
        )

        self._logger.info("Received Paillier scheme")

    async def receive_all_randomness(self) -> None:
        """
        Receive randomness from other data_owner to be used in the salted hash
        """
        await asyncio.gather(
            *[self._receive_randomness(party) for party in self._other_parties]
        )

        self._logger.info("Received randomness from all parties")

    async def receive_and_verify_data_parties(self) -> None:
        """
        Receive all data parties with accompanying addresses and ports from the helper and verify if it exactly
        (including order) matches the own data parties tuple.

        :raise ValueError: In case the data parties do not match exactly (including order).
        """
        received_data_parties = await self.receive_message(self.helper, "data_parties")
        if len(self.data_parties_and_addresses) != len(received_data_parties):
            raise ValueError(
                f"The data parties as sent by the helper '{received_data_parties}', does not equal the own"
                f" set of data parties '{self.data_parties_and_addresses}'. Each tuple in both lists is structured"
                f" as '(identifier, address, port number)'."
            )
        for party_helper, party_self in zip(
            self.data_parties_and_addresses, received_data_parties
        ):
            if party_helper[0] != party_self[0] or (
                party_helper[0] != self.identifier and party_helper != party_self
            ):
                raise ValueError(
                    f"The data parties as sent by the helper '{received_data_parties}', does not equal the own"
                    f" set of data parties '{self.data_parties_and_addresses}'. Each tuple in both lists is structured"
                    f" as '(identifier, address, port number)'."
                )

    async def receive_share(self) -> None:
        """
        Receive an additive share of your own feature values (columns)
        """
        # self._received.share refers to *own* features
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
        await self.receive_and_verify_data_parties()
        await asyncio.gather(
            *[
                self.send_paillier_scheme_to_all(),
                self.send_randomness_to_all(),
                self.send_feature_names_to_all(),
                self.receive_all_paillier_schemes(),
                self.receive_all_randomness(),
                self.receive_all_feature_names(),
            ]
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.hash_data)
        await loop.run_in_executor(None, self.encode_phonetic_data)
        await loop.run_in_executor(None, self.encode_lsh_data)
        await loop.run_in_executor(None, self.encrypt_data)
        await asyncio.gather(
            *[
                self.send_hashed_identifiers(),
                self.send_phonetic_identifiers(),
                self.send_lsh_identifiers(),
                self.send_encrypted_data(),
                self.receive_intersection_size(),
            ]
        )
        if self.intersection_size > 0:
            await loop.run_in_executor(None, self.generate_shares)
            await asyncio.gather(
                *[
                    self.send_shares(),
                    self.receive_share(),
                ]
            )
        else:
            self._logger.warning("Intersection size was nonpositive")
        self._logger.info("All done")

    async def send_encrypted_data(self) -> None:
        """
        Send the encrypted data to the helper
        """
        await self.send_message(self.helper, self._scrambled_data, "encrypted_data")
        self._logger.info(f"Sent encrypted data to {self.helper}")

    async def send_feature_names_to_all(self) -> None:
        """
        Send the feature names of the own dataset to all other data parties
        """
        await asyncio.gather(
            *[self._send_feature_names(party) for party in self._other_parties]
        )

    async def send_hashed_identifiers(self) -> None:
        """
        Send the hashed identifiers to the helper
        """
        self._logger.info(f"Sending hashed identifiers to {self.helper}")
        await self.send_message(
            self.helper, self._scrambled_identifiers, "hashed_identifiers"
        )

    async def send_paillier_scheme_to_all(self) -> None:
        """
        Send the Paillier scheme to all other parties, this enables them to encrypt values with your public key.
        The private key is NOT communicated.
        """
        await asyncio.gather(
            *[self._send_paillier_scheme(party) for party in self._other_parties]
        )

    async def send_lsh_identifiers(self) -> None:
        """
        Send the encoded Locality-Sensitive Hashing identifiers to the helper
        """
        self._logger.info(f"Sending hashed LSH identifiers to {self.helper}")
        await self.send_message(
            self.helper, self._scrambled_lsh_identifiers, "hashed_lsh_identifiers"
        )

    async def send_phonetic_identifiers(self) -> None:
        """
        Send the hashed phonetic identifiers to the helper
        """
        self._logger.info(f"Sending hashed phonetic identifiers to {self.helper}")
        await self.send_message(
            self.helper, self._scrambled_ph_identifiers, "hashed_ph_identifiers"
        )

    async def send_randomness_to_all(self) -> None:
        """
        Send randomness to other data_owners to be used in the salted hash
        """
        await asyncio.gather(
            *[self._send_randomness(party) for party in self._other_parties]
        )

    async def send_shares(self) -> None:
        """
        Send the random generated shares for all other data parties to the helper party
        """
        self._logger.info("Start sending shares")

        loop = asyncio.get_event_loop()
        encrypted_shares = await loop.run_in_executor(None, self._encrypt_all_shares)
        await self.send_message(self.helper, encrypted_shares, msg_id="random_share")
        self._logger.info("Sent shares")

    def _encrypt_attributes(self) -> None:
        """
        Encrypts attributes (feature values) stored in self.__data and stores the encryption in self._scrambled_data.
        """
        self._scrambled_data = np.ndarray(self.__data.shape, dtype=np.object_)
        self._scrambled_data = np.vectorize(self.paillier_scheme.encrypt)(self.__data)

    def _encrypt_all_shares(self) -> Dict[str, npt.NDArray[np.object_]]:
        """
        Encrypt the shares of all other data parties with their respective Paillier public keys.

        :return: Dictionary mapping other data parties to their encrypted shares.
        """
        encrypted_shares = {}

        for other_party in self._other_parties:
            encrypted_shares[other_party] = np.vectorize(
                self._encrypt_share(other_party)
            )(self._share[other_party])

        return encrypted_shares

    def _encrypt_share(self, party: str) -> Callable[[int], PaillierCiphertext]:
        """
        Return method for encrypting values with the public key of the given other party.

        :return: Method to encrypt values with a public key.
        """
        paillier_scheme = self.received_paillier_schemes[party]
        return paillier_scheme.encrypt

    def _hash_entry(self, entry: AnyStr) -> bytes:
        """
        Returns the hash (default sha256) digest of byte length 32 (256 bits) using the provided entry and the shared randomness
        as input.

        :param entry: Entry to be hashed.
        :return: The hash digest.
        """
        string = str(entry) + str(self.shared_randomness)
        bytes_string = string.encode("utf-8")
        return self.hash_fun(bytes_string)

    def _hash_identifiers(self) -> None:
        """
        Hash the identifiers of the data set, it combines identifiers into
        a single identifier when multiple are supplied.
        """
        if self.__identifiers.ndim > 1:
            self._scrambled_identifiers = np.sum(self.__identifiers, axis=1)
        else:
            self._scrambled_identifiers = self.__identifiers
        self._scrambled_identifiers = np.vectorize(self._hash_entry)(
            self._scrambled_identifiers
        )

    def _lsh_encode(self, date: str, zip6: str) -> bitarray:
        """
        Encode a date + zip6 code using Locality-Sensitive Hashing
        (LSH).

        :param date: date to encode
        :param zip6: zip6 to encode
        :return: LSH encoding
        :raise ValueError: when not ready to start LSH computation
        """
        if self._lsh_hyperplanes is None or self._lsh_bitmask is None:
            raise ValueError("LSH not properly initialized yet.")
        day, month, year = date.split("-")
        zip4 = zip6[0:4]
        hash_value = lsh_hash(
            int(day),
            int(month),
            int(year[2:4]),
            int(zip4),
            hyper_planes=self._lsh_hyperplanes,
            bit_mask=self._lsh_bitmask,
        )
        return hash_value

    def _lsh_encoding(self) -> None:
        """
        Encodes date and zip6 identifiers of the data set using
        Locality-Sensitive Hashing (LSH).
        """
        self._lsh_hyperplanes, self._lsh_bitmask = get_hyper_planes(
            amount=self.lsh_slices, seed=self.shared_randomness, mask=True
        )
        if self.__identifier_date is None or self.__identifier_zip6 is None:
            self._scrambled_lsh_identifiers = np.empty(
                [0, self.__data.shape[1]], dtype=np.object_
            )
        else:
            self._scrambled_lsh_identifiers = np.vectorize(
                self._lsh_encode,
                otypes=[bitarray],
            )(
                self.__identifier_date,
                self.__identifier_zip6,
            )

    def _phonetic_encoding(self) -> None:
        """
        Encodes and hashes the phonetic identifiers of the data set, it combines
        identifiers into a single identifier when multiple are supplied.
        Furthermore, it appends exact attributes just before hashing.

        e.g. first name Jan, last name Janssen, gender Male becomes YANYANSNMale
        and is hashed afterwards, where first and last name are phonetic, gender is exact
        """
        if self.__identifiers_phonetic is None:
            self._scrambled_ph_identifiers = np.empty(
                [0, self.__data.shape[1]], dtype=np.object_
            )
        else:
            if self.__identifiers_phonetic.ndim > 1:
                # Construct whitespace separator of phonetic attributes
                self._scrambled_ph_identifiers = np.insert(
                    self.__identifiers_phonetic,
                    range(1, self.__identifiers_phonetic.shape[1]),
                    values=" ",
                    axis=1,
                )
                # Sum into single string
                self._scrambled_ph_identifiers = np.sum(
                    self._scrambled_ph_identifiers, axis=1
                )
            else:
                self._scrambled_ph_identifiers = self.__identifiers_phonetic

            self._scrambled_ph_identifiers = np.vectorize(self.phonetic_algorithm)(
                self._scrambled_ph_identifiers
            )
            if self.__identifiers_phonetic_exact is not None:
                # Append exact attributes to string
                self._scrambled_ph_identifiers = np.sum(
                    np.column_stack(
                        (
                            self._scrambled_ph_identifiers,
                            self.__identifiers_phonetic_exact,
                        )
                    ),
                    axis=1,
                )
            self._scrambled_ph_identifiers = np.vectorize(self._hash_entry)(
                self._scrambled_ph_identifiers
            )

    async def _receive_feature_names(self, party: str) -> None:
        """
        Receive the feature names of the given party.

        :param party: Identifier of the party to receive feature names from.
        """
        self._received.feature_names[party] = await self.receive_message(
            party, msg_id="feature_names"
        )

    async def _receive_paillier_scheme(self, party: str) -> None:
        """
        Receive the Paillier scheme of the given party.

        :param party: Identifier of the party to receive Paillier scheme from.
        """
        self._received.paillier_scheme[party] = await self.receive_message(
            party, msg_id="paillier_scheme"
        )

    async def _receive_randomness(self, party: str) -> None:
        """
        Receive the randomness of the given party.

        :param party: Identifier of the party to receive randomness from.
        """
        self._received.randomness[party] = await self.receive_message(
            party, msg_id="randomness"
        )

    async def _send_feature_names(self, party: str) -> None:
        """
        Send the feature names of the own dataset to the given party.

        :param party: Identifier of the party to send the feature names to.
        """
        await self.send_message(party, self._own_feature_names, msg_id="feature_names")

    async def _send_paillier_scheme(self, party: str) -> None:
        """
        Send own Paillier scheme to the given party.

        :param party: Identifier of the party to send the Paillier scheme to.
        """
        await self.send_message(party, self.paillier_scheme, msg_id="paillier_scheme")

    async def _send_randomness(self, party: str) -> None:
        """
        Send own randomness to the given party.

        :param party: Identifier of the party to send the Paillier scheme to.
        """
        await self.send_message(party, self._own_randomness, msg_id="randomness")

    def _signed_randomness_for_party(
        self, party: str, *_args: Any, **_kwargs: Any
    ) -> Callable[[VarArg(Any), KwArg(Any)], SupportsInt]:
        """
        Return a method that produces a signed random plaintext using the Paillier public key of the given party,
        satisfying the security restraints for an additive masking.

        :param party: Identifier of the party whose Paillier public key is to be used.
        :param _args: Unused additional arguments.
        :param _kwargs: Unused additional arguments.
        :return: Method that produces random plaintext under the given Paillier public key.
        """
        paillier_scheme = self.received_paillier_schemes[party]

        def _signed_randomness(*_args: Any, **_kwargs: Any) -> SupportsInt:
            """
            Return a signed random plaintext, satisfying security restraints for an additive masking

            :return: Signed randomness.
            """
            return paillier_scheme.random_plaintext() / (len(self._other_parties))

        return _signed_randomness
