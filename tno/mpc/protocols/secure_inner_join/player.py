"""
Code for the abstract player class to implement parties
performing secure set intersection
"""

import datetime
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from tno.mpc.communication import Pool


class Player(ABC):
    """
    Class for a player
    """

    def __init__(
        self,
        identifier: str,
        pool: Pool,
        data_parties: Tuple[str, ...] = ("alice", "bob"),
        helper: str = "henri",
    ) -> None:
        """
        Initializes the database owner

        :param identifier: identifier of the player
        :param pool: instance of tno.mpc.communication.Pool
        :param data_parties: identifiers of the data_parties
        :param helper: identifier of the helper
        """
        # Initialise attributes
        self._identifier = identifier
        self._data_parties = tuple(sorted(data_parties))
        self._helper = helper
        self._pool = pool

        # Initialise logger
        self._logger = self.create_logger(self._identifier)
        self._logger.info(
            "Starting execution, time stamp:"
            f"{datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')}"
        )

    @property
    def data_parties(self) -> Tuple[str, ...]:
        """
        The identifiers of all data parties (sorted alphabetically, and the same for each player).

        :return: A tuple containing the identifiers of all data parties.
        """
        return self._data_parties

    @property
    def data_parties_and_addresses(
        self,
    ) -> Tuple[Tuple[str, Optional[str], Optional[int]], ...]:
        """
        Tuples containing (identifier, address, port) of all data parties (sorted alphabetically on identifier, and the
        same for each player).

        :return: A tuple containing the identifiers, addresses, and ports of all data parties.
        """
        return tuple(
            (
                party,
                self._pool.pool_handlers[party].addr,
                self._pool.pool_handlers[party].port,
            )
            if party != self.identifier
            else (party, None, None)
            for party in self.data_parties
        )

    @property
    def helper(self) -> str:
        """
        The identifier of the helper party.

        :return: The identifier of the helper.
        """
        return self._helper

    @property
    def identifier(self) -> str:
        """
        The identifier of this party.

        :return: The identifier of this party.
        """
        return self._identifier

    @property
    @abstractmethod
    def intersection_size(self) -> int:
        """
        Returns the size of the intersection of the identifier columns of all data parties.

        :return: The intersection size.
        """

    @staticmethod
    def create_logger(name: str) -> logging.Logger:
        """
        Create logger for class

        :param name: name of the logger
        :return: logger object
        """
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

    async def receive_message(self, party: str, msg_id: Optional[str] = None) -> Any:
        """
        Receive a message from party with the given msg_id, if no msg_id is given the message with the lowest numerical
        id is selected.

        :param party: Identifier of the party to receive message from.
        :param msg_id: Optional identifier for the message.
        :return: The received message contents.
        """
        return await self._pool.recv(party, msg_id=msg_id)

    @abstractmethod
    async def run_protocol(self) -> None:
        """
        Runs the entire protocol, start to end, in an asynchronous manner.
        """

    async def send_message(
        self, receiver: str, payload: Any, msg_id: Optional[str] = None
    ) -> None:
        """
        Sends the given payload to the receiver with the given msg_id.

        :param receiver: Identifier of the party to send the message to.
        :param payload: Data to send.
        :param msg_id: Optional identifier for the message.
        """
        await self._pool.send(receiver, payload, msg_id=msg_id)
