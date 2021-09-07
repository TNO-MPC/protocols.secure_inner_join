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
        data_parties: Tuple[str, str] = ("alice", "bob"),
        helper: str = "henri",
    ) -> None:
        """
        Initializes the database owner

        :param identifier: identifier of the player, either Alice or Bob or Henri
        :param pool: instance of tno.mpc.communication.Pool
        :param data_parties: identifiers of the data_parties
        :param helper: identifier of the helper
        """
        # Initialise
        self._identifier = identifier
        self._data_parties = data_parties
        self._helper = helper
        self.pool = pool

        self._logger = self.create_logger(self._identifier)
        self._logger.info(
            "Starting execution, time stamp:"
            f"{datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')}"
        )

    @property
    def data_parties(self) -> Tuple[str, str]:
        """
        The identifiers of the data parties

        :return: the identifiers of the data parties
        """
        return self._data_parties

    @property
    def helper(self) -> str:
        """
        The identifier of the helper player

        :return: the identifier of the helper
        """
        return self._helper

    @property
    def identifier(self) -> str:
        """
        The identifier of this player

        :return: the identifier
        """
        return self._identifier

    @property
    @abstractmethod
    def intersection_size(self) -> int:
        """
        Should return the size of the intersection between
        the identifier columns of the data parties

        :return: the intersection size
        """

    @property
    def party_alice(self) -> str:
        """
        The identifier that is used to identify the first party (Alice)

        :return: the identifier of the first party
        """
        return self.data_parties[0]

    @property
    def party_bob(self) -> str:
        """
        The identifier that is used to identify the second party (Bob)

        :return: the identifier of the second party
        """
        return self.data_parties[1]

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
        Receive a message from party with msg_id

        :param party: party to receive message from
        :param msg_id: optional identifier for the message
        :return: the received message contents
        """
        return await self.pool.recv(party, msg_id=msg_id)

    @abstractmethod
    async def run_protocol(self) -> None:
        """
        Should run the entire protocol, start to end, in an asynchronous manner
        """

    async def send_message(
        self, receiver: str, payload: Any, msg_id: Optional[str] = None
    ) -> None:
        """
        Send a payload to receiver with msg_id

        :param receiver: party to send message to
        :param payload: data to send
        :param msg_id: optional identifier for the message
        """
        await self.pool.send(receiver, payload, msg_id=msg_id)
