"""
Example usage for performing secure fuzzy matching with a variable number of players
Run three separate instances e.g.,
$ python example_usage.py -p Alice
$ python example_usage.py -p Bob
$ python example_usage.py -p Henri

This protocol can be run with any number of data parties that is greater than or equal to 2. A helper party (Henri)
is always required
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import Paillier

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper

DATA_DIR = Path(__file__).parent.parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--player",
        help="Name of the sending player",
        type=str.lower,
        required=True,
        choices=("alice", "bob", "henri"),
    )
    args = parser.parse_args()
    return args


async def main(player_instance: DatabaseOwner | Helper) -> None:
    await player_instance.run_protocol()
    if isinstance(player_instance, DatabaseOwner):
        print("Gathered shares:")
        print(player_instance.feature_names)
        print(player_instance.shares)


async def generate_instance(player: str) -> None:
    parties = {
        "alice": {"address": "127.0.0.1", "port": 8080},
        "bob": {"address": "127.0.0.1", "port": 8081},
        "henri": {"address": "127.0.0.1", "port": 8083},
    }

    port = parties[player]["port"]
    del parties[player]

    pool = Pool()
    pool.add_http_server(port=port)
    for name, party in parties.items():
        assert "address" in party
        pool.add_http_client(
            name, party["address"], port=party["port"] if "port" in party else 80
        )  # default port=80

    player_instance: DatabaseOwner | Helper
    if player == "henri":
        player_instance = Helper(
            data_parties=("alice", "bob"),
            identifier=player,
            pool=pool,
        )
    else:
        if player == "alice":
            df = pd.read_csv(DATA_DIR / "player_1.csv")
        elif player == "bob":
            df = pd.read_csv(DATA_DIR / "player_2.csv")
        paillier = Paillier.from_security_parameter(key_length=2048, precision=8)
        player_instance = DatabaseOwner(
            data_parties=("alice", "bob"),
            identifier=player,
            paillier_scheme=paillier,
            identifiers=df[
                [
                    "first_name",
                    "last_name",
                    "date_of_birth",
                    "zip6_code",
                    "gender_at_birth",
                ]
            ].to_numpy(dtype=str),
            identifiers_phonetic=df[["first_name", "last_name"]].to_numpy(dtype=str),
            identifiers_phonetic_exact=df[["gender_at_birth"]].to_numpy(dtype=str),
            identifier_date=df[["date_of_birth"]].to_numpy(dtype=str),
            identifier_zip6=df[["zip6_code"]].to_numpy(dtype=str),
            data=df.to_numpy(dtype=object)[:, -1, None],
            feature_names=(df.columns[-1],),
            pool=pool,
        )

    await main(player_instance)

    player_instance.shutdown_received_schemes()
    if isinstance(player_instance, DatabaseOwner):
        player_instance.paillier_scheme.shut_down()


if __name__ == "__main__":
    # Parse arguments and acquire configuration parameters
    args = parse_args()
    player = args.player
    loop = asyncio.new_event_loop()
    loop.run_until_complete(generate_instance(player))
