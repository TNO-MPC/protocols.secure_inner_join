"""
Example usage for performing secure set intersection with a variable number of players
Run four separate instances e.g.,
$ python example_usage_exact.py -p Alice
$ python example_usage_exact.py -p Bob
$ python example_usage_exact.py -p Charlie
$ python example_usage_exact.py -p Henri

This protocol can be run with any number of data parties that is greater than or equal to 2. A helper party (Henri)
is always required
"""

from __future__ import annotations

import argparse
import asyncio

import pandas as pd

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import Paillier

from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--player",
        help="Name of the sending player",
        type=str.lower,
        required=True,
        choices=("alice", "bob", "charlie", "henri"),
    )
    args = parser.parse_args()
    return args


async def main(player_instance: DatabaseOwner | Helper) -> None:
    await player_instance.run_protocol()
    if isinstance(player_instance, DatabaseOwner):
        print("Gathered shares:")
        print(player_instance.feature_names)
        print(player_instance.shares)


if __name__ == "__main__":
    # Parse arguments and acquire configuration parameters
    args = parse_args()
    player = args.player
    parties = {
        "alice": {"address": "127.0.0.1", "port": 8080},
        "bob": {"address": "127.0.0.1", "port": 8081},
        "charlie": {"address": "127.0.0.1", "port": 8082},
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

    df: pd.DataFrame | None = None
    player_instance: DatabaseOwner | Helper
    if player == "henri":
        player_instance = Helper(
            data_parties=("alice", "bob", "charlie"),
            identifier=player,
            pool=pool,
        )
    else:
        if player == "alice":
            df = pd.DataFrame(
                {
                    "identifier": ["Thomas", "Michiel", "Bart", "Nicole", "Alex"],
                    "feature_A1": [2, -1, 3, 1, 0],
                    "feature_A2": [12.5, 31.232, 23.11, 8.3, 20.44],
                }
            )
        elif player == "bob":
            df = pd.DataFrame(
                {
                    "identifier": [
                        "Thomas",
                        "Victor",
                        "Bart",
                        "Michiel",
                        "Tariq",
                        "Alex",
                    ],
                    "feature_B1": [5, 231, 30, 40, 42, 11],
                    "feature_B2": [10, 2, 1, 8, 6, 5],
                }
            )
        elif player == "charlie":
            df = pd.DataFrame(
                {
                    "identifier": ["Bart", "Thomas", "Michiel", "Robert"],
                    "feature_C1": [-1, -5, 100, 23.3],
                    "feature_C2": [10, 12, 8, 5],
                }
            )
        paillier = Paillier.from_security_parameter(key_length=2048, precision=8)
        player_instance = DatabaseOwner(
            data_parties=("alice", "bob", "charlie"),
            identifier=player,
            paillier_scheme=paillier,
            identifiers=df[["identifier"]].to_numpy(dtype=str),
            data=df.to_numpy(dtype=object)[:, 1:],
            feature_names=tuple(df.columns[1:]),
            pool=pool,
        )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(player_instance))

    player_instance.shutdown_received_schemes()
    if isinstance(player_instance, DatabaseOwner):
        player_instance.paillier_scheme.shut_down()
