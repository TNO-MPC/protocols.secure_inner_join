# TNO MPC Lab - Protocols - Secure Inner Join

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.protocols.secure_inner_join is part of the TNO Python Toolbox.

The research activities that led to this protocol and implementation took place in the BigMedilytics project that received funding from the European Union's Horizon 2020 research and innovation program under grant agreement No. 780495.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*

## Documentation

Documentation of the tno.mpc.protocols.secure_inner_join package can be found [here](https://docs.mpc.tno.nl/protocols/secure_inner_join/0.3.4).

## Install

Easily install the tno.mpc.protocols.secure_inner_join package using pip:
```console
$ python -m pip install tno.mpc.protocols.secure_inner_join
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.mpc.protocols.secure_inner_join[tests]'
```

### Note:
A significant performance improvement can be achieved by installing the GMPY2 library.
```console
$ python -m pip install 'tno.mpc.protocols.secure_inner_join[gmpy]'
```

## Protocol description
We consider a three-party setting with two data owners and one helper party. The protocol is passively secure and a visual representation of the protocol is shown below.

For more information see [Blog: Identifying heart failure patients at high risk using MPC](https://medium.com/applied-mpc/identifying-heart-failure-patients-at-high-risk-using-mpc-ab8900e75295) and/or [Video: Identifying heart failure risks with Multi-Party Computation](https://youtu.be/hvBb80eXuZg).

![Protocol diagram](https://raw.githubusercontent.com/TNO-MPC/protocols.secure_inner_join/main/assets/protocol_description.svg)

## Usage

The protocol is asymmetric. To run the protocol you need to run three separate instances.

*Note*: Identifiers are assumed to be unique.

>`example_usage.py`
>```python
>"""
>    Example usage for performing secure set intersection
>    Run three separate instances e.g.,
>    $ python example_usage.py -p Alice
>    $ python example_usage.py -p Bob
>    $ python example_usage.py -p Henri
>"""
>import argparse
>import asyncio
>from typing import Optional
>
>import pandas as pd
>
>from tno.mpc.communication import Pool
>from tno.mpc.protocols.secure_inner_join import DatabaseOwner, Helper
>
>
>def parse_args():
>    parser = argparse.ArgumentParser()
>    parser.add_argument(
>        "-p",
>        "--player",
>        help="Name of the sending player",
>        type=str.lower,
>        required=True,
>        choices=["alice", "bob", "henri"],
>    )
>    args = parser.parse_args()
>    return args
>
>
>async def main(player_instance):
>    await player_instance.run_protocol()
>    if player_instance.identifier in player_instance.data_parties:
>        print("Gathered shares:")
>        print(player_instance.feature_names)
>        print(player_instance.shares)
>
>
>if __name__ == "__main__":
>    # Parse arguments and acquire configuration parameters
>    args = parse_args()
>    player = args.player
>    parties = {
>        "alice": {"address": "127.0.0.1", "port": 8080},
>        "bob": {"address": "127.0.0.1", "port": 8081},
>        "henri": {"address": "127.0.0.1", "port": 8082},
>    }
>
>    port = parties[player]["port"]
>    del parties[player]
>
>    pool = Pool()
>    pool.add_http_server(port=port)
>    for name, party in parties.items():
>        assert "address" in party
>        pool.add_http_client(
>            name, party["address"], port=party["port"] if "port" in party else 80
>        )  # default port=80
>
>    df: Optional[pd.DataFrame] = None
>    if player == "henri":
>        player_instance = Helper(
>            identifier=player,
>            pool=pool,
>        )
>    else:
>        if player == "alice":
>            df = pd.DataFrame(
>                {
>                    "identifier": ["Thomas", "Michiel", "Bart", "Nicole"],
>                    "feature_A1": [2, -1, 3, 1],
>                    "feature_A2": [12.5, 31.232, 23.11, 8.3],
>                }
>            )
>        elif player == "bob":
>            df = pd.DataFrame(
>                {
>                    "identifier": ["Thomas", "Victor", "Bart", "Michiel", "Tariq"],
>                    "feature_B1": [5, 231, 30, 40, 42],
>                    "feature_B2": [10, 2, 1, 8, 6],
>                }
>            )
>        player_instance = DatabaseOwner(
>            identifier=player,
>            data=df.to_numpy(dtype="object"),
>            feature_names=tuple(df.columns[1:]),
>            pool=pool,
>        )
>
>    loop = asyncio.get_event_loop()
>    loop.run_until_complete(main(player_instance))
>```

Run three separate instances specifying the players:
```console
$ python example_usage.py -p Alice
$ python example_usage.py -p Bob
$ python example_usage.py -p Henri
```
