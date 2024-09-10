# TNO PET Lab - secure Multi-Party Computation (MPC) - Protocols - Secure Inner Join

Inspired by the work done in the BigMedilytics project.
For more information see https://youtu.be/hvBb80eXuZg.

### PET Lab

The TNO PET Lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of PET solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed PET functionalities to boost the development of new protocols and solutions.

The package `tno.mpc.protocols.secure_inner_join` is part of the [TNO Python Toolbox](https://github.com/TNO-PET).

_Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws._  
_This implementation of cryptographic software has not been audited. Use at your own risk._

## Documentation

Documentation of the `tno.mpc.protocols.secure_inner_join` package can be found
[here](https://docs.pet.tno.nl/mpc/protocols/secure_inner_join/2.0.2).

## Install

Easily install the `tno.mpc.protocols.secure_inner_join` package using `pip`:

```console
$ python -m pip install tno.mpc.protocols.secure_inner_join
```

_Note:_ If you are cloning the repository and wish to edit the source code, be
sure to install the package in editable mode:

```console
$ python -m pip install -e 'tno.mpc.protocols.secure_inner_join'
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.mpc.protocols.secure_inner_join[tests]'
```
_Note:_ A significant performance improvement can be achieved by installing the GMPY2 library.

```console
$ python -m pip install 'tno.mpc.protocols.secure_inner_join[gmpy]'

```

## Protocol description

A visual representation of the protocol, with two data parties, is shown below. The protocol as implemented in this
library can be run with any number (at least two) of data parties. One helper party is always required.

The result of the Secure Inner Join protocol with `n` data parties is an additive `n`-sharing, i.e. the table of the
resulting intersection can be constructed by summing the table shares owned by all `n` data parties.

_Note:_ more information on the secure approximate matching (fuzzy matching) protocol is found [here](#secure-approximate-matching-fuzzy-matching).

![Protocol diagram](https://raw.githubusercontent.com/TNO-MPC/protocols.secure_inner_join/main/assets/protocol_description.svg)

## Usage

The protocol is asymmetric. To run the protocol you need to run separate instances, one for the helper party and one for each data party (at least two data parties are required). Please make sure to install all required (additional) dependencies through

`python -m pip install .[scripts]`

<details>
<summary>
Exact matching

</summary>

_Note_: Identifiers are assumed to be unique.

`example_usage_exact.py`

```python
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
```

Run four separate instances specifying the players:

```console
$ python example_usage.py -p Alice
$ python example_usage.py -p Bob
$ python example_usage.py -p Charlie
$ python example_usage.py -p Henri
```

This protocol can be run with any number of data parties that is greater than or equal to 2. A helper party (Henri) is always required.

</details>

<details>
<summary>
Fuzzy/approximate matching

</summary>

> `data/player_1.csv`
>
> ```csv
> first_name,last_name,date_of_birth,zip6_code,gender_at_birth,correct_match_A
> Tomas,Roijackers,09-01-1874,1234AB,M,-1
> Tomas,Rooiakkers,06-12-1874,1232XY,M,3
> Tomas,Rooijackers,16-02-1875,5712DX,M,5
> Tomas,Roijackers,09-01-1874,7521LS,M,4
> Thomas,Rooijakkers,09-01-1874,1234AB,M,1
> Thomas,Rooijakkers,06-12-1874,1234AB,F,-2
> Thomas,Rooijakkers,09-01-1830,1234AB,M,-3
> Thomas,Someone-else,01-01-1873,6789CD,M,-4
> Victor,Li,09-01-1823,6231LI,M,-5
> Bart,Kamphoorst,07-06-1872,3412CD,M,6
> Michiel,Marcus,06-05-1874,1382SH,M,2
> Tariq,Bontekoe,24-12-1873,8394HG,M,-6
> ```

> `data/player_2.csv`
>
> ```csv
> first_name,last_name,date_of_birth,zip6_code,gender_at_birth,correct_match_B
> Michiel,Marcus,06-05-1874,1234AB,M,2
> Thomas,Rooijakkers,09-01-1874,8972ZX,M,-1
> Thomas,Rooijakkers,09-01-1874,1234AB,M,1
> Thomas,Rooijakkers,06-12-1874,1234AB,M,3
> Thomas,Rooijakkers,17-02-1876,5634AB,M,5
> Thomas,Rooijakkers,09-01-1874,7534CD,M,4
> Bart,Kamphorst,06-06-1872,3412CD,M,6
> Bart,Who,06-12-1875,3231CD,M,-2
> Nicole,Gervasoni,30-01-1877,3411AS,F,-3
> ```

`example_usage.py`

```python
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

DATA_DIR = Path(__file__).parent / "data"


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
```

Run three separate instances specifying the players:

```console
$ python example_usage.py -p Alice
$ python example_usage.py -p Bob
$ python example_usage.py -p Henri
```

This protocol can be run with any number of data parties that is greater than or equal to 2. A helper party (Henri) is always required.

Combining the shares should result in (possibly different order):

```
[
  [1 1]
  [3 3]
  [6 6]
  [5 5]
  [2 2]
  [4 4]
]
```

</details>

# Secure Approximate Matching (Fuzzy Matching)

Within the [LANCELOT](https://www.tno.nl/en/about-tno/news/2021/11/lancelot-new-collaboration-between-iknl-and-tno-to-enable-privacy-preserving-analyses-on-cancer-related-data/) project, a collaboration between TNO, IKNL and Janssen, TNO developed and implemented secure approximate matching solutions.

We wish to match two (or more) datasets based on a set of identifying attributes per records, which are available in all datasets. The following attributes were mentioned to be (often) available by the LANCELOT partners:

- First name
- Last name
- Date of birth
- Gender-at-birth
- Place of birth
- Postal code

There a couple of challenges we wish to resolve

- There is no identifier
- There is no unique pseudo-identifier
- We cannot reveal attributes of the records
- There are mistakes in the attributes
- Typos
- Slightly mismatching data

The proposed solution can cope with one-off (or more, based on configurable parameters) mistakes in date of birth, and zip2 postal code as well as for mistakes in names (speech-to-text, a subset of possible typos).

## Overview of protocol/algorithm steps

### Setting

We consider a setting with (at least) two data owners and a single helper party. The helper party learns the size of the data sets and the size of the intersection, but nothing else. The data owners learn the size of the intersection, the number of features in the other data set(s) and end up with an additive sharing of the intersection. More specifically, for every attribute (data entry) for every matched record a distributed additive share is constructed. These additive shares serve the purpose of secure input for follow-up analysis, where they can be recombined in the encrypted domain to perform computations on the secure inner join.

### High-over steps

We first aim to find exact matches securely, i.e., matches where all of the identifying attributes match exactly. On the unmatched remaining data we perform an approximate matching protocol. In summary,

1. Exact matching,
2. Approximate matching (on unmatched remaining data).

#### Exact matching

As described in https://www.tno.nl/en/tno-insights/articles/identifying-highrisk-factors-diseases and https://youtu.be/hvBb80eXuZg. Steps data owners,

1. Concatenate identifying features into single identifier
2. Hash concatenation using salted hashing
3. Encrypt your data features (those that you wish to use for follow-up analysis) using partially homomorphic encryption (Paillier scheme of which you do own the secret key)
4. Share the hashes along with the encrypted data with the helper party

7) Generate a masking table[^masking] of the size of the matched data
8) Encrypt the masking using the public key of the other data owner(s)
9) Share encrypted masking with the helper party

12. Decrypt masked encryption of own data to get your masked, matched data
13. The plaintext masking combined with the masked data equals the plaintext inner join (additive sharing between data parties)

[^masking]: A masking table is a table consisting of randomly generated numbers, which is used to mask (securely hide) the original values of another table of the same dimensions.

Steps helper:

5. Match hashes of data owners (and thereby find the encrypted inner join)
6. Communicate the size of the intersection to data parties

10) Subtract masking from the computed inner join
11) Share masked encryptions of the inner join with there respective data owners

Note: salts are shared between data owners, but **not** with the helper party.

#### Approximate matching

To cope with mismatches due to minor mistakes in the data we introduced some additional substeps in the above steps 2. and 5.

2. Prepare hashes  
   a) Hash concatenation using salted hashing  
   b) Determine phonetic encoding of the concatenation of first name and last name  
   c) Add features that should match exactly to this encoding  
   d) Hash encoding using salted hashing  
   e) Encode date of birth and 2-zipcode into a (salted) locality-sensitive hash (LSH)  
   f) Mask locality sensitive hashes (using a seeded random bit mask, all data owners should know and use the same seed)

5) Determine matching  
   a) Match exact hashes of data owners  
   b) Match encoded (approximate) hashes of data owners on remaining data  
   c) Filter encoded matches, which are not necessarily unique, by computing their respective LSH distance  
   d) Pick the "best" matches which satisfy preset constraints on the distances (thresholds)  
   e) Combine the computed exact and approximate matches into a single (encrypted) inner join

## Assumptions, choices and tackled challenges

### Phonetic encoding

Applying phonetic encodings on names can be viewed as transforming your data from a high-dimensional space to a space of smaller dimension. In this process, we lose information. This loss of information is beneficial, as this could add in finding similar names. More specifically by transforming written names to their phonetic representation, we can find similarly names based on pronunciation/speech.

An important things to note is that this helps us to match data that contains misinterpreted/misspelled names by the individual that entered the data (speech-to-text mistakes), and some (but not all) typo's. Typo's are only filtered when they do not influence the pronunciation according to the phonetic algorithm. E.g. "Jan Janssen" > 'YANYANSN', "Jan Jahnssen" > 'YANYANSN', 'Jan Jansen' > 'YANYANSN', "Jan Jasnsen" > 'YANYASNSN'.

When applying an phonetic algorithm, we have to decide what the base language should be. It seems to make most sense, as we want to catch speech-to-text errors here, that one should use the mother tongue of the individual that entered the data.

For our use-case, it is the most likely mother tongue of the person entering the data. As there is no State of the Art phonetic encoding algorithm available for the Dutch language, or Dutch names, we decided that `phonem`[^phonem] would be the best option. The phonem algorithm specifically targets German(ic) names and produces human-interpretable encodings. (Dutch is a West Germanic language)

[^phonem]: [Wilde, Georg ; Meyer, Carsten: Doppelgänger gesucht - Ein Programm fur kontext-sensitive phonetische Textumwandlung. In: ct Magazin fur Computer & Technik 25 (1988)](http://web.archive.org/web/20070209153423/http://uni-koeln.de/phil-fak/phonetik/Lehre/MA-Arbeiten/magister_wilz.pdf)

This encoding needs to be hashed (similar to the salted hashing procedure in the exact matching part), to ensure the same level of security as in the exact matching case.

### Locality-sensitive hashing

To use locality-sensitive hashing, one needs to define a mapping from input to hash in such a way that close inputs remain close in the hash. This requires context-specific reasoning about the term "close" and what close means in terms of our input type(s).

For encoding a date-of-birth we consider its elements (day, month, year) separately. Specifically, we **chose/decided** to go for a simple mapping to a modular space per element. In the future, we could consider constructing a more advanced mapping, by e.g. taking into account the days that are actually possible within a month, (e.g. only 1-28 for February).

We map date of birth (dd-mm-yy) as follows:

- Month (mm) is mapped to modular space (mod 12)
- Day (dd) is mapped to modular space (mod 31)
- Year (yy) is mapped to modular space (mod 100)

![Mapping month and day [torus]](https://raw.githubusercontent.com/TNO-MPC/protocols.secure_inner_join/main/assets/month_day_torus.svg)
![Mapping year](https://raw.githubusercontent.com/TNO-MPC/protocols.secure_inner_join/main/assets/year.svg)

For 2-zip regional codes (i.e., all postal codes with the same first two digits) we made the assumption that people often move within the same (or neighbouring/close) regions. Based on the distribution of 2-zip regions in the Netherlands we **chose/decided** to go for a simple mapping to a line [10-99]. Again, note that more accurate, complicated, mappings could improve the performance of the matching. One could even consider using statistics to base a mapping on (e.g. the likelihood of moving between regions?).

To conclude, for Dutch zipcode/postal code (1234 AB):

- 2-zip code (12) is mapped to a line $`[10-99]`$

![2-zip code](https://raw.githubusercontent.com/TNO-MPC/protocols.secure_inner_join/main/assets/zip2.svg)

To make sure that the produced LSH hashes are secure, we need to mask these with a random (collaboratively determined) bit vector, which is unknown to the helper party. If we would not do this, one (the helper) could learn/guess plaintext values based on the distributions within the hashes.

## Configurable parameters + defaults

For approximate matching we introduced some configurable parameters. More specifically one can set the thresholds for "closeness" of LSH hashes on various levels. There is an option to set an overall threshold function on the weighted distance scores. The intuition behind those scores is that if two input values of the LSH hashing algorithm are one-off (in one of its attributes), the weighted overall distance will be approximately $`1`$ (converges to $`1`$ if the number of hyperplanes goes to infinity). Depending on the (number of and the) randomly picked hyperplanes the distance might deviate a bit from $`1`$. If you wish to support one-offs it makes sense to set the threshold to $`1.5`$, if you are happy to allow two one-off, or one two-off, set it to $`2.5`$, etc. The default threshold function validates that all attributes differ at most $`1.5`$ and the sum of all differences does not surpass $`4.5`$.

One can also configure the number of slices/hyperplanes to construct for LSH, trivially this number should be equal for all data owners. A default value of $`2000`$ seems to result in sufficient accuracy.
