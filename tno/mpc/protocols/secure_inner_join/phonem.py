"""
Implementation of phonem

Based on:
Wilde, Georg ; Meyer, Carsten: Doppelgänger gesucht - Ein Programm fur
kontext-sensitive phonetische Textumwandlung. In: ct Magazin fur
Computer & Technik 25 (1988)

Inspired by Talisman's implementation:
https://raw.githubusercontent.com/Yomguithereal/talisman/4c8ac7fd9/src/phonetics/german/phonem.js
"""


import itertools
from typing import Dict, Union

SUBSTITUTIONS = {
    "SC": "C",
    "SZ": "C",
    "CZ": "C",
    "TZ": "C",
    "TS": "C",
    "KS": "X",
    "PF": "V",
    "PH": "V",
    "QU": "KW",
    "UE": "Y",
    "AE": "E",
    "OE": "Ö",
    "EI": "AY",
    "EY": "AY",
    "EU": "OY",
    "AU": "A§",
    "OU": "§",
}

TRANSLATIONS: Dict[str, Union[int, str, None]] = {
    "Z": "C",
    "K": "C",
    "G": "C",
    "Q": "C",
    "Ç": "C",
    "Ñ": "N",
    "ß": "S",
    "F": "V",
    "W": "V",
    "P": "B",
    "T": "D",
    "Á": "A",
    "À": "A",
    "Â": "A",
    "Ã": "A",
    "Å": "A",
    "Ä": "E",
    "Æ": "E",
    "É": "E",
    "È": "E",
    "Ê": "E",
    "Ë": "E",
    "I": "Y",
    "J": "Y",
    "Ì": "Y",
    "Í": "Y",
    "Î": "Y",
    "Ï": "Y",
    "Ü": "Y",
    "Ý": "Y",
    "§": "U",
    "Ú": "U",
    "Ù": "U",
    "Û": "U",
    "Ô": "O",
    "Ò": "O",
    "Ó": "O",
    "Õ": "O",
    "Ø": "Ö",
}

ACCEPTABLE_LETTERS = "ABCDLMNORSUVWXYÖ"
WHITELIST_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ -")
WHITELIST_LETTERS.update(TRANSLATIONS.keys() | SUBSTITUTIONS.keys())
WHITELIST_LETTERS.discard(
    "§"
)  # Do not allow this non-printable character in original name


def phonem_encode(string: str) -> str:
    """
    Implements the phonem phonetic algorithm suitable for Germanic names.

    Based on:
    Wilde, Georg ; Meyer, Carsten: Doppelgänger gesucht - Ein Programm fur
    kontext-sensitive phonetische Textumwandlung. In: ct Magazin fur
    Computer & Technik 25 (1988)

    Inspired by Talisman's implementation:
    https://raw.githubusercontent.com/Yomguithereal/talisman/4c8ac7fd9/src/phonetics/german/phonem.js

    Spaces and dashes (' ', '-'), are considered as word/name separators.

    :param string: string to encode
    :return: encoded string
    """
    # Strip all but acceptable letters and separators
    string = "".join(filter(lambda _: _ in WHITELIST_LETTERS, string.upper()))

    # Substitute
    for match, replacement in SUBSTITUTIONS.items():
        string = string.replace(match, replacement)

    # Translate
    string = string.translate(str.maketrans(TRANSLATIONS))

    # Squeeze and filter unacceptable letters
    output = "".join(
        char for char, _ in itertools.groupby(string) if char in ACCEPTABLE_LETTERS
    )

    return output
