"""
Tests for the phonetic algorithm
"""
import pytest

from tno.mpc.protocols.secure_inner_join.phonem import phonem_encode

correct_pairs = [
    ("", ""),
    ("müller", "MYLR"),
    ("Linda Achternaam", "LYNDAACDRNAM"),
    ("LindaAchternaam", "LYNDACDRNAM"),
    ("Linda-Achternaam", "LYNDAACDRNAM"),
    ("Linda&Achternaam", "LYNDACDRNAM"),
    ("Rooijakkers", "ROYACRS"),
    ("Thomas", "DOMAS"),
    ("ThOmAsRooijAkkErs", "DOMASROYACRS"),
    ("Thomas  _=(*-#$%^&;:`  /\\?!@ '\"    [](){}|  Rooijakkers", "DOMASROYACRS"),
    ("Thomas               Rooijakkers", "DOMASROYACRS"),
    (
        "Thom994563()(6as4  _=(*-#$%^&;:`  /\?!@ '\"    [](){}|  Ro*(o00ij2#$akkers",
        "DOMASROYACRS",
    ),
    ("Thomas Rooijakkers", "DOMASROYACRS"),
    ("schmidt", "CMYD"),
    ("schneider", "CNAYDR"),
    ("fischer", "VYCR"),
    ("weber", "VBR"),
    ("meyer", "MAYR"),
    ("wagner", "VACNR"),
    ("schulz", "CULC"),
    ("becker", "BCR"),
    ("hoffmann", "OVMAN"),
    ("schäfer", "CVR"),
    ("mair", "MAYR"),
    ("bäker", "BCR"),
    ("schaeffer", "CVR"),
    ("computer", "COMBUDR"),
    ("pfeifer", "VAYVR"),
    ("pfeiffer", "VAYVR"),
]


@pytest.mark.parametrize("input_string,expected_output", correct_pairs)
def test_phonem_encode(input_string: str, expected_output: str) -> None:
    """
    Validates the correctness of the phonetic encoding
    :param input_string: input string to test
    :param expected_output: the expected output from the phonem algorithm
    """
    assert phonem_encode(input_string) == expected_output
