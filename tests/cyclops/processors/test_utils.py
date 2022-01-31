"""Test utils module for processors."""


from cyclops.processors.utils import count_occurrences, normalize_special_characters


def test_count_occurrences():
    """Test count_occurrences fn."""
    test_case1 = ["kobe", "jordan", "magic", "durant", "kobe", "magic", "kobe"]
    test_case2 = [1, 1, 2, 4, 9]

    counts = count_occurrences(test_case1)
    assert counts[0][0] == "kobe"
    assert counts[1][0] == "magic"
    counts = count_occurrences(test_case2)
    assert counts[0][0] == 1


def test_normalize_special_characters():
    """Test normalize_special_characters fn."""
    test_input = "test% result+ & g/mol #2"
    normalized = normalize_special_characters(test_input)
    assert normalized == "test_percent_result_plus_and_g_per_mol_number_2"
