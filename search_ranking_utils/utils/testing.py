def assert_dicts_equal(expected: dict, actual: dict):
    """
    Check that two dicts are identical
    """
    for k in expected:
        if isinstance(expected[k], dict):
            assert_dicts_equal(expected[k], actual[k])
        else:
            assert (
                expected[k] == actual[k]
            ), f"Expected {expected[k]}, got {actual[k]}"
