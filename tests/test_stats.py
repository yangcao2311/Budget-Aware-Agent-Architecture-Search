import math

from hbws.stats import binomial_exact_upper


def test_binomial_exact_upper_matches_dcert_values():
    assert math.isclose(binomial_exact_upper(22, 835),
                        0.03740991490836185, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(binomial_exact_upper(2, 835),
                        0.007520500974103121, rel_tol=0, abs_tol=1e-12)


def test_binomial_exact_upper_zero_events():
    assert math.isclose(binomial_exact_upper(0, 59),
                        1 - 0.05 ** (1 / 59), rel_tol=0, abs_tol=1e-12)
