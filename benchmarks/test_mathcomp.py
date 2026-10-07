"""Checker tests for mathcomp.same (AIME/HMMT). Run: python -m pytest benchmarks/test_mathcomp.py
Cases are real model answers from HMMT Nov 2025 that the checker marked wrong before the brace fix."""
import pytest

from mathcomp import same


@pytest.mark.parametrize("pred, gold", [
    (r"6\sqrt3+5\pi", r"5\pi + 6\sqrt{3}"),            # problem 15
    (r"5\pi + 6\sqrt3", r"5\pi + 6\sqrt{3}"),
    (r"\frac{1+\sqrt7}{2}", r"\frac{\sqrt{7} + 1}{2}"),  # problem 19
    (r"\dfrac{1+\sqrt7}{2}", r"\frac{\sqrt{7} + 1}{2}"),
    (r"\sqrt3-1", r"\sqrt{3} - 1"),                      # problem 22
    (r"\frac{3\sqrt5}{7}", r"\frac{3\sqrt{5}}{7}"),      # problem 29
    (r"\dfrac{3\sqrt5}{7}", r"\frac{3\sqrt{5}}{7}"),
    (r"\sqrt 3", r"\sqrt{3}"),
    (r"\sqrt{12}", r"2\sqrt3"),
    (r"\frac12", r"\frac{1}{2}"),
])
def test_equal_answers_in_shorthand(pred, gold):
    assert same(pred, gold)


@pytest.mark.parametrize("pred, gold", [
    (r"\sqrt3", r"\sqrt{5}"),
    (r"6\sqrt3+5\pi", r"5\pi + 6\sqrt{5}"),
    (r"\frac{1+\sqrt7}{3}", r"\frac{\sqrt{7} + 1}{2}"),
    (r"\frac13", r"\frac{1}{2}"),
    (r"1320", r"1305"),
    (None, r"\sqrt{3}"),
    # parse_latex silently stops at an unbraced \sqrt: "6\sqrt3+5\pi" parses as 6 and "3\sqrt5" as 3
    (r"6\sqrt3+5\pi", r"6"),
    (r"3\sqrt5", r"3"),
])
def test_different_answers_stay_wrong(pred, gold):
    assert not same(pred, gold)


@pytest.mark.parametrize("pred, gold", [
    (r"\sqrt{3}", r"\sqrt{3}"),
    (r"\sqrt[3]{2}", r"\sqrt[3]{2}"),
    (r"\frac{1}{2}", r"0.5"),
    (r"8.5", r"\frac{17}{2}"),
])
def test_braced_answers_unchanged(pred, gold):
    assert same(pred, gold)
