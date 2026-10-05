"""The `jannus` CLI must print the 1.50 deprecation warning on every invocation (v1.50.1)."""

from __future__ import annotations

import pytest

from jannus.cli.main import DEPRECATION_WARNING, main


def _assert_warned(err: str) -> None:
    assert DEPRECATION_WARNING in err
    assert "DEPRECATED" in err and "audit/1.50-defects" in err and "JANNUS 1.60" in err


def test_warning_on_bare_invocation(capsys):
    assert main([]) == 0
    _assert_warned(capsys.readouterr().err)


@pytest.mark.parametrize("argv", [["--help"], ["--version"]])
def test_warning_precedes_argparse_exits(capsys, argv):
    with pytest.raises(SystemExit):
        main(argv)
    _assert_warned(capsys.readouterr().err)


def test_warning_is_stderr_only_and_ascii(capsys):
    main([])
    captured = capsys.readouterr()
    assert DEPRECATION_WARNING not in captured.out
    DEPRECATION_WARNING.encode("ascii")  # must survive a non-UTF-8 console
