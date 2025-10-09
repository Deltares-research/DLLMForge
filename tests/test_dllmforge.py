from dllmforge.__version__ import __version__
import pytest


class TestVersion:

    def test_version(self):
        assert __version__ == "0.1.0"
