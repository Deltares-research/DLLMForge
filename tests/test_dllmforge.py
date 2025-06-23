from dllmforge.__version__ import version
import pytest


class TestVersion:

    def test_version(self):
        assert version == "0.1.0"
