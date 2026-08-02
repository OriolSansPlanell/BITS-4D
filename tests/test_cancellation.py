import pytest

from utils.cancellation import CancellationToken, OperationCancelled


def test_cancellation_token_is_cooperative():
    token = CancellationToken()
    token.raise_if_cancelled()
    token.cancel()
    assert token.is_cancelled
    with pytest.raises(OperationCancelled):
        token.raise_if_cancelled()
