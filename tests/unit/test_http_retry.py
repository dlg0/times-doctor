"""Tests for HTTP client timeout and retry logic."""

from unittest.mock import Mock, patch

import pytest

from src.times_doctor.core.exceptions import LlmError
from src.times_doctor.core.http_client import get_http_client, make_request_with_retry


def test_get_http_client_creates_client_with_timeouts():
    """Test that get_http_client creates httpx client with proper timeouts."""
    client = get_http_client()

    assert client is not None
    assert client.timeout.connect == 10.0
    assert client.timeout.read == 300.0
    assert client.timeout.write == 10.0
    assert client.timeout.pool == 10.0
    assert client.follow_redirects is True


@patch("src.times_doctor.core.http_client.get_http_client")
def test_make_request_with_retry_success(mock_get_client):
    """Test successful request on first try."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_client.request = Mock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    response = make_request_with_retry("POST", "https://api.example.com/test")

    assert response == mock_response
    assert mock_client.request.call_count == 1


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_429_rate_limit(mock_sleep, mock_get_client):
    """Test retry on 429 rate limit with Retry-After header."""
    mock_client = Mock()

    # First two calls return 429, third succeeds
    mock_response_429 = Mock()
    mock_response_429.status_code = 429
    mock_response_429.headers = {"Retry-After": "2"}

    mock_response_200 = Mock()
    mock_response_200.status_code = 200
    mock_response_200.raise_for_status = Mock()

    mock_client.request = Mock(
        side_effect=[mock_response_429, mock_response_429, mock_response_200]
    )
    mock_get_client.return_value = mock_client

    response = make_request_with_retry("POST", "https://api.example.com/test", max_retries=3)

    assert response == mock_response_200
    assert mock_client.request.call_count == 3
    # Should sleep with jitter (2s + jitter)
    assert mock_sleep.call_count == 2


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_429_exhausted(mock_sleep, mock_get_client):
    """Test that 429 retries are eventually exhausted."""
    mock_client = Mock()

    mock_response_429 = Mock()
    mock_response_429.status_code = 429
    mock_response_429.headers = {"Retry-After": "1"}

    mock_client.request = Mock(return_value=mock_response_429)
    mock_get_client.return_value = mock_client

    with pytest.raises(LlmError, match="Rate limited after 3 retries"):
        make_request_with_retry("POST", "https://api.example.com/test", max_retries=3)

    assert mock_client.request.call_count == 3


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_5xx_server_error(mock_sleep, mock_get_client):
    """Test retry on 5xx server errors with exponential backoff."""
    mock_client = Mock()

    # First call 500, second call 503, third succeeds
    mock_response_500 = Mock()
    mock_response_500.status_code = 500

    mock_response_503 = Mock()
    mock_response_503.status_code = 503

    mock_response_200 = Mock()
    mock_response_200.status_code = 200
    mock_response_200.raise_for_status = Mock()

    mock_client.request = Mock(
        side_effect=[mock_response_500, mock_response_503, mock_response_200]
    )
    mock_get_client.return_value = mock_client

    response = make_request_with_retry(
        "POST", "https://api.example.com/test", max_retries=3, backoff_factor=2.0
    )

    assert response == mock_response_200
    assert mock_client.request.call_count == 3
    assert mock_sleep.call_count == 2

    # Check exponential backoff (with jitter tolerance)
    # First retry: ~2^0 = 1s, second retry: ~2^1 = 2s (with ±25% jitter)
    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
    assert 0.75 <= sleep_calls[0] <= 1.25  # 1s ± 25%
    assert 1.5 <= sleep_calls[1] <= 2.5  # 2s ± 25%


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_5xx_exhausted(mock_sleep, mock_get_client):
    """Test that 5xx retries are eventually exhausted."""
    mock_client = Mock()

    mock_response_500 = Mock()
    mock_response_500.status_code = 500

    mock_client.request = Mock(return_value=mock_response_500)
    mock_get_client.return_value = mock_client

    with pytest.raises(LlmError, match="Server error 500 after 3 retries"):
        make_request_with_retry("POST", "https://api.example.com/test", max_retries=3)

    assert mock_client.request.call_count == 3


@patch("src.times_doctor.core.http_client.get_http_client")
def test_make_request_with_retry_4xx_no_retry(mock_get_client):
    """Test that 4xx client errors are not retried (except 429)."""
    mock_client = Mock()

    mock_response_400 = Mock()
    mock_response_400.status_code = 400
    mock_response_400.text = "Bad request"

    mock_client.request = Mock(return_value=mock_response_400)
    mock_get_client.return_value = mock_client

    with pytest.raises(LlmError, match="Client error 400"):
        make_request_with_retry("POST", "https://api.example.com/test")

    # Should only try once, no retries for 4xx
    assert mock_client.request.call_count == 1


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_timeout(mock_sleep, mock_get_client):
    """Test retry on timeout errors."""
    import httpx

    mock_client = Mock()

    # First two calls timeout, third succeeds
    mock_response_200 = Mock()
    mock_response_200.status_code = 200
    mock_response_200.raise_for_status = Mock()

    mock_client.request = Mock(
        side_effect=[
            httpx.TimeoutException("Request timeout"),
            httpx.TimeoutException("Request timeout"),
            mock_response_200,
        ]
    )
    mock_get_client.return_value = mock_client

    response = make_request_with_retry("POST", "https://api.example.com/test", max_retries=3)

    assert response == mock_response_200
    assert mock_client.request.call_count == 3
    assert mock_sleep.call_count == 2


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_timeout_exhausted(mock_sleep, mock_get_client):
    """Test that timeout retries are eventually exhausted."""
    import httpx

    mock_client = Mock()
    mock_client.request = Mock(side_effect=httpx.TimeoutException("Request timeout"))
    mock_get_client.return_value = mock_client

    with pytest.raises(LlmError, match="Request timeout after 3 retries"):
        make_request_with_retry("POST", "https://api.example.com/test", max_retries=3)

    assert mock_client.request.call_count == 3


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_network_error(mock_sleep, mock_get_client):
    """Test retry on network errors."""
    import httpx

    mock_client = Mock()

    # First call network error, second succeeds
    mock_response_200 = Mock()
    mock_response_200.status_code = 200
    mock_response_200.raise_for_status = Mock()

    mock_client.request = Mock(
        side_effect=[httpx.NetworkError("Connection failed"), mock_response_200]
    )
    mock_get_client.return_value = mock_client

    response = make_request_with_retry("POST", "https://api.example.com/test", max_retries=3)

    assert response == mock_response_200
    assert mock_client.request.call_count == 2
    assert mock_sleep.call_count == 1


@patch("src.times_doctor.core.http_client.get_http_client")
@patch("time.sleep")
def test_make_request_with_retry_jitter_prevents_thundering_herd(mock_sleep, mock_get_client):
    """Test that jitter is applied to retry delays to prevent thundering herd."""

    mock_client = Mock()

    # Simulate multiple retry scenarios
    mock_response_500 = Mock()
    mock_response_500.status_code = 500

    mock_response_200 = Mock()
    mock_response_200.status_code = 200
    mock_response_200.raise_for_status = Mock()

    mock_client.request = Mock(side_effect=[mock_response_500, mock_response_200])
    mock_get_client.return_value = mock_client

    # Run multiple times to verify jitter varies
    sleep_times = []
    for _ in range(5):
        mock_sleep.reset_mock()
        mock_client.request = Mock(side_effect=[mock_response_500, mock_response_200])
        make_request_with_retry(
            "POST", "https://api.example.com/test", max_retries=3, backoff_factor=2.0
        )

        if mock_sleep.call_count > 0:
            sleep_times.append(mock_sleep.call_args[0][0])

    # With jitter, sleep times should not all be identical
    # (With ±25% jitter on 1s base, should have variance)
    assert len(set(sleep_times)) > 1, "Jitter should produce different sleep times"
