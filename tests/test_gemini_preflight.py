from scripts.check_gemini_api import interpret_response, mask_key


def test_gemini_key_is_masked_in_logs():
    masked = mask_key("AIzaSyExampleSecret123456")
    assert masked.startswith("AIza")
    assert masked.endswith("3456")
    assert "ExampleSecret" not in masked


def test_gemini_http_200_is_usable():
    usable, message = interpret_response(200, '{"candidates": []}')
    assert usable is True
    assert message == "OK"


def test_gemini_auth_and_quota_failures_are_actionable():
    usable_403, message_403 = interpret_response(
        403, '{"error":{"message":"API key not valid"}}'
    )
    usable_429, message_429 = interpret_response(429, "{}")
    assert usable_403 is False
    assert "API key" in message_403
    assert usable_429 is False
    assert "quota" in message_429
