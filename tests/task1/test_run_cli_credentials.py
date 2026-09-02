from __future__ import annotations

import pytest

import task1.run as run_module


def test_cli_apify_token_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "env_apify")
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini")
    args = run_module.build_parser().parse_args(
        ["--generate-from-run", "run_123", "--apify-api-token", "cli_apify", "--gemini-api-key", "cli_gemini"]
    )
    apify_token, gemini_key = run_module._configure_and_validate_credentials(args)
    assert apify_token == "cli_apify"
    assert gemini_key == "cli_gemini"
    assert run_module.os.getenv("APIFY_API_TOKEN") == "cli_apify"


def test_cli_gemini_key_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "env_apify")
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini")
    args = run_module.build_parser().parse_args(
        ["--generate-from-run", "run_123", "--gemini-api-key", "cli_gemini"]
    )
    apify_token, gemini_key = run_module._configure_and_validate_credentials(args)
    assert apify_token == "env_apify"
    assert gemini_key == "cli_gemini"
    assert run_module.os.getenv("GEMINI_API_KEY") == "cli_gemini"


def test_environment_fallback_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APIFY_API_TOKEN", "env_apify")
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini")
    args = run_module.build_parser().parse_args([])
    apify_token, gemini_key = run_module._configure_and_validate_credentials(args)
    assert apify_token == "env_apify"
    assert gemini_key == "env_gemini"


def test_missing_apify_token_error_for_process_mode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini")
    args = run_module.build_parser().parse_args([])
    with pytest.raises(RuntimeError) as exc:
        run_module._configure_and_validate_credentials(args)
    message = str(exc.value)
    assert "Missing Apify API token" in message
    captured = capsys.readouterr()
    assert "env_gemini" not in message + captured.out + captured.err


def test_missing_gemini_key_error_for_generate_only(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    args = run_module.build_parser().parse_args(["--generate-from-run", "run_123"])
    with pytest.raises(RuntimeError) as exc:
        run_module._configure_and_validate_credentials(args)
    message = str(exc.value)
    assert "Missing Gemini API key" in message
    captured = capsys.readouterr()
    assert "run_123" not in captured.out + captured.err


def test_full_pipeline_parses_with_both_keys() -> None:
    args = run_module.build_parser().parse_args(
        [
            "--full-pipeline",
            "--apify-api-token",
            "apify_x",
            "--gemini-api-key",
            "gemini_x",
        ]
    )
    assert args.full_pipeline is True
    assert args.apify_api_token == "apify_x"
    assert args.gemini_api_key == "gemini_x"


def test_generate_only_does_not_require_apify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini")
    args = run_module.build_parser().parse_args(["--generate-from-run", "run_123", "--generate-top", "3"])
    apify_token, gemini_key = run_module._configure_and_validate_credentials(args)
    assert apify_token is None
    assert gemini_key == "env_gemini"
