import os
import tempfile
from pathlib import Path
import pytest
from times_doctor.core.config import Config


def test_config_defaults():
    config = Config()
    assert config.openai_model == "gpt-4o-mini"
    assert config.openai_temperature == 0.2
    assert config.anthropic_temperature == 0.2
    assert config.amp_cli == "amp"


def test_config_load_toml(tmp_path, monkeypatch):
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AMP_API_KEY", "OPENAI_MODEL", "AMP_CLI"]:
        monkeypatch.delenv(key, raising=False)
    
    monkeypatch.chdir(tmp_path)
    
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[llm]
openai_api_key = "test-key-123"
openai_model = "gpt-4o"
openai_temperature = 0.5
amp_cli = "custom-amp"

[paths]
log_dir = "/custom/log/dir"
output_dir = "/custom/output/dir"

[gams]
gams_path = "/opt/gams/gams"
cplex_threads = 8
""")
    
    config = Config.load(config_path=config_file)
    
    assert config.openai_api_key == "test-key-123"
    assert config.openai_model == "gpt-4o"
    assert config.openai_temperature == 0.5
    assert config.amp_cli == "custom-amp"
    assert config.log_dir == Path("/custom/log/dir")
    assert config.output_dir == Path("/custom/output/dir")
    assert config.gams_path == "/opt/gams/gams"
    assert config.cplex_threads == 8


def test_config_env_override_toml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[llm]
openai_api_key = "from-toml"
openai_model = "gpt-4o"
""")
    
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")
    
    config = Config.load(config_path=config_file)
    
    assert config.openai_api_key == "from-env"
    assert config.openai_model == "gpt-5-mini"


def test_config_merge_cli_args():
    config = Config(openai_model="gpt-4o-mini", cplex_threads=4)
    
    updated = config.merge_cli_args(openai_model="gpt-5-mini", cplex_threads=8)
    
    assert updated.openai_model == "gpt-5-mini"
    assert updated.cplex_threads == 8
    
    assert config.openai_model == "gpt-4o-mini"
    assert config.cplex_threads == 4


def test_config_merge_cli_args_ignores_none():
    config = Config(openai_model="gpt-4o-mini", cplex_threads=4)
    
    updated = config.merge_cli_args(openai_model="gpt-5-mini", cplex_threads=None)
    
    assert updated.openai_model == "gpt-5-mini"
    assert updated.cplex_threads == 4


def test_config_env_with_dotenv(tmp_path, monkeypatch):
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AMP_API_KEY", "OPENAI_TEMPERATURE"]:
        monkeypatch.delenv(key, raising=False)
    
    monkeypatch.chdir(tmp_path)
    
    env_file = tmp_path / ".env"
    env_file.write_text("""
OPENAI_API_KEY=from-dotenv
ANTHROPIC_API_KEY=sk-ant-test
OPENAI_TEMPERATURE=0.7
""")
    
    config = Config.load()
    
    assert config.openai_api_key == "from-dotenv"
    assert config.anthropic_api_key == "sk-ant-test"
    assert config.openai_temperature == 0.7


def test_config_priority_cli_over_env_over_toml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[llm]
openai_model = "from-toml"
""")
    
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=from-env")
    
    monkeypatch.chdir(tmp_path)
    
    config = Config.load(config_path=config_file)
    assert config.openai_model == "from-env"
    
    updated = config.merge_cli_args(openai_model="from-cli")
    assert updated.openai_model == "from-cli"
