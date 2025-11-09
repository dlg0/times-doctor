import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.9-3.10
from typing import Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """Unified configuration for times-doctor.
    
    Priority (highest to lowest):
    1. CLI arguments (set at runtime)
    2. Environment variables
    3. config.toml file
    4. Default values
    """
    
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    amp_api_key: str | None = None
    amp_cli: str = "amp"
    
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    anthropic_temperature: float = 0.2
    
    log_dir: Path = field(default_factory=lambda: Path.cwd() / "_llm_calls")
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "times_doctor_out")
    
    gams_path: str | None = None
    cplex_threads: int | None = None
    
    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """Load configuration from all sources with proper priority.
        
        Args:
            config_path: Path to config.toml file (defaults to cwd/config.toml)
        
        Returns:
            Config instance with merged settings
        """
        config_data: dict[str, Any] = {}
        
        if config_path is None:
            config_path = Path.cwd() / "config.toml"
        
        if config_path.exists():
            config_data = cls._load_toml(config_path)
        
        config_data = cls._merge_env(config_data)
        
        return cls(**config_data)
    
    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        result: dict[str, Any] = {}
        
        if "llm" in data:
            llm = data["llm"]
            if "openai_api_key" in llm:
                result["openai_api_key"] = llm["openai_api_key"]
            if "anthropic_api_key" in llm:
                result["anthropic_api_key"] = llm["anthropic_api_key"]
            if "amp_api_key" in llm:
                result["amp_api_key"] = llm["amp_api_key"]
            if "amp_cli" in llm:
                result["amp_cli"] = llm["amp_cli"]
            if "openai_model" in llm:
                result["openai_model"] = llm["openai_model"]
            if "openai_temperature" in llm:
                result["openai_temperature"] = float(llm["openai_temperature"])
            if "anthropic_temperature" in llm:
                result["anthropic_temperature"] = float(llm["anthropic_temperature"])
        
        if "paths" in data:
            paths = data["paths"]
            if "log_dir" in paths:
                result["log_dir"] = Path(paths["log_dir"])
            if "output_dir" in paths:
                result["output_dir"] = Path(paths["output_dir"])
        
        if "gams" in data:
            gams = data["gams"]
            if "gams_path" in gams:
                result["gams_path"] = gams["gams_path"]
            if "cplex_threads" in gams:
                result["cplex_threads"] = int(gams["cplex_threads"])
        
        return result
    
    @staticmethod
    def _merge_env(config_data: dict[str, Any]) -> dict[str, Any]:
        """Merge environment variables into config (env takes priority over file)."""
        from dotenv import load_dotenv
        
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
        
        env_mappings = {
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "AMP_API_KEY": "amp_api_key",
            "AMP_CLI": "amp_cli",
            "OPENAI_MODEL": "openai_model",
            "OPENAI_TEMPERATURE": "openai_temperature",
            "ANTHROPIC_TEMPERATURE": "anthropic_temperature",
            "TIMES_DOCTOR_LOG_DIR": "log_dir",
            "TIMES_DOCTOR_OUTPUT_DIR": "output_dir",
            "GAMS_PATH": "gams_path",
            "CPLEX_THREADS": "cplex_threads",
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if config_key in ["openai_temperature", "anthropic_temperature"]:
                    config_data[config_key] = float(value)
                elif config_key == "cplex_threads":
                    config_data[config_key] = int(value)
                elif config_key in ["log_dir", "output_dir"]:
                    config_data[config_key] = Path(value)
                else:
                    config_data[config_key] = value
        
        return config_data
    
    def merge_cli_args(self, **kwargs) -> "Config":
        """Create new Config with CLI arguments merged in (CLI args have highest priority).
        
        Args:
            **kwargs: CLI arguments to override config values
        
        Returns:
            New Config instance with CLI args merged
        """
        updates = {k: v for k, v in kwargs.items() if v is not None}
        
        from dataclasses import replace
        return replace(self, **updates)


_global_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance, loading it if not already loaded."""
    global _global_config
    if _global_config is None:
        _global_config = Config.load()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _global_config
    _global_config = config
