"""
Vietnamese Food Assistant - Configuration Management

This module handles configuration for model switching, environment variables,
and system settings. Supports flexible model selection between Vistral and Qwen.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    base_url: str
    model_id: str
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class DatabaseConfig:
    """Configuration for ChromaDB and menu database."""
    chroma_persist_directory: str = "data/chroma_db"
    collection_name: str = "vietnamese_menu"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fine_tuned_model_path: Optional[str] = "models/focused-fruit-tea-model"  # Focused model for fruit vs tea distinction
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class Config:
    """
    Main configuration class for Vietnamese Food Assistant.
    
    Handles model switching, environment variables, and system settings.
    Supports both Vistral and Qwen models with flexible configuration.
    """
    
    # Model Configuration
    current_model: str = "vistral"  # Default to Vistral
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "vistral": ModelConfig(
            name="vistral",
            base_url="http://localhost:11434",  # Ollama default
            model_id="vistral:latest",
            temperature=0.1,
            max_tokens=1000
        ),
        "qwen": ModelConfig(
            name="qwen",
            base_url="http://localhost:11434",  # Ollama default
            model_id="qwen2.5:latest",
            temperature=0.1,
            max_tokens=1000
        )
    })
    
    # Database Configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Data Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    menu_data_path: Path = field(default_factory=lambda: Path("data/menu.csv"))
    correction_patterns_path: Path = field(default_factory=lambda: Path("data/correction_patterns.json"))
    constraint_rules_path: Path = field(default_factory=lambda: Path("data/constraint_rules.json"))
    
    # Session Management
    session_timeout_minutes: int = 30
    max_conversation_history: int = 50
    
    # Processing Settings
    max_retrieved_docs: int = 10
    max_filtered_docs: int = 5
    max_reranked_docs: int = 3
    confidence_threshold: float = 0.7
    
    # Language Processing
    supported_languages: list = field(default_factory=lambda: ["vi", "en"])
    default_language: str = "vi"
    
    # Error Handling
    max_retries: int = 3
    fallback_to_original_text: bool = True
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        self._load_from_environment()
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration values from environment variables."""
        # Model selection
        self.current_model = os.getenv("VFA_MODEL", self.current_model)
        
        # Model endpoints
        vistral_url = os.getenv("VFA_VISTRAL_URL", self.models["vistral"].base_url)
        qwen_url = os.getenv("VFA_QWEN_URL", self.models["qwen"].base_url)
        
        self.models["vistral"].base_url = vistral_url
        self.models["qwen"].base_url = qwen_url
        
        # Model IDs
        self.models["vistral"].model_id = os.getenv("VFA_VISTRAL_MODEL", self.models["vistral"].model_id)
        self.models["qwen"].model_id = os.getenv("VFA_QWEN_MODEL", self.models["qwen"].model_id)
        
        # Database settings
        self.database.chroma_persist_directory = os.getenv(
            "VFA_CHROMA_DIR", 
            self.database.chroma_persist_directory
        )
        
        # Data paths
        data_dir = os.getenv("VFA_DATA_DIR")
        if data_dir:
            self.data_dir = Path(data_dir)
            self.menu_data_path = self.data_dir / "menu.csv"
            self.correction_patterns_path = self.data_dir / "correction_patterns.json"
            self.constraint_rules_path = self.data_dir / "constraint_rules.json"
        
        # Processing settings
        self.confidence_threshold = float(os.getenv("VFA_CONFIDENCE_THRESHOLD", self.confidence_threshold))
        self.max_retrieved_docs = int(os.getenv("VFA_MAX_RETRIEVED_DOCS", self.max_retrieved_docs))
        
        # Session settings
        self.session_timeout_minutes = int(os.getenv("VFA_SESSION_TIMEOUT", self.session_timeout_minutes))
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.current_model not in self.models:
            raise ValueError(f"Invalid model '{self.current_model}'. Available models: {list(self.models.keys())}")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.max_retrieved_docs <= 0:
            raise ValueError("max_retrieved_docs must be positive")
    
    def get_current_model_config(self) -> ModelConfig:
        """Get configuration for the currently selected model."""
        return self.models[self.current_model]
    
    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'. Available models: {list(self.models.keys())}")
        
        self.current_model = model_name
    
    def add_model(self, name: str, config: ModelConfig) -> None:
        """Add a new model configuration."""
        self.models[name] = config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "current_model": self.current_model,
            "models": {name: {
                "name": config.name,
                "base_url": config.base_url,
                "model_id": config.model_id,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout
            } for name, config in self.models.items()},
            "database": {
                "chroma_persist_directory": self.database.chroma_persist_directory,
                "collection_name": self.database.collection_name,
                "embedding_model": self.database.embedding_model
            },
            "data_paths": {
                "data_dir": str(self.data_dir),
                "menu_data_path": str(self.menu_data_path),
                "correction_patterns_path": str(self.correction_patterns_path),
                "constraint_rules_path": str(self.constraint_rules_path)
            },
            "processing": {
                "max_retrieved_docs": self.max_retrieved_docs,
                "max_filtered_docs": self.max_filtered_docs,
                "max_reranked_docs": self.max_reranked_docs,
                "confidence_threshold": self.confidence_threshold
            },
            "session": {
                "timeout_minutes": self.session_timeout_minutes,
                "max_conversation_history": self.max_conversation_history
            }
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    config = Config()
    return config