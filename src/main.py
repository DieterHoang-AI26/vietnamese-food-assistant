"""
Vietnamese Food Assistant - Main Application Entry Point

This module provides the main application interface with comprehensive error handling,
session recovery, and system resilience features.

Requirements: 7.5 - Session Management and Data Persistence
"""

import logging
import sys
import traceback
import signal
import atexit
import json
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

from .graph.workflow import create_workflow, VietnamFoodAssistantWorkflow
from .config import get_config, Config
from .state import AgentState


class SessionRecoveryManager:
    """
    Manages session persistence and recovery for system resilience.
    
    Handles graceful session recovery after system restarts or failures,
    ensuring no data corruption or cross-session contamination.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".SessionRecovery")
        self.session_dir = Path(config.data_dir) / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Session metadata tracking
        self.active_sessions: Dict[str, datetime] = {}
        self.recovery_log_path = self.session_dir / "recovery.log"
        
        self._load_active_sessions()
    
    def save_session_state(self, session_id: str, state: AgentState) -> bool:
        """
        Persist session state to disk for recovery.
        
        Args:
            session_id: Unique session identifier
            state: Current session state
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            session_file = self.session_dir / f"{session_id}.session"
            
            # Create a serializable version of the state
            serializable_state = self._make_serializable(state)
            
            # Add metadata
            session_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "state": serializable_state,
                "version": "1.0"
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = session_file.with_suffix(".tmp")
            with open(temp_file, 'wb') as f:
                pickle.dump(session_data, f)
            
            temp_file.rename(session_file)
            
            # Update active sessions tracking
            self.active_sessions[session_id] = datetime.now()
            self._save_active_sessions()
            
            self.logger.debug(f"Session {session_id} saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    def load_session_state(self, session_id: str) -> Optional[AgentState]:
        """
        Load session state from disk.
        
        Args:
            session_id: Session identifier to load
            
        Returns:
            Loaded session state or None if not found/corrupted
        """
        try:
            session_file = self.session_dir / f"{session_id}.session"
            
            if not session_file.exists():
                self.logger.debug(f"Session file not found: {session_id}")
                return None
            
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            # Validate session data
            if not self._validate_session_data(session_data):
                self.logger.warning(f"Invalid session data for {session_id}")
                return None
            
            # Check if session is expired
            session_time = datetime.fromisoformat(session_data["timestamp"])
            if self._is_session_expired(session_time):
                self.logger.info(f"Session {session_id} expired, cleaning up")
                self.cleanup_session(session_id)
                return None
            
            # Restore state
            state = self._restore_state(session_data["state"])
            
            self.logger.debug(f"Session {session_id} loaded successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            # Move corrupted file for investigation
            self._quarantine_corrupted_session(session_id)
            return None
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up session data from disk.
        
        Args:
            session_id: Session to clean up
            
        Returns:
            True if cleanup successful
        """
        try:
            session_file = self.session_dir / f"{session_id}.session"
            
            if session_file.exists():
                session_file.unlink()
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self._save_active_sessions()
            
            self.logger.debug(f"Session {session_id} cleaned up")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup session {session_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up all expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        # Check all session files
        for session_file in self.session_dir.glob("*.session"):
            try:
                session_id = session_file.stem
                
                # Load and check expiration
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                
                session_time = datetime.fromisoformat(session_data["timestamp"])
                
                if self._is_session_expired(session_time):
                    self.cleanup_session(session_id)
                    cleaned_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error checking session {session_file}: {e}")
                # Quarantine problematic files
                self._quarantine_corrupted_session(session_file.stem)
        
        self.logger.info(f"Cleaned up {cleaned_count} expired sessions")
        return cleaned_count
    
    def get_active_sessions(self) -> Dict[str, datetime]:
        """Get list of currently active sessions."""
        return self.active_sessions.copy()
    
    def _make_serializable(self, state: AgentState) -> Dict[str, Any]:
        """Convert state to serializable format."""
        serializable = {}
        
        for key, value in state.items():
            try:
                # Handle datetime objects
                if isinstance(value, datetime):
                    serializable[key] = value.isoformat()
                # Handle lists of conversation turns
                elif key == "conversation_history" and isinstance(value, list):
                    serializable[key] = [
                        {
                            "timestamp": turn.get("timestamp", datetime.now()).isoformat() if isinstance(turn.get("timestamp"), datetime) else turn.get("timestamp"),
                            "user_input": turn.get("user_input", ""),
                            "corrected_input": turn.get("corrected_input", ""),
                            "system_response": turn.get("system_response", ""),
                            "mentioned_entities": turn.get("mentioned_entities", [])
                        }
                        for turn in value
                    ]
                # Handle other serializable types
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable[key] = value
                else:
                    # Convert to string representation for complex objects
                    serializable[key] = str(value)
                    
            except Exception as e:
                self.logger.warning(f"Could not serialize state key {key}: {e}")
                serializable[key] = None
        
        return serializable
    
    def _restore_state(self, serializable_state: Dict[str, Any]) -> AgentState:
        """Restore state from serializable format."""
        state = {}
        
        for key, value in serializable_state.items():
            try:
                # Handle conversation history
                if key == "conversation_history" and isinstance(value, list):
                    state[key] = [
                        {
                            "timestamp": datetime.fromisoformat(turn["timestamp"]) if isinstance(turn["timestamp"], str) else turn["timestamp"],
                            "user_input": turn.get("user_input", ""),
                            "corrected_input": turn.get("corrected_input", ""),
                            "system_response": turn.get("system_response", ""),
                            "mentioned_entities": turn.get("mentioned_entities", [])
                        }
                        for turn in value
                    ]
                else:
                    state[key] = value
                    
            except Exception as e:
                self.logger.warning(f"Could not restore state key {key}: {e}")
                state[key] = value
        
        return state
    
    def _validate_session_data(self, session_data: Dict[str, Any]) -> bool:
        """Validate loaded session data structure."""
        required_keys = ["session_id", "timestamp", "state", "version"]
        return all(key in session_data for key in required_keys)
    
    def _is_session_expired(self, session_time: datetime) -> bool:
        """Check if session has expired."""
        timeout = timedelta(minutes=self.config.session_timeout_minutes)
        return datetime.now() - session_time > timeout
    
    def _quarantine_corrupted_session(self, session_id: str) -> None:
        """Move corrupted session file to quarantine directory."""
        try:
            quarantine_dir = self.session_dir / "quarantine"
            quarantine_dir.mkdir(exist_ok=True)
            
            session_file = self.session_dir / f"{session_id}.session"
            if session_file.exists():
                quarantine_file = quarantine_dir / f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.corrupted"
                session_file.rename(quarantine_file)
                
        except Exception as e:
            self.logger.error(f"Failed to quarantine corrupted session {session_id}: {e}")
    
    def _load_active_sessions(self) -> None:
        """Load active sessions metadata."""
        try:
            metadata_file = self.session_dir / "active_sessions.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.active_sessions = {
                        session_id: datetime.fromisoformat(timestamp)
                        for session_id, timestamp in data.items()
                    }
        except Exception as e:
            self.logger.warning(f"Could not load active sessions metadata: {e}")
            self.active_sessions = {}
    
    def _save_active_sessions(self) -> None:
        """Save active sessions metadata."""
        try:
            metadata_file = self.session_dir / "active_sessions.json"
            data = {
                session_id: timestamp.isoformat()
                for session_id, timestamp in self.active_sessions.items()
            }
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save active sessions metadata: {e}")


class VietnamFoodAssistantApp:
    """
    Main application class for Vietnamese Food Assistant.
    
    Provides comprehensive error handling, session recovery, and system resilience.
    Implements graceful shutdown and recovery mechanisms.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.workflow: Optional[VietnamFoodAssistantWorkflow] = None
        self.session_manager: Optional[SessionRecoveryManager] = None
        self.is_running = False
        
        # Error tracking
        self.error_count = 0
        self.last_error_time: Optional[datetime] = None
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Register cleanup function
        atexit.register(self.shutdown)
    
    def initialize(self) -> bool:
        """
        Initialize the application with error handling.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Vietnamese Food Assistant...")
            
            # Initialize session recovery manager
            self.session_manager = SessionRecoveryManager(self.config)
            
            # Clean up expired sessions on startup
            cleaned_count = self.session_manager.cleanup_expired_sessions()
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired sessions on startup")
            
            # Initialize workflow
            self.workflow = create_workflow()
            
            # Test workflow initialization
            if not self._test_workflow():
                self.logger.error("Workflow test failed during initialization")
                return False
            
            self.is_running = True
            self.logger.info("Vietnamese Food Assistant initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def process_request(self, raw_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a user request with comprehensive error handling and session recovery.
        
        Args:
            raw_input: Raw user input
            session_id: Session identifier
            
        Returns:
            Response dictionary with error handling
        """
        if not self.is_running or not self.workflow:
            return {
                "response_text": "Xin lỗi, hệ thống chưa sẵn sàng. Vui lòng thử lại sau.",
                "error": "System not initialized",
                "success": False
            }
        
        try:
            self.logger.info(f"Processing request for session {session_id}: '{raw_input[:50]}...'")
            
            # Attempt session recovery if needed
            recovered_state = self._attempt_session_recovery(session_id)
            
            # Process request with retry logic
            response = self._process_with_retry(raw_input, session_id, recovered_state)
            
            # Save session state after successful processing
            if response.get("success", False) and self.session_manager:
                # Extract state from response for persistence
                session_state = self._extract_session_state(response, session_id)
                self.session_manager.save_session_state(session_id, session_state)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Unhandled error processing request: {e}")
            self.logger.debug(traceback.format_exc())
            
            self._track_error()
            
            return {
                "response_text": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                "error": str(e),
                "success": False,
                "recovery_attempted": True
            }
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up a specific session.
        
        Args:
            session_id: Session to clean up
            
        Returns:
            True if cleanup successful
        """
        if self.session_manager:
            return self.session_manager.cleanup_session(session_id)
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and health information.
        
        Returns:
            System status dictionary
        """
        status = {
            "is_running": self.is_running,
            "workflow_initialized": self.workflow is not None,
            "session_manager_initialized": self.session_manager is not None,
            "error_count": self.error_count,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "config": {
                "current_model": self.config.current_model,
                "session_timeout_minutes": self.config.session_timeout_minutes,
                "max_conversation_history": self.config.max_conversation_history
            }
        }
        
        if self.session_manager:
            active_sessions = self.session_manager.get_active_sessions()
            status["active_sessions"] = {
                "count": len(active_sessions),
                "sessions": {
                    session_id: timestamp.isoformat()
                    for session_id, timestamp in active_sessions.items()
                }
            }
        
        return status
    
    def shutdown(self) -> None:
        """Graceful shutdown with cleanup."""
        if not self.is_running:
            return
        
        self.logger.info("Shutting down Vietnamese Food Assistant...")
        
        try:
            # Clean up expired sessions
            if self.session_manager:
                self.session_manager.cleanup_expired_sessions()
            
            self.is_running = False
            self.logger.info("Vietnamese Food Assistant shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            # Create logs directory
            log_dir = Path(self.config.data_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler
            file_handler = logging.FileHandler(log_dir / "vietnamese_food_assistant.log")
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _test_workflow(self) -> bool:
        """Test workflow initialization with a simple request."""
        try:
            test_response = self.workflow.process_request("xin chào", "test_session")
            return test_response.get("success", False)
        except Exception as e:
            self.logger.error(f"Workflow test failed: {e}")
            return False
    
    def _attempt_session_recovery(self, session_id: str) -> Optional[AgentState]:
        """
        Attempt to recover session state from disk.
        
        Args:
            session_id: Session to recover
            
        Returns:
            Recovered state or None
        """
        if not self.session_manager:
            return None
        
        try:
            recovered_state = self.session_manager.load_session_state(session_id)
            if recovered_state:
                self.logger.info(f"Successfully recovered session {session_id}")
            return recovered_state
        except Exception as e:
            self.logger.warning(f"Session recovery failed for {session_id}: {e}")
            return None
    
    def _process_with_retry(self, raw_input: str, session_id: str, 
                          recovered_state: Optional[AgentState] = None) -> Dict[str, Any]:
        """
        Process request with retry logic.
        
        Args:
            raw_input: User input
            session_id: Session identifier
            recovered_state: Previously recovered session state
            
        Returns:
            Response dictionary
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # If we have recovered state, we could potentially use it
                # For now, let the workflow handle its own state management
                response = self.workflow.process_request(raw_input, session_id)
                
                # Reset error tracking on success
                if response.get("success", False):
                    self.error_count = 0
                    self.last_error_time = None
                
                return response
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request processing attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    self.logger.info(f"Retrying request processing (attempt {attempt + 2})")
        
        # All retries failed
        self._track_error()
        return {
            "response_text": "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau.",
            "error": str(last_exception) if last_exception else "Max retries exceeded",
            "success": False,
            "retries_attempted": self.config.max_retries
        }
    
    def _extract_session_state(self, response: Dict[str, Any], session_id: str) -> AgentState:
        """
        Extract session state from response for persistence.
        
        Args:
            response: Response from workflow
            session_id: Session identifier
            
        Returns:
            Session state for persistence
        """
        return {
            "session_id": session_id,
            "conversation_history": response.get("conversation_history", []),
            "mentioned_dishes": response.get("mentioned_dishes", []),
            "active_constraints": response.get("active_constraints", []),
            "session_preferences": {},
            "last_activity": datetime.now(),
            "response_metadata": {
                "last_response": response.get("response_text", ""),
                "last_clarification_needed": response.get("clarification_needed", False),
                "last_follow_up_questions": response.get("follow_up_questions", [])
            }
        }
    
    def _track_error(self) -> None:
        """Track error occurrence for monitoring."""
        self.error_count += 1
        self.last_error_time = datetime.now()


# Global application instance
app: Optional[VietnamFoodAssistantApp] = None


def get_app() -> VietnamFoodAssistantApp:
    """Get or create the global application instance."""
    global app
    if app is None:
        app = VietnamFoodAssistantApp()
    return app


def initialize_app() -> bool:
    """Initialize the application."""
    return get_app().initialize()


@contextmanager
def app_context():
    """Context manager for application lifecycle."""
    application = get_app()
    try:
        if not application.initialize():
            raise RuntimeError("Failed to initialize application")
        yield application
    finally:
        application.shutdown()


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese Food Assistant")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    parser.add_argument("--cleanup", action="store_true", help="Clean up expired sessions")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    if args.cleanup:
        print("Cleaning up expired sessions...")
        app = get_app()
        if app.initialize():
            cleaned = app.session_manager.cleanup_expired_sessions()
            print(f"Cleaned up {cleaned} expired sessions")
        sys.exit(0)
    
    if args.status:
        print("System Status:")
        app = get_app()
        if app.initialize():
            status = app.get_system_status()
            print(json.dumps(status, indent=2, default=str))
        sys.exit(0)
    
    if args.test:
        print("Running Vietnamese Food Assistant in test mode...")
        
        with app_context() as application:
            test_requests = [
                "xin chào",
                "tôi muốn ăn phở",
                "có món chay nào không",
                "tôi dị ứng tôm"
            ]
            
            for i, request in enumerate(test_requests, 1):
                print(f"\n{i}. Testing: '{request}'")
                response = application.process_request(request, f"test_session_{i}")
                print(f"   Success: {response['success']}")
                print(f"   Response: {response['response_text'][:100]}...")
                
                if response.get("errors"):
                    print(f"   Errors: {response['errors']}")
    else:
        print("Vietnamese Food Assistant - Main Application")
        print("Use --test to run in test mode")
        print("Use --cleanup to clean expired sessions")
        print("Use --status to show system status")