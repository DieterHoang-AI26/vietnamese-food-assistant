"""
Vietnamese Food Assistant - Error Handling Module

This module provides comprehensive error handling utilities for the Vietnamese
Food Assistant system, including error classification, recovery strategies,
and monitoring capabilities.

Requirements: 7.5 - Session Management and Data Persistence
"""

import logging
import traceback
import functools
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    ASR_CORRECTION = "asr_correction"
    CONTEXT_MANAGEMENT = "context_management"
    CONSTRAINT_PROCESSING = "constraint_processing"
    RETRIEVAL = "retrieval"
    FILTERING = "filtering"
    RESPONSE_GENERATION = "response_generation"
    SESSION_MANAGEMENT = "session_management"
    DATABASE = "database"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    session_id: str = ""
    traceback_info: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "session_id": self.session_id,
            "traceback_info": self.traceback_info,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "user_impact": self.user_impact
        }


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can handle the error."""
        raise NotImplementedError
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from the error."""
        raise NotImplementedError


class FallbackResponseStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that provides fallback responses."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can always provide a fallback response."""
        return True
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide appropriate fallback response based on error category."""
        fallback_responses = {
            ErrorCategory.ASR_CORRECTION: "Xin lỗi, tôi không hiểu rõ yêu cầu của bạn. Bạn có thể nói lại được không?",
            ErrorCategory.CONTEXT_MANAGEMENT: "Xin lỗi, tôi gặp sự cố với ngữ cảnh cuộc hội thoại. Chúng ta có thể bắt đầu lại được không?",
            ErrorCategory.CONSTRAINT_PROCESSING: "Xin lỗi, tôi gặp khó khăn khi xử lý yêu cầu đặc biệt của bạn. Bạn có thể nói rõ hơn không?",
            ErrorCategory.RETRIEVAL: "Xin lỗi, tôi không tìm thấy món ăn phù hợp. Bạn có thể thử từ khóa khác không?",
            ErrorCategory.FILTERING: "Xin lỗi, tôi gặp sự cố khi lọc món ăn. Bạn có thể thử lại không?",
            ErrorCategory.RESPONSE_GENERATION: "Xin lỗi, tôi gặp khó khăn khi tạo phản hồi. Vui lòng thử lại.",
            ErrorCategory.SESSION_MANAGEMENT: "Xin lỗi, tôi gặp sự cố với phiên làm việc. Chúng ta có thể bắt đầu cuộc hội thoại mới không?",
            ErrorCategory.DATABASE: "Xin lỗi, tôi gặp sự cố khi truy cập dữ liệu menu. Vui lòng thử lại sau.",
            ErrorCategory.NETWORK: "Xin lỗi, tôi gặp sự cố kết nối. Vui lòng kiểm tra kết nối và thử lại.",
            ErrorCategory.CONFIGURATION: "Xin lỗi, tôi gặp sự cố cấu hình hệ thống. Vui lòng liên hệ quản trị viên.",
            ErrorCategory.UNKNOWN: "Xin lỗi, tôi gặp sự cố không xác định. Vui lòng thử lại sau."
        }
        
        response_text = fallback_responses.get(
            error_info.category, 
            fallback_responses[ErrorCategory.UNKNOWN]
        )
        
        return {
            "response_text": response_text,
            "clarification_needed": True,
            "follow_up_questions": [
                "Bạn có thể thử lại với cách nói khác không?",
                "Bạn có cần hỗ trợ gì khác không?"
            ],
            "error_handled": True,
            "recovery_strategy": "fallback_response"
        }


class RetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries operations."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can recover from transient errors."""
        transient_categories = [
            ErrorCategory.NETWORK,
            ErrorCategory.DATABASE,
            ErrorCategory.RETRIEVAL
        ]
        return error_info.category in transient_categories
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Dict[str, Any]:
        """Indicate that retry should be attempted."""
        return {
            "should_retry": True,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "recovery_strategy": "retry"
        }


class SessionResetStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that resets session state."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can recover from session-related errors."""
        return error_info.category == ErrorCategory.SESSION_MANAGEMENT
    
    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reset session state."""
        return {
            "reset_session": True,
            "response_text": "Tôi đã khởi tạo lại phiên làm việc. Chúng ta có thể bắt đầu cuộc hội thoại mới.",
            "clarification_needed": False,
            "follow_up_questions": [
                "Bạn muốn tìm món gì hôm nay?",
                "Bạn có yêu cầu đặc biệt nào không?"
            ],
            "recovery_strategy": "session_reset"
        }


class ErrorHandler:
    """
    Comprehensive error handler for the Vietnamese Food Assistant.
    
    Provides error classification, recovery strategies, and monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.last_error_time: Optional[datetime] = None
        
        # Recovery strategies
        self.recovery_strategies: List[ErrorRecoveryStrategy] = [
            RetryStrategy(),
            SessionResetStrategy(),
            FallbackResponseStrategy()  # Always last as fallback
        ]
        
        # Error rate monitoring
        self.error_rate_window = timedelta(minutes=5)
        self.max_error_rate = 10  # errors per window
    
    def handle_error(self, 
                    exception: Exception, 
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    component: str = "",
                    session_id: str = "",
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error with classification and recovery.
        
        Args:
            exception: The exception that occurred
            category: Error category for classification
            component: Component where error occurred
            session_id: Session identifier
            context: Additional context for recovery
            
        Returns:
            Recovery response dictionary
        """
        if context is None:
            context = {}
        
        # Create error info
        error_info = self._classify_error(exception, category, component, session_id)
        
        # Log the error
        self._log_error(error_info)
        
        # Track error
        self._track_error(error_info)
        
        # Check error rate
        if self._is_error_rate_exceeded():
            self.logger.critical("Error rate exceeded, system may be unstable")
            return self._create_critical_error_response()
        
        # Attempt recovery
        recovery_response = self._attempt_recovery(error_info, context)
        
        return recovery_response
    
    def _classify_error(self, 
                       exception: Exception, 
                       category: ErrorCategory,
                       component: str,
                       session_id: str) -> ErrorInfo:
        """Classify error and determine severity."""
        
        # Determine severity based on exception type and category
        severity = self._determine_severity(exception, category)
        
        # Get user impact description
        user_impact = self._determine_user_impact(category, severity)
        
        return ErrorInfo(
            category=category,
            severity=severity,
            message=str(exception),
            component=component,
            session_id=session_id,
            traceback_info=traceback.format_exc(),
            user_impact=user_impact
        )
    
    def _determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception and category."""
        
        # Critical errors
        if isinstance(exception, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        if category in [ErrorCategory.SESSION_MANAGEMENT, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.HIGH
        
        # High severity errors
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        
        if category in [ErrorCategory.DATABASE, ErrorCategory.NETWORK]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.RETRIEVAL, ErrorCategory.FILTERING]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (processing errors that can be recovered)
        if category in [ErrorCategory.ASR_CORRECTION, ErrorCategory.RESPONSE_GENERATION]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _determine_user_impact(self, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Determine user impact description."""
        
        impact_descriptions = {
            (ErrorCategory.ASR_CORRECTION, ErrorSeverity.LOW): "User input may not be understood correctly",
            (ErrorCategory.CONTEXT_MANAGEMENT, ErrorSeverity.HIGH): "Conversation context may be lost",
            (ErrorCategory.CONSTRAINT_PROCESSING, ErrorSeverity.MEDIUM): "Dietary restrictions may not be applied",
            (ErrorCategory.RETRIEVAL, ErrorSeverity.MEDIUM): "No food recommendations available",
            (ErrorCategory.FILTERING, ErrorSeverity.MEDIUM): "Food filtering may not work correctly",
            (ErrorCategory.RESPONSE_GENERATION, ErrorSeverity.LOW): "Response may be generic or incomplete",
            (ErrorCategory.SESSION_MANAGEMENT, ErrorSeverity.HIGH): "Session data may be lost",
            (ErrorCategory.DATABASE, ErrorSeverity.HIGH): "Menu information unavailable",
            (ErrorCategory.NETWORK, ErrorSeverity.HIGH): "System connectivity issues",
            (ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL): "System may be unstable"
        }
        
        return impact_descriptions.get((category, severity), "Unknown impact")
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with appropriate level."""
        
        log_message = f"[{error_info.category.value}] {error_info.message}"
        
        if error_info.session_id:
            log_message += f" (Session: {error_info.session_id})"
        
        if error_info.component:
            log_message += f" (Component: {error_info.component})"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log traceback for high severity errors
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Traceback: {error_info.traceback_info}")
    
    def _track_error(self, error_info: ErrorInfo) -> None:
        """Track error for monitoring."""
        self.error_history.append(error_info)
        
        # Update error counts
        if error_info.category not in self.error_counts:
            self.error_counts[error_info.category] = 0
        self.error_counts[error_info.category] += 1
        
        self.last_error_time = error_info.timestamp
        
        # Keep only recent errors in memory
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.error_history = [
            error for error in self.error_history 
            if error.timestamp >= cutoff_time
        ]
    
    def _is_error_rate_exceeded(self) -> bool:
        """Check if error rate is exceeded."""
        if not self.error_history:
            return False
        
        cutoff_time = datetime.now() - self.error_rate_window
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= cutoff_time
        ]
        
        return len(recent_errors) > self.max_error_rate
    
    def _attempt_recovery(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt error recovery using available strategies."""
        
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_info):
                try:
                    recovery_response = strategy.recover(error_info, context)
                    
                    # Mark recovery attempt
                    error_info.recovery_attempted = True
                    error_info.recovery_successful = True
                    
                    self.logger.info(f"Error recovery successful using {strategy.__class__.__name__}")
                    
                    # Add error info to response
                    recovery_response["error_info"] = error_info.to_dict()
                    recovery_response["success"] = True
                    
                    return recovery_response
                    
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery strategy {strategy.__class__.__name__} failed: {recovery_error}")
                    continue
        
        # No recovery strategy worked
        error_info.recovery_attempted = True
        error_info.recovery_successful = False
        
        return self._create_unrecoverable_error_response(error_info)
    
    def _create_critical_error_response(self) -> Dict[str, Any]:
        """Create response for critical system errors."""
        return {
            "response_text": "Hệ thống đang gặp sự cố nghiêm trọng. Vui lòng liên hệ quản trị viên.",
            "error": "Critical system error - error rate exceeded",
            "success": False,
            "critical_error": True,
            "clarification_needed": False,
            "follow_up_questions": []
        }
    
    def _create_unrecoverable_error_response(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Create response for unrecoverable errors."""
        return {
            "response_text": "Xin lỗi, tôi gặp sự cố không thể khắc phục. Vui lòng thử lại sau hoặc liên hệ hỗ trợ.",
            "error": error_info.message,
            "success": False,
            "error_info": error_info.to_dict(),
            "clarification_needed": False,
            "follow_up_questions": [
                "Bạn có muốn thử lại không?",
                "Bạn có cần hỗ trợ khác không?"
            ]
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_category": dict(self.error_counts),
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "error_rate_window_minutes": self.error_rate_window.total_seconds() / 60,
            "max_error_rate": self.max_error_rate,
            "current_error_rate": len([
                error for error in self.error_history 
                if error.timestamp >= datetime.now() - self.error_rate_window
            ])
        }


def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN, 
                          component: str = ""):
    """
    Decorator for automatic error handling in functions.
    
    Args:
        category: Error category for classification
        component: Component name for logging
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler instance (assuming it's available globally)
                error_handler = ErrorHandler()
                
                # Extract session_id if available in args/kwargs
                session_id = ""
                if args and isinstance(args[0], dict) and "session_id" in args[0]:
                    session_id = args[0]["session_id"]
                elif "session_id" in kwargs:
                    session_id = kwargs["session_id"]
                
                # Handle the error
                return error_handler.handle_error(
                    exception=e,
                    category=category,
                    component=component or func.__name__,
                    session_id=session_id
                )
        
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(exception: Exception, 
                category: ErrorCategory = ErrorCategory.UNKNOWN,
                component: str = "",
                session_id: str = "",
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Global error handling function.
    
    Args:
        exception: The exception that occurred
        category: Error category for classification
        component: Component where error occurred
        session_id: Session identifier
        context: Additional context for recovery
        
    Returns:
        Recovery response dictionary
    """
    return global_error_handler.handle_error(
        exception=exception,
        category=category,
        component=component,
        session_id=session_id,
        context=context
    )


def get_error_statistics() -> Dict[str, Any]:
    """Get global error statistics."""
    return global_error_handler.get_error_statistics()