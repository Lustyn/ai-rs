use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::time::Duration;

/// Core error type for the AI SDK
#[derive(Debug, Clone, PartialEq)]
pub enum AiError {
    /// Provider-related errors
    Provider(ProviderError),

    /// Tool-related errors
    Tool(ToolError),

    /// Agent execution errors
    Agent(AgentError),

    /// Network and transport errors
    Network(NetworkError),

    /// Serialization/deserialization errors
    Serialization(SerializationError),

    /// Configuration and validation errors
    Validation(ValidationError),
}

/// Provider-specific errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProviderError {
    /// Authentication failed for a provider
    Authentication { provider: String, message: String },

    /// Rate limit exceeded
    RateLimit {
        provider: String,
        retry_after: Option<Duration>,
        message: String,
    },

    /// Model not found or not available
    ModelNotFound { provider: String, model: String },

    /// Feature not supported by provider
    UnsupportedFeature { provider: String, feature: String },

    /// Generic API error from provider
    ApiError {
        provider: String,
        status: u16,
        message: String,
    },
}

/// Tool execution errors
#[derive(Debug, Clone, PartialEq)]
pub enum ToolError {
    /// Tool not found in registry
    NotFound { name: String },

    /// Tool execution failed
    ExecutionFailed { name: String, error: String },

    /// Invalid input provided to tool
    InvalidInput {
        name: String,
        expected: String,
        received: String,
    },

    /// State mismatch or corruption
    StateMismatch { message: String },

    /// Tool has no handler (HITL scenario)
    NoHandler { name: String },

    /// Tool serialization/deserialization error
    SerializationError { name: String, error: String },
}

/// Agent-specific errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentError {
    /// Maximum steps exceeded
    MaxStepsExceeded { steps: u32, max: u32 },

    /// Invalid message sequence
    InvalidMessageSequence { message: String },

    /// Streaming error occurred
    StreamingError { message: String },

    /// Agent state error
    StateError { message: String },
}

/// Network and transport errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NetworkError {
    /// Connection failed
    ConnectionFailed { message: String },

    /// Request timeout
    Timeout { duration: Duration },

    /// HTTP error
    HttpError { status: u16, message: String },

    /// DNS resolution failed
    DnsError { message: String },
}

/// Serialization/deserialization errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SerializationError {
    /// JSON parsing failed
    JsonError { message: String },

    /// Schema validation failed
    SchemaValidation { message: String },

    /// Type mismatch
    TypeMismatch { expected: String, found: String },
}

/// Configuration and validation errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationError {
    /// Missing required field
    MissingField { field: String },

    /// Invalid value provided
    InvalidValue { field: String, message: String },

    /// Configuration error
    ConfigError { message: String },
}

/// Tool-specific execution error
#[derive(Debug, Clone, PartialEq)]
pub enum ToolExecutionError {
    /// Input validation failed
    InvalidInput(String),

    /// State-related errors
    StateError(String),

    /// Business logic errors
    ExecutionError(String),

    /// External service errors
    ExternalServiceError { service: String, error: String },

    /// Permission/authorization errors
    Unauthorized(String),

    /// Resource not found
    NotFound(String),
}

/// Tool execution result type
pub type ToolResult<T> = std::result::Result<T, ToolExecutionError>;

impl Display for AiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::Provider(e) => write!(f, "Provider error: {}", e),
            AiError::Tool(e) => write!(f, "Tool error: {}", e),
            AiError::Agent(e) => write!(f, "Agent error: {}", e),
            AiError::Network(e) => write!(f, "Network error: {}", e),
            AiError::Serialization(e) => write!(f, "Serialization error: {}", e),
            AiError::Validation(e) => write!(f, "Validation error: {}", e),
        }
    }
}

impl Display for ProviderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Authentication { provider, message } => {
                write!(f, "Authentication failed for {}: {}", provider, message)
            }
            ProviderError::RateLimit {
                provider,
                retry_after,
                message,
            } => {
                if let Some(duration) = retry_after {
                    write!(
                        f,
                        "Rate limit exceeded for {} (retry after {:?}): {}",
                        provider, duration, message
                    )
                } else {
                    write!(f, "Rate limit exceeded for {}: {}", provider, message)
                }
            }
            ProviderError::ModelNotFound { provider, model } => {
                write!(f, "Model '{}' not found for provider {}", model, provider)
            }
            ProviderError::UnsupportedFeature { provider, feature } => {
                write!(
                    f,
                    "Feature '{}' not supported by provider {}",
                    feature, provider
                )
            }
            ProviderError::ApiError {
                provider,
                status,
                message,
            } => {
                write!(
                    f,
                    "API error from {} (HTTP {}): {}",
                    provider, status, message
                )
            }
        }
    }
}

impl Display for ToolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::NotFound { name } => {
                write!(f, "Tool '{}' not found", name)
            }
            ToolError::ExecutionFailed { name, error } => {
                write!(f, "Tool '{}' execution failed: {}", name, error)
            }
            ToolError::InvalidInput {
                name,
                expected,
                received,
            } => {
                write!(
                    f,
                    "Invalid input for tool '{}': expected {}, received {}",
                    name, expected, received
                )
            }
            ToolError::StateMismatch { message } => {
                write!(f, "Tool state mismatch: {}", message)
            }
            ToolError::NoHandler { name } => {
                write!(
                    f,
                    "Tool '{}' has no handler (client-side handling required)",
                    name
                )
            }
            ToolError::SerializationError { name, error } => {
                write!(f, "Tool '{}' serialization error: {}", name, error)
            }
        }
    }
}

impl Display for AgentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::MaxStepsExceeded { steps, max } => {
                write!(
                    f,
                    "Maximum steps exceeded: {} steps taken, {} allowed",
                    steps, max
                )
            }
            AgentError::InvalidMessageSequence { message } => {
                write!(f, "Invalid message sequence: {}", message)
            }
            AgentError::StreamingError { message } => {
                write!(f, "Streaming error: {}", message)
            }
            AgentError::StateError { message } => {
                write!(f, "Agent state error: {}", message)
            }
        }
    }
}

impl Display for NetworkError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::ConnectionFailed { message } => {
                write!(f, "Connection failed: {}", message)
            }
            NetworkError::Timeout { duration } => {
                write!(f, "Request timeout after {:?}", duration)
            }
            NetworkError::HttpError { status, message } => {
                write!(f, "HTTP error {}: {}", status, message)
            }
            NetworkError::DnsError { message } => {
                write!(f, "DNS resolution failed: {}", message)
            }
        }
    }
}

impl Display for SerializationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationError::JsonError { message } => {
                write!(f, "JSON parsing error: {}", message)
            }
            SerializationError::SchemaValidation { message } => {
                write!(f, "Schema validation failed: {}", message)
            }
            SerializationError::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingField { field } => {
                write!(f, "Missing required field: {}", field)
            }
            ValidationError::InvalidValue { field, message } => {
                write!(f, "Invalid value for field '{}': {}", field, message)
            }
            ValidationError::ConfigError { message } => {
                write!(f, "Configuration error: {}", message)
            }
        }
    }
}

impl Display for ToolExecutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolExecutionError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ToolExecutionError::StateError(msg) => write!(f, "State error: {}", msg),
            ToolExecutionError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            ToolExecutionError::ExternalServiceError { service, error } => {
                write!(f, "External service '{}' error: {}", service, error)
            }
            ToolExecutionError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            ToolExecutionError::NotFound(msg) => write!(f, "Not found: {}", msg),
        }
    }
}

impl std::error::Error for AiError {}
impl std::error::Error for ProviderError {}
impl std::error::Error for ToolError {}
impl std::error::Error for AgentError {}
impl std::error::Error for NetworkError {}
impl std::error::Error for SerializationError {}
impl std::error::Error for ValidationError {}
impl std::error::Error for ToolExecutionError {}

// Conversion from serde_json errors
impl From<serde_json::Error> for AiError {
    fn from(err: serde_json::Error) -> Self {
        AiError::Serialization(SerializationError::JsonError {
            message: err.to_string(),
        })
    }
}

// Conversion from ToolExecutionError to ToolError
impl From<ToolExecutionError> for ToolError {
    fn from(err: ToolExecutionError) -> Self {
        ToolError::ExecutionFailed {
            name: String::new(), // Name should be provided by context
            error: err.to_string(),
        }
    }
}

// Builder methods for adding context
impl AiError {
    /// Add context to an error
    pub fn with_context<C: Display>(self, context: C) -> Self {
        // For now, we'll just wrap the message
        // In a production system, this could maintain a chain of contexts
        match self {
            AiError::Provider(mut e) => {
                if let ProviderError::ApiError {
                    ref mut message, ..
                } = e
                {
                    *message = format!("{}: {}", context, message);
                }
                AiError::Provider(e)
            }
            _ => self,
        }
    }
}

/// Result type for AI operations
pub type Result<T> = std::result::Result<T, AiError>;
