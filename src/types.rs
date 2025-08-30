use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

/// Content parts for system messages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemContent {
    Text { text: String },
}

/// Content parts for user messages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text { text: String },
    Image { image: ImageContent },
}

/// Content parts for assistant messages (can include tool calls)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AssistantContent {
    Text { text: String },
    ToolCall { tool_call: ToolCall },
}

/// Image content with flexible source types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageContent {
    pub url: Option<String>,
    pub base64: Option<String>,
    pub mime_type: Option<String>,
}

/// Tool call representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool execution result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub result: serde_json::Value,
    pub is_error: bool,
}

/// Message enum with role-specific content constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: Vec<SystemContent>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    User {
        content: Vec<UserContent>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    Assistant {
        content: Vec<AssistantContent>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    Tool {
        tool_results: Vec<ToolResult>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
}

impl From<&str> for AssistantContent {
    fn from(value: &str) -> Self {
        AssistantContent::Text {
            text: value.to_string(),
        }
    }
}

impl From<String> for AssistantContent {
    fn from(value: String) -> Self {
        AssistantContent::Text { text: value }
    }
}

impl From<AssistantContent> for Vec<AssistantContent> {
    fn from(value: AssistantContent) -> Self {
        vec![value]
    }
}

impl From<&str> for UserContent {
    fn from(value: &str) -> Self {
        UserContent::Text {
            text: value.to_string(),
        }
    }
}

impl From<String> for UserContent {
    fn from(value: String) -> Self {
        UserContent::Text { text: value }
    }
}

impl From<UserContent> for Vec<UserContent> {
    fn from(value: UserContent) -> Self {
        vec![value]
    }
}

impl From<&str> for SystemContent {
    fn from(value: &str) -> Self {
        SystemContent::Text {
            text: value.to_string(),
        }
    }
}

impl From<String> for SystemContent {
    fn from(value: String) -> Self {
        SystemContent::Text { text: value }
    }
}

impl From<SystemContent> for Vec<SystemContent> {
    fn from(value: SystemContent) -> Self {
        vec![value]
    }
}

impl Message {
    /// Create a system message
    pub fn system(text: impl Into<SystemContent>) -> Self {
        Self::System {
            content: vec![text.into()],
            metadata: None,
        }
    }

    /// Create a user message
    pub fn user(text: impl Into<UserContent>) -> Self {
        Self::User {
            content: vec![text.into()],
            metadata: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(text: impl Into<AssistantContent>) -> Self {
        Self::Assistant {
            content: vec![text.into()],
            metadata: None,
        }
    }

    /// Create a tool message with results
    pub fn tool(tool_results: impl Into<ToolResult>) -> Self {
        Self::Tool {
            tool_results: vec![tool_results.into()],
            metadata: None,
        }
    }

    /// Add text content (only for System and User messages)
    pub fn add_text(self, text: impl Into<String>) -> Self {
        match self {
            Self::System {
                mut content,
                metadata,
            } => {
                content.push(SystemContent::Text { text: text.into() });
                Self::System { content, metadata }
            }
            Self::User {
                mut content,
                metadata,
            } => {
                content.push(UserContent::Text { text: text.into() });
                Self::User { content, metadata }
            }
            _ => self, // Cannot add text to Assistant or Tool messages this way
        }
    }

    /// Add image content (only for User messages)
    pub fn add_image(self, image: ImageContent) -> Self {
        match self {
            Self::User {
                mut content,
                metadata,
            } => {
                content.push(UserContent::Image { image });
                Self::User { content, metadata }
            }
            _ => self, // Cannot add images to other message types
        }
    }

    /// Add tool call (only for Assistant messages)
    pub fn add_tool_call(self, tool_call: ToolCall) -> Self {
        match self {
            Self::Assistant {
                mut content,
                metadata,
            } => {
                content.push(AssistantContent::ToolCall { tool_call });
                Self::Assistant { content, metadata }
            }
            _ => self, // Cannot add tool calls to other message types
        }
    }

    /// Get the role as a string for compatibility
    pub fn role(&self) -> &'static str {
        match self {
            Self::System { .. } => "system",
            Self::User { .. } => "user",
            Self::Assistant { .. } => "assistant",
            Self::Tool { .. } => "tool",
        }
    }
}

/// Generation settings for AI providers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationSettings {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<u64>,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
        }
    }
}

/// Request for chat-based text generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub settings: GenerationSettings,
    pub tools: Option<Vec<ToolDefinition>>,
}

impl ChatRequest {
    /// Create a new empty chat request
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            settings: GenerationSettings::default(),
            tools: None,
        }
    }

    /// Add a message to the request
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Add multiple messages to the request
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Add a system message
    pub fn system(self, text: impl Into<SystemContent>) -> Self {
        self.message(Message::system(text))
    }

    /// Add a user message
    pub fn user(self, text: impl Into<UserContent>) -> Self {
        self.message(Message::user(text))
    }

    /// Add an assistant message
    pub fn assistant(self, text: impl Into<AssistantContent>) -> Self {
        self.message(Message::assistant(text))
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.settings.temperature = Some(temp);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.settings.max_tokens = Some(tokens);
        self
    }

    /// Set tools
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool definition for function calling
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

/// Response from chat-based text generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub message: Message,
    pub finish_reason: FinishReason,
    pub usage: Option<Usage>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Reason why generation finished
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
}

/// Token usage information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Delta content for streaming chunks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum MessageDelta {
    System { content: Option<UserContent> },
    User { content: Option<UserContent> },
    Assistant { content: Option<AssistantContent> },
    Tool { tool_result: Option<ToolResult> },
}

/// Streaming chunk for real-time chat generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamChunk {
    pub id: String,
    pub delta: MessageDelta,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

/// Request for embedding generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
    pub model: Option<String>,
    pub encoding_format: Option<String>,
    pub dimensions: Option<u32>,
}

/// Response from embedding generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub usage: Option<Usage>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Request for image generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRequest {
    pub prompt: String,
    pub size: Option<String>,
    pub quality: Option<String>,
    pub n: Option<u32>,
    pub response_format: Option<String>,
}

/// Response from image generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
    pub images: Vec<GeneratedImage>,
    pub usage: Option<Usage>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// A generated image
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub url: Option<String>,
    pub base64: Option<String>,
    pub revised_prompt: Option<String>,
}

/// Error types for the AI SDK
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AiError {
    InvalidRequest {
        message: String,
    },
    AuthenticationError {
        message: String,
    },
    RateLimitExceeded {
        message: String,
        retry_after: Option<u64>,
    },
    ModelNotFound {
        model: String,
    },
    NetworkError {
        message: String,
    },
    ParseError {
        message: String,
    },
    ProviderError {
        provider: String,
        message: String,
    },
    InvalidToolCall {
        message: String,
    },
}

impl Display for AiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::InvalidRequest { message } => write!(f, "Invalid request: {}", message),
            AiError::AuthenticationError { message } => {
                write!(f, "Authentication error: {}", message)
            }
            AiError::RateLimitExceeded {
                message,
                retry_after,
            } => {
                if let Some(retry) = retry_after {
                    write!(
                        f,
                        "Rate limit exceeded: {} (retry after {} seconds)",
                        message, retry
                    )
                } else {
                    write!(f, "Rate limit exceeded: {}", message)
                }
            }
            AiError::ModelNotFound { model } => write!(f, "Model not found: {}", model),
            AiError::NetworkError { message } => write!(f, "Network error: {}", message),
            AiError::ParseError { message } => write!(f, "Parse error: {}", message),
            AiError::ProviderError { provider, message } => {
                write!(f, "Provider error ({}): {}", provider, message)
            }
            AiError::InvalidToolCall { message } => write!(f, "Invalid input: {}", message),
        }
    }
}

impl std::error::Error for AiError {}

impl From<serde_json::Error> for AiError {
    fn from(err: serde_json::Error) -> Self {
        AiError::ParseError {
            message: err.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, AiError>;
