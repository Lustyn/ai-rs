use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt as FuturesStreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use ai_core::errors::{AiError, NetworkError, ProviderError, SerializationError, ValidationError};
use ai_core::{Result, provider::ChatTextGeneration, types::*};

/// Configuration for Anthropic provider
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_retries: u32,
    pub timeout_seconds: u64,
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            model: model.into(),
            max_retries: 3,
            timeout_seconds: 60,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }
}

/// Anthropic provider implementation
#[derive(Clone)]
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| {
                AiError::Network(NetworkError::ConnectionFailed {
                    message: format!("Failed to create HTTP client: {}", e),
                })
            })?;

        Ok(Self { config, client })
    }

    /// Convert our Message enum to Anthropic's message format
    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> Result<(Option<String>, Vec<AnthropicMessage>)> {
        let mut system_prompt = None;
        let mut anthropic_messages = Vec::new();

        for message in messages {
            match message {
                Message::System { content, .. } => {
                    // Anthropic uses a separate system parameter
                    let text = content
                        .iter()
                        .map(|c| match c {
                            SystemContent::Text { text } => text.as_str(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ");

                    if !text.is_empty() {
                        system_prompt = Some(text);
                    }
                }
                Message::User { content, .. } => {
                    let anthropic_content = self.convert_text_content(content)?;
                    anthropic_messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: anthropic_content,
                    });
                }
                Message::Assistant { content, .. } => {
                    let anthropic_content = self.convert_assistant_content(content)?;
                    anthropic_messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: anthropic_content,
                    });
                }
                Message::Tool { tool_results, .. } => {
                    // Convert tool results to user messages in Anthropic format
                    for result in tool_results {
                        anthropic_messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: vec![AnthropicContent::ToolResult {
                                tool_use_id: result.tool_call_id.clone(),
                                content: result.result.to_string(),
                                is_error: Some(result.is_error),
                            }],
                        });
                    }
                }
            }
        }

        Ok((system_prompt, anthropic_messages))
    }

    fn convert_text_content(&self, content: &[UserContent]) -> Result<Vec<AnthropicContent>> {
        let mut anthropic_content = Vec::new();

        for item in content {
            match item {
                UserContent::Text { text } => {
                    anthropic_content.push(AnthropicContent::Text { text: text.clone() });
                }
                UserContent::Image { image } => {
                    if let Some(base64) = &image.base64 {
                        anthropic_content.push(AnthropicContent::Image {
                            source: AnthropicImageSource {
                                r#type: "base64".to_string(),
                                media_type: image
                                    .mime_type
                                    .clone()
                                    .unwrap_or("image/jpeg".to_string()),
                                data: base64.clone(),
                            },
                        });
                    } else {
                        return Err(AiError::Validation(ValidationError::InvalidValue {
                            field: "image".to_string(),
                            message: "Anthropic requires base64 encoded images".to_string(),
                        }));
                    }
                }
            }
        }

        Ok(anthropic_content)
    }

    fn convert_assistant_content(
        &self,
        content: &[AssistantContent],
    ) -> Result<Vec<AnthropicContent>> {
        let mut anthropic_content = Vec::new();

        for item in content {
            match item {
                AssistantContent::Text { text } => {
                    if !text.is_empty() {
                        anthropic_content.push(AnthropicContent::Text { text: text.clone() });
                    }
                }
                AssistantContent::ToolCall { tool_call } => {
                    anthropic_content.push(AnthropicContent::ToolUse {
                        id: tool_call.id.clone(),
                        name: tool_call.name.clone(),
                        input: tool_call.arguments.clone(),
                    });
                }
            }
        }

        Ok(anthropic_content)
    }

    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|tool| AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.parameters.clone(),
            })
            .collect()
    }

    async fn make_request(&self, request: AnthropicRequest) -> Result<AnthropicResponse> {
        let response = self
            .client
            .post(format!("{}/v1/messages", self.config.base_url))
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                AiError::Network(NetworkError::ConnectionFailed {
                    message: format!("Request failed: {}", e),
                })
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            // Check for specific error types
            if status == 401 {
                return Err(AiError::Provider(ProviderError::Authentication {
                    provider: "anthropic".to_string(),
                    message: error_text,
                }));
            } else if status == 429 {
                // TODO: Parse retry-after header if available
                return Err(AiError::Provider(ProviderError::RateLimit {
                    provider: "anthropic".to_string(),
                    retry_after: None,
                    message: error_text,
                }));
            } else {
                return Err(AiError::Provider(ProviderError::ApiError {
                    provider: "anthropic".to_string(),
                    status: status.as_u16(),
                    message: error_text,
                }));
            }
        }

        response.json().await.map_err(|e| {
            AiError::Serialization(SerializationError::JsonError {
                message: format!("Failed to parse response: {}", e),
            })
        })
    }
}

#[async_trait]
impl ChatTextGeneration for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        self.config.model.contains("claude-3")
    }

    fn supports_system_messages(&self) -> bool {
        true
    }

    fn max_tokens(&self) -> Option<u32> {
        // Claude models have different limits, but 4096 is a safe default
        Some(4096)
    }

    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse> {
        let (system, messages) = self.convert_messages(&request.messages)?;

        let anthropic_request = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: request.settings.max_tokens.unwrap_or(1000),
            temperature: request.settings.temperature,
            system,
            messages,
            tools: request.tools.as_ref().map(|t| self.convert_tools(t)),
            stream: false,
        };

        let response = self.make_request(anthropic_request).await?;

        // Convert Anthropic response back to our format
        let mut content = Vec::new();
        for item in response.content {
            match item {
                AnthropicContent::Text { text } => {
                    content.push(AssistantContent::Text { text });
                }
                AnthropicContent::ToolUse { id, name, input } => {
                    content.push(AssistantContent::ToolCall {
                        tool_call: ToolCall {
                            id,
                            name,
                            arguments: input,
                        },
                    });
                }
                _ => {} // Skip other content types in responses
            }
        }

        let message = Message::Assistant {
            content,
            metadata: None,
        };

        let finish_reason = match response.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("tool_use") => FinishReason::ToolCalls,
            _ => FinishReason::Stop,
        };

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        Ok(ChatResponse {
            id: response.id,
            message,
            finish_reason,
            usage,
            metadata: None,
        })
    }

    async fn generate_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk>> + Send>>> {
        let (system, messages) = self.convert_messages(&request.messages)?;

        let anthropic_request = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: request.settings.max_tokens.unwrap_or(1000),
            temperature: request.settings.temperature,
            system,
            messages,
            tools: request.tools.as_ref().map(|t| self.convert_tools(t)),
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.config.base_url))
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&anthropic_request)
            .send()
            .await
            .map_err(|e| {
                AiError::Network(NetworkError::ConnectionFailed {
                    message: format!("Stream request failed: {}", e),
                })
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            if status == 401 {
                return Err(AiError::Provider(ProviderError::Authentication {
                    provider: "anthropic".to_string(),
                    message: error_text,
                }));
            } else if status == 429 {
                return Err(AiError::Provider(ProviderError::RateLimit {
                    provider: "anthropic".to_string(),
                    retry_after: None,
                    message: error_text,
                }));
            } else {
                return Err(AiError::Provider(ProviderError::ApiError {
                    provider: "anthropic".to_string(),
                    status: status.as_u16(),
                    message: error_text,
                }));
            }
        }

        // Use proper SSE parsing
        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|event_result| async move {
                match event_result {
                    Ok(event) => {
                        // Parse the SSE event data
                        match serde_json::from_str::<AnthropicStreamEvent>(&event.data) {
                            Ok(stream_event) => {
                                let result =
                                    AnthropicProvider::handle_stream_event_static(stream_event);
                                // Only return Some if it's an error or has meaningful content
                                match &result {
                                    Ok(chunk) => {
                                        let empty_delta = matches!(
                                            chunk.delta,
                                            MessageDelta::Assistant { content: None }
                                        );
                                        if !empty_delta
                                            || chunk.finish_reason.is_some()
                                            || chunk.usage.is_some()
                                        {
                                            Some(result)
                                        } else {
                                            None
                                        }
                                    }
                                    Err(_) => Some(result),
                                }
                            }
                            Err(_) => {
                                // Ignore parsing errors for unknown/ping events
                                None
                            }
                        }
                    }
                    Err(e) => Some(Err(AiError::Network(NetworkError::ConnectionFailed {
                        message: format!("Stream error: {}", e),
                    }))),
                }
            });

        Ok(Box::pin(stream))
    }
}

impl AnthropicProvider {
    fn handle_stream_event_static(event: AnthropicStreamEvent) -> Result<ChatStreamChunk> {
        match event.r#type.as_str() {
            "message_start" => {
                if let AnthropicStreamEventData::MessageStart { message } = event.data {
                    Ok(ChatStreamChunk {
                        id: message.id,
                        delta: MessageDelta::Assistant { content: None },
                        finish_reason: None,
                        usage: message.usage.map(|u| Usage {
                            prompt_tokens: u.input_tokens,
                            completion_tokens: u.output_tokens,
                            total_tokens: u.input_tokens + u.output_tokens,
                        }),
                    })
                } else {
                    Ok(ChatStreamChunk {
                        id: "stream".to_string(),
                        delta: MessageDelta::Assistant { content: None },
                        finish_reason: None,
                        usage: None,
                    })
                }
            }
            "content_block_start" => {
                // Start of a content block - no delta content yet
                Ok(ChatStreamChunk {
                    id: "stream".to_string(),
                    delta: MessageDelta::Assistant { content: None },
                    finish_reason: None,
                    usage: None,
                })
            }
            "content_block_delta" => {
                if let AnthropicStreamEventData::ContentBlockDelta { delta, .. } = event.data {
                    let content = match delta.r#type.as_str() {
                        "text_delta" => Some(AssistantContent::Text {
                            text: delta.text.unwrap_or_default(),
                        }),
                        "input_json_delta" => {
                            // For tool use streaming, we could accumulate JSON here
                            // For now, just ignore these incremental JSON updates
                            None
                        }
                        "thinking_delta" => {
                            // For extended thinking - could be handled separately
                            Some(AssistantContent::Text {
                                text: delta.thinking.unwrap_or_default(),
                            })
                        }
                        _ => None,
                    };

                    Ok(ChatStreamChunk {
                        id: "stream".to_string(),
                        delta: MessageDelta::Assistant { content },
                        finish_reason: None,
                        usage: None,
                    })
                } else {
                    Ok(ChatStreamChunk {
                        id: "stream".to_string(),
                        delta: MessageDelta::Assistant { content: None },
                        finish_reason: None,
                        usage: None,
                    })
                }
            }
            "content_block_stop" => {
                // End of content block
                Ok(ChatStreamChunk {
                    id: "stream".to_string(),
                    delta: MessageDelta::Assistant { content: None },
                    finish_reason: None,
                    usage: None,
                })
            }
            "message_delta" => {
                if let AnthropicStreamEventData::MessageDelta { delta, usage } = event.data {
                    let finish_reason = delta.stop_reason.map(|reason| match reason.as_str() {
                        "end_turn" => FinishReason::Stop,
                        "max_tokens" => FinishReason::Length,
                        "tool_use" => FinishReason::ToolCalls,
                        _ => FinishReason::Stop,
                    });

                    let usage = usage.map(|u| Usage {
                        prompt_tokens: u.input_tokens,
                        completion_tokens: u.output_tokens,
                        total_tokens: u.input_tokens + u.output_tokens,
                    });

                    Ok(ChatStreamChunk {
                        id: "stream".to_string(),
                        delta: MessageDelta::Assistant { content: None },
                        finish_reason,
                        usage,
                    })
                } else {
                    Ok(ChatStreamChunk {
                        id: "stream".to_string(),
                        delta: MessageDelta::Assistant { content: None },
                        finish_reason: None,
                        usage: None,
                    })
                }
            }
            "message_stop" => {
                // Final event - stream is complete
                Ok(ChatStreamChunk {
                    id: "stream".to_string(),
                    delta: MessageDelta::Assistant { content: None },
                    finish_reason: Some(FinishReason::Stop),
                    usage: None,
                })
            }
            "ping" => {
                // Ping events - can be ignored or used for keep-alive
                Ok(ChatStreamChunk {
                    id: "stream".to_string(),
                    delta: MessageDelta::Assistant { content: None },
                    finish_reason: None,
                    usage: None,
                })
            }
            "error" => {
                if let AnthropicStreamEventData::Error { error } = event.data {
                    Err(AiError::Provider(ProviderError::ApiError {
                        provider: "anthropic".to_string(),
                        status: 500,
                        message: error.message,
                    }))
                } else {
                    Err(AiError::Provider(ProviderError::ApiError {
                        provider: "anthropic".to_string(),
                        status: 500,
                        message: "Unknown error".to_string(),
                    }))
                }
            }
            _ => {
                // Unknown event types - ignore gracefully per Anthropic docs
                Ok(ChatStreamChunk {
                    id: "stream".to_string(),
                    delta: MessageDelta::Assistant { content: None },
                    finish_reason: None,
                    usage: None,
                })
            }
        }
    }
}

// Anthropic API types
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicImageSource {
    r#type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    r#type: String,
    #[serde(flatten)]
    data: AnthropicStreamEventData,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicStreamEventData {
    MessageStart {
        message: AnthropicStreamMessage,
    },
    ContentBlockStart {
        #[allow(dead_code)]
        index: u32,
        #[allow(dead_code)]
        content_block: AnthropicStreamContentBlock,
    },
    ContentBlockDelta {
        #[allow(dead_code)]
        index: u32,
        delta: AnthropicStreamDelta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: u32,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: Option<AnthropicUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicStreamError,
    },
    Unknown,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    id: String,
    #[allow(dead_code)]
    r#type: String,
    #[allow(dead_code)]
    role: String,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    content: Vec<serde_json::Value>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamContentBlock {
    #[allow(dead_code)]
    r#type: String,
    #[serde(flatten)]
    #[allow(dead_code)]
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    r#type: String,
    text: Option<String>,
    #[allow(dead_code)]
    partial_json: Option<String>,
    thinking: Option<String>,
    #[allow(dead_code)]
    signature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageDelta {
    stop_reason: Option<String>,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamError {
    #[allow(dead_code)]
    r#type: String,
    message: String,
}
