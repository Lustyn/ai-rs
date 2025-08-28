use async_trait::async_trait;
use futures::{Stream, StreamExt as FuturesStreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::{AiError, Result, provider::ChatTextGeneration, types::*};

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
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| AiError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
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
                        .filter_map(|c| match c {
                            SystemContent::Text { text } => Some(text.as_str()),
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
                        return Err(AiError::InvalidRequest {
                            message: "Anthropic requires base64 encoded images".to_string(),
                        });
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
                    anthropic_content.push(AnthropicContent::Text { text: text.clone() });
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
            .map_err(|e| AiError::NetworkError {
                message: format!("Request failed: {}", e),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AiError::ProviderError {
                provider: "anthropic".to_string(),
                message: format!("HTTP {}: {}", status, error_text),
            });
        }

        response.json().await.map_err(|e| AiError::ParseError {
            message: format!("Failed to parse response: {}", e),
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
            .map_err(|e| AiError::NetworkError {
                message: format!("Stream request failed: {}", e),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(AiError::ProviderError {
                provider: "anthropic".to_string(),
                message: format!("HTTP {}: {}", status, error_text),
            });
        }

        // Simplified streaming implementation
        let stream = response.bytes_stream().map(|chunk_result| {
            match chunk_result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        if let Some(json_str) = line.strip_prefix("data: ")
                            && json_str != "[DONE]"
                        {
                            if let Ok(event) =
                                serde_json::from_str::<AnthropicStreamEvent>(json_str)
                            {
                                return Ok(ChatStreamChunk {
                                    id: "stream".to_string(),
                                    delta: MessageDelta::Assistant {
                                        content: Some(AssistantContent::Text {
                                            text: event.delta.text.unwrap_or_default(),
                                        }),
                                    },
                                    finish_reason: None,
                                    usage: None,
                                });
                            }
                        }
                    }
                    // Return empty chunk if no valid data found
                    Ok(ChatStreamChunk {
                        id: "stream".to_string(),
                        delta: MessageDelta::Assistant {
                            content: Some(AssistantContent::Text {
                                text: String::new(),
                            }),
                        },
                        finish_reason: None,
                        usage: None,
                    })
                }
                Err(e) => Err(AiError::NetworkError {
                    message: format!("Stream error: {}", e),
                }),
            }
        });

        Ok(Box::pin(stream))
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
    delta: AnthropicStreamDelta,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    text: Option<String>,
}
