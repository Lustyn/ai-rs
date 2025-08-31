use futures::{Stream, StreamExt};
use std::{fmt::Debug, pin::Pin};

use ai_core::{Result, provider::ChatTextGeneration, tools::BuiltToolRouter, types::*};

/// Trait for defining execution termination strategies
pub trait RunUntil: Debug {
    /// Check if execution should continue based on current step and finish reason
    fn should_continue(&mut self, step: u32, reason: &FinishReason) -> bool;
}

/// Stop after a maximum number of steps
#[derive(Debug, Clone)]
pub struct MaxSteps {
    pub max: u32,
}

impl MaxSteps {
    pub fn new(max: u32) -> Self {
        Self { max }
    }
}

impl RunUntil for MaxSteps {
    fn should_continue(&mut self, step: u32, _reason: &FinishReason) -> bool {
        step < self.max
    }
}

/// Stop on specific finish reasons
#[derive(Debug, Clone)]
pub struct StopOnReason {
    pub reasons: Vec<FinishReason>,
}

impl StopOnReason {
    pub fn new(reasons: Vec<FinishReason>) -> Self {
        Self { reasons }
    }

    pub fn stop_on_finish() -> Self {
        Self::new(vec![FinishReason::Stop])
    }

    pub fn stop_on_length() -> Self {
        Self::new(vec![FinishReason::Length])
    }
}

impl RunUntil for StopOnReason {
    fn should_continue(&mut self, _step: u32, reason: &FinishReason) -> bool {
        !self.reasons.contains(reason)
    }
}

/// Combine multiple RunUntil strategies (first to finish logic)
#[derive(Debug)]
pub struct RunUntilFirst<A, B>
where
    A: RunUntil,
    B: RunUntil,
{
    pub first: A,
    pub second: B,
}

impl<A, B> RunUntilFirst<A, B>
where
    A: RunUntil,
    B: RunUntil,
{
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: RunUntil, B: RunUntil> RunUntil for RunUntilFirst<A, B> {
    fn should_continue(&mut self, step: u32, reason: &FinishReason) -> bool {
        self.first.should_continue(step, reason) && self.second.should_continue(step, reason)
    }
}

/// Configuration for generate_text function
#[derive(Debug)]
pub struct GenerateConfig<P, S = ()>
where
    P: ChatTextGeneration,
    S: Clone + Send + Sync + 'static,
{
    pub provider: P,
    pub messages: Vec<Message>,
    pub settings: GenerationSettings,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_router: Option<BuiltToolRouter<S>>,
    pub run_until: Box<dyn RunUntil + Send>,
}

impl<P, S> GenerateConfig<P, S>
where
    P: ChatTextGeneration,
    S: Clone + Send + Sync + 'static,
{
    pub fn settings(mut self, settings: GenerationSettings) -> Self {
        self.settings = settings;
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn run_until(mut self, run_until: impl RunUntil + Send + 'static) -> Self {
        self.run_until = Box::new(run_until);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.settings.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.settings.max_tokens = Some(tokens);
        self
    }
}

impl<P> GenerateConfig<P, ()>
where
    P: ChatTextGeneration,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            messages: Vec::new(),
            settings: GenerationSettings::default(),
            tools: None,
            tool_router: None,
            run_until: Box::new(MaxSteps::new(1)),
        }
    }

    pub fn tools<S: Clone + Send + Sync + 'static>(
        self,
        router: BuiltToolRouter<S>,
    ) -> GenerateConfig<P, S> {
        let tool_definitions = router.get_tool_definitions();
        GenerateConfig {
            provider: self.provider,
            messages: self.messages,
            settings: self.settings,
            tools: Some(tool_definitions),
            tool_router: Some(router),
            run_until: self.run_until,
        }
    }
}

/// Configuration for stream_text function
#[derive(Debug)]
pub struct StreamConfig<P, S = ()>
where
    P: ChatTextGeneration,
    S: Clone + Send + Sync + 'static,
{
    pub provider: P,
    pub messages: Vec<Message>,
    pub settings: GenerationSettings,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_router: Option<BuiltToolRouter<S>>,
    pub run_until: Box<dyn RunUntil + Send>,
}

impl<P, S> StreamConfig<P, S>
where
    P: ChatTextGeneration,
    S: Clone + Send + Sync + 'static,
{
    pub fn settings(mut self, settings: GenerationSettings) -> Self {
        self.settings = settings;
        self
    }

    pub fn tools(mut self, router: BuiltToolRouter<S>) -> Self {
        let tool_definitions = router.get_tool_definitions();
        self.tools = Some(tool_definitions);
        self.tool_router = Some(router);
        self
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn run_until(mut self, run_until: impl RunUntil + Send + 'static) -> Self {
        self.run_until = Box::new(run_until);
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.settings.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.settings.max_tokens = Some(tokens);
        self
    }
}

impl<P> StreamConfig<P, ()>
where
    P: ChatTextGeneration,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            messages: Vec::new(),
            settings: GenerationSettings::default(),
            tools: None,
            tool_router: None,
            run_until: Box::new(MaxSteps::new(1)),
        }
    }
}

/// Response from agent execution
#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub messages: Vec<Message>,
    pub final_message: Message,
    pub steps: u32,
    pub finish_reason: FinishReason,
    pub total_usage: Option<Usage>,
}

/// Streaming chunk from agent execution
#[derive(Debug, Clone)]
pub struct AgentStreamChunk {
    pub step: u32,
    pub chunk: ChatStreamChunk,
    pub is_final: bool,
}

/// Generate text using an agent with execution control
pub async fn generate_text<P, S>(config: GenerateConfig<P, S>) -> Result<AgentResponse>
where
    P: ChatTextGeneration,
    S: Clone + Send + Sync + 'static,
{
    let mut run_until = config.run_until;
    let mut messages = config.messages;
    let mut step = 0;
    let mut total_usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    let mut has_usage = false;

    loop {
        // Create request from current messages
        let request = ChatRequest {
            messages: messages.clone(),
            settings: config.settings.clone(),
            tools: config.tools.clone(),
        };

        // Generate response
        let response = config.provider.generate(request).await?;

        // Update usage tracking
        if let Some(usage) = &response.usage {
            total_usage.prompt_tokens += usage.prompt_tokens;
            total_usage.completion_tokens += usage.completion_tokens;
            total_usage.total_tokens += usage.total_tokens;
            has_usage = true;
        }

        // Handle tool calls if present
        if let Message::Assistant { content, .. } = &response.message {
            let mut tool_calls = Vec::new();
            for content_part in content {
                if let AssistantContent::ToolCall { tool_call } = content_part {
                    tool_calls.push(tool_call.clone());
                }
            }

            if !tool_calls.is_empty() && config.tool_router.is_some() {
                // Add assistant message with tool calls
                messages.push(response.message.clone());

                // Execute tool calls and collect results
                let mut tool_results = Vec::new();
                let mut should_end_loop = false;
                if let Some(router) = &config.tool_router {
                    for tool_call in tool_calls {
                        match router
                            .execute_tool(&tool_call.name, tool_call.arguments.clone())
                            .await
                        {
                            Some(Ok(result)) => {
                                tool_results.push(ToolResult {
                                    tool_call_id: tool_call.id,
                                    result,
                                    is_error: false,
                                });
                            }
                            Some(Err(e)) => {
                                tool_results.push(ToolResult {
                                    tool_call_id: tool_call.id,
                                    result: serde_json::json!({
                                        "error": e.to_string()
                                    }),
                                    is_error: true,
                                });
                            }
                            None => {
                                // Tool has no handler - end the loop to return control to client
                                should_end_loop = true;
                                break;
                            }
                        }
                    }
                }

                // If we should end the loop due to missing handler, return immediately
                if should_end_loop {
                    return Ok(AgentResponse {
                        messages: messages.clone(),
                        final_message: response.message,
                        steps: step + 1,
                        finish_reason: response.finish_reason,
                        total_usage: if has_usage { Some(total_usage) } else { None },
                    });
                }

                // Add tool results message
                if !tool_results.is_empty() {
                    messages.push(Message::Tool {
                        tool_results,
                        metadata: None,
                    });
                }
            } else {
                // No tool calls, add response normally
                messages.push(response.message.clone());
            }
        } else {
            // Not an assistant message, add normally
            messages.push(response.message.clone());
        }

        // Check if we should continue
        if !run_until.should_continue(step, &response.finish_reason) {
            return Ok(AgentResponse {
                messages: messages.clone(),
                final_message: response.message,
                steps: step + 1,
                finish_reason: response.finish_reason,
                total_usage: if has_usage { Some(total_usage) } else { None },
            });
        }

        step += 1;
    }
}

/// Stream text using an agent with execution control
pub async fn stream_text<P, S>(
    config: StreamConfig<P, S>,
) -> Result<Pin<Box<dyn Stream<Item = Result<AgentStreamChunk>> + Send + 'static>>>
where
    P: ChatTextGeneration + Send + 'static,
    S: Clone + Send + Sync + 'static,
{
    let mut run_until = config.run_until;
    let mut messages = config.messages;
    let mut step = 0;

    // Create async stream
    let stream = async_stream::stream! {
        loop {
            // Create request from current messages
            let request = ChatRequest {
                messages: messages.clone(),
                settings: config.settings.clone(),
                tools: config.tools.clone(),
            };

            // Generate streaming response
            let mut response_stream = match config.provider.generate_stream(request).await {
                Ok(stream) => stream,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mut accumulated_content = Vec::new();
            let mut accumulated_tool_calls = Vec::new();
            let mut finish_reason = FinishReason::Stop;

            // Stream chunks for this step
            while let Some(chunk_result) = response_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let is_final = chunk.finish_reason.is_some();

                        if let Some(reason) = &chunk.finish_reason {
                            finish_reason = reason.clone();
                        }

                        // Accumulate content for conversation history
                        if let MessageDelta::Assistant { content: Some(content) } = &chunk.delta {
                            accumulated_content.push(content.clone());

                            // Check for tool calls in the content
                            if let AssistantContent::ToolCall { tool_call } = content {
                                accumulated_tool_calls.push(tool_call.clone());
                            }
                        }

                        // Yield the chunk
                        yield Ok(AgentStreamChunk {
                            step,
                            chunk,
                            is_final,
                        });

                        if is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }

            // Add accumulated response to conversation
            if !accumulated_content.is_empty() {
                let assistant_message = Message::Assistant {
                    content: accumulated_content,
                    metadata: None,
                };
                messages.push(assistant_message);

                // Handle tool calls if present
                if !accumulated_tool_calls.is_empty() && config.tool_router.is_some() {
                    let mut tool_results = Vec::new();
                    let mut should_end_loop = false;
                    if let Some(router) = &config.tool_router {
                        for tool_call in accumulated_tool_calls {
                            match router.execute_tool(&tool_call.name, tool_call.arguments.clone()).await {
                                Some(Ok(result)) => {
                                    tool_results.push(ToolResult {
                                        tool_call_id: tool_call.id,
                                        result,
                                        is_error: false,
                                    });
                                }
                                Some(Err(e)) => {
                                    tool_results.push(ToolResult {
                                        tool_call_id: tool_call.id,
                                        result: serde_json::json!({
                                            "error": e.to_string()
                                        }),
                                        is_error: true,
                                    });
                                }
                                None => {
                                    // Tool has no handler - end the loop to return control to client
                                    should_end_loop = true;
                                    break;
                                }
                            }
                        }
                    }

                    // If we should end the loop due to missing handler, return immediately
                    if should_end_loop {
                        return;
                    }

                    // Add tool results message
                    if !tool_results.is_empty() {
                        messages.push(Message::Tool {
                            tool_results,
                            metadata: None,
                        });
                    }
                }
            }

            // Check if we should continue
            if !run_until.should_continue(step, &finish_reason) {
                return;
            }

            step += 1;

        }
    };

    Ok(Box::pin(stream))
}
