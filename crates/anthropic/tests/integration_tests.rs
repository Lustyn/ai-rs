use ai_anthropic::{AnthropicConfig, AnthropicProvider};
use ai_core::provider::ChatTextGeneration;
use ai_core::types::*;
use futures::StreamExt;
use std::env;

fn setup() -> AnthropicProvider {
    dotenv::dotenv().ok();

    let api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set for integration tests");

    let config = AnthropicConfig::new(api_key, "claude-3-5-haiku-20241022").with_timeout(30);

    AnthropicProvider::new(config).expect("Failed to create provider")
}

fn create_simple_request(content: &str) -> ChatRequest {
    ChatRequest {
        messages: vec![Message::User {
            content: vec![UserContent::Text {
                text: content.to_string(),
            }],
            metadata: None,
        }],
        settings: GenerationSettings {
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
        },
        tools: None,
    }
}

#[tokio::test]
#[ignore] // Only run with `cargo test -- --ignored` to avoid hitting API in normal tests
async fn test_basic_conversation() {
    let provider = setup();

    let request = create_simple_request("Hello! What's 2+2? Please be concise.");

    let response = provider
        .generate(request)
        .await
        .expect("Failed to get response from Anthropic");

    // Verify basic response structure
    assert!(!response.id.is_empty(), "Response should have an ID");

    match response.message {
        Message::Assistant { content, .. } => {
            assert!(!content.is_empty(), "Response should have content");

            // Check if we got text content
            let has_text = content
                .iter()
                .any(|c| matches!(c, AssistantContent::Text { .. }));
            assert!(has_text, "Response should contain text content");

            // Print response for manual verification
            for item in &content {
                if let AssistantContent::Text { text } = item {
                    println!("Response text: {}", text);
                    assert!(text.len() > 0, "Text content should not be empty");
                }
            }
        }
        _ => panic!("Expected assistant message"),
    }

    // Check usage information
    if let Some(usage) = response.usage {
        assert!(usage.prompt_tokens > 0, "Should have prompt tokens");
        assert!(usage.completion_tokens > 0, "Should have completion tokens");
        assert!(usage.total_tokens > 0, "Should have total tokens");
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens,
            "Total tokens should equal sum of prompt and completion"
        );
    }

    // Check finish reason
    assert_ne!(
        response.finish_reason,
        FinishReason::Length,
        "Should not hit token limit with short response"
    );
}

#[tokio::test]
#[ignore]
async fn test_streaming_conversation() {
    let provider = setup();

    let request = create_simple_request("Tell me a very short joke about programming.");

    let mut stream = provider
        .generate_stream(request)
        .await
        .expect("Failed to start stream");

    let mut chunks = Vec::new();
    let mut accumulated_text = String::new();
    let mut final_usage = None;
    let mut final_finish_reason = None;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.expect("Stream chunk should not be an error");

        // Collect all chunks for analysis
        chunks.push(chunk.clone());

        // Accumulate text content
        match chunk.delta {
            MessageDelta::Assistant {
                content: Some(AssistantContent::Text { text }),
            } => {
                accumulated_text.push_str(&text);
                println!("Streaming text: {}", text);
            }
            _ => {}
        }

        // Capture final usage and finish reason
        if chunk.usage.is_some() {
            final_usage = chunk.usage;
        }
        if chunk.finish_reason.is_some() {
            final_finish_reason = chunk.finish_reason;
        }
    }

    // Verify we got some chunks
    assert!(!chunks.is_empty(), "Should receive at least one chunk");

    // Verify we accumulated some text
    assert!(
        !accumulated_text.is_empty(),
        "Should have accumulated some text content"
    );
    assert!(
        accumulated_text.len() > 5,
        "Should have meaningful text content"
    );
    println!("Final accumulated text: {}", accumulated_text);

    // Verify usage information was provided
    assert!(final_usage.is_some(), "Should receive usage information");
    if let Some(usage) = final_usage {
        assert!(usage.prompt_tokens > 0, "Should have prompt tokens");
        assert!(usage.completion_tokens > 0, "Should have completion tokens");
    }

    // Verify finish reason was provided
    assert!(
        final_finish_reason.is_some(),
        "Should receive finish reason"
    );
}

#[tokio::test]
#[ignore]
async fn test_conversation_with_system_message() {
    let provider = setup();

    let request = ChatRequest {
        messages: vec![
            Message::System {
                content: vec![SystemContent::Text {
                    text: "You are a helpful assistant that always responds with exactly 5 words."
                        .to_string(),
                }],
                metadata: None,
            },
            Message::User {
                content: vec![UserContent::Text {
                    text: "What is the capital of France?".to_string(),
                }],
                metadata: None,
            },
        ],
        settings: GenerationSettings {
            max_tokens: Some(50),
            temperature: Some(0.1),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
        },
        tools: None,
    };

    let response = provider
        .generate(request)
        .await
        .expect("Failed to get response");

    match response.message {
        Message::Assistant { content, .. } => {
            for item in &content {
                if let AssistantContent::Text { text } = item {
                    println!("System message response: {}", text);
                    let word_count = text.split_whitespace().count();
                    // Allow some flexibility since LLMs don't always follow exact constraints
                    assert!(
                        word_count <= 10,
                        "Response should be brief due to system message constraint"
                    );
                }
            }
        }
        _ => panic!("Expected assistant message"),
    }
}

#[tokio::test]
#[ignore]
async fn test_tool_use_conversation() {
    let provider = setup();

    // Define a simple calculator tool
    let calculator_tool = ToolDefinition {
        name: "calculator".to_string(),
        description: "Perform basic mathematical operations".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }),
    };

    let request = ChatRequest {
        messages: vec![Message::User {
            content: vec![UserContent::Text {
                text: "Can you calculate 15 * 7 for me using the calculator tool?".to_string(),
            }],
            metadata: None,
        }],
        settings: GenerationSettings {
            max_tokens: Some(200),
            temperature: Some(0.1),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
        },
        tools: Some(vec![calculator_tool]),
    };

    let response = provider
        .generate(request)
        .await
        .expect("Failed to get response");

    // Verify we got a tool call
    match response.message {
        Message::Assistant { content, .. } => {
            let has_tool_call = content
                .iter()
                .any(|c| matches!(c, AssistantContent::ToolCall { .. }));

            if has_tool_call {
                println!("✓ Provider successfully generated tool calls");

                for item in &content {
                    match item {
                        AssistantContent::Text { text } => {
                            println!("Tool use text: {}", text);
                        }
                        AssistantContent::ToolCall { tool_call } => {
                            println!(
                                "Tool call: {} with args: {}",
                                tool_call.name, tool_call.arguments
                            );
                            assert_eq!(tool_call.name, "calculator", "Should call calculator tool");

                            // Verify the arguments make sense for multiplication
                            let args = &tool_call.arguments;
                            if let Some(operation) = args.get("operation").and_then(|v| v.as_str())
                            {
                                assert_eq!(operation, "multiply", "Should use multiply operation");
                            }
                        }
                    }
                }

                // Check finish reason
                assert_eq!(
                    response.finish_reason,
                    FinishReason::ToolCalls,
                    "Should finish with tool calls"
                );
            } else {
                println!(
                    "⚠ Provider did not generate tool calls (may not support tool use or chose not to use tools)"
                );
                // This is acceptable behavior - not all requests will result in tool use
            }
        }
        _ => panic!("Expected assistant message"),
    }
}

#[tokio::test]
#[ignore]
async fn test_multi_turn_conversation() {
    let provider = setup();

    // First turn
    let request1 = create_simple_request("My name is Alice. Please remember this.");
    let response1 = provider
        .generate(request1)
        .await
        .expect("Failed to get first response");

    // Second turn - test if context is maintained (though this is single request, not true multi-turn)
    let request2 = ChatRequest {
        messages: vec![
            Message::User {
                content: vec![UserContent::Text {
                    text: "My name is Alice. Please remember this.".to_string(),
                }],
                metadata: None,
            },
            response1.message.clone(),
            Message::User {
                content: vec![UserContent::Text {
                    text: "What's my name?".to_string(),
                }],
                metadata: None,
            },
        ],
        settings: GenerationSettings {
            max_tokens: Some(50),
            temperature: Some(0.1),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
        },
        tools: None,
    };

    let response2 = provider
        .generate(request2)
        .await
        .expect("Failed to get second response");

    match response2.message {
        Message::Assistant { content, .. } => {
            for item in &content {
                if let AssistantContent::Text { text } = item {
                    println!("Multi-turn response: {}", text);
                    // Check if the response mentions Alice (case insensitive)
                    assert!(
                        text.to_lowercase().contains("alice"),
                        "Should remember the name Alice from conversation context"
                    );
                }
            }
        }
        _ => panic!("Expected assistant message"),
    }
}

#[tokio::test]
#[ignore]
async fn test_error_handling() {
    // Test with invalid API key
    let bad_config = AnthropicConfig::new("invalid-key", "claude-3-5-haiku-20241022");
    let bad_provider =
        AnthropicProvider::new(bad_config).expect("Should create provider even with bad key");

    let request = create_simple_request("Hello");

    let result = bad_provider.generate(request).await;

    match result {
        Err(ai_core::errors::AiError::Provider(
            ai_core::errors::ProviderError::Authentication { .. },
        )) => {
            println!("✓ Correctly handled authentication error");
        }
        Err(other) => {
            println!("Got different error type: {:?}", other);
            // This is also acceptable - might be network error or other
        }
        Ok(_) => {
            panic!("Should have failed with invalid API key");
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_provider_capabilities() {
    let provider = setup();

    // Test provider metadata
    assert_eq!(provider.name(), "anthropic");
    assert_eq!(provider.model(), "claude-3-5-haiku-20241022");
    assert!(provider.supports_tools(), "Should support tools");
    assert!(
        provider.supports_vision(),
        "Claude 3.5 should support vision"
    );
    assert!(
        provider.supports_system_messages(),
        "Should support system messages"
    );

    if let Some(max_tokens) = provider.max_tokens() {
        assert!(max_tokens > 1000, "Max tokens should be reasonable");
        println!("Max tokens: {}", max_tokens);
    }
}

#[tokio::test]
#[ignore]
async fn test_image_conversation() {
    let provider = setup();

    // Create a simple base64 image (1x1 pixel PNG)
    let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    let request = ChatRequest {
        messages: vec![Message::User {
            content: vec![
                UserContent::Text {
                    text: "What do you see in this image?".to_string(),
                },
                UserContent::Image {
                    image: ImageContent {
                        url: None,
                        base64: Some(base64_image.to_string()),
                        mime_type: Some("image/png".to_string()),
                    },
                },
            ],
            metadata: None,
        }],
        settings: GenerationSettings {
            max_tokens: Some(100),
            temperature: Some(0.7),
            ..Default::default()
        },
        tools: None,
    };

    let response = provider
        .generate(request)
        .await
        .expect("Failed to get response with image");

    match response.message {
        Message::Assistant { content, .. } => {
            let has_text = content
                .iter()
                .any(|c| matches!(c, AssistantContent::Text { .. }));
            assert!(has_text, "Should respond to image with text");

            for item in &content {
                if let AssistantContent::Text { text } = item {
                    println!("Image response: {}", text);
                    assert!(
                        !text.is_empty(),
                        "Should have meaningful response about image"
                    );
                }
            }
        }
        _ => panic!("Expected assistant message"),
    }
}

#[tokio::test]
#[ignore]
async fn test_conversation_builder_pattern() {
    let provider = setup();

    // Test the builder pattern for requests
    let request = ChatRequest::new()
        .system("You are a helpful assistant.")
        .user("Hello!")
        .temperature(0.5)
        .max_tokens(50);

    let response = provider
        .generate(request)
        .await
        .expect("Failed to get response using builder pattern");

    match response.message {
        Message::Assistant { content, .. } => {
            assert!(!content.is_empty(), "Should have content");
            for item in &content {
                if let AssistantContent::Text { text } = item {
                    println!("Builder pattern response: {}", text);
                }
            }
        }
        _ => panic!("Expected assistant message"),
    }
}
