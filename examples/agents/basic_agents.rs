use ai_agent::*;
use ai_anthropic::*;
use ai_core::*;
use dotenv::dotenv;
use tokio_stream::StreamExt;

/// Basic agent examples without tools
pub async fn run_basic_examples() -> Result<()> {
    dotenv().ok();
    println!("=== Basic Agent Examples ===\n");

    // Create provider
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");
    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet-20241022");
    let provider = AnthropicProvider::new(config)?;

    // Initial conversation
    let messages = vec![
        Message::system("You are a helpful AI assistant. Keep responses concise."),
        Message::user("Tell me a short joke, then ask me a follow-up question."),
    ];

    println!("=== Non-Streaming Agent (Max 3 Steps) ===");

    // Use generate_text with max steps
    let config = GenerateConfig::new(provider.clone())
        .messages(messages.clone())
        .run_until(MaxSteps::new(3));
    match generate_text(config).await {
        Ok(response) => {
            println!("Final conversation ({} steps):", response.steps);
            for (i, msg) in response.messages.iter().enumerate() {
                match msg {
                    Message::System { content, .. } => {
                        let text = content
                            .iter()
                            .map(|c| match c {
                                SystemContent::Text { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        println!("{}. System: {}", i + 1, text);
                    }
                    Message::User { content, .. } => {
                        let text = content
                            .iter()
                            .filter_map(|c| match c {
                                UserContent::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        println!("{}. User: {}", i + 1, text);
                    }
                    Message::Assistant { content, .. } => {
                        let text = content
                            .iter()
                            .filter_map(|c| match c {
                                AssistantContent::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        println!("{}. Assistant: {}", i + 1, text);
                    }
                    _ => {}
                }
            }

            println!("\nFinal reason: {:?}", response.finish_reason);
            if let Some(usage) = response.total_usage {
                println!("Total tokens: {}", usage.total_tokens);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\n=== Streaming Agent (Stop on Finish) ===");

    // Use stream_text with stop on finish reason
    let config = StreamConfig::new(provider.clone())
        .messages(messages)
        .run_until(StopOnReason::stop_on_finish());
    match stream_text(config).await {
        Ok(mut stream) => {
            let mut current_step = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(agent_chunk) => {
                        // Print step header when starting new step
                        if current_step != Some(agent_chunk.step) {
                            if current_step.is_some() {
                                println!(); // New line after previous step
                            }
                            println!("Step {}: ", agent_chunk.step + 1);
                            current_step = Some(agent_chunk.step);
                        }

                        // Print streaming content
                        if let MessageDelta::Assistant {
                            content: Some(AssistantContent::Text { text }),
                        } = &agent_chunk.chunk.delta
                            && !text.is_empty() {
                                print!("{}", text);
                                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            }

                        if agent_chunk.is_final {
                            println!(); // New line after final chunk
                        }
                    }
                    Err(e) => {
                        println!("\nStream error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\n=== Combined Strategy Demo ===");

    // Combine max steps with stop on specific reasons
    let combined = RunUntilFirst::new(
        MaxSteps::new(5),
        StopOnReason::new(vec![FinishReason::Stop, FinishReason::Length]),
    );

    let simple_messages = vec![
        Message::system("You are helpful."),
        Message::user("Count from 1 to 10, one number per response."),
    ];

    let config = GenerateConfig::new(provider.clone())
        .messages(simple_messages)
        .run_until(combined);
    match generate_text(config).await {
        Ok(response) => {
            println!(
                "Completed {} steps with reason: {:?}",
                response.steps, response.finish_reason
            );

            // Show just the assistant responses
            for (i, msg) in response.messages.iter().enumerate() {
                if let Message::Assistant { content, .. } = msg {
                    let text = content
                        .iter()
                        .filter_map(|c| match c {
                            AssistantContent::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!("Response {}: {}", i - 1, text); // -1 to account for system message
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}
