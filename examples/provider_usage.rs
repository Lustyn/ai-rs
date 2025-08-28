use ai_rs::*;
use dotenv::dotenv;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    println!("AI SDK for Rust - Basic Usage Demo");

    // Create Anthropic provider configuration
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet-20241022").with_timeout(30);
    let provider = AnthropicProvider::new(config)?;

    println!(
        "Provider: {} using model: {}",
        provider.name(),
        provider.model()
    );
    println!("Supports tools: {}", provider.supports_tools());
    println!("Supports vision: {}", provider.supports_vision());

    // Example of creating a simple chat request
    let request = ChatRequest::new()
        .system("You are a helpful AI assistant.")
        .user("What is the capital of France? Please be concise.")
        .temperature(0.7)
        .max_tokens(100);

    println!("\n=== Non-Streaming Generation ===");
    match provider.generate(request.clone()).await {
        Ok(response) => {
            println!("Response: {:?}", response);
            println!("Message role: {}", response.message.role());
            println!("Finish reason: {:?}", response.finish_reason);
            if let Some(usage) = response.usage {
                println!(
                    "Token usage: {} prompt + {} completion = {} total",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        Err(e) => {
            println!("Error (expected without real API key): {}", e);
        }
    }

    // Demonstrate streaming with clean text output
    println!("\n=== Streaming Generation ===");
    match provider.generate_stream(request).await {
        Ok(mut stream) => {
            println!("Streaming response:");
            print!("Assistant: ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => {
                        if let MessageDelta::Assistant {
                            content: Some(AssistantContent::Text { text }),
                        } = chunk.delta
                        {
                            if !text.is_empty() {
                                print!("{}", text);
                                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            }
                        }

                        if chunk.finish_reason.is_some() {
                            println!(); // New line when done
                            break;
                        }
                    }
                    Err(e) => {
                        println!("\nStream error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            println!("Stream error (expected without real API key): {}", e);
        }
    }

    Ok(())
}
