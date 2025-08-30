use ai_rs::*;

pub mod basic_agents;
pub mod client_tools;
pub mod tool_calling;

pub use basic_agents::*;
pub use client_tools::*;
pub use tool_calling::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🤖 AI SDK Examples\n");

    // Run all example scenarios
    println!("Choose an example to run:");
    println!("1. Basic Agents (no tools)");
    println!("2. Client-Side Tools (HITL)");
    println!("3. Tool Calling (with handlers)");
    println!("4. All examples");

    // For demo purposes, run all examples
    // In a real CLI app, you'd get user input here
    let choice = std::env::args().nth(1).unwrap_or_else(|| "4".to_string());

    match choice.as_str() {
        "1" => {
            run_basic_examples().await?;
        }
        "2" => {
            run_client_tools_example().await?;
        }
        "3" => {
            run_tool_calling_example().await?;
        }
        "4" | _ => {
            println!("Running all examples...\n");

            run_basic_examples().await?;
            println!("\n{}\n", "=".repeat(60));

            run_client_tools_example().await?;
            println!("\n{}\n", "=".repeat(60));

            run_tool_calling_example().await?;
        }
    }

    println!("\n✅ Examples completed!");
    Ok(())
}
