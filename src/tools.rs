use crate::errors::{ToolExecutionError, ToolResult};
use schemars::{JsonSchema, schema::RootSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

/// Wrapper for functions that return ToolResult
pub struct Fallible<F>(pub F);

/// Tool metadata container
#[derive(Clone, Debug)]
pub struct ToolMetadata {
    pub name: String,
    pub description: Option<String>,
    pub parameters_schema: Option<RootSchema>,
}

/// Type-safe state wrapper
#[derive(Clone)]
pub struct State<S: Clone>(pub S);

/// Input type alias for JSON inputs
pub type Input = JsonValue;

/// Request parts containing state for extraction
pub struct ToolState<S: Clone> {
    pub state: State<S>,
}

/// Full request containing both state and input
pub struct ToolRequest<S: Clone> {
    pub state: State<S>,
    pub input: Input,
}

/// Trait for extracting values from request parts (state-only)
pub trait FromToolState<S: Clone> {
    fn from_tool_state(parts: &mut ToolState<S>) -> Self;
}

/// Trait for extracting values from full request (state + input)
pub trait FromToolRequest<S: Clone>
where
    Self: Sized,
{
    fn from_request(request: &mut ToolRequest<S>) -> ToolResult<Self>;
}

/// Implementation for extracting State from request parts
impl<S: Clone> FromToolState<S> for State<S> {
    fn from_tool_state(parts: &mut ToolState<S>) -> Self {
        parts.state.clone()
    }
}

/// Blanket implementation for types that implement Deserialize + JsonSchema
impl<T, S: Clone> FromToolRequest<S> for T
where
    T: for<'de> Deserialize<'de> + JsonSchema + Send + Sync + 'static,
{
    fn from_request(request: &mut ToolRequest<S>) -> ToolResult<Self> {
        serde_json::from_value(request.input.clone())
            .map_err(|e| ToolExecutionError::InvalidInput(format!("Failed to parse input: {}", e)))
    }
}

/// Handler trait for type-safe tool functions with serializable outputs
pub trait ToolHandler<S: Clone, T> {
    type Output: Serialize;

    fn call(&mut self, state: State<S>, input: Input) -> ToolResult<Self::Output>;

    /// Generate JSON schema for the input parameters
    fn schema() -> Option<RootSchema>;
}

// Implementation for functions that return a direct value
impl<F, S: Clone, T1, R> ToolHandler<S, (T1,)> for F
where
    F: Fn(T1) -> R,
    T1: FromToolRequest<S> + JsonSchema,
    R: Serialize,
{
    type Output = R;

    fn call(&mut self, state: State<S>, input: Input) -> ToolResult<Self::Output> {
        let parsed_input = T1::from_request(&mut ToolRequest {
            state: state.clone(),
            input,
        })?;
        let result = self(parsed_input);
        Ok(result)
    }

    fn schema() -> Option<RootSchema> {
        Some(schemars::schema_for!(T1))
    }
}

/// Implementation for Fallible wrapper - single parameter functions that return ToolResult
impl<F, S: Clone, T1, R> ToolHandler<S, (T1,)> for Fallible<F>
where
    F: Fn(T1) -> ToolResult<R>,
    T1: FromToolRequest<S> + JsonSchema,
    R: Serialize,
{
    type Output = R;

    fn call(&mut self, state: State<S>, input: Input) -> ToolResult<Self::Output> {
        let parsed_input = T1::from_request(&mut ToolRequest {
            state: state.clone(),
            input,
        })?;
        (self.0)(parsed_input)
    }

    fn schema() -> Option<RootSchema> {
        Some(schemars::schema_for!(T1))
    }
}

/// Implementation for functions with two parameters (state + input) returning a direct value
impl<F, S: Clone, T1, T2, R> ToolHandler<S, (T1, T2)> for F
where
    F: Fn(T1, T2) -> R,
    T1: FromToolState<S>,
    T2: FromToolRequest<S> + JsonSchema,
    R: Serialize,
{
    type Output = R;

    fn call(&mut self, state: State<S>, input: Input) -> ToolResult<Self::Output> {
        let parsed_input = T2::from_request(&mut ToolRequest {
            state: state.clone(),
            input,
        })?;
        let result = self(
            T1::from_tool_state(&mut ToolState {
                state: state.clone(),
            }),
            parsed_input,
        );
        Ok(result)
    }

    fn schema() -> Option<RootSchema> {
        Some(schemars::schema_for!(T2))
    }
}

/// Implementation for Fallible wrapper - functions with two parameters that return ToolResult
impl<F, S: Clone, T1, T2, R> ToolHandler<S, (T1, T2)> for Fallible<F>
where
    F: Fn(T1, T2) -> ToolResult<R>,
    T1: FromToolState<S>,
    T2: FromToolRequest<S> + JsonSchema,
    R: Serialize,
{
    type Output = R;

    fn call(&mut self, state: State<S>, input: Input) -> ToolResult<Self::Output> {
        let parsed_input = T2::from_request(&mut ToolRequest {
            state: state.clone(),
            input,
        })?;
        (self.0)(
            T1::from_tool_state(&mut ToolState {
                state: state.clone(),
            }),
            parsed_input,
        )
    }

    fn schema() -> Option<RootSchema> {
        Some(schemars::schema_for!(T2))
    }
}

/// Type-erased tool function
pub trait ErasedToolHandler<S: Clone>: Send + Sync {
    fn call_erased(&self, state: State<S>, input: Input) -> ToolResult<JsonValue>;
}

/// Wrapper to make handlers type-erased
pub struct ToolHandlerWrapper<S: Clone, T, H: ToolHandler<S, T>> {
    handler: std::sync::Mutex<H>,
    _phantom: PhantomData<(S, T)>,
}

impl<S: Clone, T, H: ToolHandler<S, T>> ToolHandlerWrapper<S, T, H> {
    pub fn new(handler: H) -> Self {
        Self {
            handler: std::sync::Mutex::new(handler),
            _phantom: PhantomData,
        }
    }
}

impl<S: Clone + Send + Sync, T: Send + Sync, H: ToolHandler<S, T> + Send + Sync>
    ErasedToolHandler<S> for ToolHandlerWrapper<S, T, H>
{
    fn call_erased(&self, state: State<S>, input: Input) -> ToolResult<JsonValue> {
        let mut handler = self.handler.lock().map_err(|_| {
            ToolExecutionError::StateError("Failed to acquire handler lock".to_string())
        })?;

        let result = handler.call(state, input)?;
        let json_result = serde_json::to_value(result).map_err(|e| {
            ToolExecutionError::ExecutionError(format!("Failed to serialize result: {}", e))
        })?;
        Ok(json_result)
    }
}

/// Type-safe tool registry (without state)
pub struct ToolRouter<S: Clone> {
    tools: HashMap<String, Box<dyn ErasedToolHandler<S>>>,
    metadata: HashMap<String, ToolMetadata>,
}

impl<S: Clone + Debug> Debug for ToolRouter<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRouter")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Built tool registry with state
pub struct BuiltToolRouter<S: Clone> {
    tools: HashMap<String, Box<dyn ErasedToolHandler<S>>>,
    metadata: HashMap<String, ToolMetadata>,
    state: S,
}

impl<S: Clone + Debug> Debug for BuiltToolRouter<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltToolRouter")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .finish()
    }
}

impl<S: Clone + Send + Sync + 'static> Default for ToolRouter<S> {
    fn default() -> Self {
        Self {
            tools: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl<S: Clone + Send + Sync + 'static> ToolRouter<S> {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool with explicit name and description
    pub fn register_infallible<
        T: Send + Sync + 'static,
        H: ToolHandler<S, T> + Send + Sync + 'static,
    >(
        mut self,
        name: impl Into<String>,
        description: Option<String>,
        handler: H,
    ) -> Self {
        let name_str = name.into();
        let wrapper = ToolHandlerWrapper::new(handler);
        self.tools.insert(name_str.clone(), Box::new(wrapper));

        // Add metadata with generated schema from handler
        self.metadata.insert(
            name_str.clone(),
            ToolMetadata {
                name: name_str.clone(),
                description,
                parameters_schema: H::schema(),
            },
        );

        self
    }

    /// Register a fallible tool (one that returns ToolResult)
    pub fn register<T: Send + Sync + 'static, H>(
        self,
        name: impl Into<String>,
        description: Option<String>,
        handler: H,
    ) -> Self
    where
        Fallible<H>: ToolHandler<S, T> + Send + Sync + 'static,
    {
        self.register_infallible(name, description, Fallible(handler))
    }

    /// Register a tool definition without a handler (will be skipped during execution)
    pub fn register_definition(
        mut self,
        name: impl Into<String>,
        description: Option<String>,
        parameters_schema: Option<RootSchema>,
    ) -> Self {
        let name_str = name.into();

        // Add metadata without handler
        self.metadata.insert(
            name_str.clone(),
            ToolMetadata {
                name: name_str.clone(),
                description,
                parameters_schema,
            },
        );

        self
    }

    /// Set the state for the registry, consuming it and returning a BuiltToolRegistry
    pub fn with_state(self, state: S) -> BuiltToolRouter<S> {
        BuiltToolRouter {
            tools: self.tools,
            metadata: self.metadata,
            state,
        }
    }
}

impl<S: Clone + Send + Sync + 'static> BuiltToolRouter<S> {
    /// Execute a single tool by name
    /// Returns None if tool has no handler (should end agent loop)
    /// Returns Some(Err) for execution errors
    /// Returns Some(Ok) for successful execution
    pub fn execute_tool(&self, name: &str, input: Input) -> Option<ToolResult<JsonValue>> {
        if let Some(tool) = self.tools.get(name) {
            let state = State(self.state.clone());
            Some(tool.call_erased(state, input))
        } else if self.metadata.contains_key(name) {
            // Tool definition exists but no handler - don't execute, return None to end loop
            None
        } else {
            // Tool not found at all - this is an error
            Some(Err(ToolExecutionError::NotFound(format!(
                "Tool '{}' not found",
                name
            ))))
        }
    }

    /// Get the current state
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Get a list of all registered tool names
    pub fn tool_names(&self) -> Vec<&String> {
        self.tools.keys().collect()
    }

    /// Get metadata for a specific tool
    pub fn tool_metadata(&self, name: &str) -> Option<&ToolMetadata> {
        self.metadata.get(name)
    }

    /// Get all tool metadata
    pub fn all_tool_metadata(&self) -> &HashMap<String, ToolMetadata> {
        &self.metadata
    }

    /// Get tool definitions for use with AI providers
    pub fn get_tool_definitions(&self) -> Vec<crate::types::ToolDefinition> {
        self.metadata
            .values()
            .map(|metadata| crate::types::ToolDefinition {
                name: metadata.name.clone(),
                description: metadata.description.clone().unwrap_or_default(),
                parameters: metadata
                    .parameters_schema
                    .as_ref()
                    .and_then(|schema| serde_json::to_value(schema).ok())
                    .unwrap_or_else(|| serde_json::json!({})),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;

    #[derive(Clone)]
    struct MyState {
        value: u64,
    }

    #[derive(Deserialize, JsonSchema)]
    struct TestInput {
        message: String,
    }

    fn test_handler_with_input(State(state): State<MyState>, input: TestInput) -> String {
        format!("State: {}, Input: {}", state.value, input.message)
    }

    fn test_handler_input_only(input: TestInput) -> String {
        format!("Input: {}", input.message)
    }

    #[test]
    fn test_registry_creation() {
        let registry = ToolRouter::default()
            .register_infallible("handler_with_input", None, test_handler_with_input)
            .register_infallible("handler_input_only", None, test_handler_input_only)
            .with_state(MyState { value: 42 });

        assert_eq!(registry.state().value, 42);
    }

    #[test]
    fn test_tool_execution_by_name() {
        let registry = ToolRouter::default()
            .register_infallible("test_tool", None, test_handler_input_only)
            .with_state(MyState { value: 42 });

        let input = serde_json::json!({"message": "Hello"});
        let result = registry.execute_tool("test_tool", input).unwrap().unwrap();
        let expected = serde_json::json!("Input: Hello");
        assert_eq!(result, expected);

        // Test non-existent tool
        let input = serde_json::json!({"message": "Hello"});
        let result = registry.execute_tool("non_existent", input);
        assert!(
            result
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("Tool 'non_existent' not found")
        );
    }

    #[test]
    fn test_get_tool_definitions() {
        let registry = ToolRouter::default()
            .register_infallible(
                "tool1",
                Some("First tool".to_string()),
                test_handler_input_only,
            )
            .register_infallible("tool2", None, test_handler_with_input)
            .with_state(MyState { value: 42 });

        let definitions = registry.get_tool_definitions();
        assert_eq!(definitions.len(), 2);

        // Find tool1
        let tool1 = definitions.iter().find(|d| d.name == "tool1").unwrap();
        assert_eq!(tool1.description, "First tool");
        // Should have a proper schema for TestInput
        assert!(tool1.parameters.is_object());
        assert!(tool1.parameters["properties"].is_object());
        assert!(tool1.parameters["properties"]["message"].is_object());

        // Find tool2
        let tool2 = definitions.iter().find(|d| d.name == "tool2").unwrap();
        assert_eq!(tool2.description, "");
        // Should also have a proper schema for TestInput
        assert!(tool2.parameters.is_object());
        assert!(tool2.parameters["properties"].is_object());
        assert!(tool2.parameters["properties"]["message"].is_object());
    }
}
