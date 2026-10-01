# allama

Provider-agnostic language for low-level LLM operations and their normalized behavior.

## Language

**Direct extraction**:
A structured extraction operation derived from the caller's complete chat request. It retains caller configuration except extraction control consumed by the extraction module and fields replaced by the selected extraction strategy.

**Dedicated extraction**:
A structured extraction operation dispatched from an extraction-only request containing a model, system prompts, and completed conversation messages. It does not execute or imply an earlier generation operation.
_Avoid_: Extraction orchestration

**Extraction pass**:
One provider chat dispatch made by an extraction operation. It counts dispatches owned by extraction, not retry or fallback attempts inside provider adapters.
_Avoid_: LLM call

**Provider**:
Anything that executes model operations for a requested model: a hosted API, a gateway, or an in-process inference runtime. It is not tied to a vendor or transport, and one provider may serve many models.
_Avoid_: Vendor, API client
