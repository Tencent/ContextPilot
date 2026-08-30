# StateLM Tools Documentation

## Overview

StateLM Tools provide a suite of stateful tools for long-context agent workflows in the VERL framework. These tools enable agents to:
- Load and analyze documents
- Build searchable chunk-based indices
- Perform semantic search using Elasticsearch
- Maintain persistent notes across conversation turns
- Track and manage context budget
- Edit context history for efficient long-context reasoning

## Architecture

### State Management

The StateLM tools use a two-tier state management system:

1. **DocStateManager**: Manages document-related state including:
   - Document content and tokenization
   - Chunk index (local and Elasticsearch)
   - Search history and keywords
   - Elasticsearch connection

2. **AgentData.notes**: Manages note-taking state:
   - Key-value store for notes
   - Persistent across tool calls within a trajectory
   - Accessible through external memory interface

### Integration with Agent Loop

The tools integrate with `ToolAgentLoop` through:
- `agent_data.state_manager`: Initialized when `document_content` is provided
- `agent_data.notes`: Always available for note-taking
- Context passed via `**kwargs` to tool `execute()` methods

## Tool Catalog

### Document Analysis Tools

#### 1. `analyzeText`
Analyze the length of the attached document.

**Parameters**: None

**Returns**:
```json
{
  "file_name": "attached_document.txt",
  "total_tokens": 15000
}
```

#### 2. `loadDocument`
Load the full document content.

**Parameters**: None

**Returns**:
```json
{
  "document_content": "Full text of the document..."
}
```

⚠️ **Warning**: Use with caution on large documents as it returns the entire content.

### Document Indexing Tools

#### 3. `buildIndex`
Split document into fixed-size chunks and build a searchable index.

**Parameters**:
- `chunk_size` (integer, optional): Chunk size in tokens. Default: 4000
- `overlap` (integer, optional): Overlapping tokens between chunks. Default: 0

**Returns**:
```json
{
  "index_id": "document_index",
  "total_chunks": 42,
  "first_chunk_id": 0,
  "last_chunk_id": 41
}
```

**Example**:
```json
{
  "chunk_size": 3000,
  "overlap": 200
}
```

#### 4. `readChunk`
Retrieve the full text of a specific chunk by ID.

**Parameters**:
- `chunk_id` (integer, required): ID of the target chunk

**Returns**:
```json
{
  "retrieved_chunk": [{
    "chunk_id": 5,
    "content": "Chunk content...",
    "start_pos": 15000,
    "end_pos": 18000
  }],
  "chunk_id": 5
}
```

#### 5. `searchEngine`
Search the document by keywords using BM25 ranking.

**Parameters**:
- `keyword` (string, required): Comma-separated keywords
- `mode` (string, optional): "and" or "or" (default: "or")
- `size` (integer, optional): Max results to return (default: 50)
- `fragment_size` (integer, optional): Highlight fragment size (default: 180)
- `num_fragments` (integer, optional): Number of fragments per match (default: 3)

**Returns**:
```json
{
  "retrieved_chunks": [
    {
      "chunk_id": 12,
      "relevance_score": 15.234,
      "highlights": [
        "...text with <em>keyword</em> highlighted..."
      ]
    }
  ],
  "keywords": ["keyword1", "keyword2"],
  "message": "Showing the most relevant 20/45 chunks."
}
```

**Example**:
```json
{
  "keyword": "machine learning, neural networks"
}
```

### Note-Taking Tools

#### 6. `note`
Create a new note to record key information.

**Parameters**:
- `key` (string, required): Unique identifier for the note
- `summary` (string, required): Short summary
- `content` (object, required): Full note content (can be any JSON object)

**Returns**:
```json
{
  "status": "success",
  "key": "key_insights"
}
```

**Example**:
```json
{
  "key": "findings_chapter_3",
  "summary": "Key findings from chapter 3 about quantum mechanics",
  "content": {
    "main_points": ["wave-particle duality", "uncertainty principle"],
    "page_refs": [45, 52, 67]
  }
}
```

#### 7. `readNote`
Read the full content of a stored note.

**Parameters**:
- `key` (string, required): Note key

**Returns**:
```json
{
  "summary": "Key findings from chapter 3",
  "full_content": "{\"main_points\": [...], \"page_refs\": [...]}"
}
```

#### 8. `updateNote`
Update an existing note (append, overwrite, or delete).

**Parameters**:
- `key` (string, required): Key of the target note
- `mode` (string, required): "append", "overwrite", or "delete"
- `new_content` (object, optional): New content (not needed for delete)
- `new_summary` (string, optional): New summary

**Returns**:
```json
{
  "status": "success",
  "key": "findings_chapter_3",
  "message": "Note 'findings_chapter_3' appended."
}
```

**Examples**:
```json
// Append to note
{
  "key": "findings",
  "mode": "append",
  "new_content": {"additional": "data"}
}

// Overwrite note
{
  "key": "findings",
  "mode": "overwrite",
  "new_content": {"new": "content"},
  "new_summary": "Updated summary"
}

// Delete note
{
  "key": "findings",
  "mode": "delete"
}
```

#### 9. `mergeNotes`
Merge multiple notes into a single note.

**Parameters**:
- `keys` (array of strings, required): List of note keys to merge
- `new_key` (string, optional): Key for merged note (auto-generated if not provided)
- `new_summary` (string, optional): Summary for merged note (auto-generated if not provided)

**Returns**:
```json
{
  "status": "success",
  "new_key": "combined_findings",
  "merged_from": ["finding1", "finding2", "finding3"]
}
```

**Example**:
```json
{
  "keys": ["chapter1_notes", "chapter2_notes"],
  "new_key": "part1_summary",
  "new_summary": "Combined notes from part 1"
}
```

### Context Management Tools

#### 10. `checkBudget`
Check the remaining context budget (tokens and conversation rounds).

**Parameters**: None

**Returns**:
```json
{
  "conv_rounds": 12,
  "available_tokens": 4500,
  "available_rounds": 38
}
```

#### 11. `getContextStats`
Get comprehensive statistics about context and state.

**Parameters**: None

**Returns**:
```json
{
  "total_notes": 5,
  "notes_keys": ["note1", "note2", "note3", "note4", "note5"],
  "index_chunks": 42,
  "document_size": 15000,
  "searched_keywords": ["machine learning", "neural networks"],
  "conv_rounds": 12,
  "available_tokens": 4500,
  "available_rounds": 38
}
```

### Completion Tool

#### 12. `finish`
Submit the final answer and terminate the agent loop.

**Parameters**:
- `answer` (string, required): Final answer in short form

**Returns**:
```json
{
  "final_answer": "The answer to the question is..."
}
```

⚠️ **Important**: This tool terminates the conversation. Always call it when you've completed the task.

## Configuration

### Tool Configuration File

Create a YAML configuration file to register the tools:

```yaml
# statelm_tools_config.yaml
tools:
  - class_name: verl.tools.statelm_tools.AnalyzeTextTool
    config:
      type: native
    tool_schema:
      type: function
      function:
        name: analyzeText
        description: Analyze the length of the attached context.
        parameters:
          type: object
          properties: {}
  # ... (see verl/tools/configs/statelm_tools_config.yaml for full config)
```

### Agent Configuration

Enable StateLM features in your rollout configuration:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      statelm_enabled: true
      tool_config_path: verl/tools/configs/statelm_tools_config.yaml
      max_assistant_turns: 50
      max_user_turns: 50
    max_model_length: 8192  # Maximum context length
    context_length_penalty: -1.0  # Penalty for exceeding context
```

### Elasticsearch Configuration

StateLM tools use Elasticsearch for efficient document search. Configure via environment variables:

```bash
# Required
export ES_HOST="https://localhost:9200"

# Authentication (choose one method)
export ES_API_KEY="your-api-key"
# OR
export ES_USER="elastic"
export ES_PASS="your-password"

# Optional
export ES_CA_CERT="/path/to/ca.crt"  # For TLS
export ES_INDEX_NAME="lc_agent_document"  # Custom index name
```

## Usage Patterns

### Pattern 1: Document Analysis and Search

```python
# 1. Analyze document
analyzeText()  # Check document size

# 2. Build searchable index
buildIndex({"chunk_size": 3000, "overlap": 200})

# 3. Search for relevant information
searchEngine({"keyword": "neural networks, deep learning"})

# 4. Read specific chunks
readChunk({"chunk_id": 5})

# 5. Submit answer
finish({"answer": "Based on the document..."})
```

### Pattern 2: Progressive Note-Taking

```python
# 1. Create initial notes from chunks
note({
  "key": "chapter1",
  "summary": "Chapter 1 overview",
  "content": {"main_points": [...]}
})

# 2. Read and append more information
readNote({"key": "chapter1"})
updateNote({
  "key": "chapter1",
  "mode": "append",
  "new_content": {"additional_points": [...]}
})

# 3. Merge related notes
mergeNotes({
  "keys": ["chapter1", "chapter2"],
  "new_key": "part1_summary"
})

# 4. Use notes to answer
readNote({"key": "part1_summary"})
finish({"answer": "..."})
```

### Pattern 3: Context Budget Management

```python
# Check budget periodically
checkBudget()  # {"available_tokens": 5000, "available_rounds": 40}

# If budget is low, consolidate notes
mergeNotes({"keys": ["note1", "note2", "note3"]})

# Get full context stats
getContextStats()
```

## Best Practices

### 1. Chunk Size Selection
- **Small chunks (1000-2000 tokens)**: Better precision, more chunks to manage
- **Medium chunks (3000-4000 tokens)**: Balanced approach (recommended)
- **Large chunks (5000-8000 tokens)**: Better context, but may include irrelevant info

### 2. Overlap Strategy
- Use 10-20% overlap for important documents to avoid missing information at boundaries
- Use 0 overlap for large documents to maximize coverage

### 3. Search Strategy
- Start with broad keywords, then refine
- Use "or" mode for exploratory search
- Use "and" mode for precise information retrieval
- Check highlights before reading full chunks

### 4. Note Management
- Create notes for important findings to save context
- Use descriptive keys and summaries
- Merge notes when you have too many
- Delete notes that are no longer needed (via `updateNote` with mode="delete")

### 5. Context Efficiency
- Use `checkBudget` to monitor remaining context
- Store findings in notes instead of keeping them in conversation
- Use `searchEngine` instead of `loadDocument` for large documents
- Consider using `deleteContext` (handled by agent loop) to prune old messages

## Error Handling

Common errors and solutions:

### "State manager not available"
**Cause**: StateLM features not enabled or document_content not provided
**Solution**: Ensure `statelm_enabled: true` in config and pass `document_content` in kwargs

### "Index not built. Please call buildIndex first."
**Cause**: Attempting to search or read chunks before building index
**Solution**: Call `buildIndex` before using `searchEngine` or `readChunk`

### "Chunk_id out of range"
**Cause**: Requesting a chunk_id that doesn't exist
**Solution**: Check the `total_chunks` from `buildIndex` or `getContextStats`

### "Elasticsearch query failed"
**Cause**: Elasticsearch not running or misconfigured
**Solution**: 
1. Check ES_HOST and authentication environment variables
2. Verify Elasticsearch is running: `curl -X GET "$ES_HOST"`
3. Check logs for detailed error messages

### "Note not found"
**Cause**: Trying to read/update a note that doesn't exist
**Solution**: Use `getContextStats` to see available note keys

## Implementation Details

### Thread Safety
- Each trajectory gets its own `DocStateManager` instance
- Notes are stored per-trajectory in `AgentData`
- Elasticsearch uses unique document IDs per trajectory

### Memory Management
- Document is tokenized once at initialization
- Chunk index is stored in memory and Elasticsearch
- Notes are kept in memory throughout the trajectory
- Elasticsearch documents are cleared on trajectory completion

### Performance Considerations
- Initial indexing: O(n) where n = document length
- Search: O(log n) via Elasticsearch BM25
- Chunk retrieval: O(1) via index lookup
- Note operations: O(1) via dictionary access

## Troubleshooting

### Debug Logging
Enable debug logging to see tool execution details:

```python
import logging
logging.getLogger("verl.tools.statelm_tools").setLevel(logging.DEBUG)
```

### Common Issues

**Issue**: Slow search performance
- Check Elasticsearch health and load
- Reduce `size` parameter in searchEngine
- Consider using smaller chunk sizes

**Issue**: High memory usage
- Use smaller chunk sizes
- Build index only when needed
- Clear old notes via updateNote(mode="delete")

**Issue**: Context length exceeded
- Monitor with `checkBudget`
- Consolidate notes with `mergeNotes`
- Use `deleteContext` to prune history (handled by agent loop)

## Examples

See the test directory for complete examples:
- `tests/tools/test_statelm_tools.py` - Unit tests for all tools
- `examples/statelm_agent_demo.py` - Full agent workflow example

## Contributing

When adding new StateLM tools:
1. Inherit from `BaseTool`
2. Implement all required methods
3. Accept `**kwargs` in `execute()` method
4. Document parameters in tool schema
5. Add comprehensive error handling
6. Update this README with the new tool

## License

Copyright 2025 Bytedance Ltd. and/or its affiliates
Licensed under the Apache License, Version 2.0

