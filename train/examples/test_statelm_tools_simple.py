#!/usr/bin/env python3
"""
Simple test script for StateLM tools.

This script demonstrates basic usage of StateLM tools without requiring
a full agent loop setup.
"""

import asyncio
import json
from transformers import AutoTokenizer

from verl.tools.statelm_tools import (
    AnalyzeTextTool,
    BuildIndexTool,
    LoadDocumentTool,
    NoteTool,
    ReadChunkTool,
    ReadNoteTool,
    DocStateManager,
    UpdateNoteTool,
)
from verl.tools.schemas import OpenAIFunctionToolSchema


async def main():
    print("=" * 80)
    print("StateLM Tools Simple Test")
    print("=" * 80)
    
    document_content = """
    Machine Learning Fundamentals
    
    Chapter 1: Introduction to Neural Networks
    Neural networks are computational models inspired by biological neural networks.
    They consist of interconnected nodes (neurons) organized in layers.
    The basic components include input layer, hidden layers, and output layer.
    
    Chapter 2: Deep Learning Architectures
    Deep learning refers to neural networks with multiple hidden layers.
    Common architectures include Convolutional Neural Networks (CNNs) for image processing,
    Recurrent Neural Networks (RNNs) for sequential data, and Transformers for NLP tasks.
    
    Chapter 3: Training Neural Networks
    Training involves forward propagation, loss calculation, and backpropagation.
    Optimization algorithms like SGD, Adam, and RMSprop are used to update weights.
    Regularization techniques prevent overfitting.
    
    Chapter 4: Applications
    Neural networks are applied in computer vision, natural language processing,
    speech recognition, and game playing. Recent advances include GPT models,
    BERT, and diffusion models for image generation.
    """
    
    print("\n[1] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    
    print("[2] Initializing StateLM state manager...")
    state_manager = DocStateManager(tokenizer, document_content)
    print(f"   Document size: {len(state_manager.encoded_doc['input_ids'])} tokens")
    
    class MockAgentData:
        def __init__(self):
            self.notes = {}
    
    agent_data = MockAgentData()
    
    print("\n[3] Testing analyzeText tool...")
    analyze_tool = AnalyzeTextTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "analyzeText",
                "description": "Analyze document",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await analyze_tool.create()
    response, reward, metrics = await analyze_tool.execute(
        instance_id, {}, state_manager=state_manager
    )
    print(f"   Result: {response.text}")
    await analyze_tool.release(instance_id)
    
    print("\n[4] Testing buildIndex tool...")
    build_tool = BuildIndexTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "buildIndex",
                "description": "Build index",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await build_tool.create()
    response, reward, metrics = await build_tool.execute(
        instance_id, {"chunk_size": 100, "overlap": 20}, state_manager=state_manager
    )
    result = json.loads(response.text)
    print(f"   Total chunks: {result.get('total_chunks')}")
    print(f"   Chunk range: {result.get('first_chunk_id')} to {result.get('last_chunk_id')}")
    await build_tool.release(instance_id)
    
    print("\n[5] Testing readChunk tool...")
    read_chunk_tool = ReadChunkTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "readChunk",
                "description": "Read chunk",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await read_chunk_tool.create()
    response, reward, metrics = await read_chunk_tool.execute(
        instance_id, {"chunk_id": 0}, state_manager=state_manager
    )
    result = json.loads(response.text)
    chunk = result['retrieved_chunk'][0]
    print(f"   Chunk ID: {chunk['chunk_id']}")
    print(f"   Content preview: {chunk['content'][:100]}...")
    await read_chunk_tool.release(instance_id)
    
    print("\n[6] Testing note tool...")
    note_tool = NoteTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "note",
                "description": "Create note",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await note_tool.create()
    response, reward, metrics = await note_tool.execute(
        instance_id,
        {
            "key": "chapter1_summary",
            "summary": "Neural network basics",
            "content": {
                "topics": ["neurons", "layers", "connections"],
                "importance": "high"
            }
        },
        agent_data=agent_data
    )
    print(f"   Note created: {response.text}")
    await note_tool.release(instance_id)
    
    print("\n[7] Testing readNote tool...")
    read_note_tool = ReadNoteTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "readNote",
                "description": "Read note",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await read_note_tool.create()
    response, reward, metrics = await read_note_tool.execute(
        instance_id, {"key": "chapter1_summary"}, agent_data=agent_data
    )
    result = json.loads(response.text)
    print(f"   Summary: {result['summary']}")
    print(f"   Content: {result['full_content'][:100]}...")
    await read_note_tool.release(instance_id)
    
    print("\n[8] Testing updateNote tool...")
    update_note_tool = UpdateNoteTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "updateNote",
                "description": "Update note",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await update_note_tool.create()
    response, reward, metrics = await update_note_tool.execute(
        instance_id,
        {
            "key": "chapter1_summary",
            "mode": "append",
            "new_content": {"additional_note": "Very important for understanding"},
            "new_summary": "Neural network basics - updated"
        },
        agent_data=agent_data
    )
    print(f"   Update result: {response.text}")
    await update_note_tool.release(instance_id)
    
    instance_id, _ = await read_note_tool.create()
    response, reward, metrics = await read_note_tool.execute(
        instance_id, {"key": "chapter1_summary"}, agent_data=agent_data
    )
    result = json.loads(response.text)
    print(f"   Updated summary: {result['summary']}")
    await read_note_tool.release(instance_id)
    
    print("\n[9] Testing loadDocument tool...")
    load_tool = LoadDocumentTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "loadDocument",
                "description": "Load document",
                "parameters": {"type": "object", "properties": {}}
            }
        )
    )
    instance_id, _ = await load_tool.create()
    response, reward, metrics = await load_tool.execute(
        instance_id, {}, state_manager=state_manager
    )
    result = json.loads(response.text)
    print(f"   Document length: {len(result.get('document_content', ''))} characters")
    await load_tool.release(instance_id)
    
    print("\n" + "=" * 80)
    print("All tests completed successfully! ✓")
    print("=" * 80)
    
    state_manager.clear_current_document()


if __name__ == "__main__":
    asyncio.run(main())

