"""ContextPilot constants shared across rollout/reward/advantage modules."""

OFFLOADING_TOOL_NAMES = (
    "deleteContext",
    "summarizeContext",
    "compressContext",
    "truncateContext",
)

MEMORY_WRITING_TOOL_NAMES = (
    "memorize",
    "updateMemory",
    "note",
    "updateNote",
)

CE_TOOL_NAMES = OFFLOADING_TOOL_NAMES + MEMORY_WRITING_TOOL_NAMES

R_PEN_TOOL_FAILURE = -0.5


def is_ce_tool(name: str) -> bool:
    return name in CE_TOOL_NAMES


def is_offloading_tool(name: str) -> bool:
    return name in OFFLOADING_TOOL_NAMES


def is_memory_writing_tool(name: str) -> bool:
    return name in MEMORY_WRITING_TOOL_NAMES
