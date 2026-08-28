const scenarios = {
  chronology: {
    runTitle: "Chronology audit",
    query: "The team changed course several times. Where was Project Aurora finally approved, and on what date?",
    search: '"Aurora" + approved + final + resolution',
    result: "12 ranked passages across minutes, memos, and appendices",
    evidence: "Chunk 128: the final resolution records approval in Reykjavík on 18 May.",
    contrast: "Chunk 61: an earlier Oslo proposal was deferred and later superseded.",
    workingNote: "Likely final approval: Reykjavík, 18 May. Verify that the Oslo proposal was superseded.",
    followupSearch: '"Aurora" + Reykjavík + Oslo + superseded',
    finalSearch: '"Aurora" + 18 May + final resolution',
    memory: "Aurora → final approval: Reykjavík, 18 May; Oslo was only a superseded proposal.",
    answer: "Project Aurora was finally approved in Reykjavík on 18 May.",
  },
  memory: {
    runTitle: "Commitment recall",
    query: "What accommodation did Maya promise to book for the December conference?",
    search: "Maya + December + conference + book / reserve",
    result: "9 candidate passages across seven conversation sessions",
    evidence: "Chunk 204: Maya confirms she will book the quiet hotel beside the venue.",
    contrast: "Chunk 77: a downtown apartment was discussed, but rejected because of noise.",
    workingNote: "Likely commitment: quiet hotel beside the venue. Check whether the apartment remained active.",
    followupSearch: "Maya + quiet hotel + apartment + rejected",
    finalSearch: "Maya + December conference + confirmed booking",
    memory: "Maya’s active commitment → quiet hotel beside the venue; apartment option rejected.",
    answer: "Maya promised to book the quiet hotel beside the conference venue.",
  },
  research: {
    runTitle: "Evidence reconciliation",
    query: "Which launch date did the project ultimately announce after the schedule changed?",
    search: "launch date + rescheduled + announcement + ultimately",
    result: "14 results from release notes, interviews, and archived announcements",
    evidence: "Chunk 241: the final announcement sets the public launch for 7 October.",
    contrast: "Chunk 96: the original 12 September target was withdrawn after testing delays.",
    workingNote: "Likely final launch: 7 October. Verify that the 12 September date was withdrawn.",
    followupSearch: '"7 October" + "12 September" + withdrawn',
    finalSearch: '"7 October" + final launch announcement',
    memory: "Final public launch → 7 October; 12 September was the withdrawn target.",
    answer: "The project ultimately announced 7 October as its public launch date.",
  },
};

const steps = [
  {
    tool: "plan",
    label: "Plan",
    title: "Set the retrieval strategy",
    description: "Turn the query into a compact sequence of retrieval, verification, and context-management decisions.",
    context: 0.9,
    memories: 0,
    offloaded: 0,
    event: "plan.ready · evidence strategy outlined",
  },
  {
    tool: "analyzeText",
    label: "Analyze",
    title: "Map the source structure",
    description: "Inspect the document layout and identify where answer-bearing evidence is most likely to appear.",
    context: 1.3,
    memories: 0,
    offloaded: 0,
    event: "document_map.ready · retrieval regions identified",
  },
  {
    tool: "searchEngine",
    label: "Search",
    title: "Start with a narrow search",
    description: "Retrieve a small candidate set while the working context still contains little more than the query.",
    context: 1.8,
    memories: 0,
    offloaded: 0,
    event: "search.pass_1 · initial candidates returned",
  },
  {
    tool: "readChunk",
    label: "Read",
    title: "Open the strongest chunk",
    description: "Read one high-ranking passage instead of loading the full source into the prompt.",
    context: 5.6,
    memories: 0,
    offloaded: 0,
    event: "chunk.read · first evidence added",
  },
  {
    tool: "note",
    label: "Note",
    title: "Capture the working lead",
    description: "Save the promising finding and the uncertainty that the next retrieval pass must resolve.",
    context: 6.1,
    memories: 0,
    offloaded: 0,
    event: "note.write · provisional lead saved",
  },
  {
    tool: "searchEngine",
    label: "Search",
    title: "Search from the new clue",
    description: "Use the first passage to form a sharper query for conflicts, revisions, or superseding evidence.",
    context: 9.7,
    memories: 0,
    offloaded: 0,
    event: "search.pass_2 · contrast candidates returned",
  },
  {
    tool: "readMultiChunks",
    label: "Read many",
    title: "Compare the competing evidence",
    description: "Read several targeted chunks together and determine which statement remained valid at the end.",
    context: 19.4,
    memories: 0,
    offloaded: 0,
    event: "chunks.read · temporal conflict resolved",
  },
  {
    tool: "memorize",
    label: "Memorize",
    title: "Store the durable relationship",
    description: "Persist the answer-bearing fact and its qualifier before removing bulky retrieval turns.",
    context: 27.8,
    memories: 1,
    offloaded: 0,
    event: "memory.write · stable key created",
  },
  {
    tool: "deleteContext",
    label: "Delete",
    title: "Remove spent retrieval turns",
    description: "Delete search and reading messages whose useful content is now safely stored outside the active prompt.",
    context: 8.9,
    memories: 1,
    offloaded: 7,
    event: "context.delete · 7 messages offloaded",
  },
  {
    tool: "searchEngine",
    label: "Search",
    title: "Search the remaining gap",
    description: "Run one focused confirmation query from the compacted working state.",
    context: 11.6,
    memories: 1,
    offloaded: 7,
    event: "search.pass_3 · confirmation candidates returned",
  },
  {
    tool: "readChunk",
    label: "Read",
    title: "Read the confirming chunk",
    description: "Open only the passage needed to close the final evidence gap.",
    context: 16.8,
    memories: 1,
    offloaded: 7,
    event: "chunk.read · final claim confirmed",
  },
  {
    tool: "note",
    label: "Note",
    title: "Update the final note",
    description: "Record the confirmed answer in a compact form that can survive another context reduction.",
    context: 17.3,
    memories: 1,
    offloaded: 7,
    event: "note.write · final evidence summarized",
  },
  {
    tool: "compressContext",
    label: "Compress",
    title: "Compress the remaining evidence",
    description: "Replace the last search and reading turns with a short answer-bearing summary.",
    context: 6.2,
    memories: 1,
    offloaded: 10,
    event: "context.compress · 3 more messages offloaded",
  },
  {
    tool: "readNote",
    label: "Recall",
    title: "Read back the saved note",
    description: "Review the durable note from the compact context before composing the final response.",
    context: 6.9,
    memories: 1,
    offloaded: 10,
    event: "note.read · answer state restored",
  },
  {
    tool: "finish",
    label: "Answer",
    title: "Finish from verified evidence",
    description: "Return a concise answer grounded in the retrieved chunks and the reviewed note.",
    context: 7.4,
    memories: 1,
    offloaded: 10,
    event: "run.complete · answer returned",
  },
];

let activeScenario = "chronology";
let activeStep = 0;
let autoplayTimer = null;

const els = {
  runTitle: document.querySelector("#run-title"),
  query: document.querySelector("#scenario-query"),
  stepList: document.querySelector("#step-list"),
  stepProgress: document.querySelector("#step-progress"),
  messageStream: document.querySelector("#message-stream"),
  activeMessageCount: document.querySelector("#active-message-count"),
  toolNumber: document.querySelector("#tool-number"),
  currentTool: document.querySelector("#current-tool"),
  stepTitle: document.querySelector("#step-title"),
  stepDescription: document.querySelector("#step-description"),
  contextValue: document.querySelector("#context-value"),
  contextMeter: document.querySelector("#context-meter"),
  memoryValue: document.querySelector("#memory-value"),
  offloadValue: document.querySelector("#offload-value"),
  eventLog: document.querySelector("#event-log"),
  stepCounter: document.querySelector("#step-counter"),
  previous: document.querySelector("#previous-step"),
  next: document.querySelector("#next-step"),
  reset: document.querySelector("#reset-demo"),
  autoplay: document.querySelector("#auto-demo"),
};

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function message(type, label, text, state = "active") {
  return { type, label, text, state };
}

function messagesFor(stepIndex, scenario) {
  const messages = [message("query", "ORIGINAL QUERY", scenario.query, "pinned")];
  if (stepIndex < 8) {
    if (stepIndex >= 0) messages.push(message("tool", "PLAN", "Retrieve narrowly, verify chronology, preserve durable facts, then prune spent context."));
    if (stepIndex >= 1) messages.push(message("tool", "DOCUMENT MAP", "Source structure mapped; full text remains outside the active prompt."));
    if (stepIndex >= 2) messages.push(message("tool", "SEARCH · PASS 1", `${scenario.search} → ${scenario.result}`));
    if (stepIndex >= 3) messages.push(message("evidence", "CHUNK · PRIMARY LEAD", scenario.evidence));
    if (stepIndex >= 4) messages.push(message("memory", "WORKING NOTE", scenario.workingNote, "stored"));
    if (stepIndex >= 5) messages.push(message("tool", "SEARCH · PASS 2", scenario.followupSearch));
    if (stepIndex >= 6) messages.push(message("evidence", "MULTI-CHUNK CHECK", `${scenario.evidence} ${scenario.contrast}`));
    if (stepIndex >= 7) messages.push(message("memory", "STRUCTURED MEMORY", scenario.memory, "stored"));
    return messages;
  }

  messages.push(message("memory", "STRUCTURED MEMORY", scenario.memory, "stored"));
  messages.push(message("cleanup", "CONTEXT DELETE", "The first two retrieval passes were removed after their durable facts were saved.", "complete"));

  if (stepIndex < 12) {
    if (stepIndex >= 9) messages.push(message("tool", "SEARCH · PASS 3", `${scenario.finalSearch} → 3 targeted confirmation passages`));
    if (stepIndex >= 10) messages.push(message("evidence", "CONFIRMING CHUNK", scenario.evidence));
    if (stepIndex >= 11) messages.push(message("memory", "FINAL NOTE", scenario.memory, "stored"));
  } else {
    messages.push(message("memory", "FINAL NOTE", scenario.memory, "compressed"));
    messages.push(message("cleanup", "CONTEXT COMPRESSION", "The confirmation search was compressed into the final note.", "complete"));
  }

  if (stepIndex >= 13) messages.push(message("memory", "READ NOTE", scenario.memory, "active"));
  if (stepIndex >= 14) {
    messages.push(message("answer", "FINAL ANSWER", scenario.answer, "complete"));
  }
  return messages;
}

function renderStepList() {
  els.stepList.innerHTML = steps
    .map((step, index) => `
      <button class="trace-step ${index < activeStep ? "done" : ""} ${index === activeStep ? "active" : ""}"
        type="button" data-step="${index}" ${index === activeStep ? 'aria-current="step"' : ""}>
        <i>${String(index + 1).padStart(2, "0")}</i>
        <span><b>${step.tool}</b><small>${step.label}</small></span>
      </button>`)
    .join("");

  document.querySelectorAll(".trace-step").forEach((button) => {
    button.addEventListener("click", () => {
      stopAutoplay();
      activeStep = Number(button.dataset.step);
      render();
    });
  });
}

function renderMessages() {
  const messages = messagesFor(activeStep, scenarios[activeScenario]);
  const activeCount = messages.filter((item) => item.state !== "offloaded").length;
  els.activeMessageCount.textContent = `${activeCount} in context`;
  els.messageStream.innerHTML = messages
    .map((item, index) => `
      <div class="trace-message ${item.type} ${item.state === "offloaded" ? "dimmed" : ""}"
        style="animation-delay:${Math.min(index * 25, 150)}ms">
        <i></i>
        <div><small>${escapeHtml(item.label)}</small><b>${escapeHtml(item.text)}</b></div>
        <span>${escapeHtml(item.state)}</span>
      </div>`)
    .join("");
}

function render() {
  const scenario = scenarios[activeScenario];
  const step = steps[activeStep];
  els.runTitle.textContent = scenario.runTitle;
  els.query.textContent = scenario.query;
  els.toolNumber.textContent = String(activeStep + 1).padStart(2, "0");
  els.currentTool.textContent = step.tool;
  els.stepTitle.textContent = step.title;
  els.stepDescription.textContent = step.description;
  els.contextValue.textContent = `${step.context.toFixed(1)}k`;
  els.contextMeter.style.width = `${Math.min((step.context / 40.96) * 100, 100)}%`;
  els.memoryValue.textContent = String(step.memories);
  els.offloadValue.textContent = String(step.offloaded);
  els.eventLog.textContent = `> ${step.event}`;
  els.stepCounter.textContent = `${activeStep + 1} / ${steps.length}`;
  els.stepProgress.style.height = `${(activeStep / (steps.length - 1)) * 100}%`;
  els.previous.disabled = activeStep === 0;
  els.next.disabled = activeStep === steps.length - 1;
  renderStepList();
  renderMessages();
}

function stopAutoplay() {
  if (autoplayTimer) {
    clearInterval(autoplayTimer);
    autoplayTimer = null;
  }
  els.autoplay.classList.remove("is-playing");
  els.autoplay.querySelector("span").textContent = "Auto play";
}

function startAutoplay() {
  if (autoplayTimer) {
    stopAutoplay();
    return;
  }
  if (activeStep === steps.length - 1) activeStep = 0;
  render();
  els.autoplay.classList.add("is-playing");
  els.autoplay.querySelector("span").textContent = "Pause";
  autoplayTimer = setInterval(() => {
    if (activeStep >= steps.length - 1) {
      stopAutoplay();
      return;
    }
    activeStep += 1;
    render();
  }, 1350);
}

document.querySelectorAll(".scenario").forEach((button) => {
  button.addEventListener("click", () => {
    stopAutoplay();
    document.querySelectorAll(".scenario").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activeScenario = button.dataset.scenario;
    activeStep = 0;
    render();
  });
});

els.previous.addEventListener("click", () => {
  stopAutoplay();
  activeStep = Math.max(0, activeStep - 1);
  render();
});

els.next.addEventListener("click", () => {
  stopAutoplay();
  activeStep = Math.min(steps.length - 1, activeStep + 1);
  render();
});

els.reset.addEventListener("click", () => {
  stopAutoplay();
  activeStep = 0;
  render();
});

els.autoplay.addEventListener("click", startAutoplay);

document.querySelector("#copy-command").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const command = [
    "bash infer/scripts/setup_environment.sh",
    "source infer/.venv/bin/activate",
    "",
    "bash infer/scripts/run_full_pipeline.sh \\",
    "  /path/to/checkpoint my-run",
  ].join("\n");
  try {
    await navigator.clipboard.writeText(command);
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  } catch {
    button.textContent = "Select text";
  }
});

render();
