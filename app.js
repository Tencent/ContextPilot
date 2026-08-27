const scenarios = {
  chronology: {
    runTitle: "Chronology audit",
    query: "The team changed course several times. Where was Project Aurora finally approved, and on what date?",
    collection: "160-page board archive · 428 indexed chunks",
    search: '"Aurora" + approved + final + resolution',
    result: "12 ranked passages across minutes, memos, and appendices",
    evidence: "Chunk 128: the final resolution records approval in Reykjavík on 18 May.",
    contrast: "Chunk 61: an earlier Oslo proposal was deferred and later superseded.",
    memory: "Aurora → final approval: Reykjavík, 18 May; Oslo was only a superseded proposal.",
    answer: "Project Aurora was finally approved in Reykjavík on 18 May.",
  },
  memory: {
    runTitle: "Commitment recall",
    query: "What accommodation did Maya promise to book for the December conference?",
    collection: "Multi-session conversation · 312 indexed chunks",
    search: "Maya + December + conference + book / reserve",
    result: "9 candidate passages across seven conversation sessions",
    evidence: "Chunk 204: Maya confirms she will book the quiet hotel beside the venue.",
    contrast: "Chunk 77: a downtown apartment was discussed, but rejected because of noise.",
    memory: "Maya’s active commitment → quiet hotel beside the venue; apartment option rejected.",
    answer: "Maya promised to book the quiet hotel beside the conference venue.",
  },
  research: {
    runTitle: "Evidence reconciliation",
    query: "Which launch date did the project ultimately announce after the schedule changed?",
    collection: "Research bundle · 286 indexed passages",
    search: "launch date + rescheduled + announcement + ultimately",
    result: "14 results from release notes, interviews, and archived announcements",
    evidence: "Chunk 241: the final announcement sets the public launch for 7 October.",
    contrast: "Chunk 96: the original 12 September target was withdrawn after testing delays.",
    memory: "Final public launch → 7 October; 12 September was the withdrawn target.",
    answer: "The project ultimately announced 7 October as its public launch date.",
  },
};

const steps = [
  {
    tool: "analyzeText",
    label: "Analyze",
    title: "Map the document",
    description: "Inspect document structure and identify the evidence shape the answer will require.",
    context: 24.8,
    memories: 0,
    offloaded: 0,
    event: "document_map.ready · sections identified",
  },
  {
    tool: "buildIndex",
    label: "Index",
    title: "Make it searchable",
    description: "Build a retrieval index over the source while keeping the full document outside the working context.",
    context: 25.3,
    memories: 0,
    offloaded: 0,
    event: "index.ready · lexical retrieval online",
  },
  {
    tool: "searchEngine",
    label: "Search",
    title: "Retrieve candidates",
    description: "Search for the subject together with chronology and decision terms—not just the first keyword match.",
    context: 31.8,
    memories: 0,
    offloaded: 0,
    event: "search.complete · ranked candidates returned",
  },
  {
    tool: "readMultiChunks",
    label: "Read",
    title: "Verify the evidence",
    description: "Open the strongest passages and resolve whether an earlier statement was later superseded.",
    context: 37.2,
    memories: 0,
    offloaded: 0,
    event: "evidence.verified · temporal conflict resolved",
  },
  {
    tool: "memorize",
    label: "Remember",
    title: "Save the durable fact",
    description: "Write the answer-bearing relationship and its temporal qualifier into structured memory.",
    context: 38.1,
    memories: 1,
    offloaded: 0,
    event: "memory.write · stable key created",
  },
  {
    tool: "compressContext",
    label: "Offload",
    title: "Clear the scaffolding",
    description: "Compress useful evidence and remove bulky search and planning messages that no longer need to stay active.",
    context: 13.6,
    memories: 1,
    offloaded: 4,
    event: "context.compact · 4 messages offloaded",
  },
  {
    tool: "finish",
    label: "Answer",
    title: "Answer from evidence",
    description: "Review the saved fact and return a concise answer grounded in the verified final passage.",
    context: 14.2,
    memories: 1,
    offloaded: 4,
    event: "run.complete · evidence retained",
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
  if (stepIndex >= 0) {
    messages.push(message("tool", "DOCUMENT MAP", scenario.collection, stepIndex >= 5 ? "offloaded" : "active"));
  }
  if (stepIndex >= 1) {
    messages.push(message("tool", "INDEX", "Searchable chunks remain external until requested.", stepIndex >= 5 ? "offloaded" : "active"));
  }
  if (stepIndex >= 2) {
    messages.push(message("tool", "SEARCH", `${scenario.search} → ${scenario.result}`, stepIndex >= 5 ? "offloaded" : "active"));
  }
  if (stepIndex >= 3) {
    messages.push(message("evidence", "VERIFIED EVIDENCE", scenario.evidence, stepIndex >= 5 ? "compressed" : "active"));
    messages.push(message("evidence", "CONTRAST CHECK", scenario.contrast, stepIndex >= 5 ? "offloaded" : "active"));
  }
  if (stepIndex >= 4) {
    messages.push(message("memory", "STRUCTURED MEMORY", scenario.memory, "stored"));
  }
  if (stepIndex >= 5) {
    messages.push(message("cleanup", "CONTEXT UPDATE", "Search scaffolding offloaded; answer-bearing evidence retained."));
  }
  if (stepIndex >= 6) {
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
