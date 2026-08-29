const elements = {
  statusPill: document.querySelector("#status-pill"),
  statusLabel: document.querySelector("#status-label"),
  documentCount: document.querySelector("#document-count"),
  chunkCount: document.querySelector("#chunk-count"),
  documentList: document.querySelector("#document-list"),
  embeddingModel: document.querySelector("#embedding-model"),
  fastAnswerModel: document.querySelector("#fast-answer-model"),
  qualityAnswerModel: document.querySelector("#quality-answer-model"),
  embeddingStatus: document.querySelector("#embedding-status"),
  fastAnswerStatus: document.querySelector("#fast-answer-status"),
  qualityAnswerStatus: document.querySelector("#quality-answer-status"),
  modelOptions: document.querySelectorAll(".model-option"),
  welcomeView: document.querySelector("#welcome-view"),
  messages: document.querySelector("#messages"),
  chatScroll: document.querySelector("#chat-scroll"),
  questionForm: document.querySelector("#question-form"),
  questionInput: document.querySelector("#question-input"),
  sendButton: document.querySelector("#send-button"),
  composerNote: document.querySelector("#composer-note"),
  themeButton: document.querySelector("#theme-button"),
  newChatButton: document.querySelector("#new-chat-button"),
};

const templates = {
  user: document.querySelector("#user-message-template"),
  loading: document.querySelector("#loading-message-template"),
  assistant: document.querySelector("#assistant-message-template"),
};

const state = {
  ready: false,
  asking: false,
  documentsLoaded: false,
  statusTimer: null,
  selectedModel: "fast",
};

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("local-rag-theme", theme);
}

function initializeTheme() {
  const saved = localStorage.getItem("local-rag-theme");
  if (saved === "light" || saved === "dark") {
    setTheme(saved);
    return;
  }
  setTheme(window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
}

function setComposerAvailability() {
  const enabled = state.ready && !state.asking;
  elements.questionInput.disabled = !enabled;
  elements.sendButton.disabled = !enabled || !elements.questionInput.value.trim();
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.disabled = !enabled;
  });
  elements.modelOptions.forEach((button) => {
    button.disabled = !state.ready || state.asking;
  });
}

function selectModel(modelId) {
  if (!['fast', 'quality'].includes(modelId)) return;
  state.selectedModel = modelId;
  localStorage.setItem("local-rag-answer-model", modelId);
  elements.modelOptions.forEach((button) => {
    const active = button.dataset.model === modelId;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function updateStatusView(status) {
  const statusLabels = {
    starting: "Starting locally",
    loading: status.stage || "Loading models",
    ready: "Ready · Local only",
    busy: status.stage || "Working locally",
    error: "Setup needs attention",
  };

  elements.statusLabel.textContent = statusLabels[status.status] || status.stage;
  elements.statusPill.className = `status-pill is-${status.status}`;
  elements.documentCount.textContent = status.document_count ?? "—";
  elements.chunkCount.textContent = status.chunk_count ?? "—";
  elements.embeddingModel.textContent = status.embedding_model;
  elements.fastAnswerModel.textContent = status.answer_models?.fast?.name || "qwen2.5-0.5b";
  elements.qualityAnswerModel.textContent = status.answer_models?.quality?.name || "qwen3-4b";

  const modelReady = status.status === "ready" || status.status === "busy";
  const modelError = status.status === "error";
  const statusItems = [
    [elements.embeddingStatus, modelReady],
    [elements.fastAnswerStatus, status.answer_models?.fast?.ready],
    [elements.qualityAnswerStatus, status.answer_models?.quality?.ready],
  ];
  statusItems.forEach(([item, ready]) => {
    item.textContent = modelError ? "Error" : ready ? "Ready" : "Loading";
    item.className = `mini-status${modelError ? " error" : ready ? " ready" : ""}`;
  });

  state.ready = modelReady;
  if (!state.asking) {
    if (status.status === "error") {
      elements.composerNote.textContent = status.error || "The local models could not be started.";
    } else if (modelReady) {
      elements.composerNote.textContent = "Answers use local models and the indexed PDF collection only.";
    } else {
      elements.composerNote.textContent = `${status.stage}. The first start can take a moment.`;
    }
  }
  setComposerAvailability();
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("Status unavailable");
    const status = await response.json();
    updateStatusView(status);

    const loadingMessage = document.querySelector(".loading-message");
    if (loadingMessage && state.asking) {
      loadingMessage.querySelector(".thinking-stage").textContent = status.stage || "Working locally";
    }

    if (status.chunk_count > 0 && !state.documentsLoaded) {
      await loadDocuments();
    }
  } catch (error) {
    state.ready = false;
    elements.statusLabel.textContent = "Local server unavailable";
    elements.statusPill.className = "status-pill is-error";
    elements.composerNote.textContent = "The local server is not responding. Restart the application window.";
    setComposerAvailability();
  }
}

function documentIcon() {
  return `
    <span class="file-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h4"/></svg>
    </span>`;
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents", { cache: "no-store" });
    if (!response.ok) throw new Error("Documents unavailable");
    const payload = await response.json();
    elements.documentList.replaceChildren();

    payload.documents.forEach((documentItem) => {
      const link = document.createElement("a");
      link.className = "document-item";
      link.href = `/documents/${encodeURIComponent(documentItem.file_name)}`;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.innerHTML = documentIcon();

      const text = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = documentItem.file_name.replace(/\.pdf$/i, "").replace(/^\d+_/, "").replaceAll("_", " ");
      const meta = document.createElement("small");
      meta.textContent = `${documentItem.chunk_count} chunks · ${documentItem.page_count} pages`;
      text.append(title, meta);

      const openIcon = document.createElement("span");
      openIcon.className = "open-icon";
      openIcon.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5M19 5l-8 8"/><path d="M19 13v6H5V5h6"/></svg>`;
      link.append(text, openIcon);
      elements.documentList.append(link);
    });
    state.documentsLoaded = true;
  } catch (error) {
    elements.documentList.textContent = "Documents could not be listed.";
  }
}

function resizeTextarea() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 130)}px`;
  setComposerAvailability();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
  });
}

function addUserMessage(question) {
  const node = templates.user.content.cloneNode(true);
  node.querySelector("p").textContent = question;
  elements.messages.append(node);
}

function addLoadingMessage() {
  const node = templates.loading.content.cloneNode(true);
  elements.messages.append(node);
}

function appendAnswerWithCitations(element, answer) {
  const parts = answer.split(/(\[Source\s+\d+\])/gi);
  parts.forEach((part) => {
    if (/^\[Source\s+\d+\]$/i.test(part)) {
      const citation = document.createElement("span");
      citation.className = "citation";
      citation.textContent = part;
      element.append(citation);
    } else {
      element.append(document.createTextNode(part));
    }
  });
}

function createSourceCard(source) {
  const link = document.createElement("a");
  link.className = `source-card${source.cited ? " is-cited" : ""}`;
  link.href = `/documents/${encodeURIComponent(source.file_name)}#page=${source.page}`;
  link.target = "_blank";
  link.rel = "noreferrer";

  const number = document.createElement("span");
  number.className = "source-number";
  number.textContent = source.number;

  const heading = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = source.file_name;
  const meta = document.createElement("small");
  meta.textContent = `Page ${source.page} · Chunk ${source.chunk_number}${source.cited ? " · Cited" : ""}`;
  heading.append(title, meta);

  const score = document.createElement("span");
  score.className = "score-badge";
  score.textContent = source.score.toFixed(3);

  const excerpt = document.createElement("span");
  excerpt.className = "source-excerpt";
  excerpt.textContent = source.excerpt;

  link.append(number, heading, score, excerpt);
  return link;
}

function addAssistantMessage(payload) {
  const node = templates.assistant.content.cloneNode(true);
  const article = node.querySelector(".assistant-message");
  const answerCard = node.querySelector(".answer-card");
  const answerText = node.querySelector(".answer-text");
  const copyButton = node.querySelector(".copy-button");
  const answerMeta = node.querySelector(".answer-meta");
  const sourceList = node.querySelector(".source-list");
  const sourceCount = node.querySelector(".sources-count");

  appendAnswerWithCitations(answerText, payload.answer);
  if (payload.fallback) answerCard.classList.add("is-fallback");

  const model = document.createElement("span");
  model.className = "meta-chip";
  model.textContent = `${payload.model.label} · ${payload.model.display_name}`;
  answerMeta.append(model);

  const grounding = document.createElement("span");
  grounding.className = "meta-chip";
  grounding.textContent = payload.fallback ? "No supporting source" : "Grounded locally";
  answerMeta.append(grounding);

  sourceCount.textContent = `${payload.sources.length} ranked`;
  payload.sources.forEach((source) => sourceList.append(createSourceCard(source)));

  copyButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(payload.answer);
    copyButton.querySelector("span").textContent = "Copied";
    window.setTimeout(() => {
      copyButton.querySelector("span").textContent = "Copy";
    }, 1400);
  });

  elements.messages.append(article);
}

function addErrorMessage(message) {
  const article = document.createElement("article");
  article.className = "message assistant-message";
  article.innerHTML = `<div class="assistant-avatar" aria-hidden="true">!</div><div class="message-content"><div class="message-label">Local RAG</div></div>`;
  const card = document.createElement("div");
  card.className = "error-card";
  card.textContent = message;
  article.querySelector(".message-content").append(card);
  elements.messages.append(article);
}

async function askQuestion(question) {
  if (!state.ready || state.asking || !question.trim()) return;

  state.asking = true;
  elements.welcomeView.hidden = true;
  elements.questionInput.value = "";
  resizeTextarea();
  addUserMessage(question);
  addLoadingMessage();
  elements.composerNote.textContent = `Using ${state.selectedModel === "fast" ? "Qwen2.5 0.5B" : "Qwen3 4B"} entirely on this device.`;
  setComposerAvailability();
  scrollToBottom();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: state.selectedModel }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The local request failed.");
    document.querySelector(".loading-message")?.remove();
    addAssistantMessage(payload);
  } catch (error) {
    document.querySelector(".loading-message")?.remove();
    addErrorMessage(error.message || "The local request failed.");
  } finally {
    state.asking = false;
    elements.composerNote.textContent = "Answers use local models and the indexed PDF collection only.";
    await fetchStatus();
    setComposerAvailability();
    elements.questionInput.focus();
    scrollToBottom();
  }
}

elements.questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(elements.questionInput.value.trim());
});

elements.questionInput.addEventListener("input", resizeTextarea);
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.questionForm.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => askQuestion(button.dataset.question));
});

elements.modelOptions.forEach((button) => {
  button.addEventListener("click", () => selectModel(button.dataset.model));
});

elements.newChatButton.addEventListener("click", () => {
  if (state.asking) return;
  elements.messages.replaceChildren();
  elements.welcomeView.hidden = false;
  elements.questionInput.focus();
});

elements.themeButton.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
});

initializeTheme();
selectModel(localStorage.getItem("local-rag-answer-model") || "fast");
fetchStatus();
state.statusTimer = window.setInterval(fetchStatus, 1000);
