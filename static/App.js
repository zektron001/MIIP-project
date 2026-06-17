const API_BASE = "";

let mode = "normal";
const messagesEl = document.getElementById("messages");
const emptyEl = document.getElementById("empty");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const modesEl = document.getElementById("modes");

// mode switching
modesEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-pill");
  if (!btn) return;
  mode = btn.dataset.mode;
  [...modesEl.children].forEach((b) =>
    b.setAttribute("aria-pressed", String(b === btn)),
  );
  inputEl.focus();
});

// auto-grow textarea
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = inputEl.scrollHeight + "px";
});

// Enter to send, Shift+Enter for newline
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendEl.addEventListener("click", sendMessage);

function addBubble(role, text, pages) {
  if (emptyEl) emptyEl.remove();
  const row = document.createElement("div");
  row.className = "row " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  if (pages && pages.length) {
    const src = document.createElement("div");
    src.className = "sources";
    pages.forEach((p) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = "p. " + p;
      src.appendChild(chip);
    });
    bubble.appendChild(src);
  }
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function addTyping() {
  if (emptyEl && emptyEl.parentNode) emptyEl.remove();
  const row = document.createElement("div");
  row.className = "row bot";
  row.innerHTML =
    '<div class="bubble"><div class="dots"><i></i><i></i><i></i></div></div>';
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  addBubble("user", text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendEl.disabled = true;

  const typing = addTyping();

  try {
    const res = await fetch(API_BASE + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, mode }),
    });
    typing.remove();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addBubble("bot", "Error: " + (err.detail || res.statusText));
    } else {
      const data = await res.json();
      addBubble("bot", data.answer, data.pages);
    }
  } catch (e) {
    typing.remove();
    addBubble("bot", "Could not reach the server. Is uvicorn running?");
  } finally {
    sendEl.disabled = false;
    inputEl.focus();
  }
}
