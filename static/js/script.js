/* VCare AI — Chat Script (index.html) */

document.addEventListener("DOMContentLoaded", () => {
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesList = document.getElementById("messagesList");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const modelStatus = document.getElementById("modelStatus");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("sidebar");

  let chatStarted = false;

  // -- Sidebar --
  sidebarToggle?.addEventListener("click", () =>
    sidebar.classList.toggle("collapsed"),
  );
  mobileMenuBtn?.addEventListener("click", () =>
    sidebar.classList.toggle("open"),
  );

  // -- Model Status --
  function checkModelStatus() {
    fetch("/health")
      .then((r) => r.json())
      .then((d) => {
        const dot = modelStatus.querySelector(".status-dot");
        const text = modelStatus.querySelector(".status-text");
        if (d.model_loaded) {
          dot.classList.add("active");
          text.textContent = "Model Ready";
        } else {
          dot.classList.remove("active");
          text.textContent = "Loading model…";
          setTimeout(checkModelStatus, 4000);
        }
      })
      .catch(() => {
        const text = modelStatus?.querySelector(".status-text");
        if (text) text.textContent = "Offline";
      });
  }

  checkModelStatus();

  // -- Auto-resize textarea --
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + "px";
  });

  // -- Send on Enter (Shift+Enter = newline) --
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      triggerSend();
    }
  });

  sendBtn.addEventListener("click", triggerSend);

  function triggerSend() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendMessage(msg);
  }

  // -- Suggestion chips --
  window.fillSuggestion = (text) => {
    chatInput.value = text;
    chatInput.focus();
    triggerSend();
  };

  // -- Send message flow --
  function sendMessage(text) {
    if (!chatStarted) {
      chatStarted = true;
      welcomeScreen.style.display = "none";
      messagesList.classList.add("visible");
    }

    appendMessage(text, "user");
    const typingEl = appendTyping();

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    })
      .then((r) => r.json())
      .then((d) => {
        removeTyping(typingEl);
        appendMessage(
          d.response || "Sorry, I couldn't generate a response.",
          "assistant",
        );
      })
      .catch(() => {
        removeTyping(typingEl);
        appendMessage(
          "Connection error. Please ensure the server is running and the model is loaded.",
          "assistant",
        );
      });
  }

  function appendMessage(text, role) {
    const msgEl = document.createElement("div");
    msgEl.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML =
      role === "user"
        ? '<i class="fas fa-user"></i>'
        : '<i class="fas fa-robot"></i>';

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = text;

    msgEl.appendChild(avatar);
    msgEl.appendChild(bubble);
    messagesList.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
  }

  function appendTyping() {
    const msgEl = document.createElement("div");
    msgEl.className = "message assistant";
    msgEl.id = "typingIndicator";

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = '<i class="fas fa-robot"></i>';

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>`;

    msgEl.appendChild(avatar);
    msgEl.appendChild(bubble);
    messagesList.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
  }

  function removeTyping(el) {
    el?.remove();
  }

  function scrollToBottom() {
    const chatArea = document.getElementById("chatArea");
    chatArea.scrollTop = chatArea.scrollHeight;
  }
});
