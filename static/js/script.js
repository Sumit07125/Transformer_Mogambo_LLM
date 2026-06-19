const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");

userInput.addEventListener("keypress", function (event) {
  if (event.key === "Enter") sendMessage();
});

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  // Append user message
  appendMessage("user", message);
  userInput.value = "";

  // Hide suggestion chips if they exist
  const chips = document.getElementById("suggestion-chips");
  if (chips) chips.style.display = "none";

  // Append a loading indicator
  const loadingId = appendLoadingIndicator();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message }),
    });

    const data = await response.json();
    
    // Remove loading indicator and append actual bot response
    removeElement(loadingId);
    appendMessage("bot", data.reply);
  } catch (error) {
    removeElement(loadingId);
    appendMessage("bot", "Error connecting to model.");
  }
}

function appendMessage(sender, text) {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message-wrapper", sender);

  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender);
  msgDiv.innerText = text;

  wrapper.appendChild(msgDiv);
  chatBox.appendChild(wrapper);
  scrollToBottom();
}

function appendLoadingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message-wrapper", "bot");
  const msgId = "msg-" + Date.now();
  wrapper.id = msgId;

  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", "bot", "typing-indicator");
  
  for(let i=0; i<3; i++) {
    const dot = document.createElement("div");
    dot.classList.add("typing-dot");
    msgDiv.appendChild(dot);
  }

  wrapper.appendChild(msgDiv);
  chatBox.appendChild(wrapper);
  scrollToBottom();

  return msgId;
}

function removeElement(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}
