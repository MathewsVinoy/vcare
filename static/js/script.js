document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.querySelector('.send-btn');
    const messagesContainer = document.getElementById('messagesContainer');
    const chatContainer = document.querySelector('.chat-container');
    let chatStarted = false;

    // Handle Enter key press
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && chatInput.value.trim()) {
            sendMessage(chatInput.value.trim());
            chatInput.value = '';
        }
    });

    // Handle send button click
    sendBtn.addEventListener('click', function() {
        if (chatInput.value.trim()) {
            sendMessage(chatInput.value.trim());
            chatInput.value = '';
        }
    });

    // Handle action button clicks
    document.querySelectorAll('.action-btn').forEach(btn => {
        if (!btn.hasAttribute('onclick')) {
            btn.addEventListener('click', function() {
                const action = this.querySelector('span').textContent;
                alert(`${action} feature - Coming soon!`);
            });
        }
    });

    function sendMessage(message) {
        // First message - transform UI
        if (!chatStarted) {
            chatStarted = true;
            
            // Add chat-active class to hide welcome section and show messages
            chatContainer.classList.add('chat-active');
            
            
        }
        
        // Add user message immediately
        addMessage(message, 'user');

        // Simulate assistant response or send to server
        setTimeout(() => {
            // Try to send to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                // Add assistant response from server
                addMessage(data.response, 'assistant');
            })
            .catch(error => {
                console.error('Error:', error);
                // Provide a mock response if server fails
                addMessage('Terima kasih atas pesannya! Saya siap membantu Anda. Silakan bertanya apa saja yang Anda butuhkan.', 'assistant');
            });
        }, 500);
    }

    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});
