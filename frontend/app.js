/**
 * HR Policy Assistant - Frontend JavaScript
 * Handles chat UI, API communication, and session management
 */

// ========== State Management ==========

const state = {
    sessionId: null,
    isConnected: false,
    isProcessing: false,
    config: {}
};

// ========== DOM Elements ==========

const elements = {
    chatContainer: document.getElementById('chatContainer'),
    questionInput: document.getElementById('questionInput'),
    sendBtn: document.getElementById('sendBtn'),
    clearBtn: document.getElementById('clearBtn'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    memoryToggle: document.getElementById('memoryToggle'),
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    sessionInfo: document.getElementById('sessionInfo'),
    settingsPanel: document.getElementById('settingsPanel'),
    settingsToggleBtn: document.getElementById('settingsToggleBtn'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    topKSlider: document.getElementById('topKSlider'),
    topKValue: document.getElementById('topKValue'),
    configDisplay: document.getElementById('configDisplay')
};

// ========== API Functions ==========

const API = {
    baseURL: window.location.origin,

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Request failed');
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    },

    async ask(question, sessionId, topK, useMemory) {
        return this.request('/api/ask', {
            method: 'POST',
            body: JSON.stringify({
                question,
                session_id: sessionId,
                top_k: topK,
                use_memory: useMemory
            })
        });
    },

    async createSession() {
        return this.request('/api/session/new', { method: 'POST' });
    },

    async clearHistory(sessionId) {
        return this.request(`/api/session/${sessionId}/clear`, { method: 'POST' });
    },

    async getConfig() {
        return this.request('/api/config');
    },

    async healthCheck() {
        return this.request('/api/health');
    }
};

// ========== UI Functions ==========

const UI = {
    updateStatus(status, text) {
        elements.statusIndicator.className = `status-indicator ${status}`;
        elements.statusText.textContent = text;
        state.isConnected = status === 'connected';
    },

    updateSessionInfo(sessionId) {
        if (sessionId) {
            const shortId = sessionId.substring(0, 8);
            elements.sessionInfo.textContent = `Session: ${shortId}...`;
        } else {
            elements.sessionInfo.textContent = '';
        }
    },

    addMessage(role, content, metadata = {}) {
        // Remove welcome message if present
        const welcomeMsg = elements.chatContainer.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const icon = role === 'user' ? '👤' : '🤖';
        const roleText = role === 'user' ? 'You' : 'Assistant';

        let html = `
            <div class="message-header ${role}">
                <span>${icon} ${roleText}</span>
            </div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;

        // Add metadata footer for assistant messages
        if (role === 'assistant' && metadata.total_time) {
            html += `
                <div class="message-footer">
                    <span>⏱️ ${metadata.total_time}s</span>
                    <span>📄 ${metadata.num_sources || 0} sources</span>
                </div>
            `;
        }

        // Add collapsible sources if available
        if (role === 'assistant' && metadata.sources && metadata.sources.length > 0) {
            const sourcesId = `sources-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

            html += `
                <div class="sources-container">
                    <button class="sources-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'; this.textContent = this.nextElementSibling.style.display === 'none' ? '📚 Show Sources (${metadata.sources.length})' : '📚 Hide Sources'">
                        📚 Show Sources (${metadata.sources.length})
                    </button>
                    <div class="sources" id="${sourcesId}" style="display: none;">
            `;

            metadata.sources.slice(0, 3).forEach((source, idx) => {
                html += `
                    <div class="source-item">
                        <strong>${idx + 1}. ${source.section_title}</strong>
                        (Section ${source.section_number}, Pages ${source.page_start}-${source.page_end})
                        <br>Score: ${source.score.toFixed(4)}
                    </div>
                `;
            });

            html += '</div></div>';
        }

        messageDiv.innerHTML = html;
        elements.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    },

    addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant';
        loadingDiv.id = 'loading-message';
        loadingDiv.innerHTML = `
            <div class="message-header assistant">
                <span>🤖 Assistant</span>
            </div>
            <div class="loading">
                <span>Thinking</span>
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        elements.chatContainer.appendChild(loadingDiv);
        this.scrollToBottom();
    },

    removeLoadingMessage() {
        const loadingMsg = document.getElementById('loading-message');
        if (loadingMsg) {
            loadingMsg.remove();
        }
    },

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `❌ Error: ${message}`;
        elements.chatContainer.appendChild(errorDiv);
        this.scrollToBottom();

        // Auto-remove after 5 seconds
        setTimeout(() => errorDiv.remove(), 5000);
    },

    scrollToBottom() {
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    },

    setProcessing(isProcessing) {
        state.isProcessing = isProcessing;
        elements.sendBtn.disabled = isProcessing || !elements.questionInput.value.trim();
        elements.questionInput.disabled = isProcessing;
        elements.clearBtn.disabled = isProcessing || !state.sessionId;
    },

    updateConfig(config) {
        state.config = config;
        let configText = '';
        for (const [key, value] of Object.entries(config)) {
            configText += `${key}: ${value}\n`;
        }
        elements.configDisplay.textContent = configText || 'No configuration loaded';
    }
};

// ========== Event Handlers ==========

async function handleSendMessage() {
    const question = elements.questionInput.value.trim();
    if (!question || state.isProcessing) return;

    // Get settings
    const topK = parseInt(elements.topKSlider.value);
    const useMemory = elements.memoryToggle.checked;

    // Add user message to UI
    UI.addMessage('user', question);
    elements.questionInput.value = '';
    UI.setProcessing(true);

    // Show loading
    UI.addLoadingMessage();

    try {
        // Send to API
        const response = await API.ask(question, state.sessionId, topK, useMemory);

        // Update session ID if new
        if (response.session_id && response.session_id !== state.sessionId) {
            state.sessionId = response.session_id;
            UI.updateSessionInfo(state.sessionId);
        }

        // Remove loading
        UI.removeLoadingMessage();

        // Add assistant response
        UI.addMessage('assistant', response.answer, {
            total_time: response.total_time,
            num_sources: response.num_sources,
            sources: response.sources
        });

        // Show warning if any
        if (response.warning) {
            console.warn('Warning:', response.warning);
        }

    } catch (error) {
        UI.removeLoadingMessage();
        UI.showError(error.message);
    } finally {
        UI.setProcessing(false);
        elements.questionInput.focus();
    }
}

async function handleClearHistory() {
    if (!state.sessionId || state.isProcessing) return;

    if (!confirm('Clear conversation history? This cannot be undone.')) {
        return;
    }

    try {
        await API.clearHistory(state.sessionId);

        // Clear UI
        elements.chatContainer.innerHTML = `
            <div class="welcome-message">
                <h2>🗑️ History Cleared</h2>
                <p>Your conversation history has been cleared. I won't remember previous questions.</p>
            </div>
        `;

        UI.updateStatus('connected', 'History cleared');
        setTimeout(() => UI.updateStatus('connected', 'Connected'), 2000);

    } catch (error) {
        UI.showError(`Failed to clear history: ${error.message}`);
    }
}

async function handleNewSession() {
    if (state.isProcessing) return;

    if (state.sessionId && !confirm('Start a new session? Current conversation will be lost.')) {
        return;
    }

    try {
        const response = await API.createSession();
        state.sessionId = response.session_id;
        UI.updateSessionInfo(state.sessionId);

        // Clear UI
        elements.chatContainer.innerHTML = `
            <div class="welcome-message">
                <h2>✨ New Session Started</h2>
                <p>Ask me anything about company policies!</p>
            </div>
        `;

        UI.updateStatus('connected', 'New session started');
        setTimeout(() => UI.updateStatus('connected', 'Connected'), 2000);
        elements.questionInput.focus();

    } catch (error) {
        UI.showError(`Failed to create session: ${error.message}`);
    }
}

function handleToggleSettings() {
    elements.settingsPanel.style.display = 'flex';
}

function handleCloseSettings() {
    elements.settingsPanel.style.display = 'none';
}

function handleExampleClick(e) {
    if (e.target.classList.contains('example-btn')) {
        const question = e.target.getAttribute('data-question');
        elements.questionInput.value = question;
        elements.questionInput.focus();
        handleInputChange(); // Enable send button
        handleInputResize(); // Resize textarea
    }
}

function handleTopKChange() {
    elements.topKValue.textContent = elements.topKSlider.value;
}

function handleInputChange() {
    const hasText = elements.questionInput.value.trim().length > 0;
    elements.sendBtn.disabled = !hasText || state.isProcessing;
}

// Auto-resize textarea
function handleInputResize() {
    elements.questionInput.style.height = 'auto';
    elements.questionInput.style.height = elements.questionInput.scrollHeight + 'px';
}

// ========== Initialization ==========

async function initialize() {
    console.log('Initializing HR Policy Assistant...');

    // Update status
    UI.updateStatus('connecting', 'Connecting...');

    try {
        // Load configuration
        const config = await API.getConfig();
        UI.updateConfig(config);

        // Set default top_k from config
        elements.topKSlider.value = config.retrieval_top_k || 5;
        elements.topKValue.textContent = elements.topKSlider.value;

        // Health check
        await API.healthCheck();

        // Create initial session
        const response = await API.createSession();
        state.sessionId = response.session_id;
        UI.updateSessionInfo(state.sessionId);

        // Update status
        UI.updateStatus('connected', 'Connected');

        // Enable controls
        elements.clearBtn.disabled = false;
        elements.questionInput.focus();

        console.log('Initialization complete');

    } catch (error) {
        console.error('Initialization failed:', error);
        UI.updateStatus('error', 'Connection failed');
        UI.showError(`Failed to connect: ${error.message}`);
    }
}

// ========== Event Listeners ==========

elements.sendBtn.addEventListener('click', handleSendMessage);
elements.clearBtn.addEventListener('click', handleClearHistory);
elements.newSessionBtn.addEventListener('click', handleNewSession);
elements.settingsToggleBtn.addEventListener('click', handleToggleSettings);
elements.closeSettingsBtn.addEventListener('click', handleCloseSettings);
elements.topKSlider.addEventListener('input', handleTopKChange);
elements.questionInput.addEventListener('input', handleInputChange);
elements.questionInput.addEventListener('input', handleInputResize);

// Example question clicks
document.addEventListener('click', handleExampleClick);

// Enter to send (Shift+Enter for new line)
elements.questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
});

// ========== Start Application ==========

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}

// Periodic health check (every 60 seconds)
setInterval(async () => {
    if (state.isConnected && !state.isProcessing) {
        try {
            await API.healthCheck();
        } catch (error) {
            console.error('Health check failed:', error);
            UI.updateStatus('error', 'Connection lost');
        }
    }
}, 60000);

console.log('HR Policy Assistant loaded');
