class LogMessage {
    constructor(message, toolNames) {
        const logLevel = this.determineLogLevel(message);
        const highlightedMessage = this.highlightToolNames(message, toolNames);
        this.$elem = $('<div>').addClass('log-' + logLevel).html(highlightedMessage + '\n');
    }

    determineLogLevel(message) {
        if (message.startsWith('DEBUG')) {
            return 'debug';
        } else if (message.startsWith('INFO')) {
            return 'info';
        } else if (message.startsWith('WARNING')) {
            return 'warning';
        } else if (message.startsWith('ERROR')) {
            return 'error';
        } else {
            return 'default';
        }
    }

    highlightToolNames(message, toolNames) {
        let highlightedMessage = message;
        toolNames.forEach(function(toolName) {
            const regex = new RegExp('\\b' + toolName + '\\b', 'gi');
            highlightedMessage = highlightedMessage.replace(regex, '<span class="tool-name">' + toolName + '</span>');
        });
        return highlightedMessage;
    }
}

class Dashboard {
    constructor() {
        let self = this;

        this.toolNames = [];
        this.currentMaxIdx = -1;
        this.pollInterval = null;
        this.$logContainer = $('#log-container');
        this.$errorContainer = $('#error-container');
        this.$loadButton = $('#load-logs');
        this.$clearButton = $('#clear-logs');

        // register event handlers
        this.$loadButton.click(this.loadLogs.bind(this));
        this.$clearButton.click(this.clearLogs.bind(this));

        // initialize the application
        this.loadToolNames().then(function() {
            // Load logs on page load after tool names are loaded
            self.loadLogs();
        });
    }

    displayLogMessage(message) {
        $('#log-container').append(new LogMessage(message, this.toolNames).$elem);
    }

    loadToolNames() {
        let self = this;
        return $.ajax({
            url: '/get_tool_names',
            type: 'GET',
            success: function(response) {
                self.toolNames = response.tool_names || [];
                console.log('Loaded tool names:', toolNames);
            },
            error: function(xhr, status, error) {
                console.error('Error loading tool names:', error);
            }
        });
    }

    loadLogs() {
        console.log("load", this);
        let self = this;

        // Disable button and show loading state
        self.$loadButton.prop('disabled', true).text('Loading...');
        self.$errorContainer.empty();

        // Make API call
        $.ajax({
            url: '/get_log_messages',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                start_idx: 0
            }),
            success: function(response) {
                // Clear existing logs
                self.$logContainer.empty();

                // Update max_idx
                self.currentMaxIdx = response.max_idx || -1;

                // Display each log message
                if (response.messages && response.messages.length > 0) {
                    response.messages.forEach(function(message) {
                        self.displayLogMessage(message);
                    });

                    // Auto-scroll to bottom
                    const logContainer = $('#log-container')[0];
                    logContainer.scrollTop = logContainer.scrollHeight;
                } else {
                    $('#log-container').html('<div class="loading">No log messages found.</div>');
                }

                // Start periodic polling for new logs
                self.startPeriodicPolling();
            },
            error: function(xhr, status, error) {
                console.error('Error loading logs:', error);
                self.$errorContainer.html('<div class="error-message">Error loading logs: ' +
                    (xhr.responseJSON ? xhr.responseJSON.detail : error) + '</div>');
            },
            complete: function() {
                // Re-enable button
                self.$loadButton.prop('disabled', false).text('Load Logs');
            }
        });
    }

    pollForNewLogs() {
        let self = this;
        console.log("poll", this);
        $.ajax({
            url: '/get_log_messages',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                start_idx: self.currentMaxIdx + 1
            }),
            success: function(response) {
                // Only append new messages if we have any
                if (response.messages && response.messages.length > 0) {
                    let wasAtBottom = false;
                    const logContainer = $('#log-container')[0];

                    // Check if user was at the bottom before adding new logs
                    if (logContainer.scrollHeight > 0) {
                        wasAtBottom = (logContainer.scrollTop + logContainer.clientHeight) >= (logContainer.scrollHeight - 10);
                    }

                    // Append new messages
                    response.messages.forEach(function(message) {
                        self.displayLogMessage(message);
                    });

                    // Update max_idx
                    self.currentMaxIdx = response.max_idx || self.currentMaxIdx;

                    // Auto-scroll to bottom if user was already at bottom
                    if (wasAtBottom) {
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                } else {
                    // Update max_idx even if no new messages
                    self.currentMaxIdx = response.max_idx || self.currentMaxIdx;
                }
            },
            error: function(xhr, status, error) {
                console.error('Error polling for new logs:', error);
            }
        });
    }

    startPeriodicPolling() {
        // Clear any existing interval
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        // Start polling every second (1000ms)
        this.pollInterval = setInterval(this.pollForNewLogs.bind(this), 1000);
    }

    stopPeriodicPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    clearLogs() {
        this.stopPeriodicPolling();
        this.$logContainer.empty();
        this.$errorContainer.empty();
        this.currentMaxIdx = -1;
    }
}