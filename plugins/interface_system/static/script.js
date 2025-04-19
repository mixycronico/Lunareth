const ws = new WebSocket(`ws://${window.location.host}/ws`);
const messages = document.getElementById('messages');
const input = document.getElementById('input');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const div = document.createElement('div');
    div.textContent = data.message;
    if (data.data) {
        div.textContent += ': ' + JSON.stringify(data.data, null, 2);
    }
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
};

function sendMessage() {
    const command = input.value;
    if (command) {
        ws.send(JSON.stringify({ comando: command }));
        input.value = '';
    }
}

function sendCommand(command) {
    ws.send(JSON.stringify({ comando: command }));
}

function promptPlugin(command) {
    const plugin = prompt('Nombre del plugin:');
    if (plugin) {
        ws.send(JSON.stringify({ comando: `${command} ${plugin}` }));
    }
}