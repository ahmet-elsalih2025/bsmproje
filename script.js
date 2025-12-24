document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictionSpan = document.getElementById('prediction');
    const confidenceSpan = document.getElementById('confidence');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Set initial canvas background to black
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Drawing settings
    ctx.strokeStyle = "white";
    ctx.lineWidth = 25; // Thick lines for better recognition
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        const [x, y] = getCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();

        [lastX, lastY] = [x, y];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        let clientX, clientY;

        if (e.type.includes('touch')) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        return [clientX - rect.left, clientY - rect.top];
    }

    // Event Listeners for Mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Event Listeners for Touch
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault(); // Prevent scrolling
        startDrawing(e);
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e);
    });
    canvas.addEventListener('touchend', stopDrawing);

    // Clear Canvas
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionSpan.textContent = "-";
        confidenceSpan.textContent = "-%";
    });

    // Predict
    predictBtn.addEventListener('click', () => {
        const dataURL = canvas.toDataURL('image/png');

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataURL }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    predictionSpan.textContent = data.prediction;
                    confidenceSpan.textContent = `%${data.confidence}`;
                }
            })
            .catch((error) => {
                console.error('Error:', error);
                alert('Something went wrong!');
            });
    });
});
