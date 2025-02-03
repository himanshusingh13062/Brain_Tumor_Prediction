document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.querySelector('input[name="file"]');
    const file = fileInput.files[0];

    if (file) {
        formData.append('file', file);

        // Send the image to the Flask server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction) {
                // Update result text and image
                document.getElementById('result-text').innerText = `Prediction: ${data.prediction} (Confidence: ${(data.confidence * 100).toFixed(2)}%)`;
                const resultImage = document.getElementById('result-image');
                resultImage.src = data.tumor_image;
                resultImage.style.display = 'block';
                document.getElementById('result-description').innerText = `Description: ${data.description}`;
            } else {
                document.getElementById('result-text').innerText = 'Error: ' + data.error;
            }
        })
        .catch(error => {
            document.getElementById('result-text').innerText = 'Error: ' + error.message;
        });
    }
});
