document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');

    // Handle file selection
    dropArea.addEventListener('click', () => fileInput.click());

    // Handle drag-and-drop
    dropArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropArea.classList.add('hover');
    });

    dropArea.addEventListener('dragleave', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropArea.classList.remove('hover');
    });

    dropArea.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropArea.classList.remove('hover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        if (file) {
            fileNameDisplay.innerText = `Selected File: ${file.name}`;
        }
    }

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const formData = new FormData();
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a file.');
            return;
        }

        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/predict/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Response data:', data);

            document.getElementById('predicted-class').innerText = `Predicted Class: ${data.predicted_class}`;
            document.getElementById('confidence').innerText = `Confidence: ${data.confidence}`;
            document.getElementById('result').classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the image. Check the console for details.');
        }
    });
});
