// // document.getElementById('upload-form').addEventListener('submit', async (event) => {
// //     event.preventDefault();

// //     const fileInput = document.getElementById('file-input');
// //     const file = fileInput.files[0];

// //     if (!file) {
// //         alert('Please select a file.');
// //         return;
// //     }

// //     const formData = new FormData();
// //     formData.append('file', file);

// //     try {
// //         const response = await fetch('http://127.0.0.1:8000/predict/', {
// //             method: 'POST',
// //             body: formData
// //         });

// //         if (!response.ok) {
// //             throw new Error('Network response was not ok.');
// //         }

// //         const result = await response.json();
// //         document.getElementById('class-name').textContent = `Class: ${result.predicted_class}`;
// //         document.getElementById('confidence').textContent = `Confidence: ${result.confidence}`;
// //     } catch (error) {
// //         console.error('There was a problem with the fetch operation:', error);
// //     }
// // });





// document.getElementById('upload-form').addEventListener('submit', async (event) => {
//     event.preventDefault(); // Prevent default form submission behavior

//     const formData = new FormData();
//     const fileInput = document.getElementById('file-input');
    
//     if (!fileInput.files.length) {
//         alert('Please select a file.');
//         return;
//     }

//     formData.append('file', fileInput.files[0]);

//     try {
//         const response = await fetch('http://localhost:8000/predict/', {
//             method: 'POST',
//             body: formData,
//         });

//         if (!response.ok) {
//             throw new Error(`HTTP error! Status: ${response.status}`);
//         }

//         const data = await response.json();
//         console.log('Response data:', data); // Debugging statement

//         document.getElementById('predicted-class').innerText = `Predicted Class: ${data.predicted_class}`;
//         document.getElementById('confidence').innerText = `Confidence: ${data.confidence}`;
//         document.getElementById('result').classList.remove('hidden'); // Show the result section
//     } catch (error) {
//         console.error('Error:', error);
//         alert('An error occurred while processing the image. Check the console for details.');
//     }
// });



// scripts.js

document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
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
        // Optional: Display a preview of the selected file
        // const reader = new FileReader();
        // reader.onload = (e) => {
        //     document.getElementById('image-preview').src = e.target.result;
        // };
        // reader.readAsDataURL(file);
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
