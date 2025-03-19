async function uploadImage() {
    try {
        let fileInput = document.getElementById("fileInput");
        let formData = new FormData();
        formData.append("file", fileInput.files[0]);

        let response = await fetch("http://127.0.0.1:8000/uploads", {
            method: "POST",
            body: formData
        });

        let data = await response.json();
        console.log(data)
        document.getElementById("result").innerText =
            `Rwandan ID: ${data.is_rwandan_id ? "Confirmed" : "Not Confirmed"}
        \nExtracted Text: ${data.extracted_texts.join(", ")}`;
    } catch (error) {
        console.log(error)
    }
}
