async function analyze() {
    const input = document.getElementById("userInput").value;
    const outputBox = document.getElementById("output");

    outputBox.innerText = "Analyzing...";

    const response = await fetch(
        "https://api-inference.huggingface.co/models/mrm8488/bert-mini-finetuned-financial-news-sentiment",
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ inputs: input })
        }
    );

    const result = await response.json();
    outputBox.innerText = JSON.stringify(result, null, 2);
}
