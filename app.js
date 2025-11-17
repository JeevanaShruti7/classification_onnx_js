
async function runModel() {
    const inputText = document.getElementById("inputData").value.trim();
    const inputArray = inputText.split(",").map(Number);

    const expectedLength = 16; // <-- will be replaced automatically

    if (inputArray.length !== expectedLength) {
        alert(`Enter exactly ${expectedLength} numbers.`);
        return;
    }

    try {
        const session = await ort.InferenceSession.create("model.onnx");

        const inputTensor = new ort.Tensor("float32",
            Float32Array.from(inputArray),
            [1, expectedLength]
        );

        const outputMap = await session.run({ input: inputTensor });
        const logits = Array.from(outputMap.output.data);

        const maxLogit = Math.max(...logits);
        const exps = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = exps.reduce((a,b)=>a+b,0);
        const probs = exps.map(x => x/sumExp);

        const labels = ["M","L","H"];
        const idx = probs.indexOf(Math.max(...probs));

        document.getElementById("output").innerText =
            `Predicted Class: ${labels[idx]} (Prob: ${probs[idx].toFixed(2)})`;

    } catch (err) {
        console.error(err);
        alert("Model failed.");
    }
}
