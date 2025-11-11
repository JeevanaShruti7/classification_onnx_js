async function runModel() {
    const inputText = document.getElementById("inputData").value.trim();
    const inputArray = inputText.split(",").map(Number);

    const expectedLength = 12; // <-- change to X_train.shape[1]
    if (inputArray.length !== expectedLength) {
        alert(`Please enter exactly ${expectedLength} numbers.`);
        return;
    }

    // Load ONNX model
    const session = await ort.InferenceSession.create("model.onnx");

    // Create input tensor
    const inputTensor = new ort.Tensor("float32", Float32Array.from(inputArray), [1, inputArray.length]);

    // Run inference
    const outputMap = await session.run({ input: inputTensor });
    const output = outputMap.output.data;

    // Map output to predicted class
    const classLabels = ["M", "L", "H"]; // <-- replace with your target_le.classes_
    const predictedClassIndex = output.indexOf(Math.max(...output));

    document.getElementById("output").innerText = `Predicted Class: ${classLabels[predictedClassIndex]}`;
}
