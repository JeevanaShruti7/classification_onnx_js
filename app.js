async function runModel() {
  const inputText = document.getElementById("inputData").value.trim();
  const inputArray = inputText.split(",").map(Number);

  // Load ONNX model
  const session = await ort.InferenceSession.create("model.onnx");

  // Create input tensor
  const inputTensor = new ort.Tensor("float32", Float32Array.from(inputArray), [1, inputArray.length]);

  // Run inference
  const outputMap = await session.run({ input: inputTensor });
  const output = outputMap.output.data;

  document.getElementById("output").innerText = `Predicted: ${output}`;
}
