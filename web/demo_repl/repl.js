const output = document.getElementById("output");

const editor = CodeMirror.fromTextArea(document.getElementById("code"), {
	mode: {
		name: "python",
		version: 3,
		singleLineStringErrors: false
	},
	lineNumbers: true,
	indentUnit: 4,
	matchBrackets: true
});

editor.setValue(`
import asyncio
await asyncio.sleep("")`);
output.value = "Initializing...\n";

async function main() {
	let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.2/full/",
        stdout: (msg) => addToOutput(msg),
        stderr: (msg) => addToOutput(msg)
    });

    // Put freewillai code in pydiode filesystem
    let zipResponse = await fetch("freewillai.zip");
    let zipBinary = await zipResponse.arrayBuffer();

    // support tensorflow in webassembly
    tf.setBackend('wasm').then(() => main());

    pyodide.unpackArchive(zipBinary, "zip");
    await pyodide.loadPackage('micropip')
    pyodide.runPythonAsync(`
        import micropip
        await micropip.install([
            "aioipfs",
            "eth_account",
            "keras",
            "numpy",
            "onnx",
            "onnxruntime",
            "polars",
            "python-dotenv",
            "skl2onnx",
            "tensorflow",
            "tf2onnx",
            "torch",
            "web3",
            "scikit-learn",
            "opencv-python",
        ])
    `)

    addToOutput("Console Ready", "")
	return pyodide;
};

let pyodideReadyPromise = main();

function isPromise(p) {
  if (typeof p === 'object' && typeof p.then === 'function') {
    return true;
  }
}

function showOutput(msg, prefix) {
    if (msg == undefined || msg == "undefined") {
        return
    }
    output.value += prefix + msg + "\n";
}

function addToOutput(s, prefix=">> ") {
    if (s == "Python initialization complete") {
        return
    } else if (isPromise(s)) {
        s.then((value) => {showOutput(value, prefix);})
        s.catch((value) => {showOutput(value, prefix);})
    } else {
        showOutput(s, prefix);
    }
}

async function evaluatePython() {
	let pyodide = await pyodideReadyPromise;
	try {
		let output = pyodide.runPythonAsync(editor.getValue());
		addToOutput(output);
	} catch (err) {
		addToOutput(err);
	}
}

