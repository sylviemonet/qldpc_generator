"""Execute each notebook in a fresh kernel using this Python environment."""
from pathlib import Path
import sys
from time import perf_counter
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
print(f"Notebook interpreter: {sys.executable}", flush=True)
for path in sorted(root.joinpath("notebooks").glob("*.ipynb")):
    started = perf_counter()
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    # A globally installed python3 kernelspec can point outside the active venv.
    # Keep nbclient's owned-kernel lifecycle, but choose the invoking interpreter.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client.execute()
    nbformat.write(notebook, path)
    print(f"Executed {path.name} ({perf_counter() - started:.2f}s)", flush=True)
