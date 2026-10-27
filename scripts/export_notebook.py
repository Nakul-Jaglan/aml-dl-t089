"""Self-contained HTML companion with useful descriptions for saved figures."""
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/00_phase1_master.ipynb"
notebook = nbformat.read(path, as_version=4)
descriptions = [
    "Seeded fit-partition examples from all ten clothing categories, each labeled with its original training ID.",
    "Category counts: 4,800 fit-pool and 1,200 validation images in each of the ten categories.",
    "A fit-partition shirt beside the fixed HOG visualization of its local edge directions.",
    "Test accuracy and macro F1 for pixels and HOG across Logistic Regression, kNN, Random Forest and RBF SVM; exact scores are in the preceding table.",
    "Recall heatmap for all ten categories and all eight frozen models, with exact percentages annotated in every cell.",
    "Row-normalized confusion matrix for the validation-selected HOG RBF SVM. True categories are rows and predictions are columns.",
    "Paired RBF SVM error examples in both-correct, only-pixels-correct, only-HOG-correct and both-wrong groups, selected by the smallest original test IDs.",
    "Original official-test image ID zero with its true label and the frozen selected-model prediction.",
]
index = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    for output in cell.get("outputs", []):
        if "image/png" in output.get("data", {}):
            output.metadata["image/png"] = {"alt": descriptions[index]}
            index += 1
assert index == len(descriptions)
nbformat.write(notebook, path)
body, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
soup = BeautifulSoup(body, "html.parser")
images = soup.find_all("img")
assert len(images) == len(descriptions)
for image, description in zip(images, descriptions):
    image["alt"] = description
target = path.with_suffix(".html")
target.write_text(str(soup))
print(f"Saved {target.name}: {len(images)} described figures")
