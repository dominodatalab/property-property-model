# Domino-Data-Lab---LS-Demos
Internship Project by Luis Chan


Proteins are chains of amino acids. Inside the cell is mostly water; the cell membrane is oily. Proteins with lots of ‘oily’ amino acids—especially near the start—tend to sit in the membrane; watery ones float inside the cell.

Our tiny ML model turns any sequence into just three numbers: overall oiliness, oiliness at the N-terminus, and length. We train a simple logistic regression on labeled examples, and then output soluble or membrane-bound with a confidence score.

This is intentionally lightweight—fast to train and easy to explain—and it shows Domino’s end-to-end flow: build in a Workspace, optionally retrain as a Job, serve as an Endpoint, and even provide a simple App UI for non-coders
