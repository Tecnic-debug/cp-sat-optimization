🛠️ Setup & Installation (From Scratch)
Follow these steps to get the optimizer running on your local machine:

1. Clone the Repository

Bash

git clone https://github.com/your-username/cng-optimizer.git
cd cng-optimizer
2. Set up a Virtual Environment (Recommended)

Bash

python -m venv venv
# On Windows:
source venv/Scripts/activate
# On Mac/Linux:
source venv/bin/activate
3. Install Dependencies You must install the Google OR-Tools and Pandas libraries:

Bash

pip install ortools pandas
4. Prepare Your Data

Ensure your input CSV matches the schema required by load_data().

Update the file paths in the script (lines 11-13) to point to your local directory.

5. Run the Optimizer

Bash

python cng_route_optimization.py
📂 Output Description
Upon a successful run, the tool generates two primary files:

Output CSV: A detailed trip-by-trip breakdown of LCV assignments, quantities, and calculated costs.

Summary JSON: A high-level report showing total dispatch achievement (%), total trips, and compressor capacity utilization.
