### Clone repository and install dependencies

Clone this repository (requires git ssh keys)

    git clone git@github.com:fraunhoferportugal/tsfel-tutorials.git
    cd tsfel-tutorials

Create a python environment

    conda create -y -n tsfel-tutorials python=3.11
    conda activate tsfel-tutorials

Install dependencies

    pip install -r requirements.txt

Register the environment as a Jupyter Kernel

    python -m ipykernel install --user --name="tsfel-tutorials"