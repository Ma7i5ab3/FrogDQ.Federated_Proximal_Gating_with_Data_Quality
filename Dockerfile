FROM continuumio/miniconda3:main

# Install OS-level deps
RUN apt update && apt install -y git gcc wget cmake build-essential && rm -rf /var/lib/apt/lists/*

# Use bash for early steps before the env exists
SHELL ["bash", "-lc"]

WORKDIR /exp

# Bring in env + project metadata needed to resolve deps
COPY environment.yaml .
COPY pyproject.toml .
COPY quail ./quail
COPY README.md .

# Create the conda environment from environment.yaml
RUN conda env create -f environment.yaml && conda clean -afy

# From here on, every RUN executes inside the 'quail' env
SHELL ["conda", "run", "-n", "quail", "/bin/bash", "-c"]

# Ensure poetry is available inside the 'quail' env (skip if environment.yaml already installs it)
RUN python -m pip install --no-cache-dir poetry && poetry config virtualenvs.create false

# Install project deps with Poetry using the env's Python
RUN poetry install --no-interaction --no-ansi --no-root

# Make runtime commands also execute inside the env
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quail"]
CMD ["bash"]
