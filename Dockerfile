FROM continuumio/miniconda3

WORKDIR /mle-training

# Create the environment:
#COPY env.yml .
#RUN conda env create -f env.yml

COPY data data

RUN pip install sklearn

RUN pip install pandas

RUN pip install six

RUN pip install scipy

RUN pip install numpy

ADD dist/housing_project-0.2-py3-none-any.whl .

RUN pip install housing_project-0.2-py3-none-any.whl


# Make RUN commands use the new environment:
#SHELL ["conda", "run", "-n","mle-dev", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure package  is installed"
RUN python -c "import housing_project"

# The code to run when container is started:
COPY test.py .
ENTRYPOINT ["python", "test.py"]
