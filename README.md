# NLP - Abbreviation Detection deployment
NLP module for the MSc AI course at the University of Surrey.

Students:
* Pavlos Aquino-Ellul
* Antoine Edy
* Mingxi Li

Module leader:
* Diptesh Kanojia


----
## Files

__This repository contains the following files:__
- `app.py`: Main file to run the web-app using Flask
- `deploy.py`: Deploy the app locally (with or without Docker)
- `Dockerfile`: Dockerfile to create a docker image
- `log.txt`: Log file
- `requirements.txt`: The list of required packages
- a `utils` folder used in `deploy.py`:
    - `simple_test.py`: Simple test to check if the app is running fine and that the model implemented gives good results
    - `stress_test.py`: Stress test to check if the app can handle fancy inputs
    - `train.py`: Train the model
- A `template` folder:
    - `index.html`: HTML page for the web-app
- A `static` folder:
    - `style.css`: CSS file for the web-app
- A `notebooks` folder:
    - `stress_testing.ipynb`: Jupyter notebook to stress-test the app
    - `test_webapp.ipynb`: Jupyter notebook to test the web-app
    - `training.ipynb`: Jupyter notebook to train the model

----

## Usage

#### 1. Deployment and use of the Docker image

For the __simplest local launch of the web app__ using Flask:
```bash
python app.py
```

For a simple __local deployment__, after passing basic tests:
```bash
python deploy.py
```
After the tests are done, the app will be running locally on the link given in the terminal.

In order to deploy it with __Docker__ (so to create a Docker image to share), you can add the following argument:
```bash
python deploy.py --create-docker-image
```

We uploaded the Docker image on __Docker Hub__, so you can pull it using the following command:
```bash
docker pull antoineedy/nlp-app
```

To run the Docker image, you can use the following command:
```bash
docker run -p 5000:5000 antoineedy/nlp-app
```

#### 2. Training the model

You can train the model using a custom dataset by running the following command:
```bash
python utils/train.py --args
```
The arguments are:
* `--dataset`: the dataset to train on (default: surrey-nlp/PLOD-CW)
* `--model_checkpoint`: the model checkpoint to train (default: antoineedy/stanford-deidentifier-base-finetuned-ner)
* `--num_train_epochs`: number of training epochs (default: 60)
* `--learning_rate`: learning rate (default: 2e-5)
* `--model_name`: model name (default: model)
* `--push_to_hub`: push the model to the hub (default: True)

Then, you can modify the `app.py` file to use the newly trained model.
