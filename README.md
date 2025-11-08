# Project Checkpoint: Running the BabyLM Experiments
This is my version of the MAO-CLIMB repository for my project.

The original code is a bit old and was made for a specific computer cluster, so I had to make a lot of fixes to get it running in a normal environment. This guide explains all the steps I took to get the experiments working.

## 1. Getting Everything Set Up

The original `setup.sh` script won't work. Instead, we have to do it manually.

#### Step 1: Clone this Repo
First, clone this repository (not the original one!).
```bash
git clone https://github.com/Peter-Shamoun/Quarter-1-Project-Checkpoint.git
cd Quarter-1-Project-Checkpoint
```

#### Step 2: Create a Python Environment
We need a special environment to install all the packages.
```bash
# Install virtualenv if you don't have it
pip install virtualenv

# Create and activate the environment
virtualenv -p python3 env
source env/bin/activate
```

#### Step 3: Install the Dependencies (The Tricky Part)
The `requirements.txt` file has old package versions that don't work anymore. We have to install them in a specific way.
```bash
# Install the packages but ignore their old dependencies
pip install -r requirements.txt --no-deps

# Now, install modern, working versions of the important libraries
pip install numpy pandas transformers datasets hydra-core wandb pre-commit nltk accelerate

# Finally, install a compatible version of PyTorch (this example uses CUDA 11.8)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Download the Data
The code can't download the data from Hugging Face anymore, so we have to get it ourselves.
```bash
# Create a folder for our data
mkdir local_data

# Download the training and validation sets
wget https://osf.io/s524d/download -O local_data/train_10M.zip
wget https://osf.io/e742v/download -O local_data/dev.zip

# Unzip them
cd local_data
unzip train_10M.zip
unzip dev.zip
cd ..
```

#### Step 5: Add Your Hugging Face Tokens
Even though we're using local data, the tokenizer still needs to be downloaded. Create a file named `.env` for your tokens.
```bash
echo 'export HF_READ_TOKEN=hf_...[your_token_here]' > .env
echo 'export HF_WRITE_TOKEN=hf_...[your_token_here]' >> .env
```

## 2. Training the Models 🚀

Now you're ready to train! The plan is to train three different models. You can run each of these commands in a separate terminal to run them at the same time.

**IMPORTANT**: In every new terminal, you have to activate the environment and load your tokens first!
```bash
source env/bin/activate
source .env
```

---

### Baseline Model
This is the standard model with no special curriculum.
```bash
python train.py \
  experiment.name="baseline-10m" \
  experiment.group="babylm-project" \
  dataset=strict_small \
  experiment.offline_run=True
```

### Data Curriculum Model
This model learns from "easier" data first.
*Note: We're using `linear_ngram` because the original `data_split` curriculum doesn't work with our local data setup.*
```bash
python train.py \
  experiment.name="data-curriculum-10m" \
  experiment.group="babylm-project" \
  dataset=strict_small \
  +data_curriculum=linear_ngram \
  experiment.offline_run=True
```

### Objective Curriculum Model
This model learns a simpler task (predicting nouns/verbs) before moving on to the main goal.
```bash
python train.py \
  experiment.name="objective-curriculum-10m" \
  experiment.group="babylm-project" \
  dataset=strict_small \
  objective_curriculum=pos_nv_mlm \
  experiment.offline_run=True
```

## 3. Evaluating the Models (Getting Perplexity)

After a model is done training, a checkpoint will be saved in the `checkpoints/` directory. You can then run the `evaluate.py` script to calculate the perplexity.

Here is a template for the command:
```bash
python evaluate.py \
  experiment.name="<your_experiment_name>" \
  experiment.group="babylm-project" \
  dataset=strict_small \
  experiment.offline_run=True \
  experiment.resume_checkpoint_path="<path_to_your_checkpoint>" \
  trainer.eval_blimp=False \
  trainer.eval_glue=False \
  trainer.eval_msgs=False \
  trainer.eval_perplexity=True
```

**Example for the baseline model:**
(Assuming the final checkpoint is at step 400000)
```bash
python evaluate.py \
  experiment.name="baseline-10m" \
  experiment.group="babylm-project" \
  dataset=strict_small \
  experiment.offline_run=True \
  experiment.resume_checkpoint_path="checkpoints/babylm-project/baseline-10m/checkpoint-400000" \
  trainer.eval_blimp=False \
  trainer.eval_glue=False \
  trainer.eval_msgs=False \
  trainer.eval_perplexity=True
```
The script will run for a bit and then print out the perplexity scores to the console.

## Original Project Info

This work is based on the original MAO-CLIMB repository and the BabyLM Challenge.

#### Citation
```
@inproceedings{salhan-etal-2024-less,
    title = " Less is More: Pre-Training Cross-Lingual Small-Scale Language Models with Cognitively-Plausible Curriculum Learning Strategies",
    author = "Salhan, Suchir  and
      Diehl Martinez, Richard
      Goriely, Zebulon  and
      Buttery, Paula",
    booktitle = "Proceedings of the BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```
