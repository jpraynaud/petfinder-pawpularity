# Imports
import os
import json
from IPython.display import display
import pandas as pd
import tensorflow as tf

# Get config


def get_config(default_config={"env": "remote"}):
    config_file = "./config.txt"
    config = default_config
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.loads(f.read())
    return config

# Get environment


def get_env(config=get_config()):
    return config["env"]

# Get mode


def get_mode(process, fallback_mode, config=get_config()):
    mode_key = "mode_%s" % process
    mode = config[mode_key] if mode_key in config.keys() else fallback_mode
    return mode

# Get settings


def get_settings(process, fallback_mode):
    config = get_config()
    env = get_env(config)
    mode = get_mode(process, fallback_mode, config)
    settingsKey = "%s-%s-%s" % (env, process, mode)
    display("Settings: %s" % settingsKey)
    settings = get_settings_map()[settingsKey]
    debug = settings["debug"]
    return settings, debug

# Get settings map


def get_settings_map():
    settingsMap = {}

    # Train
    # remote-train-full
    settingsMap["remote-train-full"] = {
        "debug": False,
        "model_load_dir": os.path.join("..", "input", "petfinder-pawpularity-train", "models"),
        "model_save_dir": os.path.join("models"),
        "dataset_dir_src": os.path.join("..", "input", "petfinder-pawpularity-score"),
        "dataset_dir_cut": os.path.join("dataset", "petfinder-pawpularity-score"),
        "dataset_dir_copy": os.path.join("dataset-copy"),
        "dataset_batch_size": 64,
        "dataset_image_size": (750, 750),
        "dataset_cut_ratio": 1.0,
        "dataset_shrink_ratio": 1.0,
        "dataset_split_ratios": [0.80, 0.10, 0.10],
        "dataset_shuffle": False,
        "dataset_shuffle_seed": 42,
        "dataset_prefetch": tf.data.AUTOTUNE,
        "train_save_checkpoint_flag": False,
        "train_fine_tuning_flag": False,
        "train_load_model_flag": True,
        "infer_load_model_flag": False,
        "synchronize_models_flag": True,
        "train_max_epochs": 25,
        "train_early_stopping": 5,
        "cleanup_data_flag": True,
    }

    # remote-train-full-empty
    settingsMap["remote-train-full-empty"] = {
        **settingsMap["remote-train-full"],
        **{
            "model_load_dir": os.path.join("..", "input", "petfinder-pawpularity-empty", "models"),
            "train_max_epochs": 1,
            "train_early_stopping": 1,
        }
    }

    # remote-train-cut
    settingsMap["remote-train-cut"] = {
        **settingsMap["remote-train-full"],
        **{
            "dataset_cut_ratio": 0.2,
            "dataset_split_ratios": [0.7, 0.20, 0.10],
            "train_max_epochs": 10,
            "train_early_stopping": 5,
        }
    }

    # local-train-cut
    settingsMap["local-train-full"] = {
        **settingsMap["remote-train-full"],
        **{
            "debug": True,
            "model_load_dir": os.path.join("models"),
            "model_save_dir": os.path.join("models"),
            "dataset_dir_src": os.path.join("..", "input", "petfinder-pawpularity-score"),
            "dataset_dir_cut": os.path.join("..", "input", "petfinder-pawpularity-score"),
            "dataset_batch_size": 64,
            "dataset_image_size": (150, 150),
            "dataset_split_ratios": [0.7, 0.2, 0.1],
            "train_max_epochs": 1,
            "train_early_stopping": 3,
            "cleanup_data_flag": False,
        }
    }

    # local-train-cut
    settingsMap["local-train-cut"] = {
        **settingsMap["local-train-full"],
        **{
            "dataset_cut_ratio": 0.2,
            "train_max_epochs": 10,
            "train_early_stopping": 3,
        }
    }

    # Predict
    # remote-predict-full
    settingsMap["remote-predict-full"] = {
        **settingsMap["remote-train-full"],
        **{
            "score_sample_size": 25,
        }
    }

    # remote-predict-full-select
    settingsMap["remote-predict-full-select"] = {
        **settingsMap["remote-predict-full"],
        **{
            "model_load_dir": os.path.join("..", "input", "petfinder-pawpularity-model", "models"),
            "score_sample_size": 25,
        }
    }

    # remote-predict-cut
    settingsMap["remote-predict-cut"] = {
        **settingsMap["remote-train-cut"],
        **{
            "score_sample_size": 10,
        }
    }

    # remote-predict-cut-select
    settingsMap["remote-predict-cut-select"] = {
        **settingsMap["remote-train-cut"],
        **{
            "model_load_dir": os.path.join("..", "input", "petfinder-pawpularity-model", "models"),
            "score_sample_size": 10,
        }
    }

    # local-predict-full
    settingsMap["local-predict-full"] = {
        **settingsMap["local-train-full"],
        **{
            "score_sample_size": 10,
        }
    }

    # local-predict-cut
    settingsMap["local-predict-cut"] = {
        **settingsMap["local-train-cut"],
        **{
            "score_sample_size": 10,
        }
    }

    # Submit
    # remote-submit-full
    settingsMap["remote-submit-full"] = {
        **settingsMap["remote-train-full"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": os.path.join("..", "input", "petfinder-pawpularity-train", "dataset-copy"),
            "submit_data_sample_flag": False,
            "submit_probability_min": 0.0,
        }
    }

    # remote-submit-full-sample
    settingsMap["remote-submit-full-sample"] = {
        **settingsMap["remote-train-full"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": os.path.join("..", "input", "petfinder-pawpularity-train", "dataset-copy"),
            "submit_data_sample_flag": True,
            "submit_probability_min": 0.0,
        }
    }

    # remote-submit-full-select
    settingsMap["remote-submit-full-select"] = {
        **settingsMap["remote-submit-full"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": os.path.join("..", "input", "petfinder-pawpularity-model", "dataset-copy"),
            "model_load_dir": os.path.join("..", "input", "petfinder-pawpularity-model", "models"),
            "score_sample_size": 25,
        }
    }

    # remote-submit-cut
    settingsMap["remote-submit-cut"] = {
        **settingsMap["remote-train-cut"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": os.path.join("..", "input", "petfinder-pawpularity-model", "dataset-copy"),
            "submit_data_sample_flag": False,
            "submit_probability_min": 0.0,
        }
    }

    # local-submit-full
    settingsMap["local-submit-full"] = {
        **settingsMap["local-train-full"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": os.path.join("dataset-copy"),
            "submit_data_sample_flag": False,
            "submit_probability_min": 0.0,
        }
    }

    # local-submit-full-sample
    settingsMap["local-submit-full-sample"] = {
        **settingsMap["local-train-full"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": None,
            "submit_data_sample_flag": True,
            "submit_probability_min": 0.0,
        }
    }

    # local-submit-cut
    settingsMap["local-submit-cut"] = {
        **settingsMap["local-train-cut"],
        **{
            "submission_dir": os.path.join("submission"),
            "dataset_dir_copy": os.path.join("dataset-copy"),
            "submit_data_sample_flag": False,
            "submit_probability_min": 0.0,
        }
    }

    # Tuner
    # remote-tune-full
    settingsMap["remote-tune-full"] = {
        **settingsMap["remote-train-full"],
        **{
            "tuner_save_dir": os.path.join("tuner"),
            "tuner_project_name": "petfinder-pawpularity",
            "tuner_type": "bayesian",
            "tuner_max_epochs": 3,
            "tuner_max_trials": 50,
            "tuner_executions_per_trial": 1,
            "tuner_seed": 42,
            "tuner_hyperparameter_model_base": ["xception", "efficientnetb7"],
            "tuner_hyperparameter_dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "tuner_hyperparameter_learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "tuner_hyperparameter_input_shape": ["75x75", "150x150", "250x250", "500x500"],
            "tuner_hyperparameter_dense_layers": ["100"],
        }
    }

    # remote-tune-cut
    settingsMap["remote-tune-cut"] = {
        **settingsMap["remote-train-cut"],
        **{
            "tuner_save_dir": os.path.join("tuner"),
            "tuner_project_name": "petfinder-pawpularity",
            "tuner_type": "bayesian",
            "tuner_max_epochs": 3,
            "tuner_max_trials": 50,
            "tuner_executions_per_trial": 1,
            "tuner_seed": 42,
            "tuner_hyperparameter_model_base": ["xception", "efficientnetb7"],
            "tuner_hyperparameter_dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "tuner_hyperparameter_learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "tuner_hyperparameter_input_shape": ["75x75", "150x150", "250x250", "500x500"],
            "tuner_hyperparameter_dense_layers": ["100"],
        }
    }

    # local-tune-full
    settingsMap["local-tune-full"] = {
        **settingsMap["local-train-full"],
        **{
            "tuner_save_dir": os.path.join("tuner"),
            "tuner_project_name": "petfinder-pawpularity",
            "tuner_type": "random",
            "tuner_max_epochs": 1,
            "tuner_max_trials": 2,
            "tuner_executions_per_trial": 1,
            "tuner_seed": 42,
            "tuner_hyperparameter_model_base": ["xception", "efficientnetb3", "efficientnetb5", "efficientnetb7"],
            "tuner_hyperparameter_dropout_rate": [0.0, 0.3, 0.5],
            "tuner_hyperparameter_learning_rate": [1e-4, 1e-3, 1e-2],
            "tuner_hyperparameter_input_shape": ["75x75"],
            "tuner_hyperparameter_dense_layers": ["0", "256x128"],
        }
    }

    # local-tune-cut
    settingsMap["local-tune-cut"] = {
        **settingsMap["local-train-cut"],
        **{
            "tuner_save_dir": os.path.join("tuner"),
            "tuner_project_name": "petfinder-pawpularity",
            "tuner_type": "random",
            "tuner_max_epochs": 1,
            "tuner_max_trials": 2,
            "tuner_executions_per_trial": 1,
            "tuner_seed": 42,
            "tuner_hyperparameter_model_base": ["xception", "efficientnetb3", "efficientnetb5", "efficientnetb7"],
            "tuner_hyperparameter_dropout_rate": [0.0, 0.3, 0.5],
            "tuner_hyperparameter_learning_rate": [1e-4, 1e-3, 1e-2],
            "tuner_hyperparameter_input_shape": ["75x75"],
            "tuner_hyperparameter_dense_layers": ["0", "256x128"],
        }
    }

    return settingsMap
