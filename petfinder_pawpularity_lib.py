# Imports
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
import shutil
from PIL import Image
from IPython.display import display
from datetime import datetime

# TF log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Matplotlib

# ML libraries

# Feature fields
feature_fields = ["Subject Focus", "Eyes", "Face", "Near", "Action",
                  "Accessory", "Group", "Collage", "Human", "Occlusion", "Info", "Blur"]

# TF strategy


def tf_strategy():
    display("TensorFlow Version: %s" % tf.__version__)
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tf_strategy = tf.distribute.experimental.TPUStrategy(resolver)
    except ValueError:
        tf_strategy = tf.distribute.get_strategy()
    display("TensorFlow Strategy: %s" % type(tf_strategy).__name__)
    # GPU informations
    if tf.test.gpu_device_name():
        os.system("nvidia-smi --query-gpu=gpu_name,memory.total --format=csv")

    with tf.device(tf.test.gpu_device_name()):
        display("Built With GPU Support: %s" %
                tf.test.is_built_with_gpu_support())
        display("Built With CUDA: %s" % tf.test.is_built_with_cuda())
        display("Device: %s" % tf.test.gpu_device_name())
        os.system("nvidia-smi --query-gpu=gpu_name,memory.total --format=csv")

    return tf_strategy

# Show dict


def show_dict(dict):
    columns = ["Key", "Value"]
    dict_data = pd.DataFrame({}, columns=columns)
    for k in dict.keys():
        dict_data_row = pd.DataFrame([[k, dict[k]]], columns=columns)
        dict_data = pd.concat([dict_data, dict_data_row], ignore_index=True)
    display(dict_data)

# Load image file


def load_image_file(image_file, root_dir, target_size, to_array=True):
    image_file = "%s.jpg" % image_file
    image_path = os.path.join(root_dir, prefix_file_name(image_file))
    image = keras.preprocessing.image.load_img(
        image_path, target_size=target_size)
    if not to_array:
        return image
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    return image_array

# Show image file


def show_image_file(image_file, root_dir, target_size=None):
    return show_image(load_image_file(image_file, root_dir, target_size=target_size, to_array=False))

# Show image


def show_image(image):
    return Image.fromarray(image.astype('uint8'), 'RGB')

# Plot images


def plot_images(images, scores, ncols=5, figsize=(20, 20)):
    if len(images) == 0:
        return
    if ncols == 1:
        ncols = 2
    nrows = math.ceil(len(images) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, subplot_kw={
                            'xticks': [], 'yticks': []})
    for index in range(nrows*ncols-len(images)):
        images.append(np.ones(images[0].shape)*255.0)
    for index, (ax, image) in enumerate(zip(axs.flat, images)):
        ax.imshow(image/255.0, cmap="binary")
        if index < len(scores):
            ax.set_title(scores[index])
    plt.tight_layout()
    plt.show()

# Plot image files


def plot_image_files(image_files, root_dir, scores=None, ncols=5, figsize=(10, 10), target_size=(150, 150)):
    if scores is None:
        scores = image_files
    if ncols > len(image_files):
        ncols = len(image_files)
    images = []
    for image_file in image_files:
        images.append(load_image_file(image_file, root_dir=root_dir,
                      target_size=target_size, to_array=True))
    plot_images(images, scores, ncols, figsize)

# Load images/scores from dataset


def load_images_scores_from_dataset(dataset):
    items = None
    I = J = None
    for item in dataset:
        if I is None:
            I = len(item)
        for i in range(I):
            if J is None:
                J = len(item[0])
            for j in range(J):
                if items is None:
                    items = np.zeros((I, J)).tolist()
                items[i][j] = item[i][j].numpy()
    return tuple(items)

# Plot images/scores from dataset


def plot_images_scores_from_dataset(dataset, ncols=5, figsize=(10, 10), slice_indexes=(None, 5), with_stats=True):
    images, scores, *_ = load_images_scores_from_dataset(dataset)
    plot_images(images=images[slice_indexes[0]:slice_indexes[1]],
                scores=scores[slice_indexes[0]:slice_indexes[1]], ncols=ncols, figsize=figsize)
    if with_stats == True:
        pd.DataFrame(scores).rename(columns={0: "Score"}).hist(
            bins=500, figsize=(18, 3))

# Prefix file name


def prefix_file_name(file_name, total_prefix=0):
    prefixes = []
    for index_prefix in range(total_prefix):
        prefixes.append(file_name[index_prefix:index_prefix+1])
    prefixed_file_name = os.path.join(*prefixes, file_name)
    return prefixed_file_name

# Cut suffix


def cut_suffix(cut_ratio):
    if cut_ratio == 1.0:
        return ""
    return "-cut-" + "%0.3f" % cut_ratio

# Copy file


def copy_file(file_src, file_dst):
    display("Copy %s to %s" % (file_src, file_dst))
    os.makedirs(os.path.dirname(file_dst), exist_ok=True)
    shutil.copy(file_src, file_dst)

# Cut training data


def cut_training_data(cut_ratio=1.0, dataset_dir_src=None, dataset_dir_cut=None):
    if cut_ratio == 1.0:
        return dataset_dir_src
    dataset_dir_cut = dataset_dir_cut + cut_suffix(cut_ratio)
    if os.path.exists(dataset_dir_cut):
        return dataset_dir_cut
    training_data = pd.read_csv(os.path.join(dataset_dir_src, "train.csv"))
    index_cut_training = int(len(training_data) * cut_ratio)
    training_data_cut = training_data.iloc[0:index_cut_training, :]
    os.makedirs(dataset_dir_cut, exist_ok=True)
    training_data_cut.to_csv(os.path.join(
        dataset_dir_cut, "train.csv"), index=False)
    test_data = pd.read_csv(os.path.join(dataset_dir_src, "test.csv"))
    index_cut_test = int(len(test_data) * cut_ratio)
    test_data_cut = test_data.iloc[0:index_cut_test, :]
    os.makedirs(dataset_dir_cut, exist_ok=True)
    test_data_cut.to_csv(os.path.join(
        dataset_dir_cut, "test.csv"), index=False)
    for file_id in training_data_cut["Id"]:
        file_name = prefix_file_name("%s.jpg" % file_id)
        file_path_src = os.path.join(dataset_dir_src, "train", file_name)
        file_path_dst = os.path.join(dataset_dir_cut, "train", file_name)
        os.makedirs(os.path.dirname(file_path_dst), exist_ok=True)
        shutil.copy(file_path_src, file_path_dst)
    test_file_names = []
    for root, dirs, files in os.walk(os.path.join(dataset_dir_src, "test"), topdown=True):
        for file_name in files:
            file_id, file_ext = os.path.splitext(file_name)
            if file_ext in [".jpg"]:
                test_file_names.append(file_name)
    index_cut_test = int(len(test_file_names) * cut_ratio)
    for file_name in test_file_names[:index_cut_test]:
        file_name = prefix_file_name(file_name)
        file_path_src = os.path.join(dataset_dir_src, "test", file_name)
        file_path_dst = os.path.join(dataset_dir_cut, "test", file_name)
        os.makedirs(os.path.dirname(file_path_dst), exist_ok=True)
        shutil.copy(file_path_src, file_path_dst)
    return dataset_dir_cut

# Load training dataset


def load_training_dataset(dataset_dir, mapping_data, batch_size=32, shuffle=False, seed=42, image_size=(224, 224)):
    dataset = load_image_dataset(
        pattern=dataset_dir+"/train/*.jpg",
        mapping_data=mapping_data,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset

# Load test dataset


def load_test_dataset(dataset_dir, mapping_data, batch_size=32, shuffle=False, seed=42, image_size=(224, 224)):
    dataset = load_image_dataset(
        pattern=dataset_dir+"/test/*.jpg",
        mapping_data=mapping_data,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset

# Process image dataset function


def process_image_func(mapping_data=None, image_size=(None, None)):
    if mapping_data is not None:
        id_values = tf.convert_to_tensor(mapping_data["Id"].values)
        score_values = [0] * len(mapping_data["Id"].values)
        if "Pawpularity" in mapping_data.columns:
            score_values = tf.convert_to_tensor(
                mapping_data["Pawpularity"].values)
        scores_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=id_values,
            values=score_values,
            key_dtype=tf.string,
            value_dtype=tf.int64
        )
        scores_table = tf.lookup.StaticHashTable(
            scores_initializer, default_value=0)
        feature_values = [",".join(
            feature) for feature in mapping_data[feature_fields].astype("string").values]
        feature_values = tf.convert_to_tensor(feature_values)
        features_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=id_values,
            values=feature_values,
            key_dtype=tf.string,
            value_dtype=tf.string
        )
        features_table = tf.lookup.StaticHashTable(
            features_initializer, default_value="")

    def _process_image_file_path(file_path):
        parts = tf.strings.split(file_path, os.sep)
        parts = tf.strings.split(parts[-1], ".")
        file_id = parts[0]
        image_string = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if image_size != (None, None):
            image = tf.image.resize(image, np.asarray(image_size))
        image = image * 255.0
        score = 0
        features_ts = [] * len(feature_fields)
        if mapping_data is not None:
            score = scores_table.lookup(file_id)
            features = features_table.lookup(file_id)
            features_ts = tf.map_fn(fn=lambda x: tf.strings.to_number(
                x), elems=tf.strings.split(features, ","), fn_output_signature=tf.float32)
        return image, features_ts, score, file_id
    return _process_image_file_path

# Load image dataset


def load_image_dataset(pattern="*.jpg", mapping_data=None, batch_size=32, shuffle=False, seed=42, image_size=(224, 224), num_parallel_calls=tf.data.AUTOTUNE):
    def _get_image_dataset(skip=0):
        dataset = tf.data.Dataset.list_files(
            pattern, shuffle=shuffle, seed=seed)
        dataset = dataset.cache()
        dataset = dataset.skip(skip*batch_size)
        dataset = dataset.map(process_image_func(
            mapping_data, image_size), num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size)
        return dataset
    return _get_image_dataset

# Make training dataset


def make_training_validate_test_data(dataset, split_ratios=[0.7, 0.15, 0.15], shrink_ratio=1.0, prefetch=tf.data.AUTOTUNE, map_fn=None):
    dataset_cardinality = dataset(skip=0).cardinality().numpy()
    train_cardinality = int(
        split_ratios[0] * shrink_ratio * dataset_cardinality)
    validate_cardinality = int(
        split_ratios[1] * shrink_ratio * dataset_cardinality)
    test_cardinality = int(
        split_ratios[2] * shrink_ratio * dataset_cardinality)

    def _skip_take_dataset(skip=0, take=1):
        result_dataset = dataset(skip=skip)
        result_dataset = result_dataset.take(take)
        result_dataset = result_dataset.prefetch(prefetch)
        if map_fn is not None:
            result_dataset = result_dataset.map(map_fn)
        return result_dataset

    def _train_dataset(skip=0):
        return _skip_take_dataset(skip=skip, take=train_cardinality)

    def _validate_dataset(skip=0):
        return _skip_take_dataset(skip=skip+train_cardinality, take=validate_cardinality)

    def _test_dataset(skip=0):
        return _skip_take_dataset(skip=skip+train_cardinality+validate_cardinality, take=test_cardinality)
    return _train_dataset, _validate_dataset, _test_dataset

# Make submission dataset


def make_submission_data(dataset, shrink_ratio=1.0, prefetch=tf.data.AUTOTUNE, map_fn=None):
    dataset_cardinality = dataset(skip=0).cardinality().numpy()
    submission_cardinality = int(shrink_ratio * dataset_cardinality)

    def _skip_take_dataset(skip=0, take=1):
        result_dataset = dataset(skip=skip)
        result_dataset = result_dataset.take(take)
        result_dataset = result_dataset.prefetch(prefetch)
        if map_fn is not None:
            result_dataset = result_dataset.map(map_fn)
        return result_dataset

    def _submission_dataset(skip=0):
        return _skip_take_dataset(skip=skip, take=submission_cardinality)
    return _submission_dataset

# Delete training data


def delete_training_data(cut_ratio=1.0, dataset_dir_cut=None):
    if cut_ratio == 1.0:
        return ""
    dataset_dir_cut = dataset_dir_cut + cut_suffix(cut_ratio)
    print(dataset_dir_cut)
    if os.path.exists(dataset_dir_cut):
        shutil.rmtree(dataset_dir_cut)
        return dataset_dir_cut

# Load training data


def load_training_data(dataset_dir, csv_file="train.csv"):
    csv_file_path = os.path.join(dataset_dir, csv_file)
    display("Load training data from %s" % csv_file_path)
    training_data = pd.read_csv(csv_file_path)
    return training_data

# Load test data


def load_test_data(dataset_dir, csv_file="test.csv"):
    csv_file_path = os.path.join(dataset_dir, csv_file)
    display("Load test data from %s" % csv_file_path)
    test_data = pd.read_csv(csv_file_path)
    return test_data

# Describe training


def describe_training(history):
    pd.DataFrame(history.history).plot(figsize=(11, 4))
    plt.grid(True)
    plt.show()

# Get model name


def get_model_name(parameters):
    model_name = "%s%s-%s-input-%s-dense-%s-%s-dropout-%0.3f" % (parameters["model_id"], parameters["model_prefix"], parameters["model_base"], "x".join(
        map(str, parameters["input_shape"])), parameters["dense_layers"], parameters["dense_layers_activation"], parameters["dropout_rate"])
    return model_name

# Get model id


def get_model_id(model):
    model_name = model.name
    model_id = model_name.split("-")[0]
    return model_id

# Model file path load


def model_file_path_load(model_name, model_load_dir):
    model_file = os.path.join(model_load_dir, "%s.h5" % model_name)
    return model_file

# Model file path save


def model_file_path_save(model_name, model_save_dir):
    model_file = os.path.join(model_save_dir, "%s.h5" % model_name)
    return model_file

# Save model


def save_model(model, model_save_dir):
    os.makedirs(model_save_dir, exist_ok=True)
    model_file = model_file_path_save(model.name, model_save_dir)
    model.save_weights(model_file)
    return model_file

# Load model


def load_model(model, model_load_dir):
    model_file = model_file_path_load(model.name, model_load_dir)
    if not os.path.exists(model_file):
        return None
    model.load_weights(model_file)
    return model_file

# Show model


def show_model(model, model_save_dir):
    display(keras.utils.plot_model(model, os.path.join(model_save_dir, "%s.png" % (
        model.name)), show_shapes=True, expand_nested=False, dpi=192))

# Synchronize models


def synchronize_models(model_load_dir, model_save_dir):
    synchronize_models = []
    if model_load_dir != model_save_dir:
        for root, dirs, files in os.walk(model_load_dir, topdown=True):
            for file_name in files:
                file_id, file_ext = os.path.splitext(file_name)
                if file_ext in [".h5", ".csv"]:
                    file_src_path = os.path.join(model_load_dir, file_name)
                    file_dst_path = os.path.join(model_save_dir, file_name)
                    os.makedirs(os.path.dirname(file_dst_path), exist_ok=True)
                    shutil.copy(file_src_path, file_dst_path)
                    synchronize_models.append(file_name)
    return synchronize_models

# Evaluate model


def evaluate_model(model, test_dataset):
    if test_dataset.cardinality().numpy() > 0:
        test_data = test_dataset.map(prepare_model_dataset_fn(model))
        return model.evaluate(test_data, verbose=1)
    return None

# Record training/evaluate model


def record_training_evaluate(model_name, model_file, model_parameters, history, evaluation, model_load_dir, model_save_dir, records_file):
    records_file_path_load = os.path.join(model_load_dir, records_file)
    records_file_path_save = os.path.join(model_save_dir, records_file)
    if model_load_dir != model_save_dir and os.path.exists(records_file_path_load):
        print("Copied records from %s" % records_file_path_load)
        shutil.copy(records_file_path_load, records_file_path_save)
    record_data = pd.DataFrame({}, columns=["model_name", "model_parameters", "iteration",
                               "epochs", "epochs_sum", "steps", "train_rmse", "val_rmse", "test_rmse", "trained_at"])
    if os.path.exists(records_file_path_save):
        record_data = pd.read_csv(records_file_path_save)
    iteration_previous = 0
    epochs_sum_previous = 0
    record_data_row_previous = record_data[record_data["model_name"]
                                           == model_name][-1:]
    if (len(record_data_row_previous) > 0):
        iteration_previous = record_data_row_previous["iteration"].values[0]
        epochs_sum_previous = record_data_row_previous["epochs_sum"].values[0]
    record_data_row = pd.DataFrame([[model_name, model_parameters, iteration_previous+1, history.params["epochs"], epochs_sum_previous+history.params["epochs"], history.params["steps"],
                                   "%0.3f" % history.history["rmse"][-1:][0], "%0.3f" % history.history["val_rmse"][-1:][0], "%0.3f" % evaluation[-1:][0], datetime.now()]], columns=record_data.columns)
    record_data = pd.concat([record_data, record_data_row], ignore_index=True)
    record_data.to_csv(records_file_path_save, index=False)
    print("Successfully written training records to %s" %
          (records_file_path_save))
    return record_data

# Prepare model dataset function


def prepare_model_dataset_fn(model):
    model_id = get_model_id(model)
    return globals()["prepare_model_dataset_fn_%s" % model_id]()

# Predict model


def predict_model(model, images, features):
    model_id = get_model_id(model)
    return globals()["predict_model_%s" % model_id](model, images, features)

# Setup model


def setup_model(parameters):
    model_id = parameters["model_id"]
    return globals()["setup_model_%s" % model_id](parameters)

# Train model


def train_model(model, train_dataset, validate_dataset, parameters):
    early_stopping_patience = parameters["early_stopping_patience"]
    save_checkpoint = parameters["save_checkpoint"]
    epochs = parameters["epoch"]
    train_data = train_dataset.map(prepare_model_dataset_fn(model))
    validation_data = None
    if validate_dataset.cardinality().numpy() > 0:
        validation_data = validate_dataset.map(prepare_model_dataset_fn(model))
    callbacks = []
    if save_checkpoint:
        model_file_path = model_file_path_save(model.name)
        model_dir_path = os.path.dirname(model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=model_file_path, save_weights_only=True, monitor='val_rmse', mode='max', save_best_only=True)
        callbacks.append(model_checkpoint)
    if early_stopping_patience > 0 and early_stopping_patience < epochs:
        early_stopping = keras.callbacks.EarlyStopping(
            patience=early_stopping_patience, restore_best_weights=True)
        callbacks.append(early_stopping)
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history

# Tune model


def tune_model(hypermodel, settings, train_dataset, validate_dataset):
    project_name = settings["tuner_project_name"]
    tuner_type = settings["tuner_type"]
    objective = kt.Objective("val_rmse", direction="min")
    max_trials = settings["tuner_max_trials"]
    executions_per_trial = settings["tuner_executions_per_trial"]
    seed = settings["tuner_seed"]
    directory = settings["tuner_save_dir"]
    max_epochs = settings["tuner_max_epochs"]
    tune_new_entries = True
    allow_new_entries = True
    overwrite = True
    if tuner_type == "random":
        tuner = kt.RandomSearch(
            hypermodel,
            objective=objective,
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            overwrite=overwrite,
            seed=seed,
            directory=directory,
            project_name=project_name,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
    elif tuner_type == "bayesian":
        bayesian_num_initial_points = settings["tuner_bayesian_num_initial_points"] if "tuner_bayesian_num_initial_points" in settings.keys(
        ) else 2
        bayesian_alpha = settings["tuner_bayesian_alpha"] if "tuner_bayesian_alpha" in settings.keys(
        ) else 0.0001
        bayesian_beta = settings["tuner_bayesian_beta"] if "tuner_bayesian_beta" in settings.keys(
        ) else 2.6
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective=objective,
            max_trials=max_trials,
            num_initial_points=bayesian_num_initial_points,
            alpha=bayesian_alpha,
            beta=bayesian_beta,
            overwrite=overwrite,
            seed=seed,
            directory=directory,
            project_name=project_name,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
    elif tuner_type == "hyperband":
        hyperband_factor = settings["tuner_hyperband_factor"] if "tuner_hyperband_factor" in settings.keys(
        ) else 3
        hyperband_iterations = settings["tuner_hyperband_iterations"] if "tuner_hyperband_iterations" in settings.keys(
        ) else 1
        tuner = kt.Hyperband(
            hypermodel,
            objective=objective,
            max_epochs=max_epochs,
            factor=hyperband_factor,
            hyperband_iterations=hyperband_iterations,
            overwrite=overwrite,
            seed=seed,
            directory=directory,
            project_name=project_name,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
    tuner.search_space_summary()
    model_temp = hypermodel(kt.HyperParameters())
    train_data = train_dataset().map(prepare_model_dataset_fn(model_temp))
    validation_data = validate_dataset().map(prepare_model_dataset_fn(model_temp))
    tuner.search(train_data, epochs=max_epochs,
                 validation_data=validation_data)
    tuner.results_summary()
    return tuner

# Infer score


def infer_score(model, images, features):
    scores = predict_model(model, np.array(images), np.array(features))
    return scores

# Predict


def predict(model, image, features, label=None, true_score=None, delta_max=10, debug=False):
    predicted_scores = infer_score(model, [image], [features])
    predicted_score = int(predicted_scores[0][0])
    if true_score is not None:
        result = "SUCCESS" if abs(
            predicted_score - true_score) <= delta_max else "FAILURE"
        print(">>>", result)
    print(">>> Predicted score:", predicted_score, "vs", true_score)
    if label is None:
        label = ""
    plot_images([image], [label], ncols=1, figsize=(5, 5))

# Make submission sample file


def make_submission_sample_file(sample_dir, submission_dir):
    submission_file_sample_path = os.path.join(
        sample_dir, "sample_submission.csv")
    submission_file_path = os.path.join(submission_dir, "submission.csv")
    submission_data = pd.read_csv(submission_file_sample_path)
    os.makedirs(submission_dir, exist_ok=True)
    submission_data.to_csv(submission_file_path,
                           index=False, line_terminator="\r\n")
    print("Successfully written sample submission to %s" %
          (submission_file_path))
    return submission_data

# Make submission file


def make_submission_file(dataset, model, submission_dir):
    submission_data = infer_submission_data(dataset, model)
    submission_file_path = os.path.join(submission_dir, "submission.csv")
    print("Preparing submission to %s" %
          (submission_file_path), end="\r", flush=True)
    os.makedirs(submission_dir, exist_ok=True)
    submission_data.sort_values(by=["Id"], inplace=True)
    submission_data.to_csv(submission_file_path,
                           index=False, line_terminator="\r\n")
    print("Successfully written submission to %s" %
          (submission_file_path), end="\r", flush=True)
    return submission_data

# Infer submission data


def infer_submission_data(dataset, model, take=-1):
    submission_data = pd.DataFrame({}, columns=["Id", "Pawpularity"])
    if take == -1:
        take = dataset().cardinality().numpy()
    take = min(take, dataset().cardinality().numpy())
    for batch_index in range(take):
        print("Preparing submission for batch %s/%s..." %
              (batch_index+1, take), end="\r", flush=True)
        batch_dataset = dataset(skip=batch_index).take(1)
        images, features, _, file_ids, * \
            _ = load_images_scores_from_dataset(batch_dataset)
        predicted_scores = infer_score(model, images, features)
        for image_index in range(len(images)):
            predicted_score = "%0.2f" % predicted_scores[image_index][0]
            query_id = file_ids[image_index].decode("utf-8")
            submission_data_row = pd.DataFrame(
                [[query_id, predicted_score]], columns=submission_data.columns)
            submission_data = pd.concat(
                [submission_data, submission_data_row], ignore_index=True)
    return submission_data

# Score submission data


def score_submission_data(submission_data, training_data):
    merged_data = submission_data.copy()
    merged_data = merged_data.rename(
        columns={"Pawpularity": "predicted_Pawpularity"})
    merged_data = training_data.merge(merged_data, on="Id", how="inner")
    merged_data = merged_data.reset_index()
    merged_data = merged_data.astype(
        {"Pawpularity": "float", "predicted_Pawpularity": "float"})
    y_true = merged_data.set_index("Id")[["Pawpularity"]].values
    y_pred = merged_data.set_index("Id")[["predicted_Pawpularity"]].values
    mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
    rmse_loss = np.sqrt(mse_loss_fn(y_true, y_pred).numpy())
    return rmse_loss, merged_data

# Model 1


def prepare_model_dataset_fn_model_1():
    return lambda image, features, score, file_id: (image, score)


def predict_model_model_1(model, images, features):
    X = prepare_model_dataset_fn(model)(images, features, None, None)[0]
    y_predicted = model.predict(X)
    return y_predicted


def setup_model_model_1(parameters):
    model_base = parameters["model_base"]
    model_name = parameters["model_name"]
    input_shape = parameters["input_shape"]
    output_size = parameters["output_size"]
    preload_weights = parameters["preload_weights"]
    image_resize = [int(param) for param in parameters["image_resize"].split(
        "x")] if "image_resize" in parameters.keys() else None
    dense_layers = [int(param) for param in parameters["dense_layers"].split(
        "x")] if "dense_layers" in parameters.keys() else []
    dense_layers_activation = parameters["dense_layers_activation"]
    dropout_rate = parameters["dropout_rate"]
    learning_rate = parameters["learning_rate"]
    fine_tuning = parameters["fine_tuning"] if "fine_tuning" in parameters.keys(
    ) else False
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    mse_loss = keras.losses.MeanSquaredError(reduction="auto", name="mse")
    rmse_metric = keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)
    inputs = keras.layers.Input(shape=input_shape, name="input")
    if image_resize is not None:
        outputs = keras.layers.Resizing(
            height=image_resize[1], width=image_resize[0])(inputs)
    else:
        outputs = inputs
    if model_base == "xception":
        model_base_layer = keras.applications.xception.Xception(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
        outputs = keras.applications.xception.preprocess_input(outputs)
    elif model_base == "efficientnetb3":
        model_base_layer = keras.applications.EfficientNetB3(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    elif model_base == "efficientnetb5":
        model_base_layer = keras.applications.EfficientNetB5(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    elif model_base == "efficientnetb7":
        model_base_layer = keras.applications.EfficientNetB7(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    model_base_layer.trainable = fine_tuning
    outputs = model_base_layer(outputs)
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    top_model_layers_index = 0
    for layer_width in dense_layers:
        if layer_width > 0:
            top_model_layers_index += 1
            if dropout_rate > 0:
                outputs = keras.layers.Dropout(dropout_rate, name="dropout_%i_r%0.3f" % (
                    top_model_layers_index, dropout_rate))(outputs)
            outputs = keras.layers.Dense(layer_width, activation=dense_layers_activation, name="dense_%i" % (
                top_model_layers_index))(outputs)
    top_model_layers_index += 1
    if top_model_layers_index == 1 and dropout_rate > 0:
        outputs = keras.layers.Dropout(dropout_rate, name="dropout_%i_r%0.3f" % (
            top_model_layers_index, dropout_rate))(outputs)
    outputs = keras.layers.Dense(int(
        output_size), activation=None, name="dense_%i" % top_model_layers_index)(outputs)
    model = keras.Model(name=model_name, inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=[mse_loss], metrics=[rmse_metric])
    model.summary()
    return model

# Model 2


def prepare_model_dataset_fn_model_2():
    return lambda image, features, score, file_id: ({"input": image, "input_features": features}, tf.cast(score, dtype=tf.float32) if score is not None else None)


def predict_model_model_2(model, images, features):
    X = prepare_model_dataset_fn(model)(images, features, None, None)[0]
    y_predicted = model.predict(X)
    return y_predicted


def setup_model_model_2(parameters):
    model_base = parameters["model_base"]
    model_name = parameters["model_name"]
    input_shape = parameters["input_shape"]
    input_shape_features = parameters["input_shape_features"]
    output_size = parameters["output_size"]
    preload_weights = parameters["preload_weights"]
    image_resize = [int(param) for param in parameters["image_resize"].split(
        "x")] if "image_resize" in parameters.keys() else None
    dense_layers = [int(param) for param in parameters["dense_layers"].split(
        "x")] if "dense_layers" in parameters.keys() else []
    dense_layers_activation = parameters["dense_layers_activation"]
    dropout_rate = parameters["dropout_rate"]
    learning_rate = parameters["learning_rate"]
    fine_tuning = parameters["fine_tuning"] if "fine_tuning" in parameters.keys(
    ) else False
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    mse_loss = keras.losses.MeanSquaredError(reduction="auto", name="mse")
    rmse_metric = keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)
    inputs = keras.layers.Input(shape=input_shape, name="input")
    inputs_features = keras.layers.Input(
        shape=input_shape_features, name="input_features")
    if image_resize is not None:
        outputs = keras.layers.Resizing(
            height=image_resize[1], width=image_resize[0])(inputs)
    else:
        outputs = inputs
    if model_base == "xception":
        model_base_layer = keras.applications.xception.Xception(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
        outputs = keras.applications.xception.preprocess_input(outputs)
    elif model_base == "efficientnetb3":
        model_base_layer = keras.applications.EfficientNetB3(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    elif model_base == "efficientnetb5":
        model_base_layer = keras.applications.EfficientNetB5(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    elif model_base == "efficientnetb7":
        model_base_layer = keras.applications.EfficientNetB7(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    model_base_layer.trainable = fine_tuning
    outputs = model_base_layer(outputs)
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = keras.layers.concatenate([outputs, inputs_features])
    outputs = keras.layers.BatchNormalization()(outputs)
    top_model_layers_index = 0
    for layer_width in dense_layers:
        if layer_width > 0:
            top_model_layers_index += 1
            if dropout_rate > 0:
                outputs = keras.layers.Dropout(dropout_rate, name="dropout_%i_r%0.3f" % (
                    top_model_layers_index, dropout_rate))(outputs)
            outputs = keras.layers.Dense(layer_width, activation=dense_layers_activation, name="dense_%i" % (
                top_model_layers_index))(outputs)
    top_model_layers_index += 1
    if top_model_layers_index == 1 and dropout_rate > 0:
        outputs = keras.layers.Dropout(dropout_rate, name="dropout_%i_r%0.3f" % (
            top_model_layers_index, dropout_rate))(outputs)
    outputs = keras.layers.Dense(int(
        output_size), activation="sigmoid", name="dense_%i" % top_model_layers_index)(outputs)
    outputs = keras.layers.Rescaling(100.0, offset=0.0)(outputs)
    model = keras.Model(name=model_name, inputs=[
                        inputs, inputs_features], outputs=outputs)
    model.compile(optimizer=optimizer, loss=[mse_loss], metrics=[rmse_metric])
    model.summary()
    return model

# Model 3


def prepare_model_dataset_fn_model_3():
    return lambda image, features, score, file_id: ({"input": image, "input_features": features}, tf.one_hot(score, 101))


def setup_model_model_3(parameters):
    model_base = parameters["model_base"]
    model_name = parameters["model_name"]
    input_shape = parameters["input_shape"]
    input_shape_features = parameters["input_shape_features"]
    output_size = parameters["output_size"]
    preload_weights = parameters["preload_weights"]
    image_resize = [int(param) for param in parameters["image_resize"].split(
        "x")] if "image_resize" in parameters.keys() else None
    dense_layers = [int(param) for param in parameters["dense_layers"].split(
        "x")] if "dense_layers" in parameters.keys() else []
    dense_layers_activation = parameters["dense_layers_activation"]
    dropout_rate = parameters["dropout_rate"]
    learning_rate = parameters["learning_rate"]
    fine_tuning = parameters["fine_tuning"] if "fine_tuning" in parameters.keys(
    ) else False
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    scc_loss = keras.losses.CategoricalCrossentropy()
    rmse_metric = keras.metrics.RootMeanSquaredError(name="rmse", dtype=None)
    inputs = keras.layers.Input(shape=input_shape, name="input")
    inputs_features = keras.layers.Input(
        shape=input_shape_features, name="input_features")
    if image_resize is not None:
        outputs = keras.layers.Resizing(
            height=image_resize[1], width=image_resize[0])(inputs)
    else:
        outputs = inputs
    if model_base == "xception":
        model_base_layer = keras.applications.xception.Xception(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
        outputs = keras.applications.xception.preprocess_input(outputs)
    elif model_base == "efficientnetb3":
        model_base_layer = keras.applications.EfficientNetB3(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    elif model_base == "efficientnetb5":
        model_base_layer = keras.applications.EfficientNetB5(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    elif model_base == "efficientnetb7":
        model_base_layer = keras.applications.EfficientNetB7(
            weights=preload_weights,
            input_shape=input_shape,
            include_top=False,
        )
    model_base_layer.trainable = fine_tuning
    outputs = model_base_layer(outputs)
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = keras.layers.concatenate([outputs, inputs_features])
    #outputs = keras.layers.BatchNormalization()(outputs)
    top_model_layers_index = 0
    for layer_width in dense_layers:
        if layer_width > 0:
            top_model_layers_index += 1
            if dropout_rate > 0:
                outputs = keras.layers.Dropout(dropout_rate, name="dropout_%i_r%0.3f" % (
                    top_model_layers_index, dropout_rate))(outputs)
            outputs = keras.layers.Dense(layer_width, activation=dense_layers_activation, name="dense_%i" % (
                top_model_layers_index))(outputs)
    top_model_layers_index += 1
    if top_model_layers_index == 1 and dropout_rate > 0:
        outputs = keras.layers.Dropout(dropout_rate, name="dropout_%i_r%0.3f" % (
            top_model_layers_index, dropout_rate))(outputs)
    outputs = keras.layers.Dense(int(
        output_size), activation="softmax", name="dense_%i" % top_model_layers_index)(outputs)
    model = keras.Model(name=model_name, inputs=[
                        inputs, inputs_features], outputs=outputs)
    model.compile(optimizer=optimizer, loss=[
                  scc_loss], metrics=["accuracy", rmse_metric])
    model.summary()
    return model
