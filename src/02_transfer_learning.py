import argparse
import os
import shutil
from six import print_
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np
import io 

STAGE = "Creating_transfer_learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_even_odd_labels(labels):
    for idx,label in enumerate(labels):
        even_condition=label%2==0
        labels[idx]=np.where(even_condition,1,0)
    return labels 

def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    y_train_bin, y_test_bin, y_valid_bin = update_even_odd_labels([y_train, y_test, y_valid])


    # set random seed
    seed = 2021
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    def log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x:stream.write(f"{x}\n"))
            summary_str=stream.getvalue()
        return summary_str
    
    
    #load the base model
    model_base_path=os.path.join('artifacts','models',"base_model.h5")
    base_model=tf.keras.models.load_model(model_base_path)
    logging.info(f"loaded base model summary: \n {log_model_summary(base_model)}")
    
    for layer in base_model.layers[:-1]:
        print(f"trainable status of {layer.name}:{layer.trainable}")
        layer.trainable=False
        print(f"trainable status of {layer.name}:{layer.trainable}")

    base_layers=base_model.layers[: -1]
    new_model=tf.keras.models.Sequential(base_layers)
    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax", name="output_layer")
    )
    logging.info(f"{STAGE} model summary: \n{log_model_summary(new_model)}")
    new_model.summary()
    
    new_model.compile(loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),metrics=['accuracy'])
    
     
    history=new_model.fit(X_train,y_train_bin,epochs=10,validation_data=(X_valid,y_valid_bin))
    
    
    model_dir_path=os.path.join("artifacts","models")
    model_file_path=os.path.join(model_dir_path,"even_odd_model.h5")
    new_model.save(model_file_path)
    # model=tf.keras.models.Sequential(Layers)
    
    # model.compile(loss='sparse_categorical_crossentropy',
    # optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),metrics=['accuracy'])
    # ##log the summary
    # def log_model_summary(model):
    #     with io.StringIO() as stream:
    #         model.summary(print_fn=lambda x:stream.write(f"{x}\n"))
    #         summary_str=stream.getvalue()
    #     return summary_str
    # logging.info(f"base model summary: \n {log_model_summary(model)}")

    # model.summary()
    # history=model.fit(X_train,y_train,epochs=10,validation_data=(X_valid,y_valid))

    # ##save the model
    # model_dir_path=os.path.join("artifacts","models")
    # create_directories([model_dir_path])
    # model_file_path=os.path.join(model_dir_path,"base_model.h5")
    # model.save(model_file_path)
    logging.info(f"base model saved at {model_file_path}")
    logging.info(f"evaluate the model {new_model.evaluate(X_test,y_test_bin)}")
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e