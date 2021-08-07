from h5py._hl.files import File
import numpy as np
import cv2
from tensorflow.python.ops.control_flow_ops import Assert
from environment.drawing_env import DrawingEnvironment
import tensorflow as tf
from utils.utils import pixel_similarity
import time
import sys
import os

from subprocess import Popen, PIPE

def run_photo_sketching(dir, file_name):
    proc = Popen(". ~/anaconda3/etc/profile.d/conda.sh && conda activate sketch && ./test.sh \"{}\" \"{}\"".format(dir, file_name),
        shell=True,
        executable="/bin/bash",
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE
        )
    print("STDOUT:\n")
    print(proc.stdout.read().decode('utf8'))
    print("STDERR:\n", proc.stderr.read().decode('utf8'))

if __name__ == "__main__":
    assert len(sys.argv)==3, "Please provide directory and file name of the image"

    _, dir, filename = sys.argv
    run_photo_sketching(dir, filename)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    # model = tf.keras.models.load_model("models/datasetlama_rsg2_morestroke_dropout2.h5")\
    model_name = "datasetlama_rsg2_morestroke_dropout2_50000"
    data_dir = "img2sketch/dataset/results"
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model("models/{}.h5".format(model_name))

    env = DrawingEnvironment(data_dir)

    path = os.path.join(data_dir, "temp.png")
    ori_path = os.path.join(dir, filename)

    print(path, ori_path)

    ori = cv2.imread(ori_path)
    kontur = cv2.imread(path)

    w,h,_ = kontur.shape

    ori = cv2.resize(ori, (h//3, w//3))
    kontur = cv2.resize(kontur, (h//3, w//3))

    images = np.vstack((ori, kontur))

    cv2.imshow('Gambar asli dan Gambar Kontur', images)
    cv2.waitKey(0)

    rewards = []
    for n in range(1):
        total_reward = 0
        env.set_reference(path)
        env.reset()
        env.drawer.set_pen_position((41,41))
        observation = env.get_observation()
        start_time = time.time()
        for i in range(500):
            pred = model.predict(observation)
            action = np.argmax(pred)
            new_observation, reward, done = env.step(action=action)
            observation = new_observation
            total_reward += reward
            images = env.show()

            if done :
                break

        exec_time = time.time() - start_time
        print("====================================")
        print("             SELESAI")
        print("====================================")
        print(f"Gambar : {ori_path}")
        print("{} step, {:.02f} seconds".format(i+1, exec_time))
        # cv2.waitKey(1)

        print("Total reward: {:.02f}".format(total_reward))

        kanvas = observation[0][0][:,:,0]
        ref = observation[0][0][:,:,1]
        mse = pixel_similarity(ref, kanvas) / 84**2
        print("Error piksel: {:.04f}".format(mse))


        cv2.waitKey(0)