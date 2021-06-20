from h5py._hl.files import File
import numpy as np
import cv2
from environment.drawing_env import DrawingEnvironment
import tensorflow as tf

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    # model = tf.keras.models.load_model("models/datasetlama_rsg2_morestroke_dropout2.h5")\
    model_names = ["datasetlama_rsg2_morestroke_dropout2"]
    data_dir = "test_set"

    for model_name in model_names:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model("models/{}.h5".format(model_name))

        env = DrawingEnvironment(data_dir)
        out = "reward_results/rl{}_{}.txt".format(model_name, data_dir)
        f = open(out, "w")

        SAVE_VIDEO = True
        # if SAVE_VIDEO:
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     out = cv2.VideoWriter('output.avi',fourcc, 20.0, (84*2,84*2))
        
        # print(observation[0].shape, observation[1].shape)
        rewards = []
        for n in range(env.get_ref_number()):
        # for n in range(10):
            total_reward = 0
            env.reset()
            env.drawer.set_pen_position((41,41))
            observation = env.get_observation()
            for i in range(500):
                # action = np.random.randint(0, 242)
                pred = model.predict(observation)
                # print(pred.tolist())
                action = np.argmax(pred)
                # print(pred[0][action])
                new_observation, reward, done = env.step(action=action)
                observation = new_observation
                total_reward += reward
                # images = env.show()


                if done :
                    # cv2.waitKey(1000)
                    break
            f.write("{}: {}\n".format(env.get_ref_image(), total_reward))
            rewards.append(total_reward)
            print(f"Episode {n+1} {env.get_ref_image()} : {total_reward}")
            cv2.waitKey(1)
        
        print("Average Rewaard : {}".format(np.mean(rewards)))
        f.write("---\n{}".format(np.mean(rewards)))
        f.close()