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
    model_names = ["datasetlama_rsg2_morestroke_dropout2_50000"]
    data_dir = "bad/data"

    for model_name in model_names:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model("models/{}.h5".format(model_name))

        env = DrawingEnvironment(data_dir)

        SAVE_VIDEO = True
        # if SAVE_VIDEO:
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     out = cv2.VideoWriter('output.avi',fourcc, 20.0, (84*2,84*2))
        
        # print(observation[0].shape, observation[1].shape)
        rewards = []
        actions = [0 for _ in range(242)]
        for n in range(env.get_ref_number()):
        # for n in range(3):
            total_reward = 0
            env.reset()
            env.drawer.set_pen_position((41,41))
            observation = env.get_observation()
            for i in range(500):
                # action = np.random.randint(0, 242)
                pred = model.predict(observation)
                # print(pred.tolist())
                action = np.argmax(pred)
                actions[action] += 1
                # print(pred[0][action])
                new_observation, reward, done = env.step(action=action)
                observation = new_observation
                total_reward += reward
                # images = env.show()


                if done :
                    # cv2.waitKey(1000)
                    break
            rewards.append(total_reward)
            print(f"Episode {n+1} {env.get_ref_image()} : {total_reward}")
            # cv2.waitKey(1)

            kanvas = observation[0][0][:,:,0]
            im = np.array(255*kanvas).astype(np.uint8)
            cv2.imwrite("bad/result/{}_kanvas200.png".format(n), im)

            ref = observation[0][0][:,:,1]
            im = np.array(255*ref).astype(np.uint8)
            cv2.imwrite("bad/result/{}_ref.png".format(n), im)
        
        print("Average Rewaard : {}".format(np.mean(rewards)))

        # with open("actions.csv", "w") as f:
        #     for i, a in enumerate(actions):
        #         f.write("{},{}\n".format(i, a))