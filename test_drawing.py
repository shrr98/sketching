import numpy as np
import cv2
from environment.drawing_env import DrawingEnvironment
import tensorflow as tf

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    model = tf.keras.models.load_model("models/dqn_1000_from0_datasets_1617417775___-30.93max_-393.89avg_-653.58min.h5")
    env = DrawingEnvironment("datasets/")
    total_reward = 0
    SAVE_VIDEO = True
    # if SAVE_VIDEO:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('output.avi',fourcc, 20.0, (84*2,84*2))
    observation = env.get_observation()
    print(observation[0].shape, observation[1].shape)

    for i in range(500):
        # action = np.random.randint(0, 242)
        pred = model.predict(observation)
        print(pred.tolist())
        action = np.argmax(pred)
        new_observation, reward, done = env.step(action=action)
        observation = new_observation
        total_reward += reward
        print(f"Step {i+1} {reward} \t {total_reward}")
        images = env.show()
        if done :
            break
    tf.keras.backend.clear_session()
    cv2.waitKey(0)
        # if SAVE_VIDEO:
        #     images = (np.vstack(
        #                 (np.hstack((new_observation[0], new_observation[1])),
        #                 np.hstack((new_observation[2], new_observation[3])))
        #     )*255).astype("uint8")
        #     images = cv2.cvtColor(images, cv2.COLOR_GRAY2BGR)

        #     cv2.imshow("Video", images)
        #     cv2.imshow("patch", new_observation[4])
            # out.write(images)
        
    # out.release()