import numpy as np
import cv2
from environment.drawing_env import DrawingEnvironment
import tensorflow as tf

if __name__ == "__main__":
    model = tf.keras.models.load_model("model/1403-minstrokes4-maxstrokes64-jumping-0_01-whiteonblack-242-6.h5")
    env = DrawingEnvironment()
    total_reward = 0
    SAVE_VIDEO = True
    # if SAVE_VIDEO:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('output.avi',fourcc, 20.0, (84*2,84*2))
    observation = env.get_observation()
    print(observation[0].shape, observation[1].shape)

    for i in range(200):
        # action = np.random.randint(0, 242)
        pred = model.predict(observation)
        action = np.argmax(pred)
        new_observation, reward, done = env.step(action=action)
        observation = new_observation
        total_reward += reward
        print(f"Step {i+1} {reward} \t {total_reward}")
        images = env.show()
        if done :
            cv2.waitKey(0)
            break
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