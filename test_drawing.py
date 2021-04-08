import numpy as np
import cv2
from environment.drawing_env import DrawingEnvironment
import tensorflow as tf

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    model = tf.keras.models.load_model("model/0405_newest4.h5")
    env = DrawingEnvironment("examples/")
    SAVE_VIDEO = True
    # if SAVE_VIDEO:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('output.avi',fourcc, 20.0, (84*2,84*2))
    
    # print(observation[0].shape, observation[1].shape)
    rewards = []
    for n in range(5):
        total_reward = 0
        env.reset()
        observation = env.get_observation()
        for i in range(500):
            # action = np.random.randint(0, 242)
            pred = model.predict(observation)
            # print(pred.tolist())
            action = np.argmax(pred)
            print(action)
            new_observation, reward, done = env.step(action=action)
            observation = new_observation
            total_reward += reward
            images = env.show()
            if done :
                break
        rewards.append(total_reward)
        print(f"Episode {n+1} {total_reward}")
        cv2.waitKey(1)
    
    print("Average Rewaard : {}".format(np.mean(rewards)))
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