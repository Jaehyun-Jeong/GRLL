import pickle
import torch

fileOne = open("./saved_models/DQN_RacingEnv_v0.obj", 'rb')
dataPickle = fileOne.read()
fileOne.close()

save_dict = pickle.loads(dataPickle)
save_dict['modelStateDict'] = save_dict.model.state_dict()
save_dict.pop('model', None)
save_dict['optimizerStateDict'] = save_dict.optmizer.state_dict()
save_dict.pop('optimizer', None)

fileTwo = open(saveDir, 'wb')
fileTwo.write(pickle.dumps(save_dict))
fileTwo.close()
