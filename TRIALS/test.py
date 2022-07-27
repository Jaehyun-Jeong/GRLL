import pickle

file = open("./saved_models/DQN_RacingEnv_v0.obj", 'rb')
dataPickle = file.read()
file.close()

save_dict = pickle.loads(dataPickle)

save_dict['modelStateDict'] = self.model.state_dict()
save_dict.pop('model', None)
save_dict['optimizerStateDict'] = self.model.state_dict()
save_dict.pop('optimizer', None)

file = open(saveDir, 'wb')
file.write(pickle.dumps(save_dict))
file.close()
