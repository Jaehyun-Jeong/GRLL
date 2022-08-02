from datetime import datetime, timedelta

class RL():

    def __init__(
        self, 
        trainEnv, 
        testEnv, 
        env,
        model,
        optimizer,
        eps,
        policy,
        device,
        isRender, 
        useTensorboard, 
        tensorboardParams,
    ):

        # set Environment
        

        if env==None and trainEnv!=None and testEnv!=None:
            self.trainEnv = trainEnv
            self.testEnv = testEnv
        elif env!=None and trainEnv==None and testEnv==None:
            self.trainEnv = env
            self.testEnv = env
        else:
            ValueError(
                "No Environment or you just use one of trainEnv, or testEnv,"
                "or you set the env with trainEnv or testEnv,"
                "or you set the trainEnv or testEnv with env"
            )    

        # init parameters
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.policy = policy
        self.isRender = isRender
        self.useTensorboard = useTensorboard
        self.tensorboardParams = tensorboardParams
        
        # Init trained Episode
        self.trainedEpisodes = 0
        self.trainedTimesteps = 0

        # check the time spent
        self.timePrevStep = datetime.now()
        self.timeSpent = timedelta(0)

        # init Summary Writer
        if self.useTensorboard:
            from tensorboardX import SummaryWriter 
            self.tensorboardWriter = SummaryWriter(self.tensorboardParams['logdir'])

        # select train, test policy
        policyDict = {'greedy': [False, False], 'stochastic': [False, True], 'eps-greedy': [True, False], 'eps-stochastic': [True, True]} # [ useEpsilon, useStochastic ]

        if ( not self.policy['train'] in policyDict.keys() ) or ( not self.policy['test'] in policyDict.keys() ):
            raise ValueError("Possible policies are 'greedy', 'eps-greedy', 'stochastic', and 'eps-stochastic'")

        trainPolicyList = policyDict[self.policy['train']]
        testPolicyList = policyDict[self.policy['test']]

        if trainPolicyList[0] or testPolicyList[0]:
            self.eps = eps

        self.useTrainEps = trainPolicyList[0]
        self.useTrainStochastic = trainPolicyList[1]
        self.useTestEps = testPolicyList[0]
        self.useTestStochastic = testPolicyList[1]

    def writeTensorboard(self, y, x: int):
        if self.useTensorboard:
            try:
                self.tensorboardWriter.add_scalar(self.tensorboardParams['tag'], y, x)
            except:
                ValueError("Can not use tensorboard!")

    def test(self, testSize=10):
        
        returns = []

        for testIdx in range(testSize):
            state = self.testEnv.reset()
            done = False
            rewards = []
            for timesteps in range(self.maxTimesteps):
                if self.isRender['test']:
                    self.testEnv.render()

                action = self.get_action(state, useEps=self.useTestEps, useStochastic=self.useTestStochastic)
                next_state, reward, done, _ = self.testEnv.step(action.tolist())

                rewards.append(reward)
                state = next_state

                if done or timesteps == self.maxTimesteps-1:
                    break
            
            returns.append(sum(rewards))
        
        if testSize > 0:
            averagedReturn = sum(returns) / testSize
        elif testSize == 0:
            averagedReturn = "no Test"
        else:
            #ERROR!
            print("error")

        return averagedReturn

    def printResult(self, episode: int, timesteps: int, averagedReturn):
        
        self.timeSpent += datetime.now() - self.timePrevStep

        results = f"| Episode / Timesteps : {str(episode)[0:10]:>10} / {str(timesteps)[0:10]:>10} | Averaged Return: {str(averagedReturn)[0:10]:>10} | Time Spent : {str(self.timeSpent):10} / {str(datetime.now()-self.timePrevStep):10} | "

        print(results)

        splited = results.split('|')[1:-1]
        frameString = "+"

        for split in splited:
            frameString += "-"*len(split) + "+"

        print(frameString)

        self.timePrevStep = datetime.now()

    def save(self, saveDir: str = str(datetime)+".obj"):
        import torch

        save_dict = self.__dict__

        # belows are impossible to dump
        save_dict.pop('tensorboardWriter', None)
        save_dict.pop('trainEnv', None)
        save_dict.pop('testEnv', None)
        
        # save model state dict
        save_dict['modelStateDict'] = save_dict['model'].state_dict()
        save_dict.pop('model', None)
        save_dict['optimizerStateDict'] = save_dict['optimizer'].state_dict()
        save_dict.pop('optimizer', None)

        torch.save(save_dict, saveDir)
        
    def load(self, loadDir):
        import torch

        #=============================================================================
        # LOAD TORCH MODEL 
        #=============================================================================

        loadedDict = torch.load(loadDir, map_location=self.device)

        self.model.load_state_dict(loadedDict.pop('modelStateDict'))
        self.optimizer.load_state_dict(loadedDict.pop('optimizerStateDict'))

        self.model.eval()

        #=============================================================================

        for key, value in loadedDict.items():
            self.__dict__[key] = value
        
        self.timePrevStep = datetime.now() # Recalculating time spent
