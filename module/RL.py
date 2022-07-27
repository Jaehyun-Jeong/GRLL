
from datetime import datetime

class RL():

    def __init__(
        self, 
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

        # init parameters
        self.env = env 
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.policy = policy
        self.isRender = isRender
        self.useTensorboard = useTensorboard
        self.tensorboardParams = tensorboardParams

        # check the time spent
        self.timeStart = datetime.now()
        self.timePrevStep = datetime.now()

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
            state = self.env.reset()
            done = False
            rewards = []
            for timesteps in range(self.maxTimesteps):
                if self.isRender['test']:
                    self.env.render()

                action = self.get_action(state, useEps=self.useTestEps, useStochastic=self.useTestStochastic)
                next_state, reward, done, _ = self.env.step(action.tolist())

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

    def printResult(self, episode: int, averagedReturn):
        results = f"| Episode: {str(episode)[0:10]:>10} | Averaged Return: {str(averagedReturn)[0:10]:>10} | Time/step: {str(datetime.now()-self.timePrevStep):10} | Time spent: {str(datetime.now()-self.timeStart):10} | "

        print(results)

        splited = results.split('|')[1:-1]
        frameString = "+"

        for split in splited:
            frameString += "-"*len(split) + "+"

        print(frameString)

        self.timePrevStep = datetime.now()

    def save(self, saveDir: str = str(datetime)+".obj"):

        import pickle

        save_dict = self.__dict__

        # belows are impossible to dump
        save_dict.pop('tensorboardWriter', None)
        save_dict.pop('env', None)

        file = open(saveDir, 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, loadDir):

        import pickle

        file = open(loadDir, 'rb')
        dataPickle = file.read()
        file.close()

        for key, value in pickle.loads(dataPickle).items():
            self.__dict__[key] = value

        self.model.eval()
        
