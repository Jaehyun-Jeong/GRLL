from grll.utils.ActionSpace.ActionSpace import ActionSpace


def overrides(interface_class):
    def overrider(method):
        assert (method.__name__ in dir(interface_class))
        return method
    return overrider


def get_action_space(
        trainEnv,
        testEnv,
        ):

    # trainEnv, testEnv exist and don't match
    if trainEnv.action_space \
            != testEnv.action_space:
        raise ValueError(
                "Action Spaces of trainEnv and testEnv don't match")

    # testEnv, trainEnv exsit and match
    else:
        actionSpace = trainEnv.action_space

    # If trainEnv, testEnv are not using same ActionSpace in this module
    if not isinstance(actionSpace, ActionSpace):
        actionSpace = ActionSpace(actionSpace=trainEnv.action_space)

    return actionSpace
