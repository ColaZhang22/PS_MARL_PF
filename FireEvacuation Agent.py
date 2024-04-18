from mlagents_envs.environment import UnityEnvironment,ActionTuple
import numpy as np




env = UnityEnvironment(file_name="UnityEnvironment", seed=1, side_channels=[])
# Start interacting with the environment.
env.reset()
behavior_names = list(env.behavior_specs.keys())
behavior_value = list(env.behavior_specs.values())
for i in range(len(behavior_names)):
    print(behavior_names[i])
    print("obs:",behavior_value[i].observation_specs, "   act:", behavior_value[0].action_spec)

DecisionSteps, TerminalSteps = env.get_steps(behavior_names[0])
agentsNum = len(DecisionSteps.agent_id)
print("exist:",DecisionSteps.agent_id,"   Dead:",TerminalSteps.agent_id)
print("reward:",DecisionSteps.reward,"reward_dead:",TerminalSteps.reward)
print("obs:",DecisionSteps.obs,"DeadObs:",TerminalSteps.obs)
print("interrupted:", TerminalSteps.interrupted)
continuous_actions = (np.random.rand(agentsNum, 2) - 0.5) * 2
discrete_actions = None
actions = ActionTuple(continuous_actions, discrete_actions)
env.set_actions(behavior_names[0],actions)


print("First Step")
