import argparse
from Runner.FireEvacuationRunner import EvacuationRunner

from  Runner.PolicyGradienRunner import PolicyBasedEvacuationRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in FireEvacuation environment")
    parser.add_argument("--max_train_steps", type=int, default=int(4e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000,help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float  , default=32, help="Evaluate times")
    parser.add_argument("--episode_limit", type=int, default=32, help="Maximum number of steps per episode")
    parser.add_argument("--save_freq", type=int, default=int(1e3), help="Save frequency")
    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN or IQL")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=10000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim",  type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The dimension of the hidd en layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")
    parser.add_argument("--env_name", type=str, default='3m', help="map smac use")
    parser.add_argument("--gpu_available", type=bool, default='True', help="train in cpu or gpu")


    args=parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps
    # env_names = './EnvironmentOneAgent/UnityEnvironment'
    # env_names = 'UnityEnvironment'
    # env_names = './EnvironmentOneAgentRandom/UnityEnvironment'
    # env_names = './EnvironmentMultiAgentRandom/UnityEnvironment'
    env_names = './EnvironmentOneAgentRandom/UnityEnvironment'
    # runner = PolicyBasedEvacuationRunner (args, env=env_names, number=1, seed=0)

    runner = EvacuationRunner(args, env=env_names, number=1, seed=0)
    runner.run()
