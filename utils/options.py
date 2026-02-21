import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--total_UE', type=int, default=20)
    parser.add_argument('--active_UE', type=int, default=10)
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--local_bs', type=int, default=32)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)

    # parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--iid', type=str, default='dirichlet') #dirichlet,iid
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=1)  
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='fedavg')
    
    parser.add_argument('--mode', type=str, default='random_selection', choices=['ideal', 'random_selection'])


    # Experiment switches
    parser.add_argument('--method', type=str, default='marl',
                        choices=['random', 'greedy_channel', 'marl'],
                        help='Client selection method')
    
    # parser.add_argument('--altitude_mode', type=str, default='fixed',
    #                     choices=['fixed', 'random', 'marl'],
    #                     help='How UAV altitudes are set')
    
    parser.add_argument('--wireless_on', action='store_true', default=True,
                    help='Enable wireless success/failure model')

    
    parser.add_argument('--snr_th', type=float, default=20.0,
                    help='SNR threshold for successful upload')
 # Dataset and models
    parser.add_argument('--dataset', type=str, default='mnist',
                     choices=['mnist', 'cifar10'],
                     help='Dataset to use')
    parser.add_argument('--model', type=str, default='cnn60k',
                    choices=['cnn', 'resnet', 'cnn60k'],
                    help='Model architecture')
 # Different Environments 
 
    parser.add_argument('--env', type=str, default='highrise',
                    choices=['suburban', 'urban','denseurban', 'highrise'],
                    help='Propagation environment type')


        # --- Altitude bounds (for MARL / scenario) ---
    parser.add_argument('--h_min', type=float, default=80.0)
    parser.add_argument('--h_max', type=float, default=500.0)
    parser.add_argument('--delta_h_max', type=float, default=20.0)

    # --- MARL training switches ---
    parser.add_argument('--train_marl', action='store_true', default=False)
    parser.add_argument('--marl_episodes', type=int, default=5000)
    parser.add_argument('--episode_len', type=int, default=100)  # wireless-only episode length
    parser.add_argument('--seed', type=int, default=20)

    # --- PPO / MAPPO hyperparams ---
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_clip', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--ppo_epochs', type=int, default=5)
    parser.add_argument('--minibatch_size', type=int, default=256)

    # --- Reward smoothing (helps PPO) ---
    parser.add_argument('--snr_kappa', type=float, default=0.1)  # smooth success prob

    parser.add_argument('--marl_policy_path', type=str, default='marl_policy.pt')
    args = parser.parse_args()
    return args

