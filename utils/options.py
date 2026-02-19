import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--round', type=int, default=100)
    parser.add_argument('--total_UE', type=int, default=20)
    parser.add_argument('--active_UE', type=int, default=10)
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)

    # parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--iid', type=str, default='dirichlet') #dirichlet,iid
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=1)  
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='fedavg')
    
    parser.add_argument('--mode', type=str, default='random_selection', choices=['ideal', 'random_selection'])


    # Experiment switches
    parser.add_argument('--method', type=str, default='random',
                        choices=['random', 'greedy_channel', 'marl'],
                        help='Client selection method')
    
    parser.add_argument('--altitude_mode', type=str, default='fixed',
                        choices=['fixed', 'random', 'marl'],
                        help='How UAV altitudes are set')
    
    parser.add_argument('--wireless_on', action='store_true', default=True,
                    help='Enable wireless success/failure model')

    
    parser.add_argument('--snr_th', type=float, default=15.0,
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

# in utils/options.py or argparse in main.py
    parser.add_argument('--policy', type=str, default='mappo',
                    choices=['fixed', 'random', 'mappo'],
                    help='UAV altitude control policy')


    args = parser.parse_args()
    return args

