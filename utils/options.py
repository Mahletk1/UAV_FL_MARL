import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--round', type=int, default=20)
    parser.add_argument('--total_UE', type=int, default=10)
    parser.add_argument('--active_UE', type=int, default=5)
    parser.add_argument('--local_ep', type=int, default=2)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--iid', type=str, default='iid')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=1)  
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='fedavg')


    args = parser.parse_args()
    return args

