import argparse
import numpy as np
import os
from nonstationarynoise.ornstein_uhlenbeck import sample_ou, sample_stationary_ou
from nonstationarynoise.oneoverf import sample_one_over_f, sample_one_over_f_equal, sample_one_over_f_same_gamma
from nonstationarynoise.oubath import sample_ou_bath

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_name', type=str, required=True, 
                        help="Name of the noise to generate (e.g., ornstein_uhlenbeck)")
    parser.add_argument('--out_path', type=str, required=True, 
                        help="File path to save the output")
    
    parser.add_argument('--extra_args', nargs='*', type=float, default=[], 
                        help="Additional parameters for the noise function")

    args = parser.parse_args()

    if args.noise_name == 'ornstein_uhlenbeck':
        signal = sample_ou(*args.extra_args)
    elif args.noise_name =='stationary_ornstein_uhlenbeck':
        signal = sample_stationary_ou(*args.extra_args)
    elif args.noise_name == 'one_over_f':
        signal = sample_one_over_f(*args.extra_args)
    elif args.noise_name == 'one_over_f_equal':
        signal = sample_one_over_f_equal(*args.extra_args)
    elif args.noise_name == 'one_over_f_same_gamma':
        signal = sample_one_over_f_same_gamma(*args.extra_args)
    elif args.noise_name == 'ou_bath':
        signal = sample_ou_bath(*args.extra_args)
    else:
        raise Exception("Noise name not recognised")    

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.savez(
        args.out_path,
        signal=signal,
        extra_args=np.array(args.extra_args),
        noise_name=np.array(args.noise_name) 
    )

if __name__ == "__main__":
    main()
