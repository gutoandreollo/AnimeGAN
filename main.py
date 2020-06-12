import argparse
import json
from utils import *

from AnimeGAN import AnimeGAN


def main():
    # Setup and parse command-line arguments
    parser = argparse.ArgumentParser(description="Transforms your pictures into animation using AI!")
    parser.add_argument("--phase", type=str, default="transform", choices=["transform", "train"], help="Phase to run, either transform pictures or train AI")
    parser.add_argument("--config", type=str, required=True, help="Path to a configuration file")
    parser.add_argument("--images", type=str, required=True, help="[transform] Path to a images to transform")
    args = parser.parse_args()

    # Open Tensorflow session
    tf.device(os.getenv('GPU_DEVICE', "/device:GPU:0"))
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8,
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    )
    with tf.compat.v1.Session(config=tf_config) as sess:
        # Load configuration file
        with open(args.config) as f:
            params = json.load(f)

        # Build model
        gan = AnimeGAN(sess, params)

        # Run command
        if args.phase == 'train':
            gan.train()
        if args.phase == 'transform':
            gan.transform()


if __name__ == '__main__':
    main()
