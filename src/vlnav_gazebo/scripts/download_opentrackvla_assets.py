#!/usr/bin/env python3
import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    print(
        'Missing Python dependency: huggingface_hub\n'
        'Install it into the same interpreter with:\n'
        '  python3 -m pip install huggingface_hub safetensors transformers accelerate',
        file=sys.stderr,
    )
    raise


ASSETS = {
    'opentrackvla': 'omlab/opentrackvla-qwen06b',
    'qwen': 'Qwen/Qwen3-0.6B',
    'dino': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'siglip': 'google/siglip-so400m-patch14-384',
}


def local_dir(root: str, repo_id: str) -> str:
    return os.path.join(root, repo_id.replace('/', '__'))


def main():
    parser = argparse.ArgumentParser(description='Download OpenTrackVLA and encoder checkpoints.')
    parser.add_argument('--root', default='/root/Desktop/vln_project/models/hf')
    parser.add_argument(
        '--asset',
        action='append',
        choices=sorted(ASSETS),
        help='Download one asset. Repeatable. Defaults to all assets.',
    )
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', ''))
    parser.add_argument(
        '--proxy',
        default='',
        help='Optional HTTP/SOCKS proxy, for example http://127.0.0.1:7890 or socks5h://127.0.0.1:1080.',
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ['HF_ENDPOINT'] = args.endpoint
    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
        os.environ['ALL_PROXY'] = args.proxy

    os.makedirs(args.root, exist_ok=True)
    assets = args.asset or list(ASSETS)
    for name in assets:
        repo_id = ASSETS[name]
        target = local_dir(args.root, repo_id)
        print(f'[download] {name}: {repo_id} -> {target}', flush=True)
        try:
            snapshot_download(repo_id, repo_type='model', local_dir=target)
        except Exception as exc:
            print(f'[error] failed to download {repo_id}: {exc}', file=sys.stderr)
            print(
                '\nNetwork troubleshooting:\n'
                '  1. Direct HuggingFace: python3 src/vlnav_gazebo/scripts/download_opentrackvla_assets.py\n'
                '  2. Mirror endpoint:     python3 src/vlnav_gazebo/scripts/download_opentrackvla_assets.py --endpoint https://hf-mirror.com\n'
                '  3. HTTP proxy:         python3 src/vlnav_gazebo/scripts/download_opentrackvla_assets.py --proxy http://HOST:PORT\n'
                '  4. SOCKS proxy:        python3 src/vlnav_gazebo/scripts/download_opentrackvla_assets.py --proxy socks5h://HOST:PORT\n',
                file=sys.stderr,
            )
            raise
        print(f'[done] {name}: {target}', flush=True)

    print('\nUse these launch overrides for local checkpoints:')
    print(f'  hf_model_dir:={local_dir(args.root, ASSETS["opentrackvla"])}')
    print(f'  qwen_model_name:={local_dir(args.root, ASSETS["qwen"])}')
    print(f'  dino_model_name:={local_dir(args.root, ASSETS["dino"])}')
    print(f'  siglip_model_name:={local_dir(args.root, ASSETS["siglip"])}')


if __name__ == '__main__':
    main()
