#!/usr/bin/env python3
"""
Download electrode configurations from Cionic API.

Requires a token.json file.
"""

__usage__ = '''
./scripts/download_electrode_config.py
    [org_shortname]
    [study_shortname]
    [--token_path <filepath to tokenfile>]

Common usage examples:

Print help
./scripts/download_electrode_config.py -h

Interactive mode - prompts for org and study selection
./scripts/download_electrode_config.py

Download all configs from cionic/demo-develop study
./scripts/download_electrode_config.py cionic demo-develop

Download all configs from all studies in cionic org
./scripts/download_electrode_config.py cionic
'''

import argparse
import json
import os

from cionic import api


def download_electrode_configs(
    org_shortname: str,
    study_shortname: str,
    token_path: str = './token.json',
):
    """
    Download electrode configurations from Cionic API.

    Args:
        org_shortname (str): The organization shortname
        study_shortname (str): Study shortname (optional, download all if not given)
        token_path (str): The path to the token file (default: ./token.json)
    """
    # Authenticate
    if not os.path.isfile(token_path):
        raise ValueError(f"Token file not found at '{token_path}'")
    orgs = api.auth(tokenpath=token_path)
    if not orgs:
        raise ValueError(
            f"Auth failed with token file '{token_path}'. Token is invalid/expired."
        )

    # Validate org exists
    org_shortnames = [org['shortname'] for org in orgs]
    if org_shortname not in org_shortnames:
        raise ValueError(
            f"Organization {org_shortname} not found. "
            f"Available orgs: {', '.join(org_shortnames)}"
        )

    # Get studies for org
    studies = api.get_cionic(f'{org_shortname}/studies')
    if not studies:
        raise ValueError(f"No studies found for organization {org_shortname}")

    # Filter studies if study_name is provided (otherwise, use all studies)
    if study_shortname is not None:
        studies = [s for s in studies if s['shortname'] == study_shortname]
    if not studies:
        raise ValueError(
            f"Study {study_shortname} not found in org {org_shortname}. "
            f"Available studies: {', '.join([s['shortname'] for s in studies])}"
        )

    base_dir = os.path.join('./recordings', org_shortname)

    configs_downloaded = 0
    for study in studies:
        study_shortname = study['shortname']
        study_xid = study['xid']

        # Create directory structure
        study_dir = os.path.join(base_dir, study_shortname)
        os.makedirs(study_dir, exist_ok=True)

        # Get configs for this study
        configs = api.get_cionic(
            f"{org_shortname}/studies/{study_xid}/deviceconfig", ver='1.16'
        )

        # Save each config
        for config in configs:
            config_name = config['config_name']
            config_data = config['config']

            # Create safe filename from config name
            safe_name = "".join(
                c for c in config_name if c.isalnum() or c in (' ', '-', '_')
            ).rstrip()
            filename = f"{safe_name}.json"
            filepath = os.path.join(study_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)

            print(f"Saved config '{config_name}' to {filepath}")
            configs_downloaded += 1

    print(f"\nDownloaded {configs_downloaded} configurations")
    return configs_downloaded


def main():
    parser = argparse.ArgumentParser(
        description='Download electrode configurations from Cionic API', usage=__usage__
    )
    parser.add_argument('org_shortname', nargs='?', help='Organization shortname')
    parser.add_argument('study_shortname', nargs='?', help='Study shortname')
    parser.add_argument(
        '--token_path',
        default='./token.json',
        help='Path to token file (default: ./token.json)',
    )

    args = parser.parse_args()

    # Initialize authentication first to get available orgs
    orgs = api.auth(tokenpath=args.token_path)

    # Handle org selection
    if args.org_shortname is None:
        print("\nAvailable organizations:")
        for i, org in enumerate(orgs):
            print(f"{i} : {org['shortname']}")
        choice = int(input("\nChoose an organization: "))
        args.org_shortname = orgs[choice]['shortname']

    # Get studies for selected org
    studies = api.get_cionic(f'{args.org_shortname}/studies')
    if studies is None:
        print(f"Studies not found for org [{args.org_shortname}]")
        return 1

    # Handle study selection if not downloading all
    if args.study_shortname is None:
        print("\nAvailable studies (or press Enter to download from all):")
        for i, study in enumerate(studies):
            print(f"{i} : {study['shortname']}")
        choice = input("\nChoose a study (or press Enter for all): ").strip()
        if choice:
            args.study_shortname = studies[int(choice)]['shortname']

    try:
        download_electrode_configs(
            args.org_shortname, args.study_shortname, args.token_path
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
