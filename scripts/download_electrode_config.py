#!/usr/bin/env python3
import argparse
import json
import os

from cionic import api


def download_electrode_configs(org_id, study_name=None, token_path=None):
    # Authenticate
    if token_path is None:
        token_path = os.path.abspath('../token.json')
    d = api.auth(tokenpath=token_path)
    if not d:
        raise ValueError("Authentication failed")

    # Validate org exists
    orgs = [org['shortname'] for org in d]
    if org_id not in orgs:
        raise ValueError(
            f"Organization {org_id} not found. Available orgs: {', '.join(orgs)}"
        )

    # Get studies for org
    studies = api.get_cionic(f'{org_id}/studies')
    if not studies:
        raise ValueError(f"No studies found for organization {org_id}")

    # Filter studies if study_name is provided
    if study_name:
        print(studies)
        studies = [s for s in studies if s['shortname'] == study_name]
        if not studies:
            raise ValueError(f"Study {study_name} not found in org {org_id}")

    base_dir = os.path.join('recordings', org_id)

    configs_downloaded = 0
    for study in studies:
        study_shortname = study['shortname']
        study_xid = study['xid']

        # Create directory structure
        study_dir = os.path.join(base_dir, study_shortname)
        os.makedirs(study_dir, exist_ok=True)

        # Get configs for this study
        configs = api.get_cionic(
            f"{org_id}/studies/{study_xid}/deviceconfig", ver='1.16'
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
        description='Download electrode configurations from Cionic API'
    )
    parser.add_argument('--org', required=True, help='Organization ID')
    parser.add_argument(
        '--study', help='Study name (optional, downloads all studies if not specified)'
    )
    parser.add_argument(
        '--token-path', help='Path to token file (default: ../token.json)'
    )

    args = parser.parse_args()

    try:
        download_electrode_configs(args.org, args.study, args.token_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
