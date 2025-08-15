#!/usr/bin/env python3
import argparse
import json
import os

from cionic import api


def upload_electrode_config(
    org_id, study_name, config_name, json_path, token_path, update=False
):
    # Authenticate
    d = api.auth(tokenpath=token_path)

    # Validate org exists
    orgs = [org['shortname'] for org in d]
    if org_id not in orgs:
        raise ValueError(
            f"Organization {org_id} not found. Available orgs: {', '.join(orgs)}"
        )

    # Get studies for org
    studies = api.get_cionic(f'{org_id}/studies')
    study_map = {s['shortname']: s['xid'] for s in studies}

    if study_name not in study_map:
        raise ValueError(
            f"Study {study_name} not found. "
            f"Available studies: {', '.join(study_map.keys())}"
        )

    study_xid = study_map[study_name]

    # Read and validate JSON config
    try:
        with open(json_path, 'r') as f:
            config_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_path}")
    except FileNotFoundError:
        raise ValueError(f"JSON file not found: {json_path}")

    # Prepare payload
    payload = {'config_name': config_name, 'config': config_data}

    if update:
        # Get existing configs to find the config_id
        configs = api.get_cionic(
            f"{org_id}/studies/{study_xid}/deviceconfig", ver='1.16'
        )

        # Find the config with matching name
        matching_config = next(
            (c for c in configs if c['config_name'] == config_name), None
        )
        if not matching_config:
            raise ValueError(
                f"No existing config named '{config_name}' "
                f"found in study '{study_name}'"
            )

        # Update existing config
        response = api.put_cionic(
            f"{org_id}/studies/{study_xid}/deviceconfig/{matching_config['xid']}",
            json=payload,
            ver='1.16',
        )
        print(
            f"Successfully updated config '{config_name}' "
            f"in study '{study_name}' in org '{org_id}'"
        )
    else:
        # Create new config
        response = api.post_cionic(
            f"{org_id}/studies/{study_xid}/deviceconfig/", json=payload, ver='1.16'
        )
        print(
            f"Successfully created new config '{config_name}' "
            f"in study '{study_name}' in org '{org_id}'"
        )

    return response


def main():
    parser = argparse.ArgumentParser(
        description='Upload electrode configuration to Cionic API'
    )
    parser.add_argument('--org', required=True, help='Organization ID')
    parser.add_argument('--study', required=True, help='Study name')
    parser.add_argument(
        '--config-name', required=True, help='Name for the configuration'
    )
    parser.add_argument(
        '--json-path', required=True, help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--token-path',
        default='../token.json',
        help='Path to token file (default: ../token.json)',
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing config instead of creating new one',
    )

    args = parser.parse_args()

    try:
        upload_electrode_config(
            args.org,
            args.study,
            args.config_name,
            args.json_path,
            os.path.abspath(args.token_path),
            update=args.update,
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
