"""
Upload electrode configuration to Cionic API. Use the --new flag to make a new config.

Requires a token.json file and a config json file to upload.

Example usage:

For a new config:
python upload_electrode_config.py \
    --org cionic \
    --study demo-develop \
    --config-name "EXAMPLE_CONFIG" \
    --json-path "../recordings/cionic/artemis/EXAMPLE_CONFIG.json" \
    --token-path "../token.json" \
    --new

For an existing config:
python upload_electrode_config.py \
    --org cionic \
    --study demo-develop \
    --config-name "EXAMPLE_CONFIG" \
    --json-path "../recordings/cionic/artemis/EXAMPLE_CONFIG.json" \
    --token-path "../token.json"

"""

import argparse
import json
from difflib import get_close_matches

from cionic import api


def find_closest_config_name(config_name: str, available_configs: list) -> str:
    """
    Find the closest matching config name using fuzzy string matching.

    Args:
        config_name (str): The config name to match
        available_configs (list): List of available config dictionaries

    Returns:
        str: The closest matching config name, or empty string if no close matches found
    """
    config_names = [c['config_name'] for c in available_configs]
    matches = get_close_matches(config_name, config_names, n=1, cutoff=0.6)
    return matches[0] if matches else ""


def upload_electrode_config(
    org_shortname, study_shortname, config_name, json_path, token_path, new=False
):
    """
    Upload electrode configuration to Cionic API.

    Args:
        org_shortname (str): The organization shortname
        study_shortname (str): The study shortname
        config_name (str): The name of the configuration
        json_path (str): The path to the JSON configuration file
        token_path (str): The path to the token file
        new (bool): Whether to create a new config (default: False)
    """

    # Authenticate
    orgs = api.auth(tokenpath=token_path)

    # Validate org exists
    org_shortnames = [org['shortname'] for org in orgs]
    if org_shortname not in org_shortnames:
        raise ValueError(
            f"Organization {org_shortname} not found. "
            f"Available orgs: {', '.join(org_shortnames)}"
        )

    # Get studies for org
    studies = api.get_cionic(f'{org_shortname}/studies')
    study_map = {s['shortname']: s['xid'] for s in studies}

    if study_shortname not in study_map:
        raise ValueError(
            f"Study {study_shortname} not found. "
            f"Available studies: {', '.join(study_map.keys())}"
        )

    study_xid = study_map[study_shortname]

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

    if new:
        # Create new config
        response = api.post_cionic(
            f"{org_shortname}/studies/{study_xid}/deviceconfig/",
            json=payload,
            ver='1.16',
        )
        if response is None:
            raise ValueError(
                f"Failed to create new config '{config_name}' "
                f"in study '{study_shortname}' in org '{org_shortname}'. "
                f"It may already exist."
            )
        else:
            print(
                f"Successfully created new config '{config_name}' "
                f"in study '{study_shortname}' in org '{org_shortname}'"
            )
    else:
        # Get existing configs to find the config_id
        configs = api.get_cionic(
            f"{org_shortname}/studies/{study_xid}/deviceconfig", ver='1.16'
        )

        # Find the config with matching name
        matching_config = next(
            (c for c in configs if c['config_name'] == config_name), None
        )
        if not matching_config:
            suggestion = find_closest_config_name(config_name, configs)
            suggestion_msg = f"\nDid you mean '{suggestion}'?" if suggestion else ""
            raise ValueError(
                f"No existing config named '{config_name}' "
                f"found in study '{study_shortname}'{suggestion_msg}"
            )

        # Update existing config
        response = api.put_cionic(
            f"{org_shortname}/studies/{study_xid}/"
            f"deviceconfig/{matching_config['xid']}",
            json=payload,
            ver='1.16',
        )
        print(
            f"Successfully updated config '{config_name}' "
            f"in study '{study_shortname}' in org '{org_shortname}'"
        )

    return response


def main():
    parser = argparse.ArgumentParser(
        description='Upload electrode configuration to Cionic API'
    )
    parser.add_argument('--org_shortname', required=True, help='Organization ID')
    parser.add_argument('--study_shortname', required=True, help='Study name')
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
        '--new',
        action='store_true',
        help='Create new config instead of updating existing one',
    )

    args = parser.parse_args()

    try:
        upload_electrode_config(
            args.org_shortname,
            args.study_shortname,
            args.config_name,
            args.json_path,
            args.token_path,
            new=args.new,
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
