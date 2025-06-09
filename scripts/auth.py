#!/usr/bin/env python3

import argparse
import sys

sys.path.append('.')
import cionic

__usage__ = '''
./scripts/auth.py [email] [org]
    [-a <admin collector analyst>]
    [-r <admin collector analyst>]
REQUIRES ORG ADMIN ROLE
'''


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=__usage__)
    parser.add_argument('email', help='email to grant permission')
    parser.add_argument('org', nargs='?', help='organization shortname')
    parser.add_argument(
        '-a', dest="add", nargs='+', help='add role flags: -a analyst collector admin'
    )
    parser.add_argument(
        '-r',
        dest="rem",
        nargs='+',
        help='remove role flags: -d analyst collector admin',
    )
    parser.add_argument(
        '-t',
        dest='token',
        default='token.json',
        help='path to auth credentials json file',
    )

    args = parser.parse_args(sys.argv[1:])
    tokenpath = args.token
    orgs = cionic.auth(tokenpath=tokenpath)
    org_names = [org['shortname'] for org in orgs]

    if args.org not in org_names:
        if args.org:
            print(f"Invalid org name [{args.org}]:")
        else:
            print("Available orgs:")
        for org in org_names:
            print(f"  {org}")
        return

    if not args.add and not args.rem:
        print("Please specify at least one role to add [-a] or remove [-r]")
        return

    users = cionic.get_user(args.email)
    if len(users) > 0:
        user = users[0]
    else:
        # prompt to create user
        answer = input(
            f'User [{args.email}] no found for [{args.org}]. Create Y/N\n'
        ).lower()
        if answer == "y":
            user = cionic.create_user(args.org, args.email)
        else:
            return

    if args.add:
        cionic.add_roles(args.org, user['xid'], args.add)
    if args.rem:
        cionic.remove_roles(args.org, user['xid'], args.rem)


if __name__ == '__main__':
    main()
