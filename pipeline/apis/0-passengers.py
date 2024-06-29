#!/usr/bin/env python3
"""Creating a method that returns the list of ships that can hold a given number of passengers"""

import requests


def availableShips(passengerCount):
    """Returns a list of ships"""
    url = https://swapi-api.alx-tools.com/api/
    
    req = requests.get(f'{url}/starships')

    output = []
    while req.status_code == 200:
        req = req.json()
        for ship in req['results']:
            passengers = ship['passengers'].replace(',', '')
            try:
                if int(passengers) >= passengerCount:
                    output.append(ship['name'])
            except ValueError:
                pass
        try:
            req = requests.get(req['next'])
        except Exception:
            break
    return output
