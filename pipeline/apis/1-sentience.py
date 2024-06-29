#!/usr/bin/env python3

"""Returns the list of names of the home planets of all sentient species"""

import requests


def sentientPlanets():
    """ Return list of names of the home planets of all sentient species"""

    url = "https://swapi-api.alx-tools.com/api/species/?page=1"
    planets = []

    while url:
        response = requests.get(url).json()

        for specy in response['results']:
            classification = specy['classification']
            designation = specy['designation']
            if classification == 'sentient' or designation == 'sentient':
                if specy['homeworld']:
                    get_planet = requests.get(specy['homeworld']).json()
                    planets.append(get_planet['name'])

        url = response['next']

    return planets
