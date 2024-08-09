# -*- coding: utf-8 -*-
"""
A small toolkit to help with betting the Euro 2024.

Created on Thu Jun 13 17:38:55 2024.

@author: Tim Wiegmann
"""


from datetime import datetime
import numpy as np


FIXTURES_GROUP_STAGE_1 = [("Germany", "Scotland"),
                          ("Hungary", "Switzerland"),
                          ("Spain", "Croatia"),
                          ("Italy", "Albania"),
                          ("Poland", "Netherlands"),
                          ("Slovenia", "Denmark"),
                          ("Serbia", "England"),
                          ("Romania", "Ukraine"),
                          ("Belgium", "Slovakia"),
                          ("Austria", "France"),
                          ("Turkey", "Georgia"),
                          ("Portugal", "Czechia")]

FIXTURES_GROUP_STAGE_2 = [("Croatia", "Albania"),
                          ("Germany", "Hungary"),
                          ("Scotland", "Switzerland"),
                          ("Slovenia", "Serbia"),
                          ("Denmark", "England"),
                          ("Spain", "Italy"),
                          ("Slovakia", "Ukraine"),
                          ("Poland", "Austria"),
                          ("Netherlands", "France"),
                          ("Georgia", "Czechia"),
                          ("Turkey", "Portugal"),
                          ("Belgium", "Romania")]

FIXTURES_GROUP_STAGE_3 = [("Switzerland", "Germany"),
                          ("Scotland", "Hungary"),
                          ("Croatia", "Italy"),
                          ("Albania", "Spain"),
                          ("Netherlands", "Austria"),
                          ("France", "Poland"),
                          ("Denmark", "Serbia"),
                          ("England", "Slovenia"),
                          ("Ukraine", "Belgium"),
                          ("Slovakia", "Romania"),
                          ("Czechia", "Turkey"),
                          ("Georgia", "Portugal")]

FIXTURES = (FIXTURES_GROUP_STAGE_1
            + FIXTURES_GROUP_STAGE_2
            + FIXTURES_GROUP_STAGE_3)

TEAM_ALIASES = {"Czech Republic": "Czechia",
                "Republic of Ireland": "Ireland"}

CONVERGENCE_MATCHES = 30


def load_match_database(file_name: str) -> dict:
    """
    Load the database of historical match results from a csv file.

    Parameters
    ----------
    file_name : str
        Name of the csv file with the match results.
        Download it from https://www.kaggle.com/martj42/datasets

    Returns
    -------
    database : dict
        A dictionary with the historical match results.

    """
    match_database = []

    with open(file_name, encoding="utf8") as csv_file:

        for i, line in enumerate(csv_file.readlines()):

            if i == 0:
                continue

            fields = line.strip().split(",")

            if fields[3] == "NA":
                continue

            def _apply_alias(team_name):
                return TEAM_ALIASES.get(team_name, team_name)

            match_database.append({"date": datetime.fromisoformat(fields[0]),
                                   "home_team": _apply_alias(fields[1]),
                                   "away_team": _apply_alias(fields[2]),
                                   "home_goals": int(fields[3]),
                                   "away_goals": int(fields[4]),
                                   "competition": fields[5]})

    return match_database


def load_elo_database(file_name: str) -> dict:
    """
    Load the database of Elo ratings from a txt file.

    Parameters
    ----------
    file_name : str
        Name of the txt file with the Elo ratings.
        Create it manually by copy-pasting from https://www.eloratings.net/

    Returns
    -------
    dict
        A dictionary with the Elo ratings.

    """
    elo_database = {}

    with open(file_name, encoding="utf8") as txt_file:

        for i, line in enumerate(txt_file.readlines()):

            if i % 2 == 0:
                current_country = line.strip()

            else:
                elo_database[current_country] = int(line)

    return elo_database


def get_participating_teams(fixtures: list) -> list:
    """
    Get a list of participating teams from a list of fixtures.

    Parameters
    ----------
    fixtures : list
        List of fixtures, with each fixture as a tuple of team names.

    Returns
    -------
    list
        List of all teams in the fixtures without duplicates.

    """
    teams = []

    for fixture in fixtures:
        teams.extend([fixture[0], fixture[1]])

    return list(dict.fromkeys(teams))


def filter_match_database(match_database: list, teams: list,
                          number_of_matches: int) -> list:
    """
    Filter match database for the n most recent matches by specified teams.

    Parameters
    ----------
    match_database : list
        The full match database.
    teams : list
        The list of teams whose matches to include.
    number_of_matches : int
        Include the n most recent matches of every team.

    Returns
    -------
    list
        The filtered match database.

    """
    matches_remaining = {team: number_of_matches for team in teams}
    filtered_match_database = []

    for match in reversed(match_database):

        if (matches_remaining.get(match["home_team"], 0) > 0
                or matches_remaining.get(match["away_team"], 0) > 0):

            filtered_match_database.append(match)

            matches_remaining[match["home_team"]] = matches_remaining.get(
                match["home_team"], 0) - 1

            matches_remaining[match["away_team"]] = matches_remaining.get(
                match["away_team"], 0) - 1

    return list(reversed(filtered_match_database))


def format_match_by_winner(match: dict) -> dict:
    """
    Format the match by winner/loser instead of home/away team.

    In case of a draw, the home team is considered the winner.

    Parameters
    ----------
    match : dict
        The match in home/away format.

    Returns
    -------
    dict
        The match in winner/loser format.

    """
    home_win = match["home_goals"] >= match["away_goals"]

    return {"date": match["date"],
            "winner_team": (match["home_team"] if home_win
                            else match["away_team"]),
            "loser_team": (match["away_team"] if home_win
                           else match["home_team"]),
            "winner_goals": (match["home_goals"] if home_win
                             else match["away_goals"]),
            "loser_goals": (match["away_goals"] if home_win
                            else match["home_goals"]),
            "competition": match["competition"]}


def result_histogram(matches: list, elo_database: dict,
                     cutoff_percentage: int = 5):
    """
    Print histograms of historical matches to the console.

    Will show the distribution of goal differences over the matches in the
    database, the average Elo difference for each goal difference, and the
    distribution of results for each goal difference.

    Parameters
    ----------
    matches : list
        Database of historical matches.
    elo_database : dict
        Database of Elo ratings.
    cutoff_percentage : int, optional
        Results with probabilities smaller than this percentage are considered
        rare and ignored. The default is 5.

    Returns
    -------
    None.

    """
    elo_histogram = {}

    for match in [format_match_by_winner(match) for match in matches]:

        goal_diff = match["winner_goals"] - match["loser_goals"]

        elo_difference = (elo_database[match["winner_team"]]
                          - elo_database[match["loser_team"]])

        elo_histogram[goal_diff] = (elo_histogram.get(goal_diff, [])
                                    + [elo_difference])

    rescaled_elo_histogram = {}
    removed_matches = 0

    for goal_diff in sorted(elo_histogram):

        percentage = round(len(elo_histogram[goal_diff]) * 100 / len(matches))

        if percentage >= cutoff_percentage:
            rescaled_elo_histogram[goal_diff] = elo_histogram[goal_diff]

        else:
            removed_matches += len(elo_histogram[goal_diff])

    for goal_diff in rescaled_elo_histogram:

        percentage = round(len(rescaled_elo_histogram[goal_diff]) * 100
                           / (len(matches) - removed_matches))

        print("GD " + str(goal_diff) + " occurs in "
              + str(percentage) + " % of matches.")

        print("The average Elo difference is "
              + str(round(np.average(elo_histogram[goal_diff]))) + " ± "
              + str(round(np.std(elo_histogram[goal_diff]))) + ".")

        results_count = {}

        for match in [format_match_by_winner(match) for match in matches]:

            if match["winner_goals"] - match["loser_goals"] == goal_diff:

                result = (str(match["winner_goals"]) + ":"
                          + str(match["loser_goals"]))

                results_count[result] = (results_count.get(result, 0) + 1)

        print("Likely results: " + str(dict(sorted(results_count.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True))))
        print()


def predict_fixtures(fixtures: list, elos: dict, goal_diff_bins: list,
                     results_dict: dict):
    """
    Print predictions to the console.

    Parameters
    ----------
    fixtures : list
        The list of fixtures to be predicted.
    elos : dict
        Database of Elo ratings.
    goal_diff_bins : list
        List of percentiles for increasing goal differences.
    results_dict : dict
        Dictionary for translating predicted goal differences to results.

    Returns
    -------
    None.

    """
    predictions = {}

    for i, fixture in enumerate(fixtures):

        predictions[i] = {"display_id": i,
                          "home_team": fixture[0],
                          "away_team": fixture[1],
                          "elo_diff": elos[fixture[0]] - elos[fixture[1]]}

    predictions = dict(sorted(predictions.items(),
                              key=lambda item: np.abs(item[1]["elo_diff"])))

    for i, bin in enumerate(goal_diff_bins):
        goal_diff_bins[i] = round(len(fixtures) * bin / 100)

    bin_id = 0
    counter = 1
    counter_reset = goal_diff_bins[bin_id]

    for prediction in predictions.values():

        winner_goals, loser_goals = results_dict[bin_id]

        if prediction["elo_diff"] >= 0:
            prediction["home_goals"] = winner_goals
            prediction["away_goals"] = loser_goals

        else:
            prediction["home_goals"] = loser_goals
            prediction["away_goals"] = winner_goals

        if counter == counter_reset and bin_id < len(goal_diff_bins) - 1:
            bin_id += 1
            counter_reset = goal_diff_bins[bin_id]
            counter = 1

        else:
            counter += 1

    predictions = dict(sorted(predictions.items(),
                              key=lambda item: item[1]["display_id"]))

    for prediction in predictions.values():
        print(prediction["home_team"] + " " + str(prediction["home_goals"])
              + " : " + str(prediction["away_goals"]) + " "
              + str(prediction["away_team"]))


if __name__ == "__main__":
    # Load the underlying data.
    MATCHES = load_match_database("historical_results.csv")
    ELOS = load_elo_database("elo_ratings.txt")
    TEAMS = get_participating_teams(FIXTURES)

    # Limit analysis to the 30 most recent matches by participating teams.
    MATCHES = filter_match_database(MATCHES, TEAMS, CONVERGENCE_MATCHES)

    # Print result histograms of historical matches.
    result_histogram(MATCHES, ELOS)

    # Based on this analysis, we translate predicted goal differences into
    # predicted goals as follows ...
    RESULTS_FROM_GOAL_DIFF = {0: (1, 1),
                              1: (1, 0),
                              2: (2, 0),
                              3: (3, 0),
                              4: (4, 0)}

    # ... and use the following percentage bins for goal differences:
    GOAL_DIFF_BINS = [23, 36, 23, 12, 7]

    predict_fixtures(FIXTURES, ELOS, GOAL_DIFF_BINS, RESULTS_FROM_GOAL_DIFF)
