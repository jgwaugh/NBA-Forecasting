import pickle
import random
from os.path import abspath, dirname
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

stats = "YR GP GS MIN FGM FGA FGP 3PM 3PA 3PP FTM FTA FTP OFF DEF TREB AST STL BLK PF TOV PTS GP GS DBLDBL TPLDBL 40P 20P 20AS Techs HOB AST_TO STL_TO FT_FGA W L WP OWS DWS WS GP GS TSP EFGP ORBP DRBP TRBP ASTP TOVP STLP BLKP USGP TOTSP PPR PPS ORtg DRtg PER"
cutoff_year = 1990


def multi_team_player_avg(l: List) -> NDArray:
    """
    If a player played on multiple teams during a season, take the average
    of his stats on each team

    Parameters
    ----------
    l : list
        List of player stats

    Returns
    -------
    numpy array
        Average of players statistics across teams if that player had a multiteam season

    """
    l_yr = l[0]
    l_other = l[1:]

    num_teams = len(l_other) // 57

    totals = np.array(l[1 : 21 * num_teams + 1]).reshape(num_teams, 21)
    misc = np.array(
        l[21 * num_teams + 1 : 21 * num_teams + 18 * num_teams + 1]
    ).reshape(num_teams, 18)
    advanced = np.array(l[21 * num_teams + 18 * num_teams + 1 :]).reshape(num_teams, -1)

    rv = np.array(l_yr)
    for stats in [totals, misc, advanced]:
        stats = np.mean(stats, axis=0)
        rv = np.hstack((rv, stats))
    return rv


def transform_to_array(info: List) -> NDArray:
    """Transforms the scraped data, in list format, into an array"""

    v_0 = info[0]
    if len(v_0) != 58:
        rv = multi_team_player_avg(v_0)
    else:
        rv = np.array(info[0])

    for i in range(1, len(info)):
        v = info[i]
        if len(v) == 58:
            v = np.array(v)
        else:
            v = multi_team_player_avg(v)

        rv = np.vstack((rv, v))
    return rv


def prepare_data(data: pd.DataFrame) -> List[Tuple[str, NDArray]]:
    """
    Gets data ready to feed to the NN

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe of player performance

    Returns
    -------
    list
        List of tuples corresponding to players + career data

    """

    players = data.PLAYER.unique()
    player_career = []
    for player in players:
        player_data = data[data.PLAYER == player]
        player_data = player_data.drop(
            ["PLAYER", "career_len", "start_yr", "YR"], axis=1
        ).values
        player_career.append((player, player_data))
    return player_career


if __name__ == "__main__":

    #################################################################################
    #
    # Load in the data
    #
    #################################################################################
    data_directory = Path(dirname(dirname(abspath(__file__)))).joinpath("scraping")

    all_names = []
    all_stats = []

    for yr in range(1980, 2023):
        yr_str = str(yr)
        name_str = yr_str + "_names.npy"
        numeric_str = yr_str + "_numeric.npy"

        name_path = data_directory.joinpath(name_str)
        numeric_path = data_directory.joinpath(numeric_str)

        numeric = transform_to_array(np.load(numeric_path, allow_pickle=True))
        names = np.load(name_path, allow_pickle=True)

        all_names.append(names)
        all_stats.append(numeric)

    all_names = np.hstack((all_names))
    all_stats = np.vstack((all_stats))

    df = pd.DataFrame(all_stats, columns=stats.split(" "))
    df.insert(0, "PLAYER", all_names)

    #################################################################################
    #
    # Scale the data for training
    #
    #################################################################################

    df_named = df[df.PLAYER != ""]

    career_len = (
        df_named.groupby(["PLAYER"])
        .agg({"YR": len})
        .rename(columns={"YR": "career_len"})
    )
    start_yr = (
        df_named.groupby(["PLAYER"]).agg({"YR": min}).rename(columns={"YR": "start_yr"})
    )
    career = career_len.join(start_yr)
    career["PLAYER"] = career.index
    career = career.reset_index(drop=True)

    df_named = df_named.merge(career, on="PLAYER", how="left")
    df_named = df_named.loc[:, ~df_named.columns.duplicated()]

    df_transformed = df_named.copy()

    columns_scale = df_named.columns[2:-2]
    scaler = MinMaxScaler()
    numeric_data = df_named.loc[:, columns_scale]
    scaled = scaler.fit_transform(numeric_data)

    df_transformed.loc[:, columns_scale] = scaled

    joblib.dump(scaler, "data/scaler")
    df_named.to_pickle("data/NBA_stats.pkl")
    df_transformed.to_pickle("data/NBA_stats_transformed.pkl")

    #################################################################################
    #
    # Order for the NN
    #
    #################################################################################

    recent_players = df_transformed[df_transformed.start_yr >= cutoff_year]
    older_players = df_transformed[df_transformed.start_yr < cutoff_year]

    # drop all players whose career was only one year
    older_players = older_players[older_players.career_len > 2]

    all_train = prepare_data(older_players)
    random.shuffle(all_train)

    # split the data into training and validation sets
    # training data is 85%, validation is 15%
    t_point = int(0.85 * len(all_train))
    train = all_train[:t_point]
    val = all_train[t_point:]

    prediction = prepare_data(recent_players)

    with open("data/train.pkl", "wb") as fp:  # Pickling train
        pickle.dump(train, fp)

    with open("data/val.pkl", "wb") as fp:  # Pickling val
        pickle.dump(val, fp)

    with open("data/prediction.pkl", "wb") as fp:  # Pickling prediction
        pickle.dump(prediction, fp)
