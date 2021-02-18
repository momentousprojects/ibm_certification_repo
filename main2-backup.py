# LOAD PACKAGES#
import csv
from typing import Optional, Tuple, Hashable
import time
import math

start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib as plt
import plotly.graph_objects as go
from pandas import Series
import math

sns.set_style("darkgrid")
sns.set_context("notebook")

# enable the input directory - come back to this when files loaded
"""
from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))
"""

# Set constants (k_factor to become dynamic later, constant for now)
mean_elo = float(1500)
elo_width = 400
k_factor = 64
pnl = 0.
existing_pnl = 0.

# Set the testing and betting parametres.
match_sample_start = 1
match_sample_size = 100000
edge_to_bet = 0.05
min_odds = 1.
max_odds = 1000.
n = 30 #minimum matches in player_count to bet
match_test_size = int(match_sample_size * 0.15)
test_range_start = int((match_sample_start + match_sample_size) - match_test_size)
print(test_range_start)
training_size = match_sample_size - match_test_size

# Clear working csv file: bet_record # W+ truncates a csv file on opening
filename = "bet_record.csv"
# truncate csv file
f = open(filename, "w+")
f.close()
# rewrite headers (agricultural workaround)
with open('bet_record.csv', mode='a') as bet_file:
    bet_writer = csv.writer(bet_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    bet_writer.writerow(["Date", "Winner", "Loser", "Winner Odds", "Loser Odds","Winner Elo","Loser Elo", "BetProfit", "RunningPnL"])

# Import matches from csv, using testing ranges defined earlier
all_matches = pd.read_csv('ELOSET_3.csv', header=0, skiprows=range(1, match_sample_start), nrows=match_sample_size)
all_matches['ID'] = all_matches['ID'].astype(int)
#all_matches.sort_values('ID',inplace=True)
# clean the dataset - spaces, drop cols, drop blank rows, rename cols, sort by date, print headers to terminal
all_matches.drop(labels=['Last1', 'Last2', 'Res1', 'Res2', 'Old1', 'Old2', 'New1', 'New2', 'Unnamed: 22', 'Unnamed: 21',
                         'Unnamed: 20'], inplace=True, axis=1)
all_matches.dropna(axis=0, subset=["Player 1"], inplace=True)
all_matches.columns = ['MatchID', 'Winner', 'Loser', 'Tournament', 'Date', 'Round', 'Surface', 'Result', 'WinnerOdds',
                       'LoserOdds', 'SetScore', 'VictoryMargin']
#all_matches = all_matches[['MatchID', 'Loser', 'Winner', 'Tournament', 'Date', 'Round', 'Surface', 'Result', 'WinnerOdds',
                       #'LoserOdds', 'SetScore', 'VictoryMargin']]

print(list(all_matches.columns.values))

#Find number of matches and print
'''count_matches = all_matches.count
print (count_matches)'''

# print((all_matches))
# Fix column headers in dataset
# all_matches.columns = all_matches.columns.str.replace('','_')

# transform player names to unique player numbers (transform back later)
'''le = LabelEncoder()
all_matches.Winner = le.fit_transform(all_matches.Winner)
all_matches.Loser = le.fit_transform(all_matches.Loser)'''

le = LabelEncoder()
le.fit(np.unique(np.concatenate((all_matches['Winner'].tolist(), all_matches['Loser'].tolist()),axis=0)))

all_matches['Winner'] = le.transform(all_matches.Winner)
all_matches['Loser'] = le.transform(all_matches['Loser'])

# FUNCTIONS

# calculate new elo ratings
def update_elo(winner_elo, loser_elo,k_factor):
    expected_win_a = expected_result(winner_elo, loser_elo)
    #expected_win_b = expected_result(loser_elo, winner_elo)
    #change_in_elo_a = k_factor * (1-expected_win_a)
    #change_in_elo_b = k_factor * (0-expected_win_b)
    winner_elo = winner_elo + (k_factor*(1-expected_win_a))
    loser_elo = loser_elo + (k_factor*(0-(1-expected_win_a)))
    print(expected_win_a," ",1-expected_win_a,winner_elo," --> ",winner_elo," ",loser_elo," ",loser_elo)
    return winner_elo, loser_elo

# find probability of Winner winning
def expected_result(elo_a, elo_b):
    expect_a = (1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width)))
    #print(expect_a)
    return expect_a

# update ELO at start of season, reverted 1/3 towards MEAN. Currently, static 1500 instead - revisit this
def update_for_new_season(elos):
    # diff_from_mean = elos - mean_elo
    # elos -= diff_from_mean/3
    elos = 1500
    return elos

#BET PLACING METHOD
def bet_gen(WinnerOdds, LoserOdds,w_elo,l_elo,expect_a, existing_pnl, edge_to_bet,w_id,l_id,match_id,match_date):
    expect_b = 1-expect_a
    #print(expect_a)
    if math.isnan(LoserOdds) == bool(False):
        implied_chance_a = float(1.0/WinnerOdds)
        implied_chance_b = float(1.0/LoserOdds)
        edge_a = float(expect_a - implied_chance_a)
        edge_b = float(expect_b - implied_chance_b)

        # Check whether some values are NaN
        '''if math.isnan(edge_a) == bool(True):
            print(match_id)'''

        if edge_a > edge_to_bet and WinnerOdds >= min_odds and WinnerOdds <= max_odds:
            profit = float(WinnerOdds - 1.0)
            newpnl = float("{:.2f}".format(existing_pnl + (WinnerOdds - 1)))  # update running PnL
            with open('bet_record.csv', mode='a') as bet_file:  # write bet to bet_record.csv (a=append)
                bet_writer = csv.writer(bet_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                bet_writer.writerow([match_date, w_id, l_id, float("{:.2f}".format(WinnerOdds)), float("{:.2f}".format(LoserOdds)), w_elo, l_elo, float("{:.2f}".format(profit)), newpnl])
            #return profit  # return the bet profit back to the loop
        elif edge_b > edge_to_bet and LoserOdds >= min_odds and LoserOdds <= max_odds:
            profit = float(-1.0)
            newpnl = float("{:.2f}".format(existing_pnl - 1))
            with open('bet_record.csv', mode='a') as bet_file:  # a=append
                bet_writer = csv.writer(bet_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                bet_writer.writerow([match_date, w_id, l_id, float("{:.2f}".format(WinnerOdds)), float("{:.2f}".format(LoserOdds)), w_elo, l_elo, float("{:.2f}".format(profit)), newpnl])
                #return profit # return bet loss back to the loop
        else:
            profit = float(0.0)
    else:
        edge_a = 0.0
        edge_b = 0.0
        profit = float(0.0)
        #return float(0)  # if no odds, return 0 back to the loop
    print(match_id," ",w_elo, " ",l_elo, " ",edge_a, " ", edge_b, " ", profit)
    return profit

# add new ELO and bet columns to all_matches dataframe and initialise with 0
all_matches["w_elo_before_match"] = float(0)
all_matches["l_elo_before_match"] = float(0)
all_matches["w_elo_after_match"] = float(0)
all_matches["l_elo_after_match"] = float(0)
all_matches["bet_result"] = float(0)
all_matches['w_running_count'] = float(0)
all_matches['l_running_count'] = float(0)
# elo_per_season = {} - for mean reversion later

# find number of players and create dataframe for current elos
n_players = len(le.classes_)
current_elos = np.ones(shape=(n_players)) * mean_elo
#create array for players match count - initialise with 0
current_match_count = np.ones(shape=(n_players)) * 0
#current_elos = ()

print(current_elos)

# Unique time/identifier
#all_matches['total_days'] = all_matches.Date

# create dataframe for all players and set their current elos from current_elos array
'''player_elos = pd.DataFrame(index=all_matches.MatchID.unique(),
                           columns=range(n_players))'''

print(n_players)
#set all players' current elos to 1500
#player_elos.iloc[0] = current_elos

#player_elos.to_csv('ambrose_test.csv')

# print(all_matches)
# print(player_elos)
previous = 0
# ELO GENERATOR LOOP

print(current_elos)

for ind in all_matches.index:
    # get current elo for each player
    # idx = row.Index
    w_id = all_matches['Winner'][ind]
    l_id = all_matches['Loser'][ind]
    w_odds = all_matches['WinnerOdds'][ind]
    l_odds = all_matches['LoserOdds'][ind]
    match_date = all_matches['MatchID'][ind]
    w_elo_before = current_elos[w_id]
    l_elo_before = current_elos[l_id]

    # update after the match result
    w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before,k_factor)

    if ind == 0:
        print(w_elo_after)
        print(l_elo_after)

    # save the new elos to the all_matches dataframe
    all_matches.at[ind, 'w_elo_before_match'] = w_elo_before
    all_matches.at[ind, 'l_elo_before_match'] = l_elo_before
    all_matches.at[ind, 'w_elo_after_match'] = w_elo_after
    all_matches.at[ind, 'l_elo_after_match'] = l_elo_after


    # save updated ELOs to player dataframe
    #match_date = all_matches['Date'][ind]
    #player_elos[w_id] = w_elo_after
    #player_elos[l_id] = l_elo_after
    current_elos[w_id] = w_elo_after
    current_elos[l_id] = l_elo_after
    #increase player match counts by 1
    current_match_count[w_id] += 1
    current_match_count[l_id] += 1
    all_matches.at[ind, 'w_running_count'] = current_match_count[w_id]
    all_matches.at[ind, 'l_running_count'] = current_match_count[l_id]

    #n = 100

    '''if current_elos[n] != previous:
       # print(current_elos[n])

    previous = current_elos[n]'''


all_matches.to_csv('Pre-betting test.csv')

#player_elos.to_csv('player_elos.csv')

# EVALUATION

# Reduce dataframe to MatchIDs after test_range_start
#all_matches[all_matches.WinnerOdds != 'NaN']
#all_matches = all_matches[all_matches.MatchID > test_range_start]
all_matches.drop(all_matches[all_matches.MatchID < test_range_start].index, inplace=True)
loss = 0
expected_list = []

all_matches.to_csv('new test1.csv')

# Loop through sample matches
# update log loss and bet each time
for ind in all_matches.index:
    #fetch required values from elo'd dataframe
    w_elo = all_matches['w_elo_before_match'][ind]
    l_elo = all_matches['l_elo_before_match'][ind]
    w_b_odds = all_matches['WinnerOdds'][ind]
    l_b_odds = all_matches['LoserOdds'][ind]
    w_b_id = all_matches['Winner'][ind]
    l_b_id = all_matches['Loser'][ind]
    match_id = all_matches['MatchID'][ind]
    match_date = all_matches['Date'][ind]
    w_r_c = all_matches['w_running_count'][ind]
    l_r_c = all_matches['l_running_count'][ind]
    #calculate historic chance of winner winning
    w_expected = expected_result(w_elo, l_elo)
    #add historic chance to list and calculate log loss of the list
    expected_list.append(w_expected)
    loss += np.log(w_expected)
    # calculate and save bet result if both players have 10 matches

    if w_r_c >= n and l_r_c >= n:
        new_bet = bet_gen(w_b_odds, l_b_odds, w_elo,l_elo,w_expected, existing_pnl, edge_to_bet, w_b_id, l_b_id,match_id,match_date)
        all_matches.at[ind, 'bet_result'] = new_bet
        pnl += float(new_bet)
    existing_pnl = float(pnl)
    #current_elos[w_id] = w_elo
    #current_elos[l_id] = l_elo

#Calculate total bets and ROI
total_bets = np.sum(all_matches['bet_result'] != 0)
roi = float(pnl / total_bets)

#Turn IDs back into names - not necessary but good for future referencing of the dataframe
all_matches.Winner = le.inverse_transform(all_matches.Winner)
all_matches.Loser = le.inverse_transform(all_matches.Loser)

#import bet_record, convert IDs back to names, write to NewBetRecord.csv
reinsert_names = pd.read_csv('bet_record.csv', header=0)
reinsert_names.Winner = le.inverse_transform(reinsert_names.Winner)
reinsert_names.Loser = le.inverse_transform(reinsert_names.Loser)
print(reinsert_names)
reinsert_names.to_csv('NewBetReport.csv')

all_matches.to_csv('All_Matches.csv')

#print(player_elos)

#Print evaluation results
print("You trained on", training_size, "matches and tested the strategy on", match_test_size, "matches. You placed ",
      total_bets, " bets.")
print("The total profit/loss was", float("{:.2f}".format(pnl)))
print("ROI:", "{0:.0000%}".format(roi))
print("The LOG LOSS was ", loss / match_test_size)
print(current_elos)

#write result to CSV
with open('para_log_1.csv', mode='a') as bet_file:  # write bet to bet_record.csv (a=append)
    bet_writer = csv.writer(bet_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    bet_writer.writerow([match_sample_size,match_test_size,k_factor,elo_width,edge_to_bet,n,min_odds,max_odds,total_bets,float("{:.2f}".format(pnl)),"{0:.00%}".format(roi),float("{:.2f}".format(time.time() - start_time))])

#write betting graph from new_bet_record
df = pd.read_csv('NewBetReport.csv')
fig = go.Figure(go.Scatter(x = df['Date'],y=df['RunningPnL'],name='PnL'))
fig.update_layout(title='PnL over time',plot_bgcolor='rgb(230, 230,230)',showlegend=True)
fig.show()

#Print time taken to execute
print("--- %s seconds ---" % (time.time() - start_time))

#Chart of logloss - not working yet
"""sns.distplot(expected_list, kde=False, bins=20)
plt.xlabel('Elo Expected Wins for Actual Winner')
plt.ylabel('Counts')

player_elos.fillna(method='ffill', inplace=True)
trans_dict = {i: 'player_{}'.format(i) for i in range(n_players)}
player_elos.rename(columns=trans_dict, inplace=True)
epoch = (player_elos.index)
player_elos['Date'] = pd.to_datetime(epoch)

player_elos.plot(x='Date',y=['Player_1','Player_2'])
plt.ylabel('Elo rating')
"""""