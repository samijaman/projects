# NBA Team Rankings Using Massey Ratings

A linear algebra project that uses the Massey ranking method to rank NBA teams and predict game outcomes based on regular-season results.

The project applies matrix methods to NBA game data to estimate a numerical strength rating for each team and evaluates how well those ratings predict future games.

## Project Overview

This project analyzes 1,231 games from the 2025–2026 NBA regular season across all 30 teams.

Using the Massey ranking method, game results are converted into a system of linear equations that estimates the relative strength of each NBA team.

The model was evaluated on held-out games using an 80/20 train-test split and achieved **68.5% accuracy** in predicting game winners.

## Methodology

The Massey ranking system represents team strength using the linear system:

**Mr = p**

where:

- **M** is a matrix representing games played between teams
- **r** is the vector of unknown team ratings
- **p** contains each team's cumulative point differential

For each game, the matrix is updated based on the two teams that played, while the point-differential vector records the scoring margin.

The resulting system is solved to obtain a numerical rating for each team.

A team with a higher Massey rating is predicted to defeat a team with a lower rating.

## Limiting the Effect of Blowouts

One challenge with using point differential is that unusually large wins can have too much influence on team ratings.

To reduce this effect, scoring margins were capped at **15 points**.

For example:

- A 7-point win is recorded as +7
- A 15-point win is recorded as +15
- A 30-point win is also recorded as +15

Applying this cap improved test-set prediction accuracy from approximately **63% to 68.5%**.

## Model Evaluation

The regular-season data was divided into an **80% training set and 20% test set**.

Massey ratings were calculated using the training games. For each game in the test set, the model predicted that the team with the higher calculated rating would win.

### Test Accuracy

**68.5%**

The project also trained a model using the full regular season and compared the resulting team rankings with external NBA power rankings.

## Technologies

- Python
- NumPy
- Pandas
- Jupyter Notebook
- Kaggle NBA game data

## Dataset

The project uses the **Historical NBA Data and Player Box Scores** dataset available on Kaggle.

The dataset contains historical NBA game and box-score information. This analysis focuses on regular-season games from the 2025–2026 season.

## Limitations

The Massey model provides a relatively simple measure of team strength and does not incorporate several factors that can affect actual game outcomes.

In particular:

- All games are weighted equally regardless of when they occurred.
- Player injuries are not represented.
- Trades and roster changes are not incorporated.
- Changes in team performance over the course of the season are not explicitly modeled.
- Predictions are based on team-level historical results rather than individual player information.

These limitations mean that a team's calculated rating may not fully represent its strength at a particular point in the season.

## Possible Improvements

Future versions of the model could:

- Give greater weight to recent games
- Incorporate player availability and injuries
- Account for roster changes
- Include home-court advantage
- Compare different point-differential caps
- Evaluate performance across multiple NBA seasons

## Files

`nba_massey_rankings.ipynb` — complete data processing, Massey ranking implementation, team rankings, and model evaluation.
