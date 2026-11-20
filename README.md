[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kuhugn/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kuhugn/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Trophy Rarity Forecaster

A make.fun **gta6-trophy-rarity-forecast-01** entry: for each of GTA VI's 55
trophies it forecasts the share of tracked players who will have earned it by
the target time, as a five-quantile band (q05, q25, q50, q75, q95) of the earn
rate in percent.

## Run it

```
python3 -P -s main.py
```

Python 3.13, standard library only, no network. The harness places each case
file in the input directory (`MAKEFUN_INPUT_DIR`, default `/input`); the
program reads every case there and writes one JSON line per case to stdout,
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`, deterministic and well
inside the per-case time and size caps.

## The model in one paragraph

An earn rate is a ratio — earners over tracked players — and launch week moves
both sides of it: new players keep syncing into the denominator while existing
cohorts play their way to each trophy in the numerator. The forecaster blends
two views in logit space: a trend view (a recency-weighted fit of logit(rate)
against log game age over the last day of distinct snapshots, each slope
shrunk toward its tier and field means by its own standard error) and a count
view (the same fit applied separately to log players and log earners, then
divided, which keeps the dilution mechanics explicit). The ensemble weights
the trend view 0.8 and the count view 0.2, keeps the wider of the two spreads
in each direction, narrows that spread for common trophies, and widens the
upper tail for still-rare ones, whose misses in backtests were almost all
upside surprises from the endgame unlock wave. Medians are disciplined at the
end: the platinum never sits above another trophy, and the story chapters
stay ordered first to last.
